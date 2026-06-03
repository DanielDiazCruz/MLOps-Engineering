"""Perfila las fases de `t_train` para ubicar el cuello de botella real.

Mide por separado lectura / fit / predicción para responder DÓNDE se van
los ~17 min del entrenamiento (que en los logs aparecen como un hueco sin
trazas entre el inicio del task y el upload a MLflow).

Pensado para ejecutarse DENTRO de un pod que tenga el paquete `pipeline`
instalado y acceso a Postgres (p. ej. `airflow-scheduler-0`). NO sube nada
a MLflow ni escribe en la DB; solo lee `clean.diabetes_clean` y entrena en
memoria para cronometrar.

Para que las cifras sean limpias, correrlo cuando el nodo esté ocioso (sin
un `t_train` del DAG corriendo en paralelo: es un solo nodo y contienden).

Uso:
    python profile_train.py                 # baseline: denso, todo el train
    python profile_train.py --sparse        # OneHotEncoder sparse_output=True
    python profile_train.py --sample 10000  # muestrea 10k filas de train
    python profile_train.py --sparse --max-iter 200
"""

from __future__ import annotations

import argparse
import logging
import resource
import time

import numpy as np
from mlflow.models.signature import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from pipeline.config import load
from pipeline.train import _read_clean, _split_xy


def _maxrss_mb() -> float:
    """Pico de memoria residente del proceso (ru_maxrss viene en KB en Linux)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _phase(label: str, fn):
    """Ejecuta `fn`, cronometra y reporta tiempo + memoria pico acumulada."""
    t0 = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t0
    print(f"[{dt:8.2f}s] {label:42s} maxRSS={_maxrss_mb():7.0f} MB")
    return out, dt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sparse", action="store_true",
                    help="OneHotEncoder sparse_output=True (matriz dispersa)")
    ap.add_argument("--sample", type=int, default=0,
                    help="entrena solo con las primeras N filas de train (0 = todas)")
    ap.add_argument("--max-iter", type=int, default=1000,
                    help="max_iter de LogisticRegression (default 1000, igual que prod)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING)
    settings = load()
    print(f"=== profile_train  modo={'SPARSE' if args.sparse else 'DENSE'} "
          f"sample={args.sample or 'todas'} max_iter={args.max_iter} ===")

    # --- Fase 1: lectura del acumulado completo desde Postgres (JSONB -> df)
    df, _ = _phase("read_clean (todo el acumulado)", lambda: _read_clean(None))
    print(f"   filas={len(df)}  splits={df['split'].value_counts().to_dict()}")

    feature_cols = [c for c in df.columns if c not in {"row_hash", "split", "target"}]
    train_df = df[df["split"] == "train"][feature_cols]
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]
    df[numeric_cols] = df[numeric_cols].astype("float64")
    onehot_dim = int(sum(df[c].nunique() for c in categorical_cols))
    print(f"   numeric={len(numeric_cols)} categorical={len(categorical_cols)} "
          f"onehot_dim~{onehot_dim} (cols del modelo si es denso)")

    X_train, y_train = _split_xy(df, "train", feature_cols)
    X_val, y_val = _split_xy(df, "val", feature_cols)
    X_test, y_test = _split_xy(df, "test", feature_cols)
    if args.sample and args.sample < len(X_train):
        X_train = X_train.iloc[: args.sample]
        y_train = y_train[: args.sample]
        print(f"   train muestreado -> {len(X_train)} filas")

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=args.sparse),
             categorical_cols),
            ("num", "passthrough", numeric_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    clf = LogisticRegression(
        max_iter=args.max_iter, random_state=settings.random_seed, class_weight="balanced",
    )
    pipe = Pipeline([("preprocessor", pre), ("classifier", clf)])

    # --- Fase 2: fit (lo que sospechamos que domina)
    _phase(f"fit ({len(X_train)} filas)", lambda: pipe.fit(X_train, y_train))
    print(f"   n_iter={getattr(clf, 'n_iter_', None)} "
          f"(si ~max_iter, NO convergió)")

    # --- Fase 3: predicciones (val/test) e infer_signature (predice sobre train)
    _phase("predict val+test", lambda: (pipe.predict(X_val), pipe.predict(X_test)))
    _phase("infer_signature (predict train)",
           lambda: infer_signature(X_train, pipe.predict(X_train)))

    val_f1 = f1_score(y_val, pipe.predict(X_val), zero_division=0)
    print(f"   val_f1={val_f1:.4f}")


if __name__ == "__main__":
    main()
