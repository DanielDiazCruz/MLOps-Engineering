"""Entrena varios modelos candidatos y los registra en MLflow.

Se entrenan dos baselines: LogisticRegression y RandomForestClassifier.
Cada uno se envuelve en un `Pipeline` de sklearn que incluye el encoder:

    Pipeline([
        ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), columnas_categoricas),
            ("num", "passthrough",                          columnas_numericas),
        ]),
        Classifier(...),
    ])

De esta forma el encoder queda **serializado junto con el modelo** en
MLflow. La API recibe features crudas (strings para categóricas, números
para numéricas) y el propio pipeline aplica el encoding en tiempo de
inferencia. Si llega un valor categórico nuevo, el flag
`handle_unknown="ignore"` lo descarta silenciosamente sin romper.

La métrica elegida para la promoción es `f1` porque el target está
fuertemente desbalanceado (~11% positivos en 130-US) y el costo clínico
de un falso negativo (un reingreso no detectado) es más alto que el de
un falso positivo. F1 balancea precision y recall sobre la clase
positiva, mientras que accuracy sería engañosa.
"""

from __future__ import annotations

import logging
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from pipeline.config import export_aws_env, load
from pipeline.db.connection import connect

logger = logging.getLogger(__name__)

# `infer_signature` de MLflow escala O(filas) cuando hay columnas `object`
# (las categóricas crudas). Inferirla sobre el train completo (decenas de
# miles de filas) tardaba ~11 min y era EL cuello de botella real de
# `t_train` — el fit en sí toma ~2 s. El esquema (nombres + tipos de
# columna) es idéntico con una muestra pequeña, así que la firma se infiere
# sobre estas pocas filas. Diagnóstico reproducible en scripts/profile_train.py.
_SIGNATURE_SAMPLE_ROWS = 200


def _read_clean(batch_id: str | None) -> pd.DataFrame:
    """Carga TODAS las filas limpias que ya tengan split asignado.

    Entrenamos sobre el acumulado completo (no solo el batch nuevo) para
    que el esquema de features que ve el OneHotEncoder sea consistente
    entre ejecuciones del DAG.
    """
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT row_hash, split, features, target FROM clean.diabetes_clean "
            "WHERE split IS NOT NULL"
        )
        rows = cur.fetchall()
    if not rows:
        raise ValueError(f"no hay filas limpias listas para entrenar (batch_id={batch_id!r})")
    # Expandimos JSONB → columnas planas para que pandas pueda manejar
    # los dtypes de cada feature de forma independiente.
    return pd.DataFrame(
        [
            {"row_hash": h, "split": s, "target": t, **f}
            for (h, s, f, t) in rows
        ]
    )


def _split_xy(df: pd.DataFrame, split: str, feature_cols: list[str]):
    """Devuelve (X, y) filtrando por el valor de la columna `split`."""
    sub = df[df["split"] == split]
    return sub[feature_cols].copy(), sub["target"].values


def _build_pipeline(estimator: Any, numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    """Construye el Pipeline (preprocesador + clasificador).

    El ColumnTransformer aplica:
      - OneHotEncoder a las categóricas, con `handle_unknown="ignore"`
        para tolerar valores nuevos en inferencia.
      - passthrough a las numéricas (sin transformación).
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", estimator),
    ])


_MODEL_ALIASES = {
    "lr": "logistic_regression",
    "logistic_regression": "logistic_regression",
    "rf": "random_forest",
    "random_forest": "random_forest",
}


def _candidates(seed: int) -> dict[str, Any]:
    """Define los modelos candidatos a entrenar.

    Ambos usan `class_weight="balanced"` porque el target tiene ~11% de
    positivos y sin esa pesa los modelos colapsan a predecir siempre 0.
    """
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, random_state=seed, class_weight="balanced",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=50, max_depth=6, random_state=seed, n_jobs=-1,
            class_weight="balanced",
        ),
    }


def run(batch_id: str | None = None, model: str | None = None) -> dict:
    """Entrena uno o varios candidatos, los registra en MLflow y devuelve el mejor.

    Args:
        batch_id: id del lote que se está procesando (solo informativo, se
            usa en el `run_name` de MLflow).
        model: nombre del candidato a entrenar (`"lr"`, `"rf"`, o sus
            aliases largos). Si es `None` (default) entrena ambos. Este
            argumento habilita el patrón de DAG con `t_train_lr` y
            `t_train_rf` en paralelo.

    Pasos:
      1. Configura MLflow (tracking URI + experimento).
      2. Lee los datos limpios + splits desde Postgres.
      3. Selecciona qué candidatos entrenar según el arg `model`.
      4. Para cada candidato: arma el pipeline, lo entrena, calcula
         métricas en val/test, loguea params/metrics/artifacts/modelo
         en MLflow y registra una nueva versión en el Model Registry.
      5. Retorna el dict del candidato ganador según `primary_metric`.
    """
    settings = load()
    export_aws_env(settings)
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.experiment_name)

    df = _read_clean(batch_id)
    feature_cols = [c for c in df.columns if c not in {"row_hash", "split", "target"}]

    # Determinamos qué columnas son numéricas vs. categóricas usando los
    # dtypes que vienen del JSONB. preprocess.py se encargó de castear
    # las categóricas a str, así que cualquier int/float aquí es genuino.
    train_df = df[df["split"] == "train"][feature_cols]
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    # Forzamos todas las numéricas a float64 para que la signature de
    # MLflow acepte tanto int como float desde los clientes (API/UI).
    # Si dejamos int64 en columnas como patient_nbr/encounter_id, MLflow
    # rechaza el downcasting de float64 que envía /predict.
    df[numeric_cols] = df[numeric_cols].astype("float64")

    X_train, y_train = _split_xy(df, "train", feature_cols)
    X_val, y_val = _split_xy(df, "val", feature_cols)
    X_test, y_test = _split_xy(df, "test", feature_cols)

    # Estado inicial del "ganador". Se irá actualizando si algún
    # candidato supera el primary_metric registrado.
    best = {"run_id": None, "version": None, "metric": -1.0, "model_name": None}

    # Resolvemos qué candidatos entrenar. Si `model` es None entrenamos
    # todos; si viene un alias específico filtramos a ese único candidato.
    all_candidates = _candidates(settings.random_seed)
    if model is None:
        selected = all_candidates
    else:
        key = _MODEL_ALIASES.get(model)
        if key is None:
            raise ValueError(
                f"modelo desconocido {model!r}; opciones válidas: {sorted(_MODEL_ALIASES)}"
            )
        selected = {key: all_candidates[key]}

    for name, estimator in selected.items():
        # Abrimos un run nuevo de MLflow por cada candidato.
        with mlflow.start_run(run_name=f"{name}-{batch_id or 'all'}") as run_:
            pipeline = _build_pipeline(estimator, numeric_cols, categorical_cols)
            pipeline.fit(X_train, y_train)

            # Predicciones para evaluación.
            y_val_pred = pipeline.predict(X_val)
            y_val_proba = (
                pipeline.predict_proba(X_val)[:, 1]
                if hasattr(pipeline, "predict_proba") else None
            )
            y_test_pred = pipeline.predict(X_test)

            # Métricas sobre validación + F1 sobre test (referencia
            # adicional). `primary_metric` se loguea como copia explícita
            # de la métrica elegida para la selección/promoción.
            metrics = {
                "accuracy": accuracy_score(y_val, y_val_pred),
                "precision": precision_score(y_val, y_val_pred, zero_division=0),
                "recall": recall_score(y_val, y_val_pred, zero_division=0),
                "f1": f1_score(y_val, y_val_pred, zero_division=0),
                "test_f1": f1_score(y_test, y_test_pred, zero_division=0),
            }
            if y_val_proba is not None and len(np.unique(y_val)) > 1:
                metrics["roc_auc"] = roc_auc_score(y_val, y_val_proba)
            metrics["primary_metric"] = metrics[settings.primary_metric]

            # Registramos hiperparámetros y métricas en MLflow.
            mlflow.log_params({
                "model_type": name,
                "batch_id": batch_id or "all",
                "seed": settings.random_seed,
                "n_numeric_features": len(numeric_cols),
                "n_categorical_features": len(categorical_cols),
            })
            mlflow.log_metrics(metrics)

            # La signature documenta las features CRUDAS que el modelo
            # espera. Dos precauciones específicas de este cluster:
            #   - No pasamos `input_example`: hace que MLflow recargue el
            #     modelo recién guardado para validarlo, y ese round-trip
            #     se cuelga bajo presión de memoria.
            #   - Fijamos `pip_requirements` a mano para evitar que MLflow
            #     intente detectar el entorno automáticamente (hace HTTP
            #     lookups que se cuelgan detrás del egress del cluster).
            #   - Inferimos la firma sobre una MUESTRA del train, no sobre
            #     el set completo: infer_signature escala O(filas) sobre
            #     columnas object y sobre 42k filas tardaba ~11 min (era el
            #     cuello de botella del task). El esquema es idéntico.
            sig_sample = X_train.head(_SIGNATURE_SAMPLE_ROWS)
            signature = infer_signature(sig_sample, pipeline.predict(sig_sample))
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                registered_model_name=settings.registered_model_name,
                signature=signature,
                pip_requirements=[
                    "mlflow",
                    "scikit-learn",
                    "pandas",
                    "numpy",
                ],
            )

            # Buscamos la versión recién registrada para retornarla.
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            versions = client.search_model_versions(f"run_id='{run_.info.run_id}'")
            version = versions[0].version if versions else None

            logger.info("run %s metrics=%s version=%s", name, metrics, version)

            # Si este candidato gana en la métrica principal, lo
            # marcamos como mejor hasta el momento.
            if metrics["primary_metric"] > best["metric"]:
                best = {
                    "run_id": run_.info.run_id,
                    "version": version,
                    "metric": metrics["primary_metric"],
                    "model_name": settings.registered_model_name,
                }

    logger.info("mejor candidato: %s", best)
    return best


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print(run())
