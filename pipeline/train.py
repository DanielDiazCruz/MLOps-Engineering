"""Entrena modelos de regresión de PRECIO y los registra en MLflow.

Se entrenan dos candidatos y se elige el mejor por MAE de validación:
  - `ridge`: baseline lineal regularizado (rápido).
  - `hist_gbr`: HistGradientBoostingRegressor (fuerte en tabular).

Cada candidato es un `Pipeline` de sklearn:

    TransformedTargetRegressor(                # log1p(price) -> expm1
        regressor = Pipeline([
            ColumnTransformer([
                ("num", StandardScaler(),          numéricas),
                ("cat", OneHotEncoder(...),        categóricas),
            ]),
            Regressor(...),
        ]),
        func=log1p, inverse_func=expm1,
    )

El encoder queda serializado junto al modelo, así la API recibe features
crudas. `OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency,
max_categories)` maneja categorías nuevas y de alta cardinalidad agrupando
las raras en un bucket "infrequent" (RF3). El target se modela en escala
log porque el precio es muy sesgado; las métricas se reportan en la escala
original (dólares).

Métrica principal de promoción: **MAE** (menor es mejor).
"""

from __future__ import annotations

import logging
import tempfile
from typing import Any

import matplotlib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # backend sin display, apto para el cluster
import matplotlib.pyplot as plt  # noqa: E402
from mlflow.models.signature import infer_signature  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.compose import TransformedTargetRegressor  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

from pipeline.config import export_aws_env, load  # noqa: E402
from pipeline.db.connection import connect  # noqa: E402
from pipeline.preprocess import CATEGORICAL_FEATURES, NUMERIC_FEATURES  # noqa: E402

logger = logging.getLogger(__name__)

# infer_signature de MLflow escala O(filas) sobre columnas object; se infiere
# sobre una muestra pequeña (esquema idéntico) para no convertirla en el
# cuello de botella del task. Ver scripts/profile_train.py.
_SIGNATURE_SAMPLE_ROWS = 200
# Tope de filas de entrenamiento: los lotes de la API son enormes (100k+),
# así que muestreamos para acotar tiempo y memoria del fit en el nodo.
_MAX_TRAIN_ROWS = 150_000


def _read_clean() -> pd.DataFrame:
    """Carga las filas limpias con split asignado desde clean.properties_clean."""
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT split, features, target FROM clean.properties_clean "
            "WHERE split IS NOT NULL"
        )
        rows = cur.fetchall()
    if not rows:
        raise ValueError("no hay filas limpias con split para entrenar")
    return pd.DataFrame([{"split": s, "target": float(t), **f} for s, f, t in rows])


def _xy(df: pd.DataFrame, split: str):
    sub = df[df["split"] == split]
    return sub[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy(), sub["target"].to_numpy()


def _build_pipeline(estimator: Any) -> TransformedTargetRegressor:
    """Pipeline preprocesador + regresor, con target en escala log."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(
                handle_unknown="infrequent_if_exist",
                min_frequency=0.01,
                max_categories=25,
                sparse_output=False,
            ), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    inner = Pipeline([("preprocessor", preprocessor), ("regressor", estimator)])
    return TransformedTargetRegressor(regressor=inner, func=np.log1p, inverse_func=np.expm1)


def _candidates(seed: int) -> dict[str, Any]:
    """Modelos candidatos a entrenar."""
    return {
        "ridge": Ridge(alpha=1.0, random_state=seed),
        "hist_gbr": HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.08, max_depth=None,
            l2_regularization=1.0, random_state=seed,
        ),
    }


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> dict[str, float]:
    """MAE / RMSE / MAPE / R² con prefijo (val_ / test_)."""
    return {
        f"{prefix}mae": float(mean_absolute_error(y_true, y_pred)),
        f"{prefix}rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        f"{prefix}mape": float(mean_absolute_percentage_error(y_true, y_pred)),
        f"{prefix}r2": float(r2_score(y_true, y_pred)),
    }


def _log_plots(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Genera y registra artefactos: predicho-vs-real y residuales (RF5)."""
    with tempfile.TemporaryDirectory() as tmp:
        # Predicho vs real
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(y_true, y_pred, s=4, alpha=0.3)
        lim = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax.plot(lim, lim, "r--", linewidth=1)
        ax.set_xlabel("precio real"); ax.set_ylabel("precio predicho")
        ax.set_title("Predicho vs Real (val)")
        fig.tight_layout(); fig.savefig(f"{tmp}/pred_vs_real.png", dpi=110); plt.close(fig)

        # Residuales
        resid = y_pred - y_true
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(resid, bins=60)
        ax.set_xlabel("residual (pred - real)"); ax.set_ylabel("frecuencia")
        ax.set_title("Distribución de residuales (val)")
        fig.tight_layout(); fig.savefig(f"{tmp}/residuals.png", dpi=110); plt.close(fig)

        mlflow.log_artifacts(tmp, artifact_path="plots")


def run(batch_id: str | None = None, reason: str | None = None) -> dict:
    """Entrena los candidatos, registra en MLflow y devuelve el mejor (por MAE).

    Args:
        batch_id: lote que disparó el entrenamiento (informativo).
        reason: motivo por el que se decidió entrenar (RF5), se loguea como tag.
    """
    settings = load()
    export_aws_env(settings)
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.experiment_name)

    df = _read_clean()
    X_train, y_train = _xy(df, "train")
    X_val, y_val = _xy(df, "val")
    X_test, y_test = _xy(df, "test")

    # Muestreo del train si excede el tope (lotes de la API son enormes).
    if len(X_train) > _MAX_TRAIN_ROWS:
        idx = np.random.RandomState(settings.random_seed).choice(
            len(X_train), _MAX_TRAIN_ROWS, replace=False
        )
        X_train = X_train.iloc[idx]
        y_train = y_train[idx]
        logger.info("train muestreado a %d filas", _MAX_TRAIN_ROWS)

    best = {"run_id": None, "version": None, "metric": float("inf"),
            "model_name": settings.registered_model_name, "metrics": None, "model_type": None}

    for name, estimator in _candidates(settings.random_seed).items():
        with mlflow.start_run(run_name=f"{name}-{batch_id or 'all'}") as run_:
            pipe = _build_pipeline(estimator)
            pipe.fit(X_train, y_train)

            y_val_pred = pipe.predict(X_val)
            y_test_pred = pipe.predict(X_test)
            metrics = {**_metrics(y_val, y_val_pred, "val_"),
                       **_metrics(y_test, y_test_pred, "test_")}
            # primary_metric = MAE de validación (lo usa promote.py).
            metrics["primary_metric"] = metrics["val_mae"]

            mlflow.log_params({
                "model_type": name,
                "batch_id": batch_id or "all",
                "seed": settings.random_seed,
                "n_train": len(X_train),
                "n_numeric": len(NUMERIC_FEATURES),
                "n_categorical": len(CATEGORICAL_FEATURES),
                "target_transform": "log1p",
            })
            mlflow.log_metrics(metrics)
            if reason:
                mlflow.set_tag("training_reason", reason)
            _log_plots(y_val, y_val_pred)

            # Firma sobre muestra (ver nota en _SIGNATURE_SAMPLE_ROWS).
            sig_sample = X_train.head(_SIGNATURE_SAMPLE_ROWS)
            signature = infer_signature(sig_sample, pipe.predict(sig_sample))
            mlflow.sklearn.log_model(
                sk_model=pipe,
                artifact_path="model",
                registered_model_name=settings.registered_model_name,
                signature=signature,
                pip_requirements=["mlflow", "scikit-learn", "pandas", "numpy"],
            )

            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            versions = client.search_model_versions(f"run_id='{run_.info.run_id}'")
            version = versions[0].version if versions else None

            logger.info("candidato %s metrics=%s version=%s", name, metrics, version)

            # Mejor = menor MAE de validación (en regresión menor es mejor).
            if metrics["primary_metric"] < best["metric"]:
                best = {
                    "run_id": run_.info.run_id,
                    "version": version,
                    "metric": metrics["primary_metric"],
                    "model_name": settings.registered_model_name,
                    "metrics": metrics,
                    "model_type": name,
                }

    logger.info("mejor candidato: %s", best)
    return best


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print(run())
