"""Servicio de inferencia FastAPI — regresión de precio inmobiliario."""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from prometheus_fastapi_instrumentator import Instrumentator

from api import db
from api.config import load
from api.metrics import (
    INFERENCE_LATENCY_SECONDS,
    PREDICTED_PRICE,
    PREDICTIONS_TOTAL,
    set_model_info,
)
from api.model_loader import LoadedModel, get_cache
from api.schemas import (
    HealthResponse,
    ModelInfo,
    PredictRequest,
    PredictResponse,
    TrainingHistoryResponse,
)

logger = logging.getLogger("api")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

settings = load()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        lm = get_cache().get()
        set_model_info(lm.name, lm.version, lm.alias)
        logger.info("model pre-loaded at startup: %s v%s", lm.name, lm.version)
    except Exception as e:  # noqa: BLE001
        logger.warning("startup model pre-load failed (will retry on first request): %s", e)
    yield


app = FastAPI(title="Real Estate Price API", version="1.0.0", lifespan=lifespan)

Instrumentator(
    excluded_handlers=["/metrics", "/health"],
).instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/model-info", response_model=ModelInfo)
def model_info() -> ModelInfo:
    try:
        lm = get_cache().get()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"no model loaded: {e}") from e
    set_model_info(lm.name, lm.version, lm.alias)
    return ModelInfo(
        model_name=lm.name,
        model_version=lm.version,
        model_alias=lm.alias,
        loaded_at=lm.loaded_at,
        cache_ttl_seconds=settings.model_cache_ttl_seconds,
    )


@app.post("/reload-model", response_model=ModelInfo)
def reload_model() -> ModelInfo:
    try:
        lm = get_cache().reload()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"reload failed: {e}") from e
    set_model_info(lm.name, lm.version, lm.alias)
    return ModelInfo(
        model_name=lm.name,
        model_version=lm.version,
        model_alias=lm.alias,
        loaded_at=lm.loaded_at,
        cache_ttl_seconds=settings.model_cache_ttl_seconds,
    )


@app.get("/training-history", response_model=TrainingHistoryResponse)
def training_history(limit: int = Query(default=20, ge=1, le=100)) -> TrainingHistoryResponse:
    """Devuelve el historial de decisiones de entrenamiento (RF9).

    Lee audit.training_history (poblada por la tarea t_notify del DAG) para
    que la UI muestre, por cada corrida, si se entrenó/promovió y por qué.
    """
    rows = db.fetch_training_history(limit=limit)
    return TrainingHistoryResponse(rows=rows)


# Mapa de tipos MLflow -> dtype de pandas, para coaccionar la entrada a la
# firma exacta del modelo (MLflow valida el esquema de forma estricta y, p. ej.,
# no convierte int64 a float64 automáticamente).
_MLFLOW_TO_PANDAS = {
    "double": "float64", "float": "float32",
    "long": "int64", "integer": "int32",
    "boolean": "bool", "string": "object",
}


def _coerce_to_schema(lm: LoadedModel, df: pd.DataFrame) -> pd.DataFrame:
    """Ajusta los dtypes del DataFrame a la firma de entrada del modelo.

    Así un cliente puede mandar `bed` como int o `prev_sold_year` como float y
    la API los convierte a lo que el modelo espera (double/long), evitando que
    la validación estricta de esquema de MLflow rechace la petición.
    """
    try:
        schema = lm.model.metadata.get_input_schema()
        if schema is None:
            return df
        mapping = {
            col.name: _MLFLOW_TO_PANDAS[col.type.name]
            for col in schema.inputs
            if col.name in df.columns and col.type.name in _MLFLOW_TO_PANDAS
        }
        return df.astype(mapping) if mapping else df
    except Exception:  # noqa: BLE001
        return df  # si no hay firma utilizable, dejamos que el modelo decida


def _predict_price(lm: LoadedModel, features: dict) -> float:
    """Ejecuta el modelo de regresión sobre un dict de features y devuelve el precio.

    El modelo es un Pipeline de sklearn que incluye su propio OneHotEncoder
    (`handle_unknown="infrequent_if_exist"`), por lo que la API solo pasa las
    features crudas y las categorías nuevas se manejan sin fallar.
    """
    df = _coerce_to_schema(lm, pd.DataFrame([features]))
    with INFERENCE_LATENCY_SECONDS.time():
        raw = lm.model.predict(df)
    value = raw[0] if hasattr(raw, "__len__") else raw
    return float(value)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    try:
        lm = get_cache().get()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"no model available: {e}") from e
    set_model_info(lm.name, lm.version, lm.alias)

    request_id = str(uuid.uuid4())
    started = time.perf_counter()
    try:
        prediction = _predict_price(lm, req.features)
    except Exception as e:  # noqa: BLE001
        logger.exception("prediction failure")
        latency_ms = (time.perf_counter() - started) * 1000.0
        PREDICTIONS_TOTAL.labels(status="error").inc()
        # Registramos también las inferencias fallidas (RF8) para auditarlas.
        db.log_inference(
            request_id=request_id,
            input_payload=req.features,
            prediction=None,
            model_name=lm.name,
            model_version=lm.version,
            latency_ms=latency_ms,
            status="error",
            error=str(e),
        )
        raise HTTPException(status_code=400, detail=f"invalid features: {e}") from e

    latency_ms = (time.perf_counter() - started) * 1000.0
    PREDICTIONS_TOTAL.labels(status="ok").inc()
    PREDICTED_PRICE.observe(prediction)

    db.log_inference(
        request_id=request_id,
        input_payload=req.features,
        prediction=prediction,
        model_name=lm.name,
        model_version=lm.version,
        latency_ms=latency_ms,
        status="ok",
    )

    return PredictResponse(
        request_id=request_id,
        prediction=prediction,
        model_name=lm.name,
        model_version=lm.version,
        model_alias=lm.alias,
        processing_time_ms=latency_ms,
    )
