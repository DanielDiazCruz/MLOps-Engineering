"""Esquemas Pydantic usados por la API de inferencia.

La entrada se modela como un diccionario abierto de features
(`Dict[str, float | int | str]`) porque la tabla `clean.diabetes_clean`
guarda las features como JSONB y el conjunto exacto de columnas depende
del dataset (Pima 8 features vs. 130-US ~40 features).

El modelo en sí es un `Pipeline` de sklearn cuyo primer paso es un
`ColumnTransformer` entrenado sobre ese mismo JSONB, por lo que acepta
un dict siempre que las llaves coincidan con las features que se vieron
en el entrenamiento.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Payload aceptado por POST /predict."""

    features: Dict[str, Any] = Field(
        ...,
        description="Diccionario de features que coincide con las usadas al entrenar.",
    )


class PredictResponse(BaseModel):
    """Respuesta de POST /predict.

    Cumple con los campos mínimos exigidos por el enunciado: predicción,
    score, modelo, versión/alias y tiempo de procesamiento.
    """

    request_id: str
    prediction: int
    score: Optional[float] = None
    model_name: str
    model_version: str
    model_alias: str
    processing_time_ms: float


class ModelInfo(BaseModel):
    """Metadatos del modelo actualmente cargado en memoria (GET /model-info)."""

    model_name: str
    model_version: str
    model_alias: str
    loaded_at: float
    cache_ttl_seconds: int


class HealthResponse(BaseModel):
    """Respuesta de GET /health (liveness/readiness probe)."""

    status: str = "ok"
