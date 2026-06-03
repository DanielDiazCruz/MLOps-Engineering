"""Esquemas Pydantic usados por la API de inferencia (regresión de precio).

La entrada se modela como un diccionario abierto de features
(`Dict[str, float | int | str]`) porque la tabla `clean.properties_clean`
guarda las features como JSONB. El modelo es un `Pipeline` de sklearn cuyo
primer paso es un `ColumnTransformer` entrenado sobre esas features, así que
acepta un dict siempre que las llaves coincidan con las del entrenamiento
(bed, bath, acre_lot, house_size, prev_sold_year + status, city, state,
zip_code). El target es el **precio** (regresión), así que la respuesta es un
número continuo, no una clase.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Payload aceptado por POST /predict."""

    features: Dict[str, Any] = Field(
        ...,
        description="Diccionario de features de la propiedad (mismas usadas al entrenar).",
    )


class PredictResponse(BaseModel):
    """Respuesta de POST /predict (regresión).

    Campos mínimos exigidos por el enunciado: predicción, modelo,
    versión/alias y tiempo de procesamiento. En regresión `prediction` es el
    **precio estimado** (float, dólares).
    """

    request_id: str
    prediction: float
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


class TrainingHistoryRow(BaseModel):
    """Una fila del historial de entrenamiento (audit.training_history, RF9)."""

    id: int
    executed_at: Optional[str] = None
    batch_id: Optional[str] = None
    n_records_batch: Optional[int] = None
    n_records_total: Optional[int] = None
    decision: Optional[str] = None
    decision_reason: Optional[str] = None
    trained: bool = False
    promoted: bool = False
    promotion_reason: Optional[str] = None
    candidate_metrics: Optional[Dict[str, Any]] = None
    champion_metrics: Optional[Dict[str, Any]] = None
    drift: Optional[Dict[str, Any]] = None
    new_categories: Optional[Dict[str, Any]] = None
    mlflow_model_version: Optional[str] = None


class TrainingHistoryResponse(BaseModel):
    """Respuesta de GET /training-history (RF9)."""

    rows: List[TrainingHistoryRow]
