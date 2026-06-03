"""Acceso a PostgreSQL desde la API: log de inferencias e historial.

Cada llamada a /predict inserta una fila en `inference.predictions` (RF8):
entrada, predicción (precio), versión del modelo, estado y error si lo hubo.
La sección de "Historial de entrenamiento" de la UI (RF9) se sirve leyendo
`audit.training_history`.

Política de errores del log de inferencia: si el INSERT falla (Postgres caído,
transacción abortada, etc.) se loguea el error pero **NO** se interrumpe la
respuesta al cliente. Servir una predicción siempre es prioritario sobre
dejarla registrada.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import psycopg2
from psycopg2.extras import Json, RealDictCursor

from api.config import load

logger = logging.getLogger(__name__)

# Sentencia INSERT plantilla. Los placeholders coinciden con el orden de
# argumentos de log_inference(). `prediction` es DOUBLE (precio) tras la
# migración de esquema en pipeline/db/migrations.py.
INSERT_SQL = """
INSERT INTO inference.predictions
    (request_id, input_payload, prediction, model_name, model_version, status, error, latency_ms)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
"""


def log_inference(
    request_id: str,
    input_payload: Dict[str, Any],
    prediction: float | None,
    model_name: str,
    model_version: str,
    latency_ms: float,
    status: str = "ok",
    error: str | None = None,
) -> None:
    """Inserta una fila en inference.predictions.

    Cualquier excepción se captura y se loguea como warning. La función nunca
    relanza para que la API responda al cliente aunque la BD esté
    momentáneamente fuera de servicio (modo degradado).
    """
    try:
        # Una conexión por inserción: simple y suficiente para el volumen
        # esperado; si la carga sube se puede cambiar por un pool sin tocar
        # el contrato.
        with psycopg2.connect(load().pg_dsn) as conn, conn.cursor() as cur:
            cur.execute(
                INSERT_SQL,
                (
                    request_id,
                    Json(input_payload),
                    None if prediction is None else float(prediction),
                    model_name,
                    model_version,
                    status,
                    error,
                    float(latency_ms),
                ),
            )
    except Exception as e:  # noqa: BLE001
        logger.warning("no se pudo registrar la inferencia %s: %s", request_id, e)


# Columnas que expone el historial de entrenamiento a la UI (RF9).
_HISTORY_SQL = """
SELECT id, executed_at, batch_id, n_records_batch, n_records_total,
       decision, decision_reason, trained, promoted, promotion_reason,
       candidate_metrics, champion_metrics, drift, new_categories,
       mlflow_model_version
FROM audit.training_history
ORDER BY id DESC
LIMIT %s
"""


def fetch_training_history(limit: int = 20) -> List[Dict[str, Any]]:
    """Devuelve las últimas corridas registradas en audit.training_history.

    Se serializa `executed_at` a ISO-8601 para que viaje como JSON. Si la
    consulta falla, devuelve lista vacía (la UI muestra "sin datos") en vez
    de romper la página.
    """
    try:
        with psycopg2.connect(load().pg_dsn) as conn, \
                conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(_HISTORY_SQL, (int(limit),))
            rows = cur.fetchall()
    except Exception as e:  # noqa: BLE001
        logger.warning("no se pudo leer el historial de entrenamiento: %s", e)
        return []

    result: List[Dict[str, Any]] = []
    for r in rows:
        row = dict(r)
        ts = row.get("executed_at")
        row["executed_at"] = ts.isoformat() if ts is not None else None
        result.append(row)
    return result
