"""Logger de inferencias hacia inference.predictions.

Cada llamada a /predict inserta una fila aquí para habilitar
reentrenamiento, monitoreo de deriva y auditoría. La tabla se crea en
`pipeline/db/migrations.py`.

Política de errores: si el INSERT falla (Postgres caído, transacción
abortada, etc.) se loguea el error pero **NO** se interrumpe la
respuesta al cliente. Servir una predicción siempre es prioritario
sobre dejarla registrada.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

import psycopg2
from psycopg2.extras import Json

from api.config import load

logger = logging.getLogger(__name__)

# Sentencia INSERT plantilla. Los placeholders coinciden con el orden
# de argumentos de log_inference().
INSERT_SQL = """
INSERT INTO inference.predictions
    (request_id, input_payload, prediction, score, model_name, model_version, latency_ms)
VALUES (%s, %s, %s, %s, %s, %s, %s)
"""


def log_inference(
    request_id: str,
    input_payload: Dict[str, Any],
    prediction: int,
    score: float | None,
    model_name: str,
    model_version: str,
    latency_ms: float,
) -> None:
    """Inserta una fila en inference.predictions.

    Cualquier excepción se captura y se loguea como warning. La función
    nunca relanza para que la API responda al cliente aunque la BD esté
    momentáneamente fuera de servicio (modo degradado).
    """
    try:
        # Abrimos una conexión por inserción. Es simple y suficiente
        # dado el volumen esperado; si la carga aumenta se puede
        # cambiar por un pool sin afectar el contrato.
        with psycopg2.connect(load().pg_dsn) as conn, conn.cursor() as cur:
            cur.execute(
                INSERT_SQL,
                (
                    request_id,
                    Json(input_payload),
                    int(prediction),
                    None if score is None else float(score),
                    model_name,
                    model_version,
                    float(latency_ms),
                ),
            )
    except Exception as e:  # noqa: BLE001
        logger.warning("no se pudo registrar la inferencia %s: %s", request_id, e)
