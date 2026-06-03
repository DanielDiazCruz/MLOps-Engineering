"""RF4/RF9: registra cada corrida del DAG en audit.training_history.

Una fila por ejecución del pipeline. Captura la decisión de entrenar y la de
promover, con su razón, las métricas del candidato y del champion, las señales
que motivaron la decisión (esquema, categorías nuevas, drift, validaciones de
calidad) y los identificadores de MLflow. Es la fuente del "Historial de
entrenamiento" que muestra la UI de Streamlit (RF9).
"""

from __future__ import annotations

import logging

from psycopg2.extras import Json

from pipeline.db.connection import connect

logger = logging.getLogger(__name__)


def _maybe_json(value):
    """Envuelve en JSONB salvo que sea None (para guardar NULL real)."""
    return Json(value) if value is not None else None


def record_run(
    *,
    batch_id: str | None,
    dag_run_id: str | None,
    n_records_batch: int | None,
    n_records_total: int | None,
    schema_info: dict | None,
    new_categories: dict | None,
    validations: dict | None,
    drift: dict | None,
    decision: str,
    decision_reason: str | None,
    trained: bool,
    candidate_metrics: dict | None = None,
    promoted: bool = False,
    promotion_reason: str | None = None,
    champion_metrics: dict | None = None,
    mlflow_run_id: str | None = None,
    mlflow_model_version: str | None = None,
) -> int:
    """Inserta una fila de auditoría y devuelve su id."""
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO audit.training_history (
                batch_id, dag_run_id, n_records_batch, n_records_total,
                schema_info, new_categories, validations, drift,
                decision, decision_reason, trained,
                candidate_metrics, promoted, promotion_reason,
                champion_metrics, mlflow_run_id, mlflow_model_version
            ) VALUES (
                %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s
            )
            RETURNING id
            """,
            (
                batch_id, dag_run_id, n_records_batch, n_records_total,
                _maybe_json(schema_info), _maybe_json(new_categories),
                _maybe_json(validations), _maybe_json(drift),
                decision, decision_reason, trained,
                _maybe_json(candidate_metrics), promoted, promotion_reason,
                _maybe_json(champion_metrics), mlflow_run_id, mlflow_model_version,
            ),
        )
        new_id = int(cur.fetchone()[0])

    logger.info(
        "auditoría registrada: id=%s batch=%s decision=%s trained=%s promoted=%s",
        new_id, batch_id, decision, trained, promoted,
    )
    return new_id


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print("audit.record_run se invoca desde el DAG (tarea t_notify)")
