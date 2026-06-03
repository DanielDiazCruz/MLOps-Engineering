"""Compara el candidato recién entrenado contra el champion actual y, si gana, lo promueve.

El modelo productivo se identifica mediante un alias en el Model Registry
de MLflow (por defecto `champion`). Desde MLflow 2.9 los `stages`
(Production/Staging) están deprecados; los aliases son la forma
recomendada. La métrica de selección se lee desde el run que respaldó la
versión candidata (se guarda como `primary_metric`).
"""

from __future__ import annotations

import logging

import mlflow
from mlflow.tracking import MlflowClient

from pipeline.config import export_aws_env, load

logger = logging.getLogger(__name__)


def _client() -> MlflowClient:
    """Crea un MlflowClient con el tracking URI y las credenciales S3 ya configurados."""
    settings = load()
    export_aws_env(settings)
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    return MlflowClient()


def _metric(client: MlflowClient, run_id: str, name: str) -> float:
    """Lee una métrica puntual desde un run específico de MLflow."""
    return float(client.get_run(run_id).data.metrics.get(name, float("-inf")))


def compare(candidate: dict) -> dict:
    """Compara el candidato vs. el champion vigente y devuelve un resumen.

    El parámetro `candidate` es el dict que retorna `train.run()` con la
    forma {run_id, version, metric, model_name}.

    Regla de decisión:
      - Si no existe champion → promover.
      - Si existe champion → promover solo si el candidato lo supera
        estrictamente en `primary_metric`.
    """
    settings = load()
    client = _client()

    raw_metric = candidate.get("metric")
    candidate_metric = float(raw_metric) if raw_metric is not None else 0.0

    # Intentamos resolver el champion actual. Si no existe (primer run
    # del proyecto) capturamos la excepción y dejamos campos en None.
    try:
        champion_version = client.get_model_version_by_alias(
            settings.registered_model_name, settings.champion_alias
        )
        champion_metric = _metric(client, champion_version.run_id, "primary_metric")
        champion_v = champion_version.version
    except Exception:
        champion_v = None
        champion_metric = None

    decision = "promote" if champion_metric is None or candidate_metric > champion_metric else "keep"
    summary = {
        "candidate_version": candidate.get("version"),
        "candidate_metric": candidate_metric,
        "champion_version": champion_v,
        "champion_metric": champion_metric,
        "decision": decision,
    }
    logger.info("comparación: %s", summary)
    return summary


def _best(candidates: list[dict]) -> dict:
    valid = [c for c in candidates if c and c.get("version") is not None]
    if not valid:
        raise ValueError("no valid candidates to promote (all missing 'version')")
    return max(valid, key=lambda c: float(c.get("metric") or float("-inf")))


def promote_best(candidates: list[dict]) -> dict:
    """Pick the best candidate by `metric` and promote it via `promote()`.

    Used by the Airflow DAG when several `train_*` tasks run in parallel and
    feed their results into a single promotion step. The winner is the one
    with the highest `primary_metric` (F1 by default).
    """
    winner = _best(candidates)
    logger.info(
        "selected best candidate: run_id=%s version=%s metric=%.4f (out of %d)",
        winner.get("run_id"), winner.get("version"), winner.get("metric"), len(candidates),
    )
    result = promote(winner)
    result["selected_run_id"] = winner.get("run_id")
    result["selected_version"] = winner.get("version")
    result["candidates_considered"] = len(candidates)
    return result


def promote(candidate: dict) -> dict:
    """Aplica la decisión de `compare()`: si gana el candidato, mueve el alias champion."""
    settings = load()
    client = _client()

    decision = compare(candidate)
    if decision["decision"] == "promote" and candidate.get("version"):
        # set_registered_model_alias es atómico: re-apunta el alias
        # `champion` a la nueva versión sin necesidad de borrar el
        # anterior. La API recogerá el cambio en el próximo refresh
        # del cache o al recibir POST /reload-model.
        client.set_registered_model_alias(
            name=settings.registered_model_name,
            alias=settings.champion_alias,
            version=str(candidate["version"]),
        )
        logger.info(
            "se promovió la versión %s al alias '%s'",
            candidate["version"],
            settings.champion_alias,
        )
        decision["promoted"] = True
    else:
        logger.info("champion sin cambios: %s", decision)
        decision["promoted"] = False
    return decision


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print("promote requiere un dict de candidato; invocar vía el comando CLI 'all'")
