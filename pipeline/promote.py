"""Compara el candidato contra el champion y lo promueve solo si mejora (RF6).

El modelo productivo se identifica con un alias en el Model Registry de
MLflow (`champion`). Problema de regresión → la métrica principal es **MAE**
(menor es mejor) y la regla de promoción es explícita:

    Promover si NO hay champion, o si el candidato baja el MAE al menos
    `MAE_IMPROVE_MIN` (3%) y su RMSE no empeora más de `RMSE_WORSEN_MAX` (1%).

Si el candidato no supera al productivo, queda registrado como experimento
pero NO reemplaza el modelo que sirve la API.
"""

from __future__ import annotations

import logging

import mlflow
from mlflow.tracking import MlflowClient

from pipeline.config import export_aws_env, load

logger = logging.getLogger(__name__)

# Reglas de promoción (configurables).
MAE_IMPROVE_MIN = 0.03   # el MAE debe bajar al menos 3%
RMSE_WORSEN_MAX = 0.01   # el RMSE no puede empeorar más de 1%


def _client() -> MlflowClient:
    settings = load()
    export_aws_env(settings)
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    return MlflowClient()


def _run_metrics(client: MlflowClient, run_id: str) -> dict:
    """Lee val_mae / val_rmse del run indicado."""
    m = client.get_run(run_id).data.metrics
    return {"mae": float(m.get("val_mae", float("inf"))),
            "rmse": float(m.get("val_rmse", float("inf")))}


def compare(candidate: dict) -> dict:
    """Compara candidato vs champion y decide si promover (RF6).

    `candidate` es el dict de `train.run()` {run_id, version, metric, metrics, ...}.
    """
    settings = load()
    client = _client()

    cand = _run_metrics(client, candidate["run_id"]) if candidate.get("run_id") else \
        {"mae": float(candidate.get("metric", float("inf"))), "rmse": float("inf")}

    try:
        champ_ver = client.get_model_version_by_alias(
            settings.registered_model_name, settings.champion_alias
        )
        champ = _run_metrics(client, champ_ver.run_id)
        champ_version = champ_ver.version
    except Exception:
        champ_version = None
        champ = None

    if champ is None:
        decision, reason = "promote", "no existía champion previo (línea base)"
    else:
        mae_impr = (champ["mae"] - cand["mae"]) / champ["mae"] if champ["mae"] else 0.0
        rmse_worse = (cand["rmse"] - champ["rmse"]) / champ["rmse"] if champ["rmse"] else 0.0
        if mae_impr >= MAE_IMPROVE_MIN and rmse_worse <= RMSE_WORSEN_MAX:
            decision = "promote"
            reason = f"MAE mejoró {mae_impr:.1%} (≥{MAE_IMPROVE_MIN:.0%}) y RMSE varió {rmse_worse:+.1%}"
        else:
            decision = "keep"
            if mae_impr < MAE_IMPROVE_MIN:
                reason = f"MAE solo mejoró {mae_impr:.1%} (<{MAE_IMPROVE_MIN:.0%})"
            else:
                reason = f"RMSE empeoró {rmse_worse:+.1%} (>{RMSE_WORSEN_MAX:.0%})"

    summary = {
        "candidate_version": candidate.get("version"),
        "candidate_metrics": cand,
        "champion_version": champ_version,
        "champion_metrics": champ,
        "decision": decision,
        "reason": reason,
    }
    logger.info("comparación: %s", summary)
    return summary


def _best(candidates: list[dict]) -> dict:
    """Mejor candidato = menor MAE (regresión: menor es mejor)."""
    valid = [c for c in candidates if c and c.get("version") is not None]
    if not valid:
        raise ValueError("no hay candidatos válidos para promover (falta 'version')")
    return min(valid, key=lambda c: float(c.get("metric") if c.get("metric") is not None else float("inf")))


def promote(candidate: dict) -> dict:
    """Aplica la decisión de compare(): si gana el candidato, mueve el alias champion."""
    settings = load()
    client = _client()

    decision = compare(candidate)
    if decision["decision"] == "promote" and candidate.get("version"):
        # set_registered_model_alias re-apunta `champion` a la nueva versión
        # de forma atómica. La API recoge el cambio en su próximo refresh o
        # vía POST /reload-model.
        client.set_registered_model_alias(
            name=settings.registered_model_name,
            alias=settings.champion_alias,
            version=str(candidate["version"]),
        )
        logger.info("promovida la versión %s al alias '%s' (%s)",
                    candidate["version"], settings.champion_alias, decision["reason"])
        decision["promoted"] = True
    else:
        logger.info("champion sin cambios: %s", decision["reason"])
        decision["promoted"] = False
    return decision


def promote_best(candidates: list[dict]) -> dict:
    """Elige el mejor candidato (menor MAE) y aplica la regla de promoción."""
    winner = _best(candidates)
    logger.info("mejor candidato: run_id=%s version=%s mae=%.2f (de %d)",
                winner.get("run_id"), winner.get("version"),
                float(winner.get("metric") or float("inf")), len(candidates))
    result = promote(winner)
    result["selected_run_id"] = winner.get("run_id")
    result["selected_version"] = winner.get("version")
    result["candidates_considered"] = len(candidates)
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print("promote requiere un dict de candidato; invocar vía el DAG")
