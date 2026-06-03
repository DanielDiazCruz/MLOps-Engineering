"""RF4: decisión automática de entrenamiento.

En lugar de reentrenar a ciegas en cada corrida, el pipeline decide SI vale
la pena entrenar a partir de tres señales calculadas sobre el lote recién
procesado en `clean.properties_clean`:

  1. `validate_schema`        — el lote trae todas las features esperadas y un
                                target válido (defensa ante cambios de esquema).
  2. `detect_new_categories`  — aparecieron valores categóricos no vistos en la
                                línea base histórica (un encoder ya entrenado
                                los trataría como "infrequent"/desconocidos).
  3. `detect_drift`           — las distribuciones numéricas y el target se
                                desplazaron respecto a la línea base, medido con
                                el Índice de Estabilidad de Población (PSI).

`decide()` cruza esas señales con el número de filas nuevas y la existencia de
un champion productivo para producir `{decision: 'train'|'skip', reason}`. El
DAG bifurca con esa decisión (BranchPythonOperator) y registra todo en
`audit.training_history` (RF4/RF9).

PSI (referencia de interpretación habitual):
  - < 0.10  -> sin cambio relevante
  - 0.10–0.25 -> cambio moderado
  - > 0.25  -> cambio significativo (se reentrena)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from pipeline.config import export_aws_env, load
from pipeline.db.connection import connect
from pipeline.preprocess import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
)

logger = logging.getLogger(__name__)

# Umbral de PSI para considerar que una feature "derivó" lo suficiente.
PSI_SIGNIFICANT = 0.25
# Filas máximas que se leen de cada lado (nuevo / línea base) para estimar el
# PSI. El PSI es robusto al muestreo, así que acotamos para no leer cientos de
# miles de filas en cada corrida (el pipeline ya es incremental).
_DRIFT_SAMPLE = 40_000
# Tamaño mínimo de lote nuevo para justificar un reentrenamiento "de rutina"
# cuando no hay drift ni categorías nuevas.
MIN_NEW_ROWS = 1_000


def validate_schema(batch_id: str | None) -> dict:
    """Valida que el lote nuevo en clean tenga todas las features esperadas.

    Como `clean.properties_clean` se construye con un set fijo de columnas
    (`FEATURE_COLUMNS`), el esquema es consistente por construcción; aun así
    se valida explícitamente para detectar cualquier regresión y alimentar la
    decisión/auditoría (RF3/RF4).
    """
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM clean.properties_clean WHERE batch_id = %s",
            (batch_id,),
        )
        n_batch = cur.fetchone()[0]
        cur.execute(
            "SELECT features FROM clean.properties_clean WHERE batch_id = %s LIMIT 1",
            (batch_id,),
        )
        row = cur.fetchone()

    if not row:
        report = {
            "valid": True,
            "n_batch": int(n_batch),
            "missing_features": [],
            "extra_features": [],
            "note": "el lote no aportó filas nuevas a clean",
        }
        logger.info("validate_schema: %s", report)
        return report

    present = set((row[0] or {}).keys())
    missing = [c for c in FEATURE_COLUMNS if c not in present]
    extra = [c for c in present if c not in FEATURE_COLUMNS]
    report = {
        "valid": not missing,
        "n_batch": int(n_batch),
        "missing_features": missing,
        "extra_features": extra,
        "expected_features": list(FEATURE_COLUMNS),
    }
    logger.info("validate_schema: %s", report)
    return report


def detect_new_categories(batch_id: str | None) -> dict:
    """Detecta valores categóricos del lote nuevo no vistos en la línea base.

    Compara, por cada feature categórica, los valores distintos del lote
    (`batch_id = N`) contra los del histórico (`batch_id <> N`) usando un
    `EXCEPT` en SQL (exacto y barato). En el primer lote no hay línea base,
    así que no se reportan categorías nuevas.
    """
    by_feature: dict[str, dict] = {}
    total_new = 0

    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM clean.properties_clean WHERE batch_id <> %s",
            (batch_id,),
        )
        n_baseline = cur.fetchone()[0]
        if n_baseline == 0:
            report = {
                "has_new": False,
                "total_new": 0,
                "by_feature": {},
                "note": "sin línea base (primer lote)",
            }
            logger.info("detect_new_categories: %s", report)
            return report

        for col in CATEGORICAL_FEATURES:
            cur.execute(
                "SELECT DISTINCT features ->> %s FROM clean.properties_clean "
                "WHERE batch_id = %s "
                "EXCEPT "
                "SELECT DISTINCT features ->> %s FROM clean.properties_clean "
                "WHERE batch_id <> %s",
                (col, batch_id, col, batch_id),
            )
            new_vals = [r[0] for r in cur.fetchall() if r[0] is not None]
            if new_vals:
                by_feature[col] = {
                    "count": len(new_vals),
                    "sample": sorted(new_vals)[:10],
                }
                total_new += len(new_vals)

    report = {
        "has_new": total_new > 0,
        "total_new": total_new,
        "by_feature": by_feature,
    }
    logger.info("detect_new_categories: total_new=%d features=%s",
                total_new, list(by_feature))
    return report


def _read_for_drift(batch_id: str | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lee (muestreado) las features numéricas + target del lote y de la base."""
    def _to_df(rows: list) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame()
        recs = []
        for feats, target in rows:
            rec = {c: (feats or {}).get(c) for c in NUMERIC_FEATURES}
            rec["target"] = target
            recs.append(rec)
        return pd.DataFrame(recs)

    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT features, target FROM clean.properties_clean "
            "WHERE batch_id = %s LIMIT %s",
            (batch_id, _DRIFT_SAMPLE),
        )
        new_rows = cur.fetchall()
        cur.execute(
            "SELECT features, target FROM clean.properties_clean "
            "WHERE batch_id <> %s LIMIT %s",
            (batch_id, _DRIFT_SAMPLE),
        )
        base_rows = cur.fetchall()
    return _to_df(new_rows), _to_df(base_rows)


def _psi(base: np.ndarray, new: np.ndarray, bins: int = 10) -> float:
    """Índice de Estabilidad de Población (PSI) entre dos muestras.

    Se binifica la línea base por cuantiles (deciles) y se compara la
    proporción de masa que cae en cada bin. Devuelve 0.0 si no hay datos
    suficientes o la columna es casi constante.
    """
    base = base[~np.isnan(base)]
    new = new[~np.isnan(new)]
    if len(base) < bins or len(new) < bins:
        return 0.0

    edges = np.unique(np.quantile(base, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # columna casi constante: PSI no es informativo
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    base_hist, _ = np.histogram(base, bins=edges)
    new_hist, _ = np.histogram(new, bins=edges)
    eps = 1e-6
    base_pct = np.clip(base_hist / base_hist.sum(), eps, None)
    new_pct = np.clip(new_hist / new_hist.sum(), eps, None)
    return float(np.sum((new_pct - base_pct) * np.log(new_pct / base_pct)))


def detect_drift(batch_id: str | None) -> dict:
    """Calcula el PSI del lote nuevo vs. la línea base por feature numérica + target."""
    new_df, base_df = _read_for_drift(batch_id)
    if new_df.empty or base_df.empty:
        report = {
            "has_drift": False,
            "max_psi": 0.0,
            "threshold": PSI_SIGNIFICANT,
            "by_feature": {},
            "note": "sin línea base (primer lote)" if base_df.empty else "lote vacío",
        }
        logger.info("detect_drift: %s", report)
        return report

    by_feature: dict[str, float] = {}
    for col in NUMERIC_FEATURES + ["target"]:
        psi = _psi(
            pd.to_numeric(base_df[col], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(new_df[col], errors="coerce").to_numpy(dtype=float),
        )
        by_feature[col] = round(psi, 4)

    max_psi = max(by_feature.values()) if by_feature else 0.0
    drifted = [k for k, v in by_feature.items() if v >= PSI_SIGNIFICANT]
    report = {
        "has_drift": max_psi >= PSI_SIGNIFICANT,
        "max_psi": round(max_psi, 4),
        "threshold": PSI_SIGNIFICANT,
        "by_feature": by_feature,
        "drifted_features": drifted,
    }
    logger.info("detect_drift: max_psi=%.4f drifted=%s", max_psi, drifted)
    return report


def champion_exists() -> bool:
    """True si existe un modelo productivo bajo el alias `champion` en MLflow."""
    import mlflow
    from mlflow.tracking import MlflowClient

    settings = load()
    export_aws_env(settings)
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    try:
        MlflowClient().get_model_version_by_alias(
            settings.registered_model_name, settings.champion_alias
        )
        return True
    except Exception:
        return False


def _total_rows() -> int:
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM clean.properties_clean")
        return int(cur.fetchone()[0])


def decide(
    batch_id: str | None,
    schema_info: dict,
    new_categories: dict,
    drift: dict,
) -> dict:
    """Cruza las señales y decide entrenar o saltar (RF4).

    Reglas, en orden de prioridad:
      1. Esquema inválido            -> skip (no entrenar con datos rotos).
      2. No hay champion             -> train (entrenamiento base).
      3. Lote sin filas nuevas       -> skip (nada que aprender).
      4. Drift significativo         -> train (la distribución cambió).
      5. Categorías nuevas           -> train (el encoder debe verlas).
      6. Lote grande (>= MIN_NEW)    -> train (refresco de rutina).
      7. En otro caso                -> skip.
    """
    n_new = int(schema_info.get("n_batch", 0))
    n_total = _total_rows()
    champ = champion_exists()

    if not schema_info.get("valid", True):
        decision = "skip"
        reason = (f"esquema inválido, faltan {schema_info.get('missing_features')}: "
                  "no se entrena")
    elif not champ:
        decision = "train"
        reason = "no hay champion productivo: entrenamiento base"
    elif n_new == 0:
        decision = "skip"
        reason = "el lote no aportó filas nuevas a clean"
    elif drift.get("has_drift"):
        decision = "train"
        reason = (f"drift significativo (PSI máx {drift.get('max_psi')} en "
                  f"{drift.get('drifted_features')})")
    elif new_categories.get("has_new"):
        feats = list(new_categories.get("by_feature", {}))
        decision = "train"
        reason = f"{new_categories.get('total_new')} categorías nuevas en {feats}"
    elif n_new >= MIN_NEW_ROWS:
        decision = "train"
        reason = f"lote grande ({n_new} filas nuevas ≥ {MIN_NEW_ROWS})"
    else:
        decision = "skip"
        reason = f"lote pequeño ({n_new} filas) sin drift ni categorías nuevas"

    result = {
        "decision": decision,
        "reason": reason,
        "n_new": n_new,
        "n_total": n_total,
        "champion_exists": champ,
        "min_new_rows": MIN_NEW_ROWS,
    }
    logger.info("decisión de entrenamiento: %s", result)
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    schema = validate_schema(None)
    cats = detect_new_categories(None)
    dr = detect_drift(None)
    print(decide(None, schema, cats, dr))
