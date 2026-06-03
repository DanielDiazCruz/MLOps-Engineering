"""Preprocesa raw.properties_raw → clean.properties_clean (regresión de precio).

Diseño Patrón 1: las categóricas se dejan como **strings**; el one-hot
encoding vive dentro del `Pipeline` de sklearn al entrenar, así el encoder
queda serializado junto al modelo en MLflow y la API recibe features crudas.

Ingeniería de características (target = `price`):
  - Numéricas: bed, bath, acre_lot, house_size + `prev_sold_year` derivado de
    prev_sold_date. Imputadas con la mediana del batch.
  - Categóricas (como str): status, city, state, zip_code. La cardinalidad la
    acota el encoder al entrenar (max_categories / handle_unknown), por eso
    aquí no se descartan por cardinalidad.
  - Se descartan `street` y `brokered_by`: son identificadores casi únicos
    (cientos de miles de valores) con poca señal individual y romperían el
    encoding. `prev_sold_date` se reemplaza por `prev_sold_year`.

Procesamiento INCREMENTAL: solo procesa las filas raw nuevas
(`status='loaded'`) y las marca como `'processed'` al terminar. Como el
conjunto de features es fijo y explícito, el esquema de `clean` es
consistente entre batches por construcción.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from psycopg2.extras import Json, execute_values

from pipeline.db.connection import connect

logger = logging.getLogger(__name__)

TARGET = "price"
NUMERIC_FEATURES = ["bed", "bath", "acre_lot", "house_size", "prev_sold_year"]
CATEGORICAL_FEATURES = ["status", "city", "state", "zip_code"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _read_unprocessed() -> pd.DataFrame:
    """Lee solo las filas raw aún sin procesar (status='loaded').

    Devuelve un DataFrame vacío si no hay nada nuevo (CSV/API agotado o batch
    duplicado); `run` lo maneja sin fallar.
    """
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT row_hash, batch_id, payload FROM raw.properties_raw "
            "WHERE status = 'loaded'"
        )
        rows = cur.fetchall()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([{"row_hash": h, "batch_id": b, **p} for h, b, p in rows])


def _cat_to_str(series: pd.Series) -> pd.Series:
    """Normaliza una categórica a string limpio.

    Algunas categóricas vienen codificadas como float (p. ej. zip_code=6016.0);
    se elimina el sufijo `.0` para que '6016' sea estable. Los nulos pasan a
    'unknown'.
    """
    s = series.astype(str).str.replace(r"\.0$", "", regex=True)
    return s.replace({"None": "unknown", "nan": "unknown", "": "unknown"})


def run(batch_id: str | None = None) -> dict:
    """Procesa el batch nuevo y lo persiste en clean.properties_clean."""
    df = _read_unprocessed()
    if df.empty:
        summary = {"batch_id": batch_id, "rows": 0, "nota": "sin filas nuevas (status=loaded)"}
        logger.info("preprocess: nada que procesar — %s", summary)
        return summary

    # --- Target: precio válido (> 0). Las filas sin precio válido no sirven
    # para entrenar, así que no entran a clean (pero igual se marcan processed).
    price = pd.to_numeric(df.get(TARGET), errors="coerce")
    valid = price.notna() & (price > 0)
    df_valid = df.loc[valid].copy()
    price_valid = price.loc[valid]

    feature_df = pd.DataFrame(index=df_valid.index)
    if not df_valid.empty:
        # --- Numéricas base (coerción robusta a número).
        for col in ["bed", "bath", "acre_lot", "house_size"]:
            feature_df[col] = pd.to_numeric(df_valid.get(col), errors="coerce")
        # --- Derivada: año de venta previa (NaN si no hay fecha).
        feature_df["prev_sold_year"] = pd.to_datetime(
            df_valid.get("prev_sold_date"), errors="coerce"
        ).dt.year
        # Imputación de numéricas con la mediana del batch.
        for col in NUMERIC_FEATURES:
            median = feature_df[col].median()
            feature_df[col] = feature_df[col].fillna(median if pd.notna(median) else 0.0)
        # --- Categóricas como string limpio.
        for col in CATEGORICAL_FEATURES:
            src = df_valid[col] if col in df_valid.columns else pd.Series(index=df_valid.index, dtype=object)
            feature_df[col] = _cat_to_str(src)
        feature_df = feature_df[FEATURE_COLUMNS]

    # --- Upsert de las filas válidas a clean (en una sola transacción que
    # además marca el batch como procesado).
    rows = [
        (rh, batch_id or rb, Json(feat), float(tgt))
        for rh, rb, feat, tgt in zip(
            df_valid["row_hash"], df_valid["batch_id"],
            feature_df.to_dict(orient="records"), price_valid,
        )
    ]
    upsert_sql = (
        "INSERT INTO clean.properties_clean (row_hash, batch_id, features, target) "
        "VALUES %s "
        "ON CONFLICT (row_hash) DO UPDATE SET "
        "  batch_id = EXCLUDED.batch_id, "
        "  features = EXCLUDED.features, "
        "  target = EXCLUDED.target, "
        "  processed_at = now()"
    )
    with connect() as conn, conn.cursor() as cur:
        if rows:
            execute_values(cur, upsert_sql, rows, template="(%s,%s,%s,%s)", page_size=5_000)
        # Marca TODO lo leído (válido o no) como procesado para no reprocesarlo.
        cur.execute("UPDATE raw.properties_raw SET status = 'processed' WHERE status = 'loaded'")

    summary = {
        "batch_id": batch_id,
        "rows": len(rows),
        "discarded_invalid_price": int((~valid).sum()),
        "feature_count": len(FEATURE_COLUMNS),
        "numeric_count": len(NUMERIC_FEATURES),
        "categorical_count": len(CATEGORICAL_FEATURES),
        "target_col": TARGET,
    }
    logger.info("preprocess finalizado: %s", summary)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print(run())
