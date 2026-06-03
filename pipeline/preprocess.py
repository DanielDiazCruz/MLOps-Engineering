"""Preprocesa las filas de raw.diabetes_raw y las escribe en clean.diabetes_clean.

Diseño Patrón 1: las columnas categóricas se dejan como **strings** (no se
hace one-hot aquí). El one-hot encoding vive dentro del `Pipeline` de
sklearn al momento de entrenar, así el encoder queda serializado junto
con el modelo en MLflow y la API recibe features crudas en lugar de un
vector pre-codificado.

Transformaciones aplicadas en este módulo:
  - Detección automática del target (`Outcome` o `readmitted`) y
    binarización a {0, 1}.
  - Imputación de columnas numéricas con la mediana de cada una.
  - Descarte de categóricas con más de 20 valores únicos
    (p. ej. códigos ICD con 700+ valores que harían explotar el
    OneHotEncoder al entrenar).
  - Las features se persisten como JSONB para que el esquema pueda
    evolucionar sin necesidad de DDL.

Procesamiento INCREMENTAL: cada run procesa únicamente las filas raw nuevas
(`status='loaded'`) y las marca como `'processed'` al terminar. Antes se
reprocesaba todo el acumulado en cada run, reescribiendo las decenas de
miles de filas de `clean` cada vez (el costo dominante del task). Para que
el esquema de features sea consistente entre batches —ahora que cada uno se
procesa por separado— las columnas se ciñen al conjunto ya establecido en
`clean` (ver `_established_feature_cols`).
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd
from psycopg2.extras import Json, execute_batch

from pipeline.db.connection import connect

logger = logging.getLogger(__name__)

# Posibles nombres del target soportados por el pipeline.
TARGET_CANDIDATES = ("Outcome", "outcome", "readmitted", "target")
# Umbral máximo de cardinalidad para conservar una categórica.
LOW_CARD_MAX = 20


def _detect_target(columns: Iterable[str]) -> str:
    """Devuelve el nombre de la columna que cumple el rol de target."""
    for c in TARGET_CANDIDATES:
        if c in columns:
            return c
    raise ValueError(f"no se encontró columna target en {list(columns)}")


def _binarize_target(series: pd.Series) -> pd.Series:
    """Convierte el target a entero binario {0, 1}.

    Maneja tres casos según el tipo de dato:
      - Entero/booleano → se acota a {0, 1}.
      - Flotante → se considera positivo si es > 0.
      - Categórico (caso 130-US): "<30" significa reingreso dentro de
        30 días (clase positiva); "NO" y ">30" se consideran negativos.
    """
    if series.dtype.kind in {"i", "u", "b"}:
        return series.astype(int).clip(0, 1)
    if series.dtype.kind == "f":
        return (series.fillna(0).astype(int) > 0).astype(int)
    # Caso categórico: mapeamos a 0 por defecto y solo elevamos a 1 las
    # categorías que representan reingreso temprano.
    mapping = {v: 0 for v in series.unique()}
    if "<30" in mapping:
        mapping["<30"] = 1
    elif "YES" in mapping:
        mapping["YES"] = 1
    return series.map(mapping).fillna(0).astype(int)


def _read_unprocessed() -> pd.DataFrame:
    """Lee SOLO las filas raw aún sin procesar (status='loaded').

    Como preprocess marca cada fila como 'processed' al terminar, estas son
    exactamente las del batch nuevo. Si no hay ninguna (CSV agotado o batch
    100% duplicado) devuelve un DataFrame vacío y `run` lo maneja sin fallar:
    el DAG sigue y train reentrena sobre el acumulado existente.
    """
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT row_hash, batch_id, payload FROM raw.diabetes_raw "
            "WHERE status = 'loaded'"
        )
        rows = cur.fetchall()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([{"row_hash": h, "batch_id": b, **p} for h, b, p in rows])


def _established_feature_cols() -> list[str] | None:
    """Columnas de features ya establecidas en clean (None si está vacío).

    Garantiza que cada batch nuevo produzca EXACTAMENTE el mismo conjunto de
    features que los ya procesados. Como ahora preprocesamos batch por batch,
    sin esto la decisión de descartar categóricas de alta cardinalidad
    podría variar entre batches y dejar filas con esquemas distintos, lo que
    rompería el entrenamiento (NaNs en columnas presentes solo en unas filas).
    """
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT features FROM clean.diabetes_clean LIMIT 1")
        row = cur.fetchone()
    return list(row[0].keys()) if row else None


def run(batch_id: str | None = None) -> dict:
    """Preprocesa el batch nuevo y lo persiste en clean.diabetes_clean.

    Incremental: solo toca las filas raw con `status='loaded'`, las escribe
    en `clean` y luego las marca como `'processed'` (en la misma transacción)
    para que el próximo run lea únicamente el siguiente batch.
    """
    df = _read_unprocessed()
    if df.empty:
        summary = {"batch_id": batch_id, "rows": 0, "nota": "sin filas nuevas (status=loaded)"}
        logger.info("preprocess: nada que procesar — %s", summary)
        return summary

    target_col = _detect_target(df.columns)

    # Separamos target y features.
    y = _binarize_target(df[target_col])
    feature_df = df.drop(columns=["row_hash", "batch_id", target_col])

    # Identificamos numéricas vs. categóricas por el dtype inferido del JSONB.
    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()

    # Imputación de numéricas con la mediana del batch (robusta a outliers).
    for c in numeric_cols:
        feature_df[c] = feature_df[c].fillna(feature_df[c].median())

    # Selección de columnas. Si clean ya tiene un esquema establecido, nos
    # ceñimos EXACTAMENTE a él (consistencia entre batches). En el primer
    # batch lo definimos descartando las categóricas de alta cardinalidad
    # (diag_1/2/3, etc., con 700+ valores únicos que explotarían el one-hot).
    established = _established_feature_cols()
    if established is not None:
        missing = [c for c in established if c not in feature_df.columns]
        if missing:
            raise ValueError(
                f"el batch nuevo no trae columnas del esquema establecido: {missing}"
            )
        feature_df = feature_df[established]
    else:
        categorical_cols = [c for c in feature_df.columns if c not in numeric_cols]
        high_card = [c for c in categorical_cols if feature_df[c].nunique() > LOW_CARD_MAX]
        if high_card:
            feature_df = feature_df.drop(columns=high_card)

    # Recalculamos categóricas sobre el set final y las casteamos a string
    # para que JSONB las serialice como texto y el OneHotEncoder las trate
    # de forma consistente.
    categorical_cols = feature_df.select_dtypes(exclude=[np.number]).columns.tolist()
    numeric_cols = [c for c in feature_df.columns if c not in categorical_cols]
    for c in categorical_cols:
        feature_df[c] = feature_df[c].astype(str)

    # Upsert por row_hash. Como el batch es nuevo, en la práctica son INSERTs
    # (split queda NULL por defecto y lo asigna split.py). El ON CONFLICT es
    # solo una red de seguridad ante reejecuciones.
    upsert_sql = (
        "INSERT INTO clean.diabetes_clean (row_hash, batch_id, features, target) "
        "VALUES (%s, %s, %s, %s) "
        "ON CONFLICT (row_hash) DO UPDATE SET "
        "  batch_id = EXCLUDED.batch_id, "
        "  features = EXCLUDED.features, "
        "  target = EXCLUDED.target, "
        "  processed_at = now()"
    )
    rows = []
    for row_hash, raw_batch, features, target in zip(
        df["row_hash"], df["batch_id"], feature_df.to_dict(orient="records"), y
    ):
        rows.append((row_hash, batch_id or raw_batch, Json(features), int(target)))

    with connect() as conn, conn.cursor() as cur:
        execute_batch(cur, upsert_sql, rows, page_size=1_000)
        # Marcamos lo que acabamos de procesar para no reprocesarlo. Misma
        # transacción que el upsert: si algo falla, no quedan filas a medias.
        cur.execute("UPDATE raw.diabetes_raw SET status = 'processed' WHERE status = 'loaded'")

    summary = {
        "batch_id": batch_id,
        "rows": len(rows),
        "feature_count": len(feature_df.columns),
        "numeric_count": len(numeric_cols),
        "categorical_count": len(categorical_cols),
        "target_col": target_col,
    }
    logger.info("preprocess finalizado: %s", summary)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print(run())
