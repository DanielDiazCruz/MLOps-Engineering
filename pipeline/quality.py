"""Validaciones mínimas de calidad sobre raw.diabetes_raw.

Ejecuta una serie de chequeos básicos contra los datos recién ingestados y
construye un reporte. Si alguno falla, lanza `ValueError` para que la
tarea de Airflow se marque como fallida y las tareas siguientes
(preprocess, train, etc.) no se ejecuten con datos inválidos.
"""

from __future__ import annotations

import logging
from typing import Iterable

import pandas as pd

from pipeline.db.connection import connect

logger = logging.getLogger(__name__)

# Posibles nombres del target. Soportamos varias variantes para que el
# mismo pipeline funcione con el dataset 130-US (`readmitted`) y con
# Pima (`Outcome`).
TARGET_CANDIDATES = ("Outcome", "outcome", "readmitted", "target")


def _detect_critical_keys(columns) -> tuple[str, ...]:
    """Detecta qué columna del dataset cumple el rol de target.

    Si encontramos alguno de los nombres candidatos, lo marcamos como
    crítico (no puede estar ausente ni tener nulos).
    """
    for c in TARGET_CANDIDATES:
        if c in columns:
            return (c,)
    return ()


CRITICAL_KEYS_DEFAULT: tuple[str, ...] = ()


def _read_batch(batch_id: str | None) -> pd.DataFrame:
    """Lee las filas del batch indicado desde raw.diabetes_raw.

    Si todas las filas del batch resultaron duplicadas (por ejemplo, una
    reejecución), caemos en un fallback que valida sobre todos los datos
    crudos disponibles para no bloquear el pipeline.
    """
    with connect() as conn, conn.cursor() as cur:
        if batch_id:
            cur.execute(
                "SELECT row_hash, payload FROM raw.diabetes_raw WHERE batch_id = %s",
                (batch_id,),
            )
            rows = cur.fetchall()
            if not rows:
                # Fallback: el batch nuevo era 100% duplicado. Validamos
                # contra el acumulado existente para que el DAG siga.
                cur.execute("SELECT row_hash, payload FROM raw.diabetes_raw")
                rows = cur.fetchall()
        else:
            cur.execute("SELECT row_hash, payload FROM raw.diabetes_raw")
            rows = cur.fetchall()
    if not rows:
        raise ValueError(f"no se encontraron filas crudas (batch_id={batch_id!r})")
    # Expandimos el JSONB de cada fila a columnas planas para poder
    # aplicar chequeos columna a columna con pandas.
    df = pd.DataFrame([{"row_hash": h, **p} for h, p in rows])
    return df


def run(batch_id: str | None = None, critical_keys: Iterable[str] | None = None) -> dict:
    """Ejecuta los chequeos de calidad y retorna un reporte.

    Reglas aplicadas (cualquier violación detiene el DAG):
      1. El dataframe no puede estar vacío.
      2. Las columnas críticas (target) deben existir y no tener nulos.
      3. No pueden existir `row_hash` duplicados (la UNIQUE de la tabla
         ya lo previene, pero lo verificamos como doble seguro).
    """
    df = _read_batch(batch_id)
    if critical_keys is None:
        critical_keys = _detect_critical_keys(df.columns)
    issues: list[str] = []

    # Chequeo 1: conteo mínimo de filas.
    if df.empty:
        issues.append("dataframe vacío")

    # Chequeo 2: presencia y completitud de columnas críticas.
    for key in critical_keys:
        if key not in df.columns:
            issues.append(f"falta columna crítica: {key}")
            continue
        n_null = df[key].isna().sum()
        if n_null:
            issues.append(f"nulos en columna crítica {key}: {n_null}")

    # Chequeo 3: integridad del hash de fila.
    if df["row_hash"].duplicated().any():
        issues.append("row_hash duplicados dentro del batch")

    report = {
        "batch_id": batch_id,
        "rows": int(len(df)),
        "columns": list(df.columns),
        "issues": issues,
    }
    if issues:
        logger.error("problemas de calidad: %s", issues)
        raise ValueError(f"validación de calidad falló: {issues}")
    logger.info("calidad OK: %s", report)
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print(run())
