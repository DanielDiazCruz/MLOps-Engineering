"""Validaciones de calidad sobre raw.properties_raw (lote inmobiliario).

Ejecuta chequeos básicos contra el lote recién ingestado y construye un
reporte. Si alguno crítico falla, lanza `ValueError` para que la tarea de
Airflow se marque como fallida y no se entrene con datos inválidos.

El reporte (columnas, conteos, nulos, problemas) se devuelve para que la
tarea de auditoría lo registre en `audit.training_history` (RF3/RF4).
"""

from __future__ import annotations

import logging

import pandas as pd

from pipeline.db.connection import connect

logger = logging.getLogger(__name__)

# Columna objetivo del problema de regresión.
TARGET = "price"
# Columnas que el esquema debe traer siempre.
EXPECTED_COLUMNS = (
    "brokered_by", "status", "price", "bed", "bath", "acre_lot",
    "street", "city", "state", "zip_code", "house_size", "prev_sold_date",
)


def _read_batch(batch_id: str | None) -> pd.DataFrame:
    """Lee las filas del batch indicado desde raw.properties_raw.

    Si el batch no tiene filas (p. ej. 100% duplicado) cae a validar sobre
    todo el acumulado para no bloquear el pipeline.
    """
    with connect() as conn, conn.cursor() as cur:
        if batch_id:
            cur.execute(
                "SELECT row_hash, payload FROM raw.properties_raw WHERE batch_id = %s",
                (batch_id,),
            )
            rows = cur.fetchall()
            if not rows:
                cur.execute("SELECT row_hash, payload FROM raw.properties_raw")
                rows = cur.fetchall()
        else:
            cur.execute("SELECT row_hash, payload FROM raw.properties_raw")
            rows = cur.fetchall()
    if not rows:
        raise ValueError(f"no se encontraron filas crudas (batch_id={batch_id!r})")
    return pd.DataFrame([{"row_hash": h, **p} for h, p in rows])


def run(batch_id: str | None = None) -> dict:
    """Ejecuta los chequeos de calidad y retorna un reporte.

    Reglas críticas (detienen el DAG):
      1. El dataframe no puede estar vacío.
      2. La columna target `price` debe existir.
      3. No pueden existir `row_hash` duplicados dentro del batch.
    Advertencias (no detienen, quedan en el reporte):
      - Columnas esperadas ausentes.
      - Proporción de precios nulos o no positivos.
    """
    df = _read_batch(batch_id)
    issues: list[str] = []
    warnings: list[str] = []

    # 1. Conteo mínimo.
    if df.empty:
        issues.append("dataframe vacío")

    # 2. Target presente.
    if TARGET not in df.columns:
        issues.append(f"falta la columna target '{TARGET}'")
    else:
        price = pd.to_numeric(df[TARGET], errors="coerce")
        n_null = int(price.isna().sum())
        n_nonpos = int((price <= 0).sum())
        if n_null:
            warnings.append(f"precios nulos/no numéricos: {n_null}")
        if n_nonpos:
            warnings.append(f"precios <= 0: {n_nonpos}")
        # Si TODO el target es inválido, no hay nada que aprender.
        if n_null + n_nonpos >= len(df):
            issues.append("ningún precio válido en el batch")

    # 3. Integridad de row_hash.
    if "row_hash" in df.columns and df["row_hash"].duplicated().any():
        issues.append("row_hash duplicados dentro del batch")

    # Advertencia por columnas esperadas ausentes (cambio de esquema).
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        warnings.append(f"columnas esperadas ausentes: {missing_cols}")

    report = {
        "batch_id": batch_id,
        "rows": int(len(df)),
        "columns": [c for c in df.columns if c != "row_hash"],
        "missing_expected": missing_cols,
        "issues": issues,
        "warnings": warnings,
    }
    if issues:
        logger.error("problemas de calidad: %s", issues)
        raise ValueError(f"validación de calidad falló: {issues}")
    logger.info("calidad OK: rows=%s warnings=%s", report["rows"], warnings)
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print(run())
