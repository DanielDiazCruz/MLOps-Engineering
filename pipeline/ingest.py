"""Ingesta incremental del CSV hacia raw.diabetes_raw.

Cada llamada a `load_batch` carga el siguiente bloque de hasta `batch_size`
filas (con tope obligatorio de 15.000 según el enunciado del proyecto).
Para saber por dónde va el cursor, se cuenta cuántas filas ya existen en
`raw.diabetes_raw` para el mismo `source_file` y se usa ese número como
`skiprows` al leer el CSV con pandas. De este modo, ejecutar el DAG varias
veces avanza el cursor de forma natural y el CSV de 101k filas se consume
en aproximadamente 7 ejecuciones.

La idempotencia se garantiza con un `row_hash` SHA-256 determinístico
sobre el contenido canónico de la fila, más una restricción
`ON CONFLICT (row_hash) DO NOTHING` al insertar.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid

import pandas as pd
from psycopg2.extras import Json, execute_batch

from pipeline.config import load
from pipeline.db.connection import connect

logger = logging.getLogger(__name__)

# Tope duro de filas por ejecución exigido por el enunciado del proyecto.
MAX_BATCH_SIZE = 15_000


def _row_hash(row: dict) -> str:
    """Calcula un hash SHA-256 determinístico sobre el contenido de la fila.

    Sirve como clave única en `raw.diabetes_raw`, asegurando que aunque el
    DAG se reejecute, los mismos datos no se inserten dos veces.

    Para que el hash sea estable se ordenan las llaves del dict, los
    valores nulos se reemplazan por cadena vacía y todo se serializa a
    JSON antes de hacer el hash.
    """
    canonical = json.dumps(
        {k: ("" if pd.isna(v) else str(v)) for k, v in sorted(row.items())},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def check_source(path: str | None = None) -> str:
    """Verifica que el CSV fuente exista y sea accesible.

    Lanza `FileNotFoundError` si la ruta no apunta a un archivo válido,
    haciendo que la tarea de Airflow falle de forma visible antes de
    intentar cargar nada.
    """
    settings = load()
    csv_path = path or settings.source_csv
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV fuente no encontrado: {csv_path}")
    size = os.path.getsize(csv_path)
    logger.info("csv fuente OK: %s (%d bytes)", csv_path, size)
    return csv_path



def load_batch(path: str | None = None, batch_id: str | None = None) -> dict:
    """Carga el siguiente lote de filas del CSV a raw.diabetes_raw.

    El cursor se calcula automáticamente: cuenta cuántas filas existen ya
    en raw para este `source_file` y usa ese número como offset (skiprows)
    al leer el CSV. Cada ejecución avanza el cursor en máximo `chunk_size`
    filas (cap a 15.000).

    Retorna un resumen con la cantidad insertada vs. duplicada para que
    Airflow pueda registrar la métrica en los XComs.
    """
    settings = load()
    csv_path = check_source(path)
    # Aplicamos el tope obligatorio del enunciado por si la configuración
    # llegara con un valor mayor.
    chunk_size = min(settings.batch_size, MAX_BATCH_SIZE)
    batch_id = batch_id or uuid.uuid4().hex
    source_file = os.path.basename(csv_path)

    # Paso 1: averiguar el offset actual contando las filas ya cargadas
    # para ESTE source_file. Esto hace que el cursor avance entre runs.
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM raw.diabetes_raw WHERE source_file = %s",
            (source_file,),
        )
        offset = cur.fetchone()[0]

    logger.info("offset actual para %s: %d filas ya cargadas", source_file, offset)

    # Paso 2: leer del CSV únicamente las siguientes `chunk_size` filas.
    # Leemos primero la cabecera y luego saltamos `offset + 1` líneas
    # (la +1 corresponde a la línea del header del CSV).
    col_names = pd.read_csv(csv_path, nrows=0).columns.tolist()
    chunk = pd.read_csv(
        csv_path,
        skiprows=offset + 1,
        nrows=chunk_size,
        names=col_names,
        header=None,
    )

    # Caso borde: ya no quedan filas por ingestar. Retornamos un resumen
    # vacío sin fallar para que el DAG siga corriendo (preprocess + train
    # se ejecutan sobre lo acumulado).
    if chunk.empty:
        logger.info("no hay filas nuevas que ingestar — CSV consumido (offset=%d)", offset)
        summary = {
            "batch_id": batch_id,
            "source_file": source_file,
            "inserted": 0,
            "duplicates": 0,
            "total_rows": 0,
        }
        logger.info("ingesta finalizada: %s", summary)
        return summary

    # Paso 3: construir el INSERT con ON CONFLICT para que las filas
    # duplicadas (mismo row_hash) sean ignoradas silenciosamente.
    insert_sql = (
        "INSERT INTO raw.diabetes_raw "
        "(row_hash, batch_id, source_file, status, payload) "
        "VALUES (%s, %s, %s, 'loaded', %s) "
        "ON CONFLICT (row_hash) DO NOTHING"
    )

    # Paso 4: armar las tuplas a insertar calculando el row_hash y
    # serializando el payload completo como JSONB.
    rows = []
    for _, raw_row in chunk.iterrows():
        row_dict = raw_row.to_dict()
        rh = _row_hash(row_dict)
        # Convertimos NaN a None para que JSONB lo guarde como null
        # en lugar de la cadena "NaN".
        clean = {k: (None if pd.isna(v) else v) for k, v in row_dict.items()}
        rows.append((rh, batch_id, source_file, Json(clean)))

    # Paso 5: ejecutar el batch insert. Comparamos el conteo antes/después
    # para saber cuántas filas eran realmente nuevas vs. duplicadas.
    with connect() as conn, conn.cursor() as cur:
        before = _count_rows(cur)
        execute_batch(cur, insert_sql, rows, page_size=1_000)
        after = _count_rows(cur)

    inserted = after - before
    duplicates = len(rows) - inserted

    logger.info(
        "lote procesado: %d filas (insertadas=%d duplicadas=%d offset=%d)",
        len(rows), inserted, duplicates, offset,
    )

    summary = {
        "batch_id": batch_id,
        "source_file": source_file,
        "inserted": inserted,
        "duplicates": duplicates,
        "total_rows": len(rows),
    }
    logger.info("ingesta finalizada: %s", summary)
    return summary


def _count_rows(cur) -> int:
    """Cuenta total de filas en raw.diabetes_raw (usado para medir el delta)."""
    cur.execute("SELECT count(*) FROM raw.diabetes_raw")
    return cur.fetchone()[0]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print(load_batch())
