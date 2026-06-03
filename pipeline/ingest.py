"""Ingesta incremental desde la API externa de datos hacia raw.properties_raw.

La fuente de datos es una API HTTP (imagen `cristiandiaz13/mlops-puj`) que
entrega los datos por lotes y es *stateful*: cada llamada a
`GET /data?group_number=N` devuelve el siguiente lote (`batch_number` 0,1,2,…)
para ese grupo. `GET /restart_data_generation?group_number=N` reinicia el
cursor del grupo.

Diseño del cliente (RF1 + RF "cliente robusto"):
  - Sesión `requests` con reintentos y backoff para errores transitorios
    (timeouts, errores de conexión, 5xx).
  - Manejo explícito de fin de datos: si la API devuelve un lote vacío, se
    retorna un resumen con `inserted=0` sin fallar, para que el DAG decida
    no entrenar en vez de romperse.
  - Idempotencia: cada fila se identifica con un `row_hash` SHA-256 y se
    inserta con `ON CONFLICT (row_hash) DO NOTHING`, así un reproceso no
    duplica datos.
"""

from __future__ import annotations

import hashlib
import json
import logging

import requests
from psycopg2.extras import Json, execute_values
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from pipeline.config import load
from pipeline.db.connection import connect

logger = logging.getLogger(__name__)


def _row_hash(row: dict) -> str:
    """Hash SHA-256 determinístico sobre el contenido de la fila.

    Sirve como clave única en `raw.properties_raw`. Para que sea estable se
    ordenan las llaves, los nulos se normalizan a cadena vacía y todo se
    serializa a JSON antes de hashear.
    """
    canonical = json.dumps(
        {k: ("" if v is None else str(v)) for k, v in sorted(row.items())},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _session(retries: int) -> requests.Session:
    """Crea una sesión HTTP con reintentos + backoff para errores transitorios."""
    retry = Retry(
        total=retries,
        connect=retries,
        read=retries,
        backoff_factor=1.0,
        status_forcelist=(500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )
    sess = requests.Session()
    adapter = HTTPAdapter(max_retries=retry)
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    return sess


def check_source(url: str | None = None) -> str:
    """Verifica que la API de datos esté accesible (tarea t_check_source).

    Hace GET /health y lanza una excepción si no responde 200, para que la
    tarea de Airflow falle de forma visible antes de intentar cargar nada.
    """
    settings = load()
    base = (url or settings.data_api_url).rstrip("/")
    sess = _session(settings.data_api_retries)
    resp = sess.get(f"{base}/health", timeout=settings.data_api_timeout)
    resp.raise_for_status()
    logger.info("API de datos OK: %s/health -> %s", base, resp.status_code)
    return base


def fetch_batch(group: int, url: str, timeout: int, retries: int) -> dict:
    """Pide el siguiente lote a la API y devuelve el JSON parseado.

    Estructura esperada: {"group_number": N, "batch_number": K, "data": [...]}.
    """
    sess = _session(retries)
    resp = sess.get(f"{url}/data", params={"group_number": group}, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()
    if "data" not in payload:
        raise ValueError(f"respuesta inesperada de la API (sin 'data'): {list(payload)[:5]}")
    return payload


def load_batch(group: int | None = None, batch_id: str | None = None) -> dict:
    """Consume un lote de la API y lo almacena en raw.properties_raw.

    Cada ejecución del DAG solicita el siguiente lote del grupo (la API lleva
    el cursor). Retorna un resumen con la cantidad insertada vs. duplicada
    para que las tareas siguientes lo lean por XCom.
    """
    settings = load()
    group = group if group is not None else settings.data_api_group
    base = check_source()

    payload = fetch_batch(group, base, settings.data_api_timeout, settings.data_api_retries)
    batch_number = payload.get("batch_number")
    rows_data = payload.get("data") or []
    # batch_id estable: identifica el grupo + el número de lote del servidor.
    batch_id = batch_id or f"g{group}-b{batch_number}"

    # Fin de datos: la API ya no entrega más filas. No es un error; el DAG
    # debe poder decidir "no entrenar" en vez de fallar.
    if not rows_data:
        summary = {
            "batch_id": batch_id,
            "batch_number": batch_number,
            "group": group,
            "inserted": 0,
            "duplicates": 0,
            "total_rows": 0,
            "end_of_data": True,
        }
        logger.info("no hay más datos disponibles (group=%s batch=%s)", group, batch_number)
        return summary

    source = f"data-api:g{group}"
    rows = []
    for raw_row in rows_data:
        # Normalizamos NaN/None a None para que JSONB guarde null.
        clean = {k: (None if v is None else v) for k, v in raw_row.items()}
        rows.append((_row_hash(raw_row), batch_id, source, "loaded", Json(clean)))

    insert_sql = (
        "INSERT INTO raw.properties_raw "
        "(row_hash, batch_id, source, status, payload) VALUES %s "
        "ON CONFLICT (row_hash) DO NOTHING"
    )
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM raw.properties_raw")
        before = cur.fetchone()[0]
        execute_values(cur, insert_sql, rows, template="(%s,%s,%s,%s,%s)", page_size=5_000)
        cur.execute("SELECT count(*) FROM raw.properties_raw")
        after = cur.fetchone()[0]

    inserted = after - before
    duplicates = len(rows) - inserted

    summary = {
        "batch_id": batch_id,
        "batch_number": batch_number,
        "group": group,
        "inserted": inserted,
        "duplicates": duplicates,
        "total_rows": len(rows),
        "end_of_data": False,
    }
    logger.info(
        "lote procesado: %d filas (insertadas=%d duplicadas=%d group=%s batch=%s)",
        len(rows), inserted, duplicates, group, batch_number,
    )
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print(load_batch())
