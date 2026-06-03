"""Helper para crear conexiones a PostgreSQL.

Centraliza la lectura del DSN para que el resto del paquete `pipeline`
no tenga que conocer credenciales ni hostnames.
"""

from __future__ import annotations

import contextlib
from typing import Iterator

import psycopg2
from psycopg2.extensions import connection as PgConnection

from pipeline.config import load


@contextlib.contextmanager
def connect() -> Iterator[PgConnection]:
    """Context manager que entrega una conexión psycopg2.

    Comportamiento al salir del bloque `with`:
      - Si no hubo excepción, hace `commit()` de la transacción.
      - Si hubo excepción, hace `rollback()` y la re-lanza.
      - En cualquier caso, cierra la conexión.

    Uso típico:
        with connect() as conn, conn.cursor() as cur:
            cur.execute("...")
    """
    conn = psycopg2.connect(load().pg_dsn)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
