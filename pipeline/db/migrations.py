"""DDL idempotente para las capas raw / clean / inference.

Aplica `CREATE SCHEMA IF NOT EXISTS` y `CREATE TABLE IF NOT EXISTS`, por lo
que se puede invocar de forma segura en cada ejecución del DAG sin riesgo
de duplicar objetos ni borrar datos.

Decisiones de modelado:
  - `raw.diabetes_raw` guarda la fila original como JSONB (`payload`) más
    columnas de auditoría. Usar JSONB hace que la ingesta sea robusta
    frente a cambios de esquema en el CSV fuente sin tener que tocar el
    DDL.
  - `clean.diabetes_clean` mantiene el `target` tipado y un mapa JSONB
    `features` con el resto de variables. Así las decisiones de
    ingeniería de características viven en el módulo `preprocess` y no
    en el DDL.
  - `inference.predictions` corresponde a la fase de inferencia; se
    declara aquí para que el esquema esté listo cuando arranque la API.
"""

from __future__ import annotations

import logging

from pipeline.db.connection import connect

logger = logging.getLogger(__name__)

# Lista de sentencias DDL que se ejecutan en orden. Todas son idempotentes.
DDL = [
    # Schemas: separan claramente las tres capas que pide el enunciado.
    "CREATE SCHEMA IF NOT EXISTS raw",
    "CREATE SCHEMA IF NOT EXISTS clean",
    "CREATE SCHEMA IF NOT EXISTS inference",
    # Tabla de datos crudos. `row_hash` PK garantiza idempotencia: una
    # misma fila del CSV nunca se inserta dos veces aunque el DAG se
    # reejecute con el mismo offset.
    """
    CREATE TABLE IF NOT EXISTS raw.diabetes_raw (
        row_hash        TEXT PRIMARY KEY,
        batch_id        TEXT NOT NULL,
        load_timestamp  TIMESTAMPTZ NOT NULL DEFAULT now(),
        source_file     TEXT NOT NULL,
        status          TEXT NOT NULL DEFAULT 'loaded',
        payload         JSONB NOT NULL
    )
    """,
    # Índice por batch_id: facilita auditar qué filas cargó cada ejecución.
    "CREATE INDEX IF NOT EXISTS ix_raw_batch ON raw.diabetes_raw (batch_id)",
    # Tabla de datos procesados. La FK a raw.diabetes_raw permite trazar
    # cada fila limpia hasta su origen crudo (auditoría y reprocesamiento).
    """
    CREATE TABLE IF NOT EXISTS clean.diabetes_clean (
        id           BIGSERIAL PRIMARY KEY,
        row_hash     TEXT UNIQUE NOT NULL REFERENCES raw.diabetes_raw(row_hash),
        batch_id     TEXT NOT NULL,
        processed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        split        TEXT,
        features     JSONB NOT NULL,
        target       INT  NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_clean_batch ON clean.diabetes_clean (batch_id)",
    "CREATE INDEX IF NOT EXISTS ix_clean_split ON clean.diabetes_clean (split)",
    # Tabla de log de inferencias. Cada llamada a /predict inserta aquí
    # una fila para habilitar reentrenamiento, monitoreo y auditoría.
    """
    CREATE TABLE IF NOT EXISTS inference.predictions (
        request_id    UUID PRIMARY KEY,
        created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
        input_payload JSONB NOT NULL,
        prediction    INT,
        score         DOUBLE PRECISION,
        model_name    TEXT,
        model_version TEXT,
        latency_ms    DOUBLE PRECISION
    )
    """,
]


def run() -> None:
    """Aplica todas las sentencias DDL en una sola transacción.

    Como cada sentencia usa `IF NOT EXISTS`, ejecutarla varias veces no
    produce errores ni borra datos existentes.
    """
    with connect() as conn, conn.cursor() as cur:
        for stmt in DDL:
            cur.execute(stmt)
    logger.info("migraciones aplicadas: %d sentencias", len(DDL))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run()
