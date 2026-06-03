"""DDL idempotente para las capas raw / clean / inference / audit.

Aplica `CREATE SCHEMA IF NOT EXISTS` y `CREATE TABLE IF NOT EXISTS`, por lo
que se puede invocar de forma segura en cada ejecución del DAG sin riesgo
de duplicar objetos ni borrar datos.

Dominio: predicción de PRECIO de propiedades inmobiliarias (regresión). Los
datos llegan por lotes desde una API externa y se persisten primero crudos
(RAW) y luego procesados (CLEAN), manteniendo trazabilidad por lote.

Decisiones de modelado:
  - `raw.properties_raw` guarda la fila original como JSONB (`payload`) más
    columnas de auditoría. Usar JSONB hace que la ingesta sea robusta frente
    a cambios de esquema en la API sin tener que tocar el DDL.
  - `clean.properties_clean` mantiene el `target` (precio) tipado como
    DOUBLE PRECISION (regresión) y un mapa JSONB `features` con el resto de
    variables. Así la ingeniería de características vive en `preprocess` y no
    en el DDL.
  - `inference.predictions` registra cada llamada a /predict (RF8): entrada,
    predicción (precio), versión del modelo, estado y error si lo hubo.
  - `audit.training_history` registra la decisión por lote (RF4/RF9): si se
    entrenó o no y por qué, métricas del candidato, si se promovió o no, el
    delta frente al productivo y los identificadores de MLflow. Es la fuente
    del "Historial de entrenamiento" que muestra Streamlit.
"""

from __future__ import annotations

import logging

from pipeline.db.connection import connect

logger = logging.getLogger(__name__)

# Lista de sentencias DDL que se ejecutan en orden. Todas son idempotentes.
DDL = [
    # Schemas: separan claramente las capas que pide el enunciado.
    "CREATE SCHEMA IF NOT EXISTS raw",
    "CREATE SCHEMA IF NOT EXISTS clean",
    "CREATE SCHEMA IF NOT EXISTS inference",
    "CREATE SCHEMA IF NOT EXISTS audit",
    # --- RAW: lotes tal como llegan de la API ---
    # `row_hash` PK garantiza idempotencia: una misma fila nunca se inserta
    # dos veces aunque el DAG reprocese un lote.
    """
    CREATE TABLE IF NOT EXISTS raw.properties_raw (
        row_hash        TEXT PRIMARY KEY,
        batch_id        TEXT NOT NULL,
        load_timestamp  TIMESTAMPTZ NOT NULL DEFAULT now(),
        source          TEXT NOT NULL,
        status          TEXT NOT NULL DEFAULT 'loaded',
        payload         JSONB NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_raw_batch ON raw.properties_raw (batch_id)",
    "CREATE INDEX IF NOT EXISTS ix_raw_status ON raw.properties_raw (status)",
    # --- CLEAN: datos procesados y listos para entrenar ---
    # FK a raw.properties_raw para trazar cada fila limpia hasta su origen.
    # `target` (precio) es DOUBLE PRECISION porque el problema es regresión.
    """
    CREATE TABLE IF NOT EXISTS clean.properties_clean (
        id           BIGSERIAL PRIMARY KEY,
        row_hash     TEXT UNIQUE NOT NULL REFERENCES raw.properties_raw(row_hash),
        batch_id     TEXT NOT NULL,
        processed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        split        TEXT,
        features     JSONB NOT NULL,
        target       DOUBLE PRECISION NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_clean_batch ON clean.properties_clean (batch_id)",
    "CREATE INDEX IF NOT EXISTS ix_clean_split ON clean.properties_clean (split)",
    # --- INFERENCE: log de cada predicción (RF8) ---
    # `prediction` es DOUBLE PRECISION (precio estimado). Guardamos estado y
    # error para poder auditar peticiones fallidas, no solo las exitosas.
    """
    CREATE TABLE IF NOT EXISTS inference.predictions (
        request_id    UUID PRIMARY KEY,
        created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
        input_payload JSONB NOT NULL,
        prediction    DOUBLE PRECISION,
        model_name    TEXT,
        model_version TEXT,
        status        TEXT NOT NULL DEFAULT 'ok',
        error         TEXT,
        latency_ms    DOUBLE PRECISION
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_inf_created ON inference.predictions (created_at)",
    # Convergencia de esquema: la BD trae la tabla del proyecto anterior
    # (clasificación: prediction INTEGER + columna score, sin status/error). El
    # CREATE de arriba no la modifica porque ya existe, así que la migramos al
    # esquema de regresión con ALTERs idempotentes (solo actúan una vez).
    """
    DO $$
    BEGIN
        IF EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_schema = 'inference' AND table_name = 'predictions'
              AND column_name = 'prediction' AND data_type = 'integer'
        ) THEN
            ALTER TABLE inference.predictions
                ALTER COLUMN prediction TYPE DOUBLE PRECISION
                USING prediction::double precision;
        END IF;
    END $$
    """,
    "ALTER TABLE inference.predictions ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'ok'",
    "ALTER TABLE inference.predictions ADD COLUMN IF NOT EXISTS error TEXT",
    "ALTER TABLE inference.predictions DROP COLUMN IF EXISTS score",
    # --- AUDIT: historial de decisiones por lote (RF4/RF9) ---
    # Una fila por ejecución del DAG. Captura la decisión de entrenar y de
    # promover, con su razón, las métricas y los IDs de MLflow asociados.
    """
    CREATE TABLE IF NOT EXISTS audit.training_history (
        id                   BIGSERIAL PRIMARY KEY,
        batch_id             TEXT,
        dag_run_id           TEXT,
        executed_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
        n_records_batch      INT,
        n_records_total      INT,
        schema_info          JSONB,
        new_categories       JSONB,
        validations          JSONB,
        drift                JSONB,
        decision             TEXT,          -- 'train' | 'skip'
        decision_reason      TEXT,
        trained              BOOLEAN NOT NULL DEFAULT FALSE,
        candidate_metrics    JSONB,
        promoted             BOOLEAN NOT NULL DEFAULT FALSE,
        promotion_reason     TEXT,
        champion_metrics     JSONB,
        mlflow_run_id        TEXT,
        mlflow_model_version TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_audit_executed ON audit.training_history (executed_at)",
    "CREATE INDEX IF NOT EXISTS ix_audit_batch ON audit.training_history (batch_id)",
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
