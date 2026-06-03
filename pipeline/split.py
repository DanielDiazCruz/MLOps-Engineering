"""Split aleatorio train / val / test (70 / 15 / 15) para clean.properties_clean.

El resultado se persiste en la columna `split` para que el entrenamiento lo
recupere sin rehacer la partición. Se usa la semilla configurada
(`random_seed`) para reproducibilidad.

Como el problema es de REGRESIÓN (precio continuo), el split es aleatorio
simple — no estratificado (la estratificación requiere clases discretas).

Procesamiento INCREMENTAL: solo particiona las filas nuevas (`split IS NULL`,
recién escritas por preprocess) y preserva las asignaciones existentes. Esto
es más rápido (no reescribe todo clean) y evita fuga de datos entre runs
(una fila de train nunca termina en test en una corrida posterior). Cada
batch nuevo se reparte 70/15/15, así la proporción global se mantiene.
"""

from __future__ import annotations

import logging

from psycopg2.extras import execute_values
from sklearn.model_selection import train_test_split

from pipeline.config import load
from pipeline.db.connection import connect

logger = logging.getLogger(__name__)


def run(batch_id: str | None = None) -> dict:
    """Asigna train / val / test a las filas nuevas de clean.properties_clean."""
    settings = load()

    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT id FROM clean.properties_clean WHERE split IS NULL")
        ids = [r[0] for r in cur.fetchall()]

    if not ids:
        summary = {"batch_id": batch_id, "train": 0, "val": 0, "test": 0,
                   "nota": "sin filas nuevas (split IS NULL)"}
        logger.info("split: nada que particionar — %s", summary)
        return summary

    # Corte 1: 70% train vs 30% (val+test). Corte 2: ese 30% mitad y mitad.
    ids_train, ids_tmp = train_test_split(
        ids, test_size=0.30, random_state=settings.random_seed,
    )
    ids_val, ids_test = train_test_split(
        ids_tmp, test_size=0.50, random_state=settings.random_seed,
    )

    assignments = (
        [(i, "train") for i in ids_train]
        + [(i, "val") for i in ids_val]
        + [(i, "test") for i in ids_test]
    )

    with connect() as conn, conn.cursor() as cur:
        # UPDATE masivo vía VALUES + join: rápido incluso con lotes grandes.
        execute_values(
            cur,
            "UPDATE clean.properties_clean AS c SET split = v.split "
            "FROM (VALUES %s) AS v(id, split) WHERE c.id = v.id",
            [(i, s) for i, s in assignments],
            template="(%s,%s)",
            page_size=5_000,
        )

    summary = {
        "batch_id": batch_id,
        "train": len(ids_train),
        "val": len(ids_val),
        "test": len(ids_test),
        "seed": settings.random_seed,
    }
    logger.info("split finalizado: %s", summary)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print(run())
