"""Split estratificado train / val / test (70 / 15 / 15).

El resultado se persiste en la columna `split` de clean.diabetes_clean
para que el módulo de entrenamiento pueda recuperarlo después sin tener
que rehacer la partición. Usamos la semilla configurada (`random_seed`)
para garantizar reproducibilidad.

Procesamiento INCREMENTAL: solo se particionan las filas nuevas (las que
tienen `split IS NULL`, recién escritas por preprocess). Las asignaciones
existentes se preservan. Esto no solo es más rápido (no reescribe todo
clean en cada run), sino más correcto: antes se re-particionaba todo en
cada ejecución, de modo que una fila usada para entrenar podía pasar a test
en el siguiente run (fuga de datos entre runs). Cada batch nuevo se reparte
70/15/15 de forma independiente, así que la proporción global se mantiene.
"""

from __future__ import annotations

import logging

from psycopg2.extras import execute_batch
from sklearn.model_selection import train_test_split

from pipeline.config import load
from pipeline.db.connection import connect

logger = logging.getLogger(__name__)


def run(batch_id: str | None = None) -> dict:
    """Asigna train / val / test a las filas nuevas de clean.diabetes_clean.

    Estratifica por la variable target para preservar la proporción de
    clases en cada subconjunto (importa porque solo ~11% de las filas son
    positivos). Si el batch nuevo tuviera una sola clase, cae a un split no
    estratificado para no fallar.
    """
    settings = load()

    # Solo las filas sin split (las que acaba de escribir preprocess).
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT id, target FROM clean.diabetes_clean WHERE split IS NULL")
        rows = cur.fetchall()

    if not rows:
        summary = {"batch_id": batch_id, "train": 0, "val": 0, "test": 0,
                   "nota": "sin filas nuevas (split IS NULL)"}
        logger.info("split: nada que particionar — %s", summary)
        return summary

    ids = [r[0] for r in rows]
    y = [r[1] for r in rows]
    # Estratificamos solo si hay al menos dos clases en el batch nuevo.
    stratify = y if len(set(y)) > 1 else None

    # Primer corte: 70% train del 30% restante (val + test).
    ids_train, ids_tmp, y_train, y_tmp = train_test_split(
        ids, y, test_size=0.30, random_state=settings.random_seed, stratify=stratify,
    )
    # Segundo corte: dividimos ese 30% por la mitad → 15% val, 15% test.
    stratify_tmp = y_tmp if len(set(y_tmp)) > 1 else None
    ids_val, ids_test, _, _ = train_test_split(
        ids_tmp, y_tmp, test_size=0.50, random_state=settings.random_seed, stratify=stratify_tmp,
    )

    assignments = (
        [(row_id, "train") for row_id in ids_train]
        + [(row_id, "val") for row_id in ids_val]
        + [(row_id, "test") for row_id in ids_test]
    )

    # Persistimos el split solo de estas filas (execute_batch en bloques de
    # 1000 para reducir el round-trip con Postgres).
    with connect() as conn, conn.cursor() as cur:
        execute_batch(
            cur,
            "UPDATE clean.diabetes_clean SET split = %s WHERE id = %s",
            [(s, i) for i, s in assignments],
            page_size=1_000,
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
