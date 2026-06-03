"""Perfila las fases de t_preprocess y t_split para ubicar su costo.

Mide por separado lectura / transformación / escritura. Las escrituras se
ejecutan de verdad (para medir su costo real) pero se hace ROLLBACK al
final, así NO muta la base.

Ejecutar dentro de un pod con el paquete `pipeline` y acceso a Postgres
(p. ej. airflow-scheduler-0), con el nodo ocioso.

Uso:  python profile_preprocess.py
"""

from __future__ import annotations

import time

import numpy as np
import psycopg2
from psycopg2.extras import Json, execute_batch

from pipeline.config import load
from pipeline.preprocess import _binarize_target, _detect_target, _read_batch, LOW_CARD_MAX


def _phase(label, fn):
    t0 = time.perf_counter()
    out = fn()
    print(f"[{time.perf_counter()-t0:8.2f}s] {label}")
    return out


def main() -> None:
    s = load()
    print("=== profile_preprocess + split (escrituras con ROLLBACK) ===")

    # ---------- PREPROCESS ----------
    df = _phase("preprocess.read (todo raw status=loaded)", lambda: _read_batch(None))
    print(f"   filas_raw={len(df)}")
    target_col = _detect_target(df.columns)

    def _transform():
        y = _binarize_target(df[target_col])
        fdf = df.drop(columns=["row_hash", "batch_id", target_col])
        num = fdf.select_dtypes(include=[np.number]).columns.tolist()
        cat = [c for c in fdf.columns if c not in num]
        for c in num:
            fdf[c] = fdf[c].fillna(fdf[c].median())
        high = [c for c in cat if fdf[c].nunique() > LOW_CARD_MAX]
        fdf = fdf.drop(columns=high)
        cat = [c for c in cat if c not in high]
        for c in cat:
            fdf[c] = fdf[c].astype(str)
        return y, fdf

    y, fdf = _phase("preprocess.transform (impute/drop highcard/astype)", _transform)
    print(f"   features={len(fdf.columns)}")

    upsert_sql = (
        "INSERT INTO clean.diabetes_clean (row_hash, batch_id, features, target) "
        "VALUES (%s, %s, %s, %s) ON CONFLICT (row_hash) DO UPDATE SET "
        "batch_id=EXCLUDED.batch_id, features=EXCLUDED.features, "
        "target=EXCLUDED.target, processed_at=now(), split=NULL"
    )
    rows = [(h, b, Json(f), int(t)) for h, b, f, t in
            zip(df["row_hash"], df["batch_id"], fdf.to_dict(orient="records"), y)]

    conn = psycopg2.connect(s.pg_dsn)
    try:
        cur = conn.cursor()
        _phase(f"preprocess.write UPSERT {len(rows)} filas (ROLLBACK)",
               lambda: execute_batch(cur, upsert_sql, rows, page_size=1000))
        conn.rollback()
    finally:
        conn.close()

    # ---------- SPLIT ----------
    from sklearn.model_selection import train_test_split

    conn = psycopg2.connect(s.pg_dsn)
    try:
        cur = conn.cursor()
        rows2 = _phase("split.read (id,target de todo clean)",
                       lambda: (cur.execute("SELECT id, target FROM clean.diabetes_clean"),
                                cur.fetchall())[1])
        print(f"   filas_clean={len(rows2)}")
        ids = [r[0] for r in rows2]
        yv = [r[1] for r in rows2]

        def _do_split():
            itr, itmp, ytr, ytmp = train_test_split(ids, yv, test_size=0.30,
                                                    random_state=s.random_seed, stratify=yv)
            iv, ite, _, _ = train_test_split(itmp, ytmp, test_size=0.50,
                                             random_state=s.random_seed, stratify=ytmp)
            return ([(i, "train") for i in itr] + [(i, "val") for i in iv]
                    + [(i, "test") for i in ite])

        assigns = _phase("split.compute train_test_split", _do_split)
        _phase(f"split.write UPDATE {len(assigns)} filas (ROLLBACK)",
               lambda: execute_batch(cur, "UPDATE clean.diabetes_clean SET split=%s WHERE id=%s",
                                     [(sp, i) for i, sp in assigns], page_size=1000))
        conn.rollback()
    finally:
        conn.close()


if __name__ == "__main__":
    main()
