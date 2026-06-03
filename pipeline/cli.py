"""Punto de entrada por línea de comandos para todas las etapas del pipeline.

Permite ejecutar cualquier paso de forma aislada (útil para depurar) o el
pipeline completo de extremo a extremo. Cada subcomando llama al módulo
correspondiente del paquete `pipeline`.

Uso:
    python -m pipeline.cli migrate
    python -m pipeline.cli ingest [--source PATH] [--batch-id ID]
    python -m pipeline.cli quality [--batch-id ID]
    python -m pipeline.cli preprocess [--batch-id ID]
    python -m pipeline.cli split [--batch-id ID]
    python -m pipeline.cli train [--batch-id ID] [--model {lr|rf}]
    python -m pipeline.cli promote
    python -m pipeline.cli all [--source PATH] [--batch-id ID]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from pipeline import ingest, preprocess, promote, quality, split, train
from pipeline.db import migrations


def _parser() -> argparse.ArgumentParser:
    """Construye el parser con todos los subcomandos disponibles."""
    p = argparse.ArgumentParser(prog="pipeline")
    sub = p.add_subparsers(dest="command", required=True)

    # Aplica las migraciones DDL (CREATE SCHEMA / CREATE TABLE).
    sub.add_parser("migrate", help="aplica las migraciones DDL")

    # Ingesta incremental de un lote del CSV a raw.diabetes_raw.
    p_ing = sub.add_parser("ingest", help="ingesta el CSV en la capa raw")
    p_ing.add_argument("--source")
    p_ing.add_argument("--batch-id", dest="batch_id")

    # Subcomandos que solo aceptan un batch-id opcional.
    for name in ("quality", "preprocess", "split"):
        sp = sub.add_parser(name)
        sp.add_argument("--batch-id", dest="batch_id")

    # train acepta además `--model` para entrenar un único candidato.
    # Sin la opción se entrenan ambos (backward-compat); con `--model lr`
    # o `--model rf` se entrena solo uno — útil para paralelizar como
    # `t_train_lr` y `t_train_rf` desde Airflow.
    p_train = sub.add_parser("train", help="entrena uno o todos los candidatos")
    p_train.add_argument("--batch-id", dest="batch_id")
    p_train.add_argument(
        "--model",
        choices=["lr", "rf", "logistic_regression", "random_forest"],
        help="entrena solo este candidato (default: ambos)",
    )

    sub.add_parser("promote", help="(requiere un candidato; usar `all` en su lugar)")

    # Pipeline completo: migrate → ingest → quality → preprocess →
    # split → train → promote, encadenando outputs por batch_id.
    p_all = sub.add_parser("all", help="pipeline completo (migrate→promote)")
    p_all.add_argument("--source")
    p_all.add_argument("--batch-id", dest="batch_id")

    return p


def _print(obj) -> None:
    """Imprime un dict como JSON formateado a stdout."""
    print(json.dumps(obj, default=str, indent=2))


def main(argv: list[str] | None = None) -> int:
    """Dispatcher principal: lee los args y llama al módulo correcto."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parser().parse_args(argv)

    if args.command == "migrate":
        migrations.run()
        _print({"ok": True})
        return 0

    if args.command == "ingest":
        _print(ingest.load_batch(path=args.source, batch_id=args.batch_id))
        return 0

    if args.command == "quality":
        _print(quality.run(batch_id=args.batch_id))
        return 0

    if args.command == "preprocess":
        _print(preprocess.run(batch_id=args.batch_id))
        return 0

    if args.command == "split":
        _print(split.run(batch_id=args.batch_id))
        return 0

    if args.command == "train":
        _print(train.run(batch_id=args.batch_id, model=getattr(args, "model", None)))
        return 0

    if args.command == "promote":
        # promote necesita el dict de candidato producido por train. No
        # se puede invocar aislado desde el CLI; sugerimos usar `all`.
        print(
            "error: promote requiere un candidato; usar `all` para encadenar train→promote",
            file=sys.stderr,
        )
        return 2

    if args.command == "all":
        # Ejecutamos el pipeline completo en orden. El batch_id que
        # genera la ingesta se propaga a las etapas siguientes para que
        # todas trabajen sobre el mismo lote.
        migrations.run()
        ingest_summary = ingest.load_batch(path=args.source, batch_id=args.batch_id)
        batch = ingest_summary["batch_id"]
        quality.run(batch_id=batch)
        preprocess.run(batch_id=batch)
        split.run(batch_id=batch)
        candidate = train.run(batch_id=batch)
        result = promote.promote(candidate)
        _print({"ingest": ingest_summary, "candidate": candidate, "promotion": result})
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
