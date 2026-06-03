"""MLOps training pipeline for the Diabetes dataset.

Modules are designed to be invoked individually (CLI) or composed by an
Airflow DAG. Each step is idempotent and isolated by `batch_id`.
"""

__all__ = [
    "config",
    "ingest",
    "quality",
    "preprocess",
    "split",
    "train",
    "promote",
]
