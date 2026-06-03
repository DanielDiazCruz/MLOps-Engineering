"""DAG de entrenamiento MLOps de Diabetes.

Cada tarea es un envoltorio delgado alrededor de una función del paquete
`pipeline` del proyecto. Los envoltorios existen solo para:
  - propagar `batch_id` entre tareas vía XCom
  - mantener el archivo DAG declarativo

Si una tarea falla, las tareas descendentes no se ejecutan (regla por
defecto `all_success`), el error es visible en la UI de Airflow, y el DAG
puede ser reejecutado de forma segura porque cada paso es idempotente
(`row_hash` UNIQUE en raw, upserts en clean, los runs de MLflow son
append-only).
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow.sdk import dag, task

from pipeline import ingest, preprocess, promote, quality, split, train
from pipeline.db import migrations


DEFAULT_ARGS = {
    "owner": "mlops",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


@dag(
    dag_id="diabetes_mlops_pipeline",
    description="Ingest -> quality -> preprocess -> split -> train -> promote",
    schedule="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["mlops", "diabetes"],
)
def diabetes_mlops_pipeline():

    @task()
    def t_migrate() -> None:
        migrations.run()

    @task()
    def t_check_source() -> str:
        return ingest.check_source()

    @task()
    def t_load_batch(_source: str) -> dict:
        return ingest.load_batch()

    @task()
    def t_quality(load_summary: dict) -> dict:
        return quality.run(batch_id=load_summary["batch_id"])

    @task()
    def t_preprocess(load_summary: dict) -> dict:
        return preprocess.run(batch_id=load_summary["batch_id"])

    @task()
    def t_split(load_summary: dict) -> dict:
        return split.run(batch_id=load_summary["batch_id"])

    @task()
    def t_train(load_summary: dict) -> dict:
        # Solo LogisticRegression: el fit es de segundos y evita el segundo
        # upload a MLflow, que es el cuello de botella real del entrenamiento
        # en este cluster. Para volver a entrenar ambos en paralelo, restaura
        # t_train_lr/t_train_rf y promote_best([lr, rf]).
        return train.run(batch_id=load_summary["batch_id"], model="lr")

    @task()
    def t_promote(candidate: dict) -> dict:
        return promote.promote_best([candidate])

    migrate = t_migrate()
    src = t_check_source()
    loaded = t_load_batch(src)
    qual = t_quality(loaded)
    prep = t_preprocess(loaded)
    sp = t_split(loaded)
    trained = t_train(loaded)
    promotion = t_promote(trained)

    migrate >> src >> loaded >> qual >> prep >> sp >> trained >> promotion


diabetes_mlops_pipeline()
