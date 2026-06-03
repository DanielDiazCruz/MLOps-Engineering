"""DAG de entrenamiento MLOps — regresión de precio inmobiliario.

Flujo:

    migrate -> check_source -> load_batch -> quality -> preprocess -> split
            -> [validate_schema, detect_new_categories, detect_drift]
            -> decide -> branch ──► train -> promote ──┐
                              └──► skip_training ───────┴─► notify

Cada tarea es un envoltorio delgado alrededor de una función del paquete
`pipeline`. Los envoltorios existen para propagar `batch_id`/decisiones entre
tareas vía XCom y mantener el DAG declarativo.

Decisión automática de entrenar (RF4): tras procesar el lote, se calculan tres
señales (esquema, categorías nuevas, drift PSI) y `decide` resuelve si entrenar
o saltar. La tarea `branch` (BranchPythonOperator) dirige el flujo; `notify`
corre en ambas ramas (trigger_rule) y registra toda la corrida en
`audit.training_history` (RF4/RF9).

Idempotencia: `row_hash` UNIQUE en raw, upserts en clean, runs de MLflow
append-only; el DAG se puede reejecutar de forma segura.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow.sdk import dag, get_current_context, task

from pipeline import audit, decision, ingest, preprocess, promote, quality, split, train
from pipeline.db import migrations


DEFAULT_ARGS = {
    "owner": "mlops",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


@dag(
    dag_id="diabetes_mlops_pipeline",
    description="ingest -> quality -> preprocess -> split -> decisión -> train/skip -> promote -> audit",
    schedule="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["mlops", "real-estate", "regression"],
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

    # --- Señales para la decisión de entrenar (RF4) ---
    @task()
    def t_validate_schema(load_summary: dict) -> dict:
        return decision.validate_schema(batch_id=load_summary["batch_id"])

    @task()
    def t_detect_new_categories(load_summary: dict) -> dict:
        return decision.detect_new_categories(batch_id=load_summary["batch_id"])

    @task()
    def t_detect_drift(load_summary: dict) -> dict:
        return decision.detect_drift(batch_id=load_summary["batch_id"])

    @task()
    def t_decide(load_summary: dict, schema_info: dict,
                 new_categories: dict, drift: dict) -> dict:
        return decision.decide(
            batch_id=load_summary["batch_id"],
            schema_info=schema_info,
            new_categories=new_categories,
            drift=drift,
        )

    @task.branch()
    def t_branch(decision_result: dict) -> str:
        """Dirige el flujo: entrena el modelo o salta el entrenamiento."""
        return "t_train" if decision_result["decision"] == "train" else "t_skip_training"

    @task()
    def t_train(load_summary: dict, decision_result: dict) -> dict:
        # Entrena los candidatos de regresión (Ridge + HistGBR) y devuelve el
        # mejor por MAE. Se registra el motivo (RF5) como tag del run.
        return train.run(
            batch_id=load_summary["batch_id"],
            reason=decision_result.get("reason"),
        )

    @task()
    def t_promote(candidate: dict) -> dict:
        return promote.promote_best([candidate])

    @task()
    def t_skip_training() -> None:
        """Rama de 'no entrenar': no hace nada; la auditoría registra el motivo."""
        return None

    @task(trigger_rule="none_failed_min_one_success")
    def t_notify(load_summary: dict, quality_report: dict, schema_info: dict,
                 new_categories: dict, drift: dict, decision_result: dict) -> dict:
        """Registra la corrida completa en audit.training_history (RF4/RF9).

        Corre tanto si se entrenó como si se saltó (trigger_rule). Los datos de
        las tareas que pueden quedar 'skipped' (train/promote) se leen por XCom
        y serán None en la rama de salto.
        """
        ctx = get_current_context()
        ti = ctx["ti"]
        best = ti.xcom_pull(task_ids="t_train")       # None si se saltó
        promotion = ti.xcom_pull(task_ids="t_promote")  # None si se saltó

        trained = bool(best)
        audit_id = audit.record_run(
            batch_id=load_summary.get("batch_id"),
            dag_run_id=ctx.get("run_id"),
            n_records_batch=decision_result.get("n_new"),
            n_records_total=decision_result.get("n_total"),
            schema_info=schema_info,
            new_categories=new_categories,
            validations=quality_report,
            drift=drift,
            decision=decision_result.get("decision"),
            decision_reason=decision_result.get("reason"),
            trained=trained,
            candidate_metrics=(best or {}).get("metrics") if trained else None,
            promoted=bool(promotion and promotion.get("promoted")),
            promotion_reason=(promotion or {}).get("reason") if promotion
                             else decision_result.get("reason"),
            champion_metrics=(promotion or {}).get("champion_metrics") if promotion else None,
            mlflow_run_id=(best or {}).get("run_id") if trained else None,
            mlflow_model_version=(str((best or {}).get("version"))
                                  if trained and (best or {}).get("version") is not None
                                  else None),
        )
        return {"audit_id": audit_id, "trained": trained,
                "decision": decision_result.get("decision")}

    # --- Cableado ---
    migrate = t_migrate()
    src = t_check_source()
    loaded = t_load_batch(src)
    qual = t_quality(loaded)
    prep = t_preprocess(loaded)
    sp = t_split(loaded)

    schema = t_validate_schema(loaded)
    cats = t_detect_new_categories(loaded)
    drift = t_detect_drift(loaded)
    decided = t_decide(loaded, schema, cats, drift)
    branch = t_branch(decided)

    trained = t_train(loaded, decided)
    promotion = t_promote(trained)
    skipped = t_skip_training()

    notify = t_notify(loaded, qual, schema, cats, drift, decided)

    # Orden de ingesta/preparación.
    migrate >> src >> loaded >> qual >> prep >> sp
    # Las señales se calculan sobre el lote ya en clean.
    sp >> [schema, cats, drift] >> decided >> branch
    # Bifurcación: o se entrena+promueve, o se salta.
    branch >> trained >> promotion
    branch >> skipped
    # La auditoría cierra ambas ramas.
    [promotion, skipped] >> notify


diabetes_mlops_pipeline()
