"""
Smoke test para la fase add-mlflow-tracking.

Uso (desde fuera del cluster, con port-forward activo):
    kubectl -n mlops port-forward svc/mlflow-service 5000:5000 &
    export MLFLOW_TRACKING_URI=http://localhost:5000
    export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000   # port-forward minio-service 9000:9000
    export AWS_ACCESS_KEY_ID=minioadmin
    export AWS_SECRET_ACCESS_KEY=minioadmin123
    python scripts/mlflow_smoke.py

Exit 0 = smoke test passed. Exit 1 = fallo (ver mensaje).
"""
from __future__ import annotations

import os
import sys
import tempfile
import pathlib
import mlflow
from mlflow.tracking import MlflowClient


EXPERIMENT_NAME = "smoke-test"
MODEL_NAME = "smoke-model"


def main() -> None:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    print(f"[smoke] Tracking URI: {tracking_uri}")

    # 1. Crear / recuperar experimento
    experiment = mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"[smoke] Experimento '{EXPERIMENT_NAME}' id={experiment.experiment_id}")

    # 2. Loguear parámetros, métrica y artefacto
    with mlflow.start_run(run_name="smoke-run") as run:
        run_id = run.info.run_id
        mlflow.log_param("param_test", "hello")
        mlflow.log_metric("metric_test", 0.42)

        with tempfile.TemporaryDirectory() as tmp:
            artifact_path = pathlib.Path(tmp) / "smoke.txt"
            artifact_path.write_text("smoke test artifact")
            mlflow.log_artifact(str(artifact_path))

        print(f"[smoke] Run completado: run_id={run_id}")

    # 3. Verificar que el run existe en el backend
    run_data = client.get_run(run_id)
    assert run_data.data.params.get("param_test") == "hello", "Parámetro no encontrado en backend"
    assert abs(run_data.data.metrics.get("metric_test", -1) - 0.42) < 1e-6, "Métrica no encontrada"
    print("[smoke] Metadatos verificados en Postgres OK")

    # 4. Verificar que el artefacto existe (lista > 0 items)
    artifacts = client.list_artifacts(run_id)
    assert len(artifacts) > 0, "Artefacto no encontrado en MinIO"
    print(f"[smoke] Artefactos en MinIO: {[a.path for a in artifacts]}")

    print("[smoke] PASSED — MLflow tracking + Postgres + MinIO OK")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"[smoke] FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"[smoke] ERROR inesperado: {exc}", file=sys.stderr)
        sys.exit(1)
