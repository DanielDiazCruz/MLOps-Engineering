"""Configuración de la API cargada desde variables de entorno.

Mantiene la misma filosofía que `pipeline/config.py`: defaults pensados
para correr dentro del cluster de Kubernetes, sobreescribibles vía env
para ejecución local con port-forwards.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # DSN de PostgreSQL para registrar cada inferencia.
    pg_dsn: str
    # Tracking URI de MLflow desde donde se resuelve el modelo champion.
    mlflow_tracking_uri: str
    # Endpoint S3-compatible (MinIO) para descargar los artefactos.
    s3_endpoint_url: str
    aws_access_key_id: str
    aws_secret_access_key: str
    # Nombre del modelo registrado y alias que marca la versión productiva.
    registered_model_name: str
    champion_alias: str
    # TTL del cache en memoria del modelo; tras vencerse, la siguiente
    # predicción re-resuelve el alias y recarga el modelo si cambió.
    model_cache_ttl_seconds: int


def load() -> Settings:
    """Construye un Settings leyendo variables de entorno.

    Como efecto colateral, vuelca las credenciales AWS/MinIO al entorno
    del proceso. boto3 (que usa MLflow para descargar artefactos) los
    lee de allí directamente.
    """
    s = Settings(
        pg_dsn=os.environ.get(
            "PG_DSN",
            "postgresql://mlops_user:mlops_pass_2026@postgres-service.mlops.svc.cluster.local:5432/mlops",
        ),
        mlflow_tracking_uri=os.environ.get(
            "MLFLOW_TRACKING_URI",
            "http://mlflow-service.mlops.svc.cluster.local:5000",
        ),
        s3_endpoint_url=os.environ.get(
            "MLFLOW_S3_ENDPOINT_URL",
            "http://minio-service.mlops.svc.cluster.local:9000",
        ),
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin123"),
        registered_model_name=os.environ.get("MLFLOW_MODEL_NAME", "diabetes-classifier"),
        champion_alias=os.environ.get("CHAMPION_ALIAS", "champion"),
        model_cache_ttl_seconds=int(os.environ.get("MODEL_CACHE_TTL_SECONDS", "300")),
    )
    # Publicamos las credenciales en el entorno para que boto3 / mlflow
    # las usen al leer artefactos desde MinIO.
    os.environ["AWS_ACCESS_KEY_ID"] = s.aws_access_key_id
    os.environ["AWS_SECRET_ACCESS_KEY"] = s.aws_secret_access_key
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = s.s3_endpoint_url
    return s
