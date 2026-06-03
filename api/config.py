"""Configuración de la API cargada desde variables de entorno.

Mantiene la misma filosofía que `pipeline/config.py`: defaults pensados
para correr dentro del cluster de Kubernetes, sobreescribibles vía env
para ejecución local con port-forwards.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _require(name: str) -> str:
    """Lee una variable de entorno sensible o falla con un mensaje claro.

    Las credenciales (DSN de Postgres, llaves de MinIO) NO tienen valor por
    defecto en el código: se inyectan vía Secret de Kubernetes (`secretRef`).
    Así no queda ninguna credencial hardcodeada en el repositorio.
    """
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"falta la variable de entorno requerida '{name}'. "
            "Inyéctala con un Secret de Kubernetes o expórtala en local."
        )
    return value


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
    # Token para proteger el endpoint admin /reload-model (RF7). Si está vacío,
    # el endpoint queda abierto (modo dev); en el cluster se inyecta por Secret.
    reload_token: str


def load() -> Settings:
    """Construye un Settings leyendo variables de entorno.

    Como efecto colateral, vuelca las credenciales AWS/MinIO al entorno
    del proceso. boto3 (que usa MLflow para descargar artefactos) los
    lee de allí directamente.
    """
    s = Settings(
        # Credenciales: requeridas, sin default en código (vienen de Secrets).
        pg_dsn=_require("PG_DSN"),
        mlflow_tracking_uri=os.environ.get(
            "MLFLOW_TRACKING_URI",
            "http://mlflow-service.mlops.svc.cluster.local:5000",
        ),
        s3_endpoint_url=os.environ.get(
            "MLFLOW_S3_ENDPOINT_URL",
            "http://minio-service.mlops.svc.cluster.local:9000",
        ),
        aws_access_key_id=_require("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=_require("AWS_SECRET_ACCESS_KEY"),
        registered_model_name=os.environ.get("MLFLOW_MODEL_NAME", "property-price-regressor"),
        champion_alias=os.environ.get("CHAMPION_ALIAS", "champion"),
        model_cache_ttl_seconds=int(os.environ.get("MODEL_CACHE_TTL_SECONDS", "300")),
        reload_token=os.environ.get("RELOAD_TOKEN", ""),
    )
    # Publicamos las credenciales en el entorno para que boto3 / mlflow
    # las usen al leer artefactos desde MinIO.
    os.environ["AWS_ACCESS_KEY_ID"] = s.aws_access_key_id
    os.environ["AWS_SECRET_ACCESS_KEY"] = s.aws_secret_access_key
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = s.s3_endpoint_url
    return s
