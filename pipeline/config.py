"""Configuración centralizada leída desde variables de entorno.

Los valores por defecto asumen que el código corre dentro del cluster de
Kubernetes (servicios resueltos por DNS interno). Para ejecutar localmente,
abre los port-forwards y exporta las variables correspondientes antes de
invocar el CLI.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _require(name: str) -> str:
    """Lee una variable de entorno sensible o falla con un mensaje claro.

    Las credenciales (DSN de Postgres, llaves de MinIO) NO tienen valor por
    defecto en el código: deben inyectarse vía Secret de Kubernetes
    (`secretRef`/`envFrom`) o exportarse a mano en local. Así no queda ninguna
    credencial hardcodeada en el repositorio.
    """
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"falta la variable de entorno requerida '{name}'. "
            "Inyéctala con un Secret de Kubernetes (en el cluster) o expórtala "
            "antes de ejecutar localmente."
        )
    return value


@dataclass(frozen=True)
class Settings:
    # Conexión a PostgreSQL (datos raw / clean / inferencias / auditoría)
    pg_dsn: str
    # URL del servidor MLflow para tracking y registry
    mlflow_tracking_uri: str
    # Endpoint S3-compatible (MinIO) donde MLflow guarda los artefactos
    s3_endpoint_url: str
    aws_access_key_id: str
    aws_secret_access_key: str

    # API externa de datos (fuente de los lotes inmobiliarios)
    data_api_url: str
    # Número de grupo que la API usa para entregar el subconjunto de datos
    data_api_group: int
    # Timeout y reintentos del cliente HTTP de la API de datos
    data_api_timeout: int
    data_api_retries: int

    # Semilla para reproducibilidad del split y los modelos
    random_seed: int

    # Identificadores usados en MLflow
    experiment_name: str
    registered_model_name: str
    champion_alias: str
    # Nombre de la métrica que decide qué modelo se promueve. Para regresión
    # de precios usamos MAE (menor es mejor); promote.py conoce la dirección.
    primary_metric: str


def load() -> Settings:
    """Construye un objeto Settings leyendo cada variable de entorno.

    Si la variable no existe, se usa un valor por defecto pensado para el
    cluster de Kubernetes (resoluciones tipo `*.mlops.svc.cluster.local`).
    """
    return Settings(
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
        data_api_url=os.environ.get(
            "DATA_API_URL",
            "http://data-api-service.mlops.svc.cluster.local",
        ),
        data_api_group=int(os.environ.get("DATA_API_GROUP", "1")),
        data_api_timeout=int(os.environ.get("DATA_API_TIMEOUT", "60")),
        data_api_retries=int(os.environ.get("DATA_API_RETRIES", "4")),
        random_seed=int(os.environ.get("RANDOM_SEED", "42")),
        experiment_name=os.environ.get("MLFLOW_EXPERIMENT", "real-estate-price"),
        registered_model_name=os.environ.get("MLFLOW_MODEL_NAME", "property-price-regressor"),
        champion_alias=os.environ.get("CHAMPION_ALIAS", "champion"),
        primary_metric=os.environ.get("PRIMARY_METRIC", "mae"),
    )


def export_aws_env(settings: Settings) -> None:
    """Publica las credenciales de S3/MinIO como variables de entorno.

    Boto3 y MLflow leen estas variables directamente del entorno del
    proceso para autenticarse contra el bucket de artefactos. Llamar a
    esta función garantiza que los valores resueltos en `Settings`
    queden disponibles para esas librerías aunque hayan sido cargados
    desde otra fuente (p. ej. un Secret de Kubernetes).
    """
    os.environ["AWS_ACCESS_KEY_ID"] = settings.aws_access_key_id
    os.environ["AWS_SECRET_ACCESS_KEY"] = settings.aws_secret_access_key
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = settings.s3_endpoint_url
