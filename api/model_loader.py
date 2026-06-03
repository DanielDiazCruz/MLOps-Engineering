"""Cache thread-safe del modelo MLflow con TTL y recarga manual.

Estrategia de carga del modelo en la API:
  - En la primera petición (o en el startup vía lifespan) se descarga
    desde MLflow el modelo identificado por el alias `champion`. Los
    artefactos vienen de MinIO a través del endpoint S3-compatible.
  - El modelo cargado + el número de versión del registry quedan en
    memoria como un `LoadedModel`.
  - El cache expira tras `model_cache_ttl_seconds`. La siguiente
    predicción dispara una nueva resolución del alias y recarga el
    modelo si cambió. Así, una promoción se propaga automáticamente
    sin redeploy, con un retraso máximo igual al TTL.
  - `POST /reload-model` fuerza un refresh inmediato (útil al terminar
    el DAG para no esperar el TTL).

El contrato evita deliberadamente cualquier ruta a archivos locales:
la única fuente de verdad del modelo es el Model Registry de MLflow.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from api.config import Settings, load

logger = logging.getLogger(__name__)


@dataclass
class LoadedModel:
    """Representa un modelo que está cargado en memoria."""

    model: Any
    name: str
    version: str
    alias: str
    loaded_at: float  # epoch (segundos) en el que se cargó


class ModelCache:
    """Cache singleton con lock para que múltiples requests concurrentes
    no recarguen el modelo al mismo tiempo.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loaded: LoadedModel | None = None
        self._settings: Settings = load()
        mlflow.set_tracking_uri(self._settings.mlflow_tracking_uri)
        self._client = MlflowClient()

    def get(self) -> LoadedModel:
        """Devuelve el modelo cacheado; lo recarga si está vencido o ausente."""
        with self._lock:
            if self._loaded is None or self._is_expired(self._loaded):
                self._loaded = self._load()
            return self._loaded

    def reload(self) -> LoadedModel:
        """Fuerza una recarga inmediata, ignorando el TTL."""
        with self._lock:
            self._loaded = self._load()
            return self._loaded

    # ----- internos -------------------------------------------------------

    def _is_expired(self, lm: LoadedModel) -> bool:
        """¿El modelo cacheado superó el TTL configurado?

        Un TTL <= 0 desactiva la expiración (el modelo se mantiene en
        memoria hasta un reload manual o reinicio del pod).
        """
        ttl = self._settings.model_cache_ttl_seconds
        if ttl <= 0:
            return False
        return (time.time() - lm.loaded_at) > ttl

    def _load(self) -> LoadedModel:
        """Resuelve el alias champion en MLflow y descarga el modelo."""
        s = self._settings
        # Resolución del alias → versión concreta (entero).
        version = self._client.get_model_version_by_alias(
            s.registered_model_name, s.champion_alias
        )
        # Descarga el modelo como pyfunc para que .predict() acepte un
        # DataFrame sin que tengamos que conocer el flavor exacto.
        uri = f"models:/{s.registered_model_name}@{s.champion_alias}"
        logger.info("cargando modelo %s versión=%s", uri, version.version)
        model = mlflow.pyfunc.load_model(uri)
        return LoadedModel(
            model=model,
            name=s.registered_model_name,
            version=str(version.version),
            alias=s.champion_alias,
            loaded_at=time.time(),
        )


# Instancia singleton del cache. Se inicializa perezosamente para que
# importar este módulo no produzca side-effects (útil en tests).
_cache: ModelCache | None = None


def get_cache() -> ModelCache:
    """Devuelve el ModelCache singleton, creándolo en la primera llamada."""
    global _cache
    if _cache is None:
        _cache = ModelCache()
    return _cache
