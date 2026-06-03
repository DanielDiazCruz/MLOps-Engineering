"""Envoltorio de cliente HTTP para la API de inferencia.

Todas las llamadas de red desde la UI pasan por este módulo para que el
manejo de errores, timeouts y la URL base se definan en un solo lugar.
La UI nunca importa mlflow, psycopg2 o ningún driver de BD — toda la
inferencia y el historial pasan por la API.
"""

from __future__ import annotations

import os
from typing import Any

import requests

_DEFAULT_API_URL = "http://api:8000"


def _base() -> str:
    return os.environ.get("API_URL", _DEFAULT_API_URL).rstrip("/")


def get_model_info(timeout: int = 30) -> dict[str, Any]:
    """Devuelve el payload de /model-info o lanza requests.RequestException."""
    resp = requests.get(f"{_base()}/model-info", timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def predict(features: dict[str, Any], timeout: int = 30) -> dict[str, Any]:
    """Llama a POST /predict y devuelve el dict de respuesta.

    Lanza:
        requests.HTTPError: en respuestas 4xx/5xx.
        requests.RequestException: en fallos de red.
    """
    resp = requests.post(
        f"{_base()}/predict",
        json={"features": features},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def get_training_history(limit: int = 20, timeout: int = 30) -> list[dict[str, Any]]:
    """Devuelve las últimas corridas de entrenamiento (RF9) vía /training-history."""
    resp = requests.get(
        f"{_base()}/training-history",
        params={"limit": limit},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json().get("rows", [])
