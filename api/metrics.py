"""Métricas Prometheus personalizadas para la API de inferencia.

Las métricas genéricas de HTTP (histograma de latencia, códigos de
respuesta, RPS, etc.) las emite `prometheus_fastapi_instrumentator`
automáticamente desde `main.py`. Aquí solo declaramos series específicas
del negocio de inferencia para enriquecer el dashboard de Grafana.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# Contador de predicciones servidas, etiquetado por estado (ok / error).
# En regresión la salida es continua (precio), así que NO se etiqueta por
# valor (haría explotar la cardinalidad de la métrica).
PREDICTIONS_TOTAL = Counter(
    "inference_predictions_total",
    "Total de predicciones servidas, etiquetadas por estado.",
    ["status"],
)

# Histograma del precio predicho (dólares). Permite ver en Grafana la
# distribución de las estimaciones que sirve el modelo.
PREDICTED_PRICE = Histogram(
    "inference_predicted_price",
    "Distribución del precio predicho (USD).",
    buckets=(50_000, 100_000, 200_000, 300_000, 500_000,
             750_000, 1_000_000, 2_000_000, 5_000_000),
)

# Histograma del tiempo que pasa dentro de model.predict (excluye el
# overhead de FastAPI/serialización). Los buckets están elegidos para
# capturar latencias entre 5 ms y 5 s.
INFERENCE_LATENCY_SECONDS = Histogram(
    "inference_latency_seconds",
    "Tiempo gastado dentro de la llamada model.predict.",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

# Gauge constante que expone nombre/versión/alias del modelo cargado.
# Es útil para correlacionar en Grafana qué modelo estaba sirviendo
# cada ventana de tiempo (especialmente tras una promoción).
MODEL_INFO = Gauge(
    "inference_model_info",
    "Gauge constante con el nombre/versión/alias del modelo activo.",
    ["model_name", "model_version", "model_alias"],
)


def set_model_info(name: str, version: str, alias: str) -> None:
    """Publica el modelo activo como gauge=1, descartando la etiqueta anterior.

    Se llama tras cargar/recargar el modelo. Como las labels cambian
    cuando se promueve una nueva versión, primero limpiamos los valores
    previos y luego registramos el nuevo conjunto de labels en 1.
    """
    MODEL_INFO.clear()
    MODEL_INFO.labels(model_name=name, model_version=version, model_alias=alias).set(1)
