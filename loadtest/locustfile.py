"""Prueba de carga Locust para la API de inferencia de precio inmobiliario.

Endpoint objetivo: POST /predict
Payload: features crudas de una propiedad (numéricas + categóricas como string).
El Pipeline de sklearn del modelo hace su propio one-hot encoding y la API
coacciona los tipos a la firma del modelo, así que basta con enviar valores
razonables.

Ejecutar desde la UI de Locust o sin interfaz:
    locust -f locustfile.py --headless -u 50 -r 10 --run-time 5m \
           --host http://api:8000 --html report.html
"""

from __future__ import annotations

import random

from locust import HttpUser, between, task

# Ubicaciones representativas (city, state, zip_code) para variar las peticiones.
_LOCATIONS = [
    ("New York", "New York", "10001"),
    ("Los Angeles", "California", "90001"),
    ("Miami", "Florida", "33101"),
    ("Houston", "Texas", "77001"),
    ("San Juan", "Puerto Rico", "00901"),
    ("Chicago", "Illinois", "60601"),
]
_STATUS = ["for_sale", "ready_to_build", "sold"]

# Features de una propiedad (mismas que produce pipeline/preprocess.py).
_BASE_SAMPLE: dict = {
    "bed": 3.0,
    "bath": 2.0,
    "acre_lot": 0.25,
    "house_size": 1800.0,
    "prev_sold_year": 2015,
    "status": "for_sale",
    "city": "New York",
    "state": "New York",
    "zip_code": "10001",
}


class PredictUser(HttpUser):
    """Simula un cliente llamando a /predict repetidamente."""

    wait_time = between(0.1, 0.5)

    @task(4)
    def predict(self) -> None:
        payload = dict(_BASE_SAMPLE)
        # Variar las features para que cada request sea distinto.
        payload["bed"] = float(random.randint(1, 6))
        payload["bath"] = float(random.randint(1, 5))
        payload["acre_lot"] = round(random.uniform(0.05, 2.0), 2)
        payload["house_size"] = float(random.randint(500, 5000))
        payload["prev_sold_year"] = random.randint(1990, 2024)
        city, state, zip_code = random.choice(_LOCATIONS)
        payload["city"], payload["state"], payload["zip_code"] = city, state, zip_code
        payload["status"] = random.choice(_STATUS)

        with self.client.post(
            "/predict",
            json={"features": payload},
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"status={resp.status_code} body={resp.text[:120]}")

    @task(1)
    def health(self) -> None:
        """Prueba ligera para confirmar que la API está viva."""
        self.client.get("/health", name="/health")
