"""Prueba de carga Locust para la API de Inferencia de Diabetes.

Endpoint objetivo: POST /predict
Payload: esquema de features crudos (~40 columnas: categóricas como strings,
numéricas como floats). El Pipeline de sklearn del modelo hace su propio
encoding one-hot internamente.

Ejecutar desde la UI de Locust o sin interfaz:
    locust -f locustfile.py --headless -u 50 -r 10 --run-time 5m \
           --host http://api:8000 --html report.html
"""

from __future__ import annotations

import random

from locust import HttpUser, between, task

# Dict de features crudos que coincide con el esquema producido por
# pipeline/preprocess.py después del refactor de Patrón 1: ~13 numéricos
# + ~32 categóricos string.
_BASE_SAMPLE: dict = {
    # Numéricos
    "encounter_id": 2278392.0,
    "patient_nbr": 8222157.0,
    "admission_type_id": 1.0,
    "discharge_disposition_id": 1.0,
    "admission_source_id": 7.0,
    "time_in_hospital": 3.0,
    "num_lab_procedures": 41.0,
    "num_procedures": 0.0,
    "num_medications": 11.0,
    "number_outpatient": 0.0,
    "number_emergency": 0.0,
    "number_inpatient": 0.0,
    "number_diagnoses": 9.0,
    # Categóricas (strings — el encoder las maneja)
    "race": "Caucasian",
    "gender": "Female",
    "age": "[60-70)",
    "weight": "?",
    "payer_code": "MC",
    "max_glu_serum": "None",
    "A1Cresult": "None",
    "metformin": "Steady",
    "repaglinide": "No",
    "nateglinide": "No",
    "chlorpropamide": "No",
    "glimepiride": "No",
    "acetohexamide": "No",
    "glipizide": "No",
    "glyburide": "No",
    "tolbutamide": "No",
    "pioglitazone": "No",
    "rosiglitazone": "No",
    "acarbose": "No",
    "miglitol": "No",
    "troglitazone": "No",
    "tolazamide": "No",
    "examide": "No",
    "citoglipton": "No",
    "insulin": "Steady",
    "glyburide-metformin": "No",
    "glipizide-metformin": "No",
    "glimepiride-pioglitazone": "No",
    "metformin-rosiglitazone": "No",
    "metformin-pioglitazone": "No",
    "change": "No",
    "diabetesMed": "Yes",
}


class PredictUser(HttpUser):
    """Simula un cliente llamando a /predict repetidamente."""

    wait_time = between(0.1, 0.5)

    @task
    def predict(self) -> None:
        payload = dict(_BASE_SAMPLE)
        # Perturba las features numéricas clave para que cada request se vea ligeramente diferente.
        payload["num_lab_procedures"] = float(random.randint(1, 80))
        payload["num_medications"] = float(random.randint(1, 30))
        payload["time_in_hospital"] = float(random.randint(1, 14))
        payload["number_diagnoses"] = float(random.randint(1, 16))
        payload["number_inpatient"] = float(random.randint(0, 5))
        payload["number_emergency"] = float(random.randint(0, 5))

        with self.client.post(
            "/predict",
            json={"features": payload},
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"status={resp.status_code} body={resp.text[:120]}")

    @task(weight=1)
    def health(self) -> None:
        """Prueba ligera para confirmar que la API está viva."""
        self.client.get("/health", name="/health")
