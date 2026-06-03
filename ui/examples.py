"""Payloads de ejemplo predefinidos para el formulario de inferencia.

Estos coinciden con el conjunto de features producido por pipeline/preprocess.py
para el dataset de hospitales Diabetes 130-US (columnas numéricas / codificadas).
El payload cubre los features más discriminantes usados por el modelo entrenado;
cualquier feature adicional que espere el modelo se comportará según su propio
imputador / manejo de defaults.
"""

from __future__ import annotations

# Perfil típico de paciente reingresado
SAMPLE_PAYLOAD: dict = {
    "age": 65,
    "time_in_hospital": 5,
    "num_lab_procedures": 44,
    "num_procedures": 1,
    "num_medications": 17,
    "number_outpatient": 0,
    "number_emergency": 0,
    "number_inpatient": 1,
    "number_diagnoses": 9,
    "max_glu_serum": 0,
    "a1c_result": 0,
    "change": 1,
    "diabetes_med": 1,
    "gender": 1,
    "admission_type_id": 1,
    "discharge_disposition_id": 1,
    "admission_source_id": 7,
}

# Payload minimalista estilo Pima (8 features) — usado como fallback si el
# modelo fue entrenado con el dataset Pima en su lugar.
PIMA_SAMPLE_PAYLOAD: dict = {
    "Pregnancies": 6,
    "Glucose": 148,
    "BloodPressure": 72,
    "SkinThickness": 35,
    "Insulin": 0,
    "BMI": 33.6,
    "DiabetesPedigreeFunction": 0.627,
    "Age": 50,
}
