"""UI de inferencia Streamlit para el proyecto MLOps de diabetes.

La UI se comunica exclusivamente con el servicio de inferencia FastAPI.
NO importa mlflow, psycopg2 ni ningún driver de base de datos.
"""

from __future__ import annotations

import streamlit as st

import client
from examples import PIMA_SAMPLE_PAYLOAD, SAMPLE_PAYLOAD

# ---------------------------------------------------------------------------
# Defaults de columnas crudas — las mismas ~40 columnas que el Pipeline
# de sklearn del modelo espera. Las categóricas son strings planos; las
# numéricas son floats. El OneHotEncoder del modelo tiene
# handle_unknown="ignore", por lo que cualquier valor no visto será
# descartado silenciosamente sin fallar.
# ---------------------------------------------------------------------------
_RAW_DEFAULTS: dict = {
    # Numerics
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
    # Categóricas — defaults coinciden con el valor más frecuente en el dataset
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

_AGE_BUCKETS = [
    (0,  10, "[0-10)"),  (10, 20, "[10-20)"), (20, 30, "[20-30)"),
    (30, 40, "[30-40)"), (40, 50, "[40-50)"), (50, 60, "[50-60)"),
    (60, 70, "[60-70)"), (70, 80, "[70-80)"), (80, 90, "[80-90)"),
    (90, 200, "[90-100)"),
]

_GENDER_MAP = {0: "Female", 1: "Male", 2: "Unknown/Invalid"}
_CHANGE_MAP = {0: "No", 1: "Ch"}
_DIABETES_MED_MAP = {0: "No", 1: "Yes"}
_GLU_MAP = {0: "None", 1: ">200", 2: ">300", 3: "Norm"}
_A1C_MAP = {0: "None", 1: ">7", 2: ">8", 3: "Norm"}


def _to_model_features(form: dict) -> dict:
    """Superpone valores del formulario sobre el dict de defaults crudos.

    El modelo es un Pipeline de sklearn que hace su propio encoding one-hot,
    así que solo necesitamos enviar strings categóricos crudos + numéricos crudos.
    """
    payload = dict(_RAW_DEFAULTS)

    # Direct numeric overrides
    for key in (
        "admission_type_id", "discharge_disposition_id", "admission_source_id",
        "time_in_hospital", "num_lab_procedures", "num_procedures",
        "num_medications", "number_outpatient", "number_emergency",
        "number_inpatient", "number_diagnoses",
    ):
        if key in form:
            payload[key] = float(form[key])

    # edad (entero) → string de rango
    age = int(form.get("age", 60))
    for lo, hi, bucket in _AGE_BUCKETS:
        if lo <= age < hi:
            payload["age"] = bucket
            break

    # Categóricas mediante tabla de mapeo
    payload["gender"] = _GENDER_MAP.get(int(form.get("gender", 1)), "Female")
    payload["change"] = _CHANGE_MAP.get(int(form.get("change", 0)), "No")
    payload["diabetesMed"] = _DIABETES_MED_MAP.get(int(form.get("diabetes_med", 1)), "Yes")
    payload["max_glu_serum"] = _GLU_MAP.get(int(form.get("max_glu_serum", 0)), "None")
    payload["A1Cresult"] = _A1C_MAP.get(int(form.get("a1c_result", 0)), "None")

    return payload


# ---------------------------------------------------------------------------
# Configuración de página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Diabetes Predictor",
    page_icon="🩺",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Barra lateral: información del modelo
# ---------------------------------------------------------------------------
st.sidebar.title("Diabetes Predictor")
st.sidebar.markdown("---")

try:
    model_info = client.get_model_info()
    st.sidebar.subheader("Modelo activo")
    st.sidebar.markdown(f"**Nombre:** {model_info.get('model_name', 'N/A')}")
    st.sidebar.markdown(f"**Version:** {model_info.get('model_version', 'N/A')}")
    st.sidebar.markdown(f"**Alias:** {model_info.get('model_alias', 'N/A')}")
except Exception as exc:  # noqa: BLE001
    st.sidebar.warning(f"No se pudo obtener info del modelo: {exc}")

st.sidebar.markdown("---")
st.sidebar.caption("UI conectada a la API FastAPI via variable de entorno API_URL.")

# ---------------------------------------------------------------------------
# Inicialización del estado de sesión
# ---------------------------------------------------------------------------
_DEFAULT_VALUES: dict = {
    "age": 50,
    "time_in_hospital": 3,
    "num_lab_procedures": 40,
    "num_procedures": 1,
    "num_medications": 14,
    "number_outpatient": 0,
    "number_emergency": 0,
    "number_inpatient": 0,
    "number_diagnoses": 7,
    "max_glu_serum": 0,
    "a1c_result": 0,
    "change": 0,
    "diabetes_med": 1,
    "gender": 1,
    "admission_type_id": 1,
    "discharge_disposition_id": 1,
    "admission_source_id": 7,
}

for k, v in _DEFAULT_VALUES.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------------------------------------------------------------------
# Encabezado
# ---------------------------------------------------------------------------
st.title("Prediccion de Reingreso Hospitalario")
st.markdown(
    "Ingrese los valores clinicos del paciente o cargue un ejemplo predefinido "
    "y presione **Predecir** para obtener la probabilidad de reingreso."
)

# ---------------------------------------------------------------------------
# Botones para cargar ejemplos
# ---------------------------------------------------------------------------
col_ex1, col_ex2, _ = st.columns([1, 1, 4])

with col_ex1:
    if st.button("Cargar valores de ejemplo (130-US)"):
        st.session_state.update(SAMPLE_PAYLOAD)
        st.rerun()

with col_ex2:
    if st.button("Cargar valores de ejemplo (Pima)"):
        st.session_state.update(PIMA_SAMPLE_PAYLOAD)
        st.rerun()

st.markdown("---")

# ---------------------------------------------------------------------------
# Formulario de entrada
# ---------------------------------------------------------------------------
st.subheader("Datos del paciente")

_pima_keys = {"Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
              "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"}
_using_pima = any(k in st.session_state for k in _pima_keys)

with st.form("predict_form"):
    if _using_pima:
        # ---- Diseño Pima ----
        c1, c2, c3, c4 = st.columns(4)
        pregnancies = c1.number_input(
            "Pregnancies", min_value=0, max_value=20,
            value=int(st.session_state.get("Pregnancies", 0)), step=1,
        )
        glucose = c2.number_input(
            "Glucose", min_value=0, max_value=300,
            value=int(st.session_state.get("Glucose", 120)), step=1,
        )
        blood_pressure = c3.number_input(
            "BloodPressure", min_value=0, max_value=200,
            value=int(st.session_state.get("BloodPressure", 70)), step=1,
        )
        skin_thickness = c4.number_input(
            "SkinThickness", min_value=0, max_value=100,
            value=int(st.session_state.get("SkinThickness", 20)), step=1,
        )
        c5, c6, c7, c8 = st.columns(4)
        insulin = c5.number_input(
            "Insulin", min_value=0, max_value=900,
            value=int(st.session_state.get("Insulin", 0)), step=1,
        )
        bmi = c6.number_input(
            "BMI", min_value=0.0, max_value=70.0,
            value=float(st.session_state.get("BMI", 25.0)), step=0.1,
        )
        dpf = c7.number_input(
            "DiabetesPedigreeFunction", min_value=0.0, max_value=3.0,
            value=float(st.session_state.get("DiabetesPedigreeFunction", 0.5)),
            step=0.001, format="%.3f",
        )
        age_pima = c8.number_input(
            "Age", min_value=1, max_value=120,
            value=int(st.session_state.get("Age", 30)), step=1,
        )

        features_payload: dict = {
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": dpf,
            "Age": age_pima,
        }
        use_conversion = False

    else:
        # ---- Diseño Diabetes 130-US ----
        c1, c2, c3 = st.columns(3)

        age = c1.number_input(
            "Edad (anos)", min_value=1, max_value=120,
            value=int(st.session_state.get("age", 50)), step=1,
        )
        time_in_hospital = c2.number_input(
            "Dias en hospital", min_value=1, max_value=30,
            value=int(st.session_state.get("time_in_hospital", 3)), step=1,
        )
        num_lab_procedures = c3.number_input(
            "Num procedimientos lab", min_value=0, max_value=200,
            value=int(st.session_state.get("num_lab_procedures", 40)), step=1,
        )

        c4, c5, c6 = st.columns(3)
        num_procedures = c4.number_input(
            "Num procedimientos", min_value=0, max_value=20,
            value=int(st.session_state.get("num_procedures", 1)), step=1,
        )
        num_medications = c5.number_input(
            "Num medicamentos", min_value=0, max_value=100,
            value=int(st.session_state.get("num_medications", 14)), step=1,
        )
        number_diagnoses = c6.number_input(
            "Num diagnosticos", min_value=1, max_value=20,
            value=int(st.session_state.get("number_diagnoses", 7)), step=1,
        )

        c7, c8, c9 = st.columns(3)
        number_outpatient = c7.number_input(
            "Visitas ambulatorias", min_value=0, max_value=50,
            value=int(st.session_state.get("number_outpatient", 0)), step=1,
        )
        number_emergency = c8.number_input(
            "Visitas emergencia", min_value=0, max_value=50,
            value=int(st.session_state.get("number_emergency", 0)), step=1,
        )
        number_inpatient = c9.number_input(
            "Hospitalizaciones previas", min_value=0, max_value=20,
            value=int(st.session_state.get("number_inpatient", 0)), step=1,
        )

        c10, c11, c12 = st.columns(3)
        max_glu_serum = c10.selectbox(
            "Max glucosa serica (0=No medido, 1=>200, 2=>300, 3=Normal)",
            options=[0, 1, 2, 3],
            index=int(st.session_state.get("max_glu_serum", 0)),
        )
        a1c_result = c11.selectbox(
            "HbA1c (0=No medido, 1=>7, 2=>8, 3=Normal)",
            options=[0, 1, 2, 3],
            index=int(st.session_state.get("a1c_result", 0)),
        )
        gender = c12.selectbox(
            "Genero (0=Femenino, 1=Masculino)",
            options=[0, 1],
            index=int(st.session_state.get("gender", 1)),
        )

        c13, c14, c15 = st.columns(3)
        change = c13.selectbox(
            "Cambio medicacion (0=No, 1=Si)",
            options=[0, 1],
            index=int(st.session_state.get("change", 0)),
        )
        diabetes_med = c14.selectbox(
            "Medicacion diabetes (0=No, 1=Si)",
            options=[0, 1],
            index=int(st.session_state.get("diabetes_med", 1)),
        )
        admission_type_id = c15.number_input(
            "Tipo admision (1-8)", min_value=1, max_value=8,
            value=int(st.session_state.get("admission_type_id", 1)), step=1,
        )

        c16, c17 = st.columns(2)
        discharge_disposition_id = c16.number_input(
            "Tipo alta (1-30)", min_value=1, max_value=30,
            value=int(st.session_state.get("discharge_disposition_id", 1)), step=1,
        )
        admission_source_id = c17.number_input(
            "Fuente admision (1-25)", min_value=1, max_value=25,
            value=int(st.session_state.get("admission_source_id", 7)), step=1,
        )

        features_payload = {
            "age": age,
            "time_in_hospital": time_in_hospital,
            "num_lab_procedures": num_lab_procedures,
            "num_procedures": num_procedures,
            "num_medications": num_medications,
            "number_outpatient": number_outpatient,
            "number_emergency": number_emergency,
            "number_inpatient": number_inpatient,
            "number_diagnoses": number_diagnoses,
            "max_glu_serum": max_glu_serum,
            "a1c_result": a1c_result,
            "change": change,
            "diabetes_med": diabetes_med,
            "gender": gender,
            "admission_type_id": admission_type_id,
            "discharge_disposition_id": discharge_disposition_id,
            "admission_source_id": admission_source_id,
        }
        use_conversion = True

    submitted = st.form_submit_button("Predecir", type="primary")

# ---------------------------------------------------------------------------
# Resultado de la predicción
# ---------------------------------------------------------------------------
if submitted:
    api_payload = _to_model_features(features_payload) if use_conversion else features_payload

    with st.spinner("Consultando la API..."):
        try:
            result = client.predict(api_payload)
        except __import__("requests").HTTPError as exc:
            status = exc.response.status_code
            try:
                detail = exc.response.json().get("detail", exc.response.text)
            except Exception:  # noqa: BLE001
                detail = exc.response.text
            st.error(f"La API respondio {status}: {detail}")
            result = None
        except __import__("requests").RequestException as exc:
            st.error(f"No se pudo contactar la API: {exc}")
            result = None

    if result is not None:
        pred = result.get("prediction")
        score = result.get("score")
        model_name = result.get("model_name", "N/A")
        model_version = result.get("model_version", "N/A")
        request_id = result.get("request_id", "N/A")
        latency = result.get("processing_time_ms")

        st.markdown("---")
        st.subheader("Resultado de la prediccion")

        res_col1, res_col2, res_col3 = st.columns(3)

        label = "REINGRESO" if pred == 1 else "NO reingreso"
        color = "red" if pred == 1 else "green"
        res_col1.markdown(
            f"<h2 style='color:{color}'>{label}</h2>", unsafe_allow_html=True
        )

        if score is not None:
            res_col2.metric("Probabilidad de reingreso", f"{score:.1%}")
        else:
            res_col2.info("Score no disponible para este modelo.")

        if latency is not None:
            res_col3.metric("Latencia de inferencia", f"{latency:.1f} ms")

        with st.expander("Detalle de la respuesta"):
            st.json(result)

        st.caption(
            f"Modelo: **{model_name}** v{model_version} | "
            f"Request ID: `{request_id}`"
        )
