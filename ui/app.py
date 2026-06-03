"""UI Streamlit — estimador de precio inmobiliario.

La UI se comunica exclusivamente con el servicio de inferencia FastAPI.
NO importa mlflow, psycopg2 ni ningún driver de base de datos. Tiene dos
pestañas: la predicción de precio y el historial de entrenamiento (RF9),
ambas servidas por la API.
"""

from __future__ import annotations

import streamlit as st

import client
from examples import SAMPLE_CONDO, SAMPLE_HOUSE

# Features que espera el modelo (deben coincidir con pipeline/preprocess.py).
_NUMERIC = ["bed", "bath", "acre_lot", "house_size", "prev_sold_year"]
_CATEGORICAL = ["status", "city", "state", "zip_code"]

# Estados de venta más comunes del dataset (la lista no es exhaustiva; el
# modelo maneja valores no vistos por su encoder).
_STATUS_OPTIONS = ["for_sale", "ready_to_build", "sold", "foreclosure"]

_DEFAULTS: dict = dict(SAMPLE_HOUSE)


# ---------------------------------------------------------------------------
# Configuración de página
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Estimador de Precio Inmobiliario", page_icon="🏠", layout="wide")

# Inicialización del estado de sesión con un ejemplo por defecto.
for k, v in _DEFAULTS.items():
    st.session_state.setdefault(k, v)

# ---------------------------------------------------------------------------
# Barra lateral: información del modelo
# ---------------------------------------------------------------------------
st.sidebar.title("🏠 Estimador de Precio")
st.sidebar.markdown("---")
try:
    model_info = client.get_model_info()
    st.sidebar.subheader("Modelo activo")
    st.sidebar.markdown(f"**Nombre:** {model_info.get('model_name', 'N/A')}")
    st.sidebar.markdown(f"**Versión:** {model_info.get('model_version', 'N/A')}")
    st.sidebar.markdown(f"**Alias:** {model_info.get('model_alias', 'N/A')}")
except Exception as exc:  # noqa: BLE001
    st.sidebar.warning(f"No se pudo obtener info del modelo: {exc}")
st.sidebar.markdown("---")
st.sidebar.caption("UI conectada a la API FastAPI vía la variable de entorno API_URL.")

tab_pred, tab_hist = st.tabs(["💲 Predicción de precio", "📊 Historial de entrenamiento"])

# ===========================================================================
# Pestaña 1: predicción de precio
# ===========================================================================
with tab_pred:
    st.title("Predicción de precio de propiedades")
    st.markdown(
        "Ingrese las características de la propiedad o cargue un ejemplo y presione "
        "**Estimar precio** para obtener el valor estimado."
    )

    col_ex1, col_ex2, _ = st.columns([1, 1, 4])
    if col_ex1.button("Cargar ejemplo: casa"):
        st.session_state.update(SAMPLE_HOUSE)
        st.rerun()
    if col_ex2.button("Cargar ejemplo: apartamento"):
        st.session_state.update(SAMPLE_CONDO)
        st.rerun()

    st.markdown("---")
    st.subheader("Datos de la propiedad")

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        bed = c1.number_input(
            "Habitaciones (bed)", min_value=0, max_value=30,
            value=int(st.session_state.get("bed", 3)), step=1,
        )
        bath = c2.number_input(
            "Baños (bath)", min_value=0, max_value=30,
            value=int(st.session_state.get("bath", 2)), step=1,
        )
        house_size = c3.number_input(
            "Área construida (house_size, sqft)", min_value=100, max_value=100_000,
            value=int(st.session_state.get("house_size", 1800)), step=50,
        )

        c4, c5, c6 = st.columns(3)
        acre_lot = c4.number_input(
            "Tamaño del lote (acre_lot, acres)", min_value=0.0, max_value=1000.0,
            value=float(st.session_state.get("acre_lot", 0.25)), step=0.05, format="%.2f",
        )
        prev_sold_year = c5.number_input(
            "Año de venta previa (prev_sold_year)", min_value=1900, max_value=2026,
            value=int(st.session_state.get("prev_sold_year", 2015)), step=1,
        )
        status = c6.selectbox(
            "Estado (status)", options=_STATUS_OPTIONS,
            index=_STATUS_OPTIONS.index(st.session_state["status"])
            if st.session_state.get("status") in _STATUS_OPTIONS else 0,
        )

        c7, c8, c9 = st.columns(3)
        city = c7.text_input("Ciudad (city)", value=str(st.session_state.get("city", "New York")))
        state = c8.text_input("Estado/Región (state)", value=str(st.session_state.get("state", "New York")))
        zip_code = c9.text_input("Código postal (zip_code)", value=str(st.session_state.get("zip_code", "10001")))

        submitted = st.form_submit_button("Estimar precio", type="primary")

    if submitted:
        features = {
            "bed": float(bed),
            "bath": float(bath),
            "acre_lot": float(acre_lot),
            "house_size": float(house_size),
            "prev_sold_year": float(prev_sold_year),
            "status": str(status),
            "city": str(city),
            "state": str(state),
            "zip_code": str(zip_code),
        }
        with st.spinner("Consultando la API..."):
            try:
                result = client.predict(features)
            except __import__("requests").HTTPError as exc:
                status_code = exc.response.status_code
                try:
                    detail = exc.response.json().get("detail", exc.response.text)
                except Exception:  # noqa: BLE001
                    detail = exc.response.text
                st.error(f"La API respondió {status_code}: {detail}")
                result = None
            except __import__("requests").RequestException as exc:
                st.error(f"No se pudo contactar la API: {exc}")
                result = None

        if result is not None:
            price = result.get("prediction")
            latency = result.get("processing_time_ms")
            st.markdown("---")
            st.subheader("Precio estimado")
            r1, r2, r3 = st.columns(3)
            r1.markdown(
                f"<h2 style='color:#2e7d32'>${price:,.0f}</h2>", unsafe_allow_html=True
            )
            if latency is not None:
                r2.metric("Latencia de inferencia", f"{latency:.1f} ms")
            r3.metric("Modelo", f"v{result.get('model_version', 'N/A')}")
            with st.expander("Detalle de la respuesta"):
                st.json(result)
            st.caption(
                f"Modelo: **{result.get('model_name', 'N/A')}** "
                f"v{result.get('model_version', 'N/A')} | "
                f"Request ID: `{result.get('request_id', 'N/A')}`"
            )

# ===========================================================================
# Pestaña 2: historial de entrenamiento (RF9)
# ===========================================================================
with tab_hist:
    st.title("Historial de entrenamiento")
    st.markdown(
        "Cada fila es una corrida del pipeline en Airflow: si se decidió entrenar, "
        "por qué, y si el candidato fue promovido a producción (champion)."
    )
    if st.button("🔄 Refrescar"):
        st.rerun()

    try:
        history = client.get_training_history(limit=30)
    except Exception as exc:  # noqa: BLE001
        st.error(f"No se pudo obtener el historial: {exc}")
        history = []

    if not history:
        st.info("Aún no hay corridas registradas en audit.training_history.")
    else:
        def _fmt_mae(metrics: dict | None, key: str) -> str:
            if not metrics or metrics.get(key) is None:
                return "—"
            return f"${float(metrics[key]):,.0f}"

        table = []
        for row in history:
            cand = row.get("candidate_metrics") or {}
            champ = row.get("champion_metrics") or {}
            drift = row.get("drift") or {}
            cats = row.get("new_categories") or {}
            table.append({
                "fecha": (row.get("executed_at") or "")[:19].replace("T", " "),
                "batch": row.get("batch_id"),
                "filas_lote": row.get("n_records_batch"),
                "decisión": row.get("decision"),
                "motivo": row.get("decision_reason"),
                "entrenó": "✅" if row.get("trained") else "—",
                "promovió": "✅" if row.get("promoted") else "—",
                "cand_val_mae": _fmt_mae(cand, "val_mae"),
                "champ_mae": _fmt_mae(champ, "mae"),
                "drift_psi": drift.get("max_psi"),
                "cats_nuevas": cats.get("total_new"),
                "modelo_v": row.get("mlflow_model_version"),
            })
        st.dataframe(table, use_container_width=True, hide_index=True)

        last = history[0]
        st.markdown("#### Última corrida")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Decisión", last.get("decision") or "—")
        m2.metric("Entrenó", "Sí" if last.get("trained") else "No")
        m3.metric("Promovió", "Sí" if last.get("promoted") else "No")
        m4.metric("Filas del lote", last.get("n_records_batch") or 0)
        st.caption(f"Motivo: {last.get('decision_reason') or '—'}")
        if last.get("promotion_reason"):
            st.caption(f"Promoción: {last.get('promotion_reason')}")
