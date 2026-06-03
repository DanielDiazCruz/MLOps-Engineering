# Build context = repo root: docker build -f airflow/Dockerfile -t <user>/mlops-airflow:<tag> .
ARG AIRFLOW_BASE_TAG=3.1.8
FROM apache/airflow:${AIRFLOW_BASE_TAG}

# pip install PRIMERO: su capa es la pesada, así que la dejamos arriba para
# que quede cacheada. Cambios en pipeline/, en los DAGs o en el CSV ya no
# re-disparan la reinstalación de dependencias (antes el COPY de pipeline/
# iba antes del pip install e invalidaba la caché en cada cambio de código).
COPY airflow/requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Código y datos al final: capas baratas que cambian con frecuencia.
USER root
COPY pipeline/ /opt/airflow/pipeline/
COPY airflow/dags/ /opt/airflow/dags/
COPY data/data/Diabetes.csv /opt/airflow/data/Diabetes.csv
ENV PYTHONPATH=/opt/airflow

# Commit del código que entró en la imagen. El CI lo pasa como build-arg y
# train.py lo registra en MLflow (RF5: versión del código asociada al modelo).
ARG GIT_COMMIT=unknown
ENV GIT_COMMIT=${GIT_COMMIT}

USER airflow
