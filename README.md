# MLOps Diabetes — Proyecto 2

Plataforma MLOps end-to-end sobre Kubernetes para predecir reingreso hospitalario de pacientes diabéticos usando el dataset **Diabetes 130-US Hospitals (1999-2008)** (~101k registros).

El sistema cubre el ciclo completo: ingesta incremental por lotes → almacenamiento crudo → preprocesamiento → entrenamiento + experimentación → registro y promoción automática del modelo → servicio de inferencia → UI → observabilidad → pruebas de carga.

## Tabla de contenidos

- [Arquitectura](#arquitectura)
- [Componentes](#componentes)
- [Stack técnico](#stack-técnico)
- [Quickstart con Makefile](#quickstart-con-makefile) — un solo comando (Linux/macOS/WSL)
- [Quickstart desde cero](#quickstart-desde-cero) — guía completa paso a paso
- [Flujo del DAG](#flujo-del-dag)
- [API de inferencia](#api-de-inferencia)
- [Modelo de datos](#modelo-de-datos)
- [Observabilidad y pruebas de carga](#observabilidad-y-pruebas-de-carga)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Variables de configuración](#variables-de-configuración)
- [Operación local](#operación-local)
- [Decisiones técnicas](#decisiones-técnicas)
- [Dificultades encontradas](#dificultades-encontradas)
- [Solución de problemas](#solución-de-problemas)

## Arquitectura

```mermaid
flowchart LR
    subgraph K8s["Kubernetes (namespace mlops)"]
        subgraph Training["Fase de entrenamiento"]
            CSV[(Diabetes.csv<br/>101k filas)]
            AF["Airflow<br/>(LocalExecutor)"]
            PG[("PostgreSQL<br/>raw / clean / inference")]
            ML["MLflow Server"]
            MN["MinIO<br/>(artifact store)"]
            CSV --> AF
            AF -->|"15k filas/run"| PG
            AF -->|"params, metrics,<br/>model + signature"| ML
            ML -.->|"metadata"| PG
            ML -.->|"artifacts"| MN
        end
        subgraph Inference["Fase de inferencia"]
            API["FastAPI<br/>/predict /metrics"]
            UI["Streamlit"]
            LC["Locust<br/>(load testing)"]
            UI -->|"HTTP POST"| API
            LC -->|"carga concurrente"| API
            API -->|"alias champion"| ML
            API -->|"log inferencia"| PG
        end
        subgraph Obs["Observabilidad"]
            PR["Prometheus"]
            GR["Grafana"]
            PR -->|"scrape /metrics"| API
            GR -->|"query"| PR
        end
    end
```

**Dos fases claramente separadas:**

- **Entrenamiento (offline):** Airflow ejecuta un DAG diario que carga 15k filas nuevas, preprocesa todo el acumulado, entrena LR + RF, registra ambos en MLflow y promueve automáticamente al ganador por F1.
- **Inferencia (online):** la API resuelve el modelo `champion` en MLflow al arrancar (con cache de 5 min y recarga manual vía `POST /reload-model`), responde predicciones a la UI y registra cada llamada en `inference.predictions`.

## Componentes

| Componente | Tecnología | Imagen DockerHub | Propósito |
|---|---|---|---|
| Orquestador | Apache Airflow 3.1.8 | `dandiazc/mlops-airflow:v2.1.0` | DAG de 8 tareas (entrena solo LR); embebe `pipeline/` y el CSV |
| BD relacional | PostgreSQL 16 | `postgres:16` (oficial) | Schemas `raw`, `clean`, `inference` + DB `mlflow`, `airflow` |
| Artifact store | MinIO | `minio/minio` (oficial, DockerHub) | Bucket `mlflow-artifacts` |
| Registro ML | MLflow 2.13.0 | `dandiazc/mlops-mlflow:v0.1.0` | Backend PG + artifacts S3-compatible |
| API inferencia | FastAPI + uvicorn | `dandiazc/mlops-api:v0.4.0` | `/health /predict /model-info /metrics /reload-model` |
| UI | Streamlit | `dandiazc/mlops-ui:v0.4.0` | Formulario clínico → POST `/predict` |
| Pruebas de carga | Locust (master+workers) | `locustio/locust` (oficial) | Escenario contra `/predict` |
| Métricas | Prometheus | `prom/prometheus` (oficial) | Scrape vía anotaciones del pod |
| Dashboards | Grafana | `grafana/grafana` (oficial) | 1 dashboard con latencias p50/p95/p99 + CPU/mem |

## Stack técnico

| Capa | Versiones |
|---|---|
| Runtime | Python 3.11 (API/UI), Python 3.12 (Airflow base) |
| ML | scikit-learn 1.4.2, pandas 2.x |
| Models | LogisticRegression + RandomForestClassifier (ambos con `class_weight=balanced`) |
| Persistencia | PostgreSQL 16 (StatefulSet + PVC) |
| Object store | MinIO (StatefulSet + PVC) |
| Despliegue | kustomize para todo excepto Airflow (Helm chart oficial `apache-airflow/airflow`) |

## Quickstart con Makefile

Si estás en **Linux, macOS o WSL** con `make`, `kubectl`, `helm` y `kustomize` instalados, puedes levantar todo el cluster con un solo comando:

```bash
make up
```

Esto ejecuta secuencialmente lo mismo que el [Quickstart desde cero](#quickstart-desde-cero), pero automatizado:

1. Aplica `k8s/foundations` (Postgres + MinIO) y espera al job `minio-bootstrap`.
2. Aplica `k8s/mlflow`.
3. Aplica el secret de Airflow y hace `helm upgrade --install airflow` con `airflow/values/values-local.yaml`.
4. Aplica `k8s/api`, `k8s/ui`, `k8s/prometheus`, `k8s/grafana` y `k8s/locust`.
5. Espera a que `airflow-api-server` esté Ready y **dispara automáticamente el DAG** `diabetes_mlops_pipeline` (target `trigger-dag`) — sin entrar a la UI de Airflow.

Tras `make up`, abre los port-forwards:

```bash
make forward          # Linux/macOS/WSL — corre en background, usa /tmp/pf.pids
make stop-forward     # detener todos los forwards
```

> En **Windows PowerShell** los targets `forward` / `stop-forward` no funcionan (usan `xargs`, `kill` y rutas Unix). Usa el script PowerShell equivalente: `.\scripts\port-forward-all.ps1`.

Otros targets útiles:

| Comando | Qué hace |
|---|---|
| `make up` | Despliega todo el stack y dispara el DAG |
| `make down` | Elimina todos los recursos en orden inverso (Helm uninstall + `kustomize delete`) |
| `make forward` | Levanta los 9 port-forwards en background (Linux/macOS) |
| `make stop-forward` | Mata todos los port-forwards |
| `make status` | `kubectl get pods,svc -n mlops` |
| `make trigger-dag` | Espera al api-server de Airflow y dispara el DAG manualmente |

> **Importante:** `make up` usa las imágenes ya publicadas en DockerHub bajo `dandiazc/*` (ver tabla de [Componentes](#componentes)). No necesitas construir ni publicar nada.

## Quickstart desde cero

Esta sección está pensada para alguien que **nunca ha visto el repo** y quiere levantar todo el sistema en su máquina.

### Prerrequisitos

| Software | Versión mínima | Para qué |
|---|---|---|
| Docker Desktop | 4.x | Construir imágenes + correr Kubernetes local |
| Kubernetes local | cualquiera | `kind`, `minikube`, `microk8s` o Docker Desktop K8s. Recomendado: Docker Desktop con K8s activado |
| `kubectl` | 1.27+ | Cliente de Kubernetes |
| `helm` | 3.x | Para instalar Airflow |
| `kustomize` | 5.x | Procesar overlays de los manifiestos (`make` lo invoca; `kubectl apply -k` también funciona) |
| `make` (opcional) | GNU Make | Para usar el [Quickstart con Makefile](#quickstart-con-makefile) en Linux/macOS/WSL |
| PowerShell 5.1+ o Bash | — | Para correr los scripts del repo |

Verificación rápida:

```powershell
docker info     # debe responder sin error
kubectl version --client
helm version
kubectl get nodes   # debe listar al menos un nodo
```

### Paso 1 — Clonar el repositorio

```bash
git clone https://github.com/CamiloGarcia06/Mlops_talleres.git
cd Mlops_talleres
git checkout DanielDiaz/protecto2/IMP-k8s
```

### Paso 2 — Desplegar fundaciones (Postgres + MinIO)

```bash
kubectl apply -k k8s/foundations
kubectl -n mlops rollout status statefulset/postgres
kubectl -n mlops rollout status statefulset/minio
kubectl -n mlops logs job/minio-bootstrap   # confirma "bootstrap done" al final
```

### Paso 3 — Desplegar MLflow

```bash
kubectl apply -k k8s/mlflow
kubectl -n mlops rollout status deployment/mlflow
```

### Paso 4 — Desplegar Airflow

```bash
helm repo add apache-airflow https://airflow.apache.org
helm repo update
helm upgrade --install airflow apache-airflow/airflow \
  -n mlops -f airflow/values/values-local.yaml
```

Espera ~2 minutos a que estén `Running` todos los pods (`scheduler`, `api-server`, `dag-processor`, `triggerer`):

```bash
kubectl -n mlops get pods -l release=airflow -w
```

### Paso 5 — Desplegar API, UI y observabilidad

```bash
kubectl apply -k k8s/api
kubectl apply -k k8s/ui
kubectl apply -k k8s/prometheus
kubectl apply -k k8s/grafana
kubectl apply -k k8s/locust
```

> **Nota:** la API arrancará en `CrashLoopBackOff` o fallará el readiness probe hasta que el DAG haya promovido un modelo `champion`. Es esperado en esta etapa.

### Paso 6 — Abrir todos los port-forwards

En PowerShell (Windows):

```powershell
.\scripts\port-forward-all.ps1
```

Esto expone localmente todos los servicios. Quick links:

| Servicio | URL | Credenciales |
|---|---|---|
| Airflow UI | http://localhost:8080 | `admin / admin` |
| MLflow UI | http://localhost:5000 | — |
| MinIO Console | http://localhost:9001 | `minioadmin / minioadmin123` |
| API (docs Swagger) | http://localhost:8000/docs | — |
| Streamlit UI | http://localhost:8501 | — |
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3000 | `admin / mlops2026` |
| Locust UI | http://localhost:8089 | — |
| PostgreSQL | `localhost:5432` | `mlops_user / mlops_pass_2026` (db: `mlops`) |

### Paso 7 — Disparar el DAG por primera vez

1. Entra a Airflow UI: http://localhost:8080
2. Busca el DAG `diabetes_mlops_pipeline` y enciéndelo (toggle)
3. Click en **Trigger DAG** (▶️)
4. Espera 2-3 minutos a que todas las tareas estén en verde

Cada ejecución carga los siguientes 15k registros del CSV. Para procesar los 101k completos, dispara el DAG **7 veces** o deja que el schedule diario haga su trabajo.

### Paso 8 — Validar end-to-end

Tras la primera ejecución exitosa del DAG:

```bash
# Modelo registrado y promovido
curl http://localhost:5000/api/2.0/mlflow/registered-models/get?name=diabetes-classifier

# La API ya debería estar Healthy
curl http://localhost:8000/health
curl http://localhost:8000/model-info

# Hacer una predicción desde la UI
# Abre http://localhost:8501, click "Cargar valores de ejemplo (130-US)", luego "Predecir"
```

### Paso 9 — Probar la carga

1. Abre Locust: http://localhost:8089
2. Configura `Number of users = 50`, `Spawn rate = 5`, `Host = http://api.mlops.svc.cluster.local:8000`
3. Start
4. En otra pestaña abre Grafana (http://localhost:3000) y observa las gráficas de latencia y rate

## Flujo del DAG

```mermaid
flowchart LR
    A[t_migrate] --> B[t_check_source]
    B --> C[t_load_batch<br/>15k filas]
    C --> D[t_quality]
    D --> E[t_preprocess]
    E --> F[t_split<br/>70/15/15]
    F --> G[t_train<br/>LR + RF]
    G --> H[t_compare<br/>vs champion]
    H --> I[t_promote<br/>set alias]
```

| Tarea | Módulo | Qué hace | Idempotencia |
|---|---|---|---|
| `t_migrate` | `pipeline/db/migrations.py` | `CREATE SCHEMA/TABLE IF NOT EXISTS` para raw/clean/inference | ✓ (IF NOT EXISTS) |
| `t_check_source` | `pipeline/ingest.py:check_source` | Valida que el CSV exista en `/opt/airflow/data/Diabetes.csv` | ✓ |
| `t_load_batch` | `pipeline/ingest.py:load_batch` | Cuenta filas en raw → usa como offset → carga las siguientes 15k | ✓ (`row_hash UNIQUE`) |
| `t_quality` | `pipeline/quality.py` | Valida columnas y tipos | ✓ |
| `t_preprocess` | `pipeline/preprocess.py` | One-hot encoding (sobre **todo** lo acumulado), imputación, persiste como JSONB | ✓ (upsert por `row_hash`) |
| `t_split` | `pipeline/split.py` | Split estratificado 70/15/15 sobre `clean.diabetes_clean` | ✓ (re-asigna `split`) |
| `t_train` | `pipeline/train.py` | Entrena LR + RF con `class_weight=balanced`, logea params/metrics/model con signature en MLflow | ✓ (cada run es nuevo) |
| `t_compare` | `pipeline/promote.py:compare` | Compara F1 del candidato vs champion actual | ✓ |
| `t_promote` | `pipeline/promote.py:promote` | Si gana, mueve alias `champion` a la versión nueva | ✓ |

**Lógica del cursor de batches** ([pipeline/ingest.py](pipeline/ingest.py)): cada ejecución cuenta cuántas filas existen ya en `raw.diabetes_raw` con el `source_file` actual, usa ese número como `skiprows` y lee las siguientes 15k. Cuando el CSV se agota (~7ma ejecución), `load_batch` reporta `inserted=0` sin fallar.

**Promoción automática** ([pipeline/promote.py:51](pipeline/promote.py#L51)): si no existe champion → se promueve el candidato; si existe → solo se promueve si `candidate.F1 > champion.F1`.

## API de inferencia

Implementada con FastAPI ([api/main.py](api/main.py)).

| Método | Endpoint | Descripción |
|---|---|---|
| GET | `/health` | Liveness/readiness simple |
| GET | `/model-info` | Nombre, versión, alias y `loaded_at` del modelo en caché |
| POST | `/predict` | Recibe `{"features": {...}}`, retorna `{prediction, score, model_name, model_version, model_alias, request_id, processing_time_ms}` |
| POST | `/reload-model` | Fuerza recarga del champion desde MLflow (útil tras una promoción) |
| GET | `/metrics` | Métricas Prometheus (HTTP + custom de inferencia) |

**Estrategia de carga del modelo:**

1. **Pre-carga en startup** vía `lifespan` de FastAPI ([api/main.py:30-38](api/main.py#L30-L38)) → evita cold-start en la primera petición.
2. **Cache en memoria con TTL** ([api/model_loader.py](api/model_loader.py)) — default 300s configurable vía `MODEL_CACHE_TTL_SECONDS`. Tras el TTL, la siguiente petición refresca el modelo desde MLflow.
3. **Recarga manual** vía `POST /reload-model` para propagación inmediata tras una promoción.

**Alineación dinámica de features** ([api/main.py:85-105](api/main.py#L85-L105)): la UI envía 141 features one-hot, pero el modelo puede esperar menos (si fue entrenado con menos batches del CSV). `_align_features()` lee el `signature` que MLflow guarda con el modelo, reindexea el DataFrame al esquema esperado (rellena con 0 lo faltante, descarta lo desconocido) y predice sin romper. Fallback: si no hay signature, intenta `feature_names_in_` del sklearn nativo.

**Persistencia de inferencias** ([api/db.py](api/db.py)): cada `/predict` inserta una fila en `inference.predictions` con:

```
request_id (UUID) | created_at | input_payload (JSONB) | prediction
                    score | model_name | model_version | latency_ms
```

## Modelo de datos

Tres schemas en PostgreSQL con responsabilidades separadas ([pipeline/db/migrations.py](pipeline/db/migrations.py)):

### `raw.diabetes_raw`

| Columna | Tipo | Descripción |
|---|---|---|
| `row_hash` | `TEXT PK` | SHA-1 de la fila — garantiza idempotencia |
| `batch_id` | `TEXT` | UUID del batch que la cargó |
| `load_timestamp` | `TIMESTAMPTZ` | Cuándo se cargó |
| `source_file` | `TEXT` | Path del CSV origen |
| `status` | `TEXT` | `loaded` por defecto |
| `payload` | `JSONB` | La fila original sin transformar |

### `clean.diabetes_clean`

| Columna | Tipo | Descripción |
|---|---|---|
| `id` | `BIGSERIAL PK` | — |
| `row_hash` | `TEXT FK → raw` | Trazabilidad al dato crudo |
| `batch_id` | `TEXT` | — |
| `processed_at` | `TIMESTAMPTZ` | — |
| `split` | `TEXT` | `train` / `val` / `test` |
| `features` | `JSONB` | One-hot + numéricas imputadas |
| `target` | `INT` | Binarizado: `1` = readmitido en <30 días |

### `inference.predictions`

| Columna | Tipo | Descripción |
|---|---|---|
| `request_id` | `UUID PK` | Generado por la API |
| `created_at` | `TIMESTAMPTZ` | — |
| `input_payload` | `JSONB` | Features que llegaron al endpoint |
| `prediction` | `INT` | 0 o 1 |
| `score` | `DOUBLE` | Probabilidad clase positiva (si `predict_proba` existe) |
| `model_name` | `TEXT` | `diabetes-classifier` |
| `model_version` | `TEXT` | Versión MLflow que sirvió la predicción |
| `latency_ms` | `DOUBLE` | Tiempo de procesamiento |

**Por qué JSONB para `features` y `payload`:** el esquema del dataset puede variar (Pima 8 features vs. 130-US 50+ features) y `pd.get_dummies` produce un número variable de columnas según los valores categóricos presentes. JSONB tolera esa variabilidad sin DDL.

## Observabilidad y pruebas de carga

### Métricas expuestas por la API

Vía `prometheus-fastapi-instrumentator` ([api/main.py:43-45](api/main.py#L43-L45)) + custom counters ([api/metrics.py](api/metrics.py)):

| Métrica | Tipo | Etiquetas |
|---|---|---|
| `http_requests_total` | Counter | `method`, `handler`, `status` |
| `http_request_duration_seconds` | Histogram | `method`, `handler` |
| `predictions_total` | Counter | `prediction` (valor 0 o 1) |
| `inference_latency_seconds` | Histogram | — |
| `model_info` | Gauge | `name`, `version`, `alias` |

### Scrape de Prometheus

Configurado por anotaciones en el pod ([k8s/api/deployment.yaml:17-20](k8s/api/deployment.yaml#L17-L20)):

```yaml
prometheus.io/scrape: "true"
prometheus.io/port: "8000"
prometheus.io/path: "/metrics"
```

### Dashboard de Grafana

Source of truth: [observability/dashboards/api.json](observability/dashboards/api.json) (sincronizado a `k8s/grafana/configmap-dashboard-json.yaml`).

Paneles incluidos:

- Requests totales y RPS
- Latencias p50 / p95 / p99
- Tasa de errores (4xx + 5xx / total)
- Predicciones por clase
- CPU y memoria del pod de la API

### Locust

Workers desplegados en el cluster ([k8s/locust/](k8s/locust/)). El payload del escenario está en [k8s/locust/configmap.yaml](k8s/locust/configmap.yaml) (141 features one-hot, valores representativos).

> ⚠️ **Mantener sincronizados:** `k8s/locust/configmap.yaml` (cluster) y `loadtest/locustfile.py` (run local) contienen el mismo payload. Si cambia el esquema de features hay que actualizar ambos.

**Cómo correr una prueba:**

1. http://localhost:8089
2. Number of users = 50, spawn rate = 5
3. Host = `http://api.mlops.svc.cluster.local:8000`
4. Click **Start swarming**
5. Observar Grafana en paralelo para ver el efecto

## Estructura del repositorio

```
.
├── README.md
├── CLAUDE.md                       # Guía para asistente Claude (gitignored)
├── airflow/
│   ├── Dockerfile                  # Imagen propia: base apache/airflow + pipeline/ + CSV
│   ├── dags/diabetes_training_dag.py
│   ├── requirements.txt
│   └── values/values-local.yaml    # Helm values
├── api/                            # Código FastAPI
│   ├── main.py                     # endpoints + _align_features()
│   ├── model_loader.py             # cache con TTL del modelo MLflow
│   ├── db.py                       # log de inferencias a PG
│   ├── metrics.py                  # custom Prometheus counters
│   ├── schemas.py                  # Pydantic models
│   └── requirements.txt
├── ui/                             # Streamlit
│   ├── app.py                      # formulario + mapeo 17 fields → 141 features
│   └── requirements.txt
├── pipeline/                       # Lógica de entrenamiento (sin deps de Airflow)
│   ├── config.py
│   ├── db/
│   ├── ingest.py                   # cursor por offset, batches de 15k
│   ├── quality.py
│   ├── preprocess.py               # one-hot, imputación, filtro alta cardinalidad
│   ├── split.py                    # 70/15/15 estratificado
│   ├── train.py                    # LR + RF + infer_signature
│   ├── promote.py                  # alias champion
│   └── cli.py                      # entry point para CLI
├── docker/
│   ├── api/Dockerfile
│   ├── ui/Dockerfile
│   └── mlflow/Dockerfile
├── k8s/
│   ├── foundations/                # postgres + minio (kustomize)
│   ├── mlflow/
│   ├── api/
│   ├── ui/
│   ├── prometheus/
│   ├── grafana/
│   └── locust/
├── observability/
│   ├── dashboards/api.json         # Grafana dashboard (source of truth)
│   └── REPORT.md
├── loadtest/locustfile.py          # locust local (mantener sync con k8s/locust/configmap.yaml)
├── scripts/
│   └── port-forward-all.ps1        # Levanta forwards de los 10 servicios (Windows)
├── Makefile                        # `make up | down | forward | trigger-dag` (Linux/macOS/WSL)
└── data/data/Diabetes.csv          # Dataset (gitignored si es grande)
```

## Variables de configuración

Toda la configuración runtime pasa por env vars resueltas en `pipeline/config.py` o vía ConfigMap/Secret en k8s.

### Pipeline / DAG

| Variable | Default in-cluster | Para qué |
|---|---|---|
| `PG_DSN` | `postgresql://mlops_user:...@postgres-service.mlops:5432/mlops` | Conexión a Postgres |
| `MLFLOW_TRACKING_URI` | `http://mlflow-service.mlops:5000` | MLflow |
| `MLFLOW_S3_ENDPOINT_URL` | `http://minio-service.mlops:9000` | MinIO como S3 |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | `minioadmin / minioadmin123` | MinIO creds |
| `SOURCE_CSV` | `/opt/airflow/data/Diabetes.csv` | CSV fuente |
| `BATCH_SIZE` | `15000` | Cap obligatorio por enunciado |
| `RANDOM_SEED` | `42` | Reproducibilidad |
| `MLFLOW_EXPERIMENT` | `diabetes-classification` | — |
| `MLFLOW_MODEL_NAME` | `diabetes-classifier` | Nombre del registered model |
| `CHAMPION_ALIAS` | `champion` | Alias para el modelo productivo |
| `PRIMARY_METRIC` | `f1` | Métrica para seleccionar champion |

### API

Define en [k8s/api/configmap.yaml](k8s/api/configmap.yaml) + [k8s/api/secret.yaml](k8s/api/secret.yaml):

| Variable | Default | Para qué |
|---|---|---|
| `MODEL_CACHE_TTL_SECONDS` | `300` | TTL del cache del modelo en memoria |
| `MLFLOW_TRACKING_URI` | (idem) | — |
| `MLFLOW_S3_ENDPOINT_URL` | (idem) | — |
| `AWS_*` | (idem) | Para descargar artefactos de MinIO |
| `PG_DSN` | (idem) | Para log de inferencias |

### UI

| Variable | Default | Para qué |
|---|---|---|
| `API_URL` | `http://api.mlops.svc.cluster.local:8000` | Endpoint de la API |
| `API_CLIENT_TIMEOUT_SECONDS` | `30` | Timeout HTTP del cliente |

## Operación local

### Levantar todos los port-forwards de una vez

**Linux / macOS / WSL** (usa el Makefile, procesos en background con PIDs en `/tmp/pf.pids`):

```bash
make forward
make stop-forward     # para detenerlos
```

**Windows PowerShell** (los targets `make forward` usan `xargs`/`kill` y no funcionan en PS):

```powershell
.\scripts\port-forward-all.ps1
Get-Job | Stop-Job | Remove-Job   # detenerlos
```

### Reiniciar la API tras un cambio de imagen

```bash
kubectl -n mlops rollout restart deployment/api
kubectl -n mlops rollout status deployment/api
```

### Forzar a la API a recargar el modelo

Después de una promoción manual o tras correr el DAG, sin esperar el TTL:

```bash
curl -X POST http://localhost:8000/reload-model
```

### Inspeccionar el estado del pipeline

```sql
-- Cuántos batches se han cargado
SELECT batch_id, COUNT(*) FROM raw.diabetes_raw GROUP BY 1 ORDER BY 1;

-- Distribución de target por split
SELECT split, target, COUNT(*)
FROM clean.diabetes_clean GROUP BY 1, 2 ORDER BY 1, 2;

-- Cuántas features tiene el modelo actual
SELECT COUNT(DISTINCT k)
FROM clean.diabetes_clean, jsonb_object_keys(features) k;

-- Últimas inferencias
SELECT created_at, prediction, score, model_version, latency_ms
FROM inference.predictions ORDER BY created_at DESC LIMIT 20;
```

### Resetear los datos (sin tocar MLflow)

```bash
kubectl -n mlops exec postgres-0 -- psql -U mlops_user -d mlops -c \
  "TRUNCATE raw.diabetes_raw CASCADE; TRUNCATE clean.diabetes_clean CASCADE; TRUNCATE inference.predictions;"
```

### Resetear MLflow Model Registry

Vía API REST de MLflow:

```bash
curl -X DELETE http://localhost:5000/api/2.0/mlflow/registered-models/delete \
  -H "Content-Type: application/json" \
  -d '{"name":"diabetes-classifier"}'
```

## Decisiones técnicas

### Métrica principal: F1

Dataset clínicamente desbalanceado (~11% positivos en 130-US). Un modelo que prediga siempre 0 sacaría ~89% accuracy con valor clínico nulo — no detectaría ningún reingreso, que es justamente lo que queremos identificar. Reportamos accuracy, precision, recall, F1 y ROC-AUC en cada run, pero la **promoción usa F1** porque balancea precision y recall sobre la clase positiva, castigando tanto falsos positivos como falsos negativos (estos últimos clínicamente más costosos).

Detalle de implementación: en lugar de hardcodear `f1` en el comparador, [`train.py`](pipeline/train.py) loguea el valor de la métrica elegida con el alias genérico `primary_metric` y [`promote.py`](pipeline/promote.py) compara por ese alias. Cambiar a otra métrica (`PRIMARY_METRIC=roc_auc`) no requiere editar código, solo el env var.

### `class_weight=balanced` en ambos modelos

Sin balance, tanto LR como RF descubren rápido que la mejor estrategia es **predecir siempre 0**: 89% accuracy a costa de recall ≈ 0. Con `class_weight="balanced"`, sklearn pondera la pérdida inversamente a la frecuencia de cada clase (la positiva pesa ~9× más), forzando al modelo a tomar en cuenta la clase minoritaria.

Elegimos esto sobre SMOTE/oversampling por dos razones: (1) `balanced` se aplica como peso durante el fit, sin alterar la distribución de `clean.diabetes_clean` (preservamos trazabilidad fila-a-fila); (2) SMOTE sintetiza filas que no existen, lo cual complica el debugging clínico de predicciones individuales.

### Split estratificado y reproducible

Dos `train_test_split` consecutivos con `stratify=y` y la misma `random_state=RANDOM_SEED` ([`pipeline/split.py`](pipeline/split.py)) mantienen la proporción de ~11% de positivos en `train` (70%), `val` (15%) y `test` (15%). Sin estratificar, con clases tan desbalanceadas, val/test pueden quedar con casi cero positivos por azar y la F1 se vuelve ruidosa entre runs.

Cada run del DAG **reasigna toda la columna `split`**, no solo las filas nuevas. Importante porque cada batch agrega filas a `clean`, y si solo asignáramos las nuevas manteniendo las viejas, terminaríamos con un sesgo por orden de llegada (filas del primer batch sobre-representadas en train). Reasignar todo garantiza estratificación sobre la población actual y, con la misma semilla, dos runs idénticos producen el mismo split exacto.

### Promoción automática con gate de no degradación

[`promote.py`](pipeline/promote.py) **nunca degrada** el modelo en producción:

```python
decision = "promote" if champion_metric is None or candidate_metric > champion_metric else "keep"
```

Dos escenarios cubiertos por la misma rama: en el primer despliegue no hay champion → `champion_metric=None` → el candidato se promueve automáticamente; en despliegues posteriores solo se promueve si supera **estrictamente** al champion actual (`>`, no `≥`). En empate gana el modelo ya desplegado — preferimos estabilidad sobre cambios sin valor demostrable.

Detalle defensivo: `_metric()` usa `metrics.get("primary_metric", float("-inf"))`. Una run con métricas faltantes (OOM mid-fit, error humano editando `train.py`) queda automáticamente al fondo del ranking en lugar de tumbar la cadena con `TypeError`. El sistema queda **anti-frágil**: pierde precisión ante datos parciales pero nunca falla con excepción, lo cual importa cuando el DAG corre de noche sin alguien mirando.

### `train.run()` polimórfico para paralelización

Una sola función entrena uno o todos los candidatos según el argumento:

```python
def run(batch_id=None, model: str | None = None):
    if model is None: selected = all_candidates              # ambos (backward-compat)
    else:             selected = {_MODEL_ALIASES[model]: ...} # solo LR o solo RF
```

Esto cumple tres requisitos sin duplicar código: (1) **backward-compat** (`train.run()` sigue entrenando ambos como antes); (2) habilita el patrón "dos tareas `t_train_lr` y `t_train_rf` paralelas → un `t_promote_best`" en el DAG, bajando el tiempo total de `LR + RF` a `max(LR, RF)` (~3-4 min en lugar de ~8); (3) `--model lr` en la CLI para debug aislado. Sin esta API polimórfica habría tocado duplicar módulos o magia con `partial`/`lambda` que dificulta testing.

### Log del modelo a MLflow: sin `input_example` y con `pip_requirements` explícito

Dos decisiones tomadas tras debug doloroso en el cluster ([`train.py:172-183`](pipeline/train.py#L172-L183)):

1. **NO pasar `input_example`** a `mlflow.sklearn.log_model`. Cuando se pasa, MLflow recarga el modelo recién guardado desde el artefacto para validar que el ejemplo produce predicción consistente. Ese roundtrip serialización → deserialización funciona con holgura de memoria pero **cuelga indefinidamente** bajo presión (pod con 1-2 GiB durante el train).
2. **`pip_requirements` explícito** (`["mlflow", "scikit-learn", "pandas", "numpy"]`). Por defecto MLflow ejecuta `infer_pip_requirements` que hace **HTTP requests a PyPI** para resolver versiones exactas. Si el cluster tiene egress restringido o proxy mal configurado, esos requests cuelgan sin mensaje de error claro.

Patrón general: MLflow tiene varias capas "inteligentes" (validación con ejemplos, autodetección de deps, wrapping automático) que asumen entorno cómodo y rompen en cluster real. Vale la pena bypasearlas con configuración explícita en producción.

### Ingesta incremental e idempotente

El enunciado pide carga en lotes de **máximo 15k**. La implementación ([`pipeline/ingest.py`](pipeline/ingest.py)) combina dos mecanismos:

1. **Cursor por offset**: cuenta filas existentes en `raw.diabetes_raw` con el `source_file` actual y usa ese número como `skiprows` en `pd.read_csv(skiprows=offset+1, nrows=chunk_size)`. Cada DAG run procesa los siguientes 15k; tras ~7 runs el CSV se agota y `load_batch` reporta `inserted=0` sin fallar — el DAG sigue corriendo sobre el acumulado.
2. **Hash determinista como PK**: cada fila se identifica con `SHA-256` sobre su contenido canonicalizado (`sorted(items)`, `NaN→""`). Es `PRIMARY KEY` en `raw.diabetes_raw` con `ON CONFLICT (row_hash) DO NOTHING`.

La canonización importa: sin `sorted()`, el orden inestable de keys en el dict de pandas produciría hashes distintos para la misma fila; sin `NaN→""`, dos NaNs en posiciones idénticas generarían hashes distintos porque `NaN != NaN`. Con ambos en su sitio, cualquier reintento de Airflow (timeout transitorio, catch-up del scheduler) queda como no-op trivial.

### Features como JSONB en lugar de columnas tipadas

`pd.get_dummies` y la lista de categóricas observadas producen un número variable de columnas según los valores presentes en el batch. Tres opciones consideradas:

1. **Schema rígido con N columnas** → rompe cada vez que aparece una categoría nueva.
2. **Columna `features TEXT` con CSV serializado** → no consultable.
3. **JSONB con `features` como objeto** ← elegida. Consultable con `jsonb_object_keys`, sin DDL, tolerante a drift.

### Filtro de alta cardinalidad en preprocess

`diag_1/2/3` del dataset 130-US tienen >700 valores únicos (ICD codes), lo que produciría miles de columnas one-hot post-fit. El preprocess descarta categóricas con `>20` valores únicos ([`pipeline/preprocess.py:87-90`](pipeline/preprocess.py#L87-L90)) para mantener acotado el espacio de features. Trade-off: perdemos la señal de los diagnósticos detallados, pero el modelo queda manejable en memoria y tiempo de entrenamiento.

### Índices explícitos en queries hot

[`migrations.py`](pipeline/db/migrations.py) declara índices secundarios sobre las columnas que el pipeline filtra con frecuencia:

```sql
CREATE INDEX IF NOT EXISTS ix_raw_batch   ON raw.diabetes_raw    (batch_id);
CREATE INDEX IF NOT EXISTS ix_clean_batch ON clean.diabetes_clean (batch_id);
CREATE INDEX IF NOT EXISTS ix_clean_split ON clean.diabetes_clean (split);
```

`ix_raw_batch` acelera el filtro `WHERE batch_id = %s` de `quality.run()`; `ix_clean_split` acelera la lectura de los subsets de train/val/test en `train.py`. Declararlos en `migrations.py` (no via psql manual) garantiza reproducibilidad: quien clone el repo obtiene el mismo schema con `make up`, y `IF NOT EXISTS` los hace seguros de re-ejecutar.

Sin estos índices, a partir de ~50k filas en `clean` cada query del DAG empezaría a tomar segundos en lugar de milisegundos, y eso se acumula en cada tarea.

### Imagen Airflow propia (no sidecar / git-sync)

Embebemos `pipeline/`, los DAGs y el CSV en la imagen ([airflow/Dockerfile](airflow/Dockerfile)). Trade-off: cada cambio en `pipeline/` requiere rebuild + helm upgrade. Beneficio: cero dependencias en runtime (no git-sync, no PVC para datos), 100% reproducible.

### `webserverSecretKey` fijo en Airflow

Por defecto Airflow 3.x regenera el JWT secret en cada restart del webserver, lo que invalida los tokens de los task workers y falla los DAGs. Fijamos uno en [airflow/values/values-local.yaml:14](airflow/values/values-local.yaml#L14).

## Dificultades encontradas

Las cuatro dificultades más relevantes durante el desarrollo, qué impacto tuvieron y cómo se resolvieron.

### 1. Categorías nuevas entre batches rompían el esquema

**Problema.** El dataset se ingiere en lotes incrementales de 15k filas (no todo de una vez). Detectamos que el **primer batch** podía contener ~100 valores únicos en una variable categórica (por ejemplo `diag_1`, `payer_code`, `medical_specialty`), pero el **segundo batch** introducía valores categóricos **nunca vistos antes**. Esto rompía cualquier intento de fijar un esquema de features estático: el `OneHotEncoder` entrenado en el primer batch desconocía las categorías nuevas y fallaba en runtime, o peor, las ignoraba silenciosamente produciendo predicciones sesgadas.

**Impacto.** Cada vez que el DAG procesaba un batch nuevo y volvía a entrenar, había riesgo de:
- Fallar el entrenamiento (`ValueError: Found unknown categories`).
- Romper la API si el encoder se serializaba por separado del modelo (el `predict` recibía un shape distinto al esperado).
- Tener que rehacer el esquema manualmente, perdiendo idempotencia.

**Solución.** Adoptamos el **Patrón 1**: serializar el `OneHotEncoder` **dentro** del mismo `sklearn.Pipeline` que el clasificador, configurado con `handle_unknown="ignore"`:

```python
Pipeline([
    ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ("num", "passthrough", numeric_cols),
    ]),
    Classifier(...),
])
```

Así el encoder y el modelo se publican como **una única unidad** en MLflow (un solo `mlflow.sklearn.log_model`). En inferencia, cualquier categoría desconocida produce un vector de ceros en su columna en vez de romper el pipeline. Se aceptó una pequeña pérdida de información (las categorías nuevas no aportan al score) a cambio de **robustez ante drift de categorías sin reentrenamiento manual**. Ver [`pipeline/train.py`](pipeline/train.py) (función `_build_pipeline`).

### 2. `batch_id` perdido entre tareas del DAG (XCom)

**Problema.** El DAG procesa un batch nuevo en cada run (`ingest → quality → preprocess → split → train`). El `batch_id` lo genera la tarea `t_load_batch` al insertar en `raw.diabetes_raw`, pero las tareas siguientes (`quality`, `preprocess`, `split`, `train`) lo necesitan para filtrar **solo las filas de ese batch** y no reprocesar todo el histórico cada vez. Sin propagarlo correctamente, cada run reprocesaba 100% del acumulado, generando entrenamientos duplicados, runs de MLflow inflados y violaciones de `row_hash UNIQUE` en upserts a `clean.diabetes_clean`.

**Impacto.**
- Entrenamientos cada vez más lentos (cada run re-procesaba todo lo previo).
- Métricas de MLflow inconsistentes entre runs (la población cambiaba sin control).
- Riesgo de duplicados en `clean.diabetes_clean` cuando dos runs procesaban la misma fila.

**Solución.** Cada tarea devuelve un `dict` con el `batch_id` y las downstream lo reciben como input vía **XCom** (mecanismo nativo de Airflow para pasar datos entre tareas):

```python
@task() def t_load_batch(_src): return ingest.load_batch()           # {"batch_id": "...", ...}
@task() def t_quality(load_summary):     return quality.run(batch_id=load_summary["batch_id"])
@task() def t_preprocess(load_summary):  return preprocess.run(batch_id=load_summary["batch_id"])
...
```

El grafo declarativo (`migrate >> src >> loaded >> qual >> prep >> sp >> [lr, rf] >> promotion`) deja explícita la dependencia y Airflow inyecta el XCom automáticamente. Combinado con `row_hash UNIQUE` en la capa `raw` y upserts idempotentes en `clean`, el DAG quedó **rerun-safe**: si una tarea falla a mitad y se reintenta, no duplica ni corrompe nada.

### 3. Entrenamientos demasiado lentos (>30 min por iteración)

**Problema.** Los primeros entrenamientos llegaron a tardar **más de 30 minutos por intento**, lo cual mataba la iteración (cada cambio de hiperparámetro requería esperar media hora para validar). Identificamos tres causas concurrentes:

1. El scheduler de Airflow tenía `limits.cpu: 1` y `memory: 3Gi`, y como usamos `LocalExecutor` las tareas corren **dentro del pod del scheduler**, con esos límites.
2. El `RandomForest` estaba configurado con `n_estimators=100, max_depth=8, n_jobs=2` — costoso para un dataset de ~100k filas × ~140 features post-OneHot.
3. Los dos candidatos (`LogisticRegression` y `RandomForest`) corrían **en serie** en una sola tarea `train`, sumando sus tiempos.

**Impacto.** Imposible iterar en tiempo razonable durante el desarrollo. Cada smoke-test del pipeline costaba ~30 min, frenando el debugging de los otros componentes (API, UI, observabilidad) que dependen del champion.

**Solución.** Tres cambios combinados:

1. **Subir los `limits` del scheduler** a `cpu: 12, memory: 6Gi` (ajustado al límite de 16 GiB / 20 CPUs de Docker Desktop local) — ver [`airflow/values/values-local.yaml`](airflow/values/values-local.yaml).
2. **Reducir el `RandomForest`** a `n_estimators=50, max_depth=6, n_jobs=-1`. La pérdida de F1 fue marginal (<1 pp) pero el tiempo de fit cayó a la mitad.
3. **Paralelizar los dos candidatos** en el DAG: `t_train_lr` y `t_train_rf` corren en simultáneo después de `t_split`, y una nueva tarea `t_promote_best` recibe ambos resultados vía XCom y promueve el mejor por F1. Con `AIRFLOW__CORE__PARALLELISM=8` Airflow ejecuta ambas tareas concurrentemente.

Resultado: el tiempo total de entrenamiento bajó de **~30 min a ~3–4 min** (≈ max(LR, RF) en vez de LR + RF), permitiendo iterar con normalidad.

### 4. Clase objetivo desbalanceada (~11% positivos)

**Problema.** En el dataset Diabetes 130-US la variable objetivo `readmitted` (binarizada a "readmitido en <30 días" vs el resto) tiene solo **~11% de positivos**. Sin ajustes:

- Los modelos colapsaban a predecir siempre la clase mayoritaria.
- La **accuracy salía engañosamente alta** (~89%) sin valor clínico: el modelo nunca detectaba reingresos, que son justamente los casos que queremos identificar.
- La selección automática de champion basada en accuracy promovería modelos inútiles.

**Impacto.** Si elegíamos modelos por accuracy, el sistema en producción habría tenido recall ≈ 0 en la clase positiva: cero detección de pacientes en riesgo, lo cual contradice el objetivo clínico (un falso negativo —un reingreso no detectado— es más costoso que un falso positivo).

**Solución.** Dos cambios alineados con la naturaleza del problema:

1. **`class_weight="balanced"`** en ambos clasificadores. Sklearn pondera el costo de los errores inversamente a la frecuencia de cada clase durante el entrenamiento, forzando al modelo a "preocuparse" por la clase minoritaria.
2. **F1 como métrica primaria de promoción** (no accuracy). F1 = media armónica de precision y recall sobre la clase positiva, balanceando ambos errores. Configurada vía `PRIMARY_METRIC=f1` (env var), leída por `pipeline/promote.py`. Reportamos también accuracy, precision, recall y ROC-AUC para diagnóstico, pero el **`champion` se elige por F1**.

Ver [`pipeline/train.py`](pipeline/train.py) y [`pipeline/promote.py`](pipeline/promote.py).

### 5. Despliegue en una máquina nueva fallaba por restricción de Kustomize

**Problema.** Al intentar levantar el cluster en un equipo distinto al de desarrollo (`kubectl apply -k k8s/foundations`), el despliegue fallaba inmediatamente con:

```
error: accumulating resources: accumulation err='accumulating resources from
'../namespace.yaml': security; file '.../k8s/namespace.yaml' is not in or
below '.../k8s/foundations''
```

El `kustomization.yaml` original de `foundations/` referenciaba archivos sueltos con rutas relativas hacia arriba (`../namespace.yaml`, `../postgres/secret.yaml`, `../minio/pvc.yaml`, etc.). Funcionaba en la máquina original — donde el desarrollador lo probó — pero Kustomize por defecto activa la restricción de seguridad `LoadRestrictionsRootOnly`, que prohíbe referenciar archivos **fuera del directorio raíz** del `kustomization.yaml`. En máquinas con versiones más estrictas de `kubectl`/`kustomize`, eso bloqueaba el `make up` desde el primer comando, antes incluso de crear el namespace.

**Impacto.** Bloqueador total para reproducir el proyecto: el profesor o cualquier evaluador que clone el repo no podía pasar del primer `kubectl apply -k`. Workarounds existían (`--load-restrictor=LoadRestrictionsNone`), pero exigían que **el usuario recordara un flag específico cada vez**, contradiciendo el requisito de "instrucciones claras de despliegue" del rubro.

**Solución.** Refactor estructural a **bases por carpeta** (patrón idiomático de Kustomize):

1. `k8s/namespace.yaml` se movió a `k8s/namespace/namespace.yaml` con su propio `kustomization.yaml`.
2. `k8s/postgres/` y `k8s/minio/` recibieron cada uno un `kustomization.yaml` listando los archivos de **su propia** carpeta.
3. `k8s/foundations/kustomization.yaml` quedó reducido a tres referencias a **carpetas** (no archivos sueltos):

   ```yaml
   apiVersion: kustomize.config.k8s.io/v1beta1
   kind: Kustomization
   namespace: mlops
   resources:
     - ../namespace
     - ../postgres
     - ../minio
   ```

Cuando Kustomize ve **carpetas** como recursos, las trata como **bases** independientes con su propio root: la restricción de "fuera del directorio raíz" no aplica porque cada base se evalúa por separado. Verificamos que `kubectl kustomize k8s/foundations` produce **exactamente la misma salida** que antes (diff vacío) — refactor sin cambio funcional pero portable a cualquier máquina sin flags adicionales.

Lección: **probar el despliegue en una máquina limpia** desde temprano evita que diferencias de versión o configuración local enmascaren un bug de portabilidad.

## Solución de problemas

| Síntoma | Causa probable | Solución |
|---|---|---|
| API en `CrashLoopBackOff` justo después del primer deploy | No hay modelo `champion` en MLflow todavía | Corre el DAG al menos una vez |
| API devuelve `503 no model available` | El champion fue borrado o MLflow está caído | Confirma con `kubectl get pods -n mlops`, luego `POST /reload-model` |
| API devuelve `400 invalid features: X has N features...` | Modelo champion sin signature + sin `feature_names_in_` | Borrar registered model y volver a correr el DAG con la imagen Airflow `>= v1.2.0` |
| DAG falla con `Invalid auth token: Signature verification failed` | `webserverSecretKey` no fijo, JWT se regeneró | Ya está fijo en `airflow/values/values-local.yaml`; haz `helm upgrade` |
| `t_load_batch` reporta `inserted=0, duplicates=15000` | El CSV ya se cargó completamente | Esperado. El DAG sigue corriendo sobre el acumulado |
| MLflow no carga un modelo entrenado en otra versión de sklearn | Diferencia entre `sklearn==1.4.2` (train) vs `1.5.0` (API) | Tolerable (warning); para reproducibilidad estricta, alinea versiones en `pipeline/requirements.txt` y `api/requirements.txt` |
| Grafana dashboard vacío | Prometheus no scrapea la API | Verifica anotaciones `prometheus.io/scrape` en el pod de la API |
| Streamlit sale en blanco / timeout | API tarda en pre-cargar el modelo (cold start) | Aumenta `API_CLIENT_TIMEOUT_SECONDS` en `k8s/ui/configmap.yaml`, ya está en 30s |
| `helm upgrade` queda colgado | PVCs con `Pending` por falta de StorageClass | Verifica `kubectl get sc`, en Docker Desktop suele ser `hostpath` |

---

**Repositorio:** https://github.com/CamiloGarcia06/Mlops_talleres
**Rama del proyecto:** `DanielDiaz/proyecto2`
**Dataset:** [Diabetes 130-US Hospitals (1999-2008) — UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008)
