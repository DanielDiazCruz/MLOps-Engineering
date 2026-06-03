# CI/CD — GitHub Actions

Dos workflows componen el pipeline de integración y entrega continua del
proyecto. El despliegue del entregable usa **imágenes construidas por CI**, no
`docker build` manuales.

## Workflows

| Workflow | Archivo | Dispara en | Qué hace |
|---|---|---|---|
| **CI** | [`ci.yml`](ci.yml) | push y PR a cualquier rama | Compila (sintaxis) `pipeline/`, `api/`, `ui/`, `scripts/` y verifica la **paridad de versiones** (numpy/scikit-learn/mlflow/pandas) entre `airflow/requirements.txt` y `api/requirements.txt`. |
| **Build & Push** | [`build-push.yml`](build-push.yml) | push a `main` y `workflow_dispatch` | Construye y publica en DockerHub las 3 imágenes propias, etiquetadas por **SHA** del commit y `latest`. |

## Imágenes publicadas

A partir de `DOCKERHUB_USERNAME/mlops-<componente>`:

- `mlops-pf-airflow` — pipeline + DAG (Dockerfile: `airflow/Dockerfile`)
- `mlops-pf-api` — API de inferencia FastAPI (`docker/api/Dockerfile`)
- `mlops-pf-ui` — UI Streamlit (`docker/ui/Dockerfile`)

> La API de datos del docente (`cristiandiaz13/mlops-puj:data-api-pf-v1`) **no**
> se reconstruye ni republica.

## Secrets requeridos

Configurar en *Settings → Secrets and variables → Actions* del repositorio:

| Secret | Descripción |
|---|---|
| `DOCKERHUB_USERNAME` | Usuario/namespace de DockerHub (p. ej. `dandiazc`). |
| `DOCKERHUB_TOKEN` | Access token de DockerHub con permiso de escritura. |

## Tagging y GitOps

Cada imagen se etiqueta con `sha-<commit>` (formato largo) para trazabilidad y
para que el despliegue GitOps (Argo CD, fase 6) ancle una versión inmutable.
En `main` también se publica `latest`.
