# Guion del video de sustentación (≤ 10 min)

Proyecto Final — Estimador de precio inmobiliario (MLOps Nivel 4).
El enunciado exige un video en YouTube de **máximo 10 minutos**. Este guion lo
ajusta a ~9:30 y cubre el checklist obligatorio (§10 del enunciado).

> Checklist previo a grabar: cluster `Running`, `data-api` arriba, un `champion`
> ya promovido, ≥1 fila en `audit.training_history`, y los port-forwards
> abiertos (UI, API/docs, MLflow, Grafana, Airflow, Argo CD). Ten a mano una
> pestaña con DockerHub y otra con GitHub → Actions.

| # | Bloque | Tiempo | Qué mostrar / decir |
|---|---|---|---|
| 1 | **Intro + repo + arquitectura** | 0:00–1:15 | Diagrama del README. Recorre la organización del repo por componentes (`pipeline/`, `api/`, `ui/`, `k8s/`, `.github/`). "Datos por API externa → decisión automática de reentrenar → API/UI, todo con CI/CD y GitOps." |
| 2 | **CI/CD (GitHub Actions)** | 1:15–2:15 | Pestaña Actions: workflow **Build & Push** en verde y el job **bump-manifests**. Muestra en DockerHub las 4 imágenes `dandiazc/mlops-pf-*` con tag `sha-...`. "No hay builds manuales; las imágenes salen del CI por SHA." |
| 3 | **GitOps (Argo CD)** | 2:15–3:30 | UI de Argo CD: `mlops-root` + hijas **Synced/Healthy**. **Demo selfHeal:** `kubectl scale deploy/ui -n mlops --replicas=2` → Argo lo revierte a 1. "El estado real sigue a Git; el tag de imagen lo fija el CI." |
| 4 | **DAG + bifurcaciones** | 3:30–5:30 | Airflow UI: grafo del DAG con `t_decide → t_branch → t_train/t_skip → t_notify`. Dispara un run. Explica las 3 señales (esquema, categorías nuevas, **drift PSI**). Muestra `audit.training_history` (decisión + motivo). |
| 5 | **MLflow + promoción** | 5:30–6:45 | Experimento `real-estate-price`: Ridge vs HistGBR con MAE/RMSE/MAPE/R², artefactos (plots), `git_commit` en params, y el alias `champion` en el registry. **Caso entrena+promueve** (línea base) y **caso entrena+NO promueve** (MAE no mejoró 3%) desde el historial. |
| 6 | **FastAPI + Streamlit** | 6:45–8:15 | Swagger `POST /predict` → precio; `GET /training-history`. Streamlit: pestaña **Predicción** (formulario → precio + versión del modelo) y pestaña **Historial** (RF9). Menciona `/reload-model` protegido y el log en `inference.predictions`. |
| 7 | **Locust + Grafana** | 8:15–9:15 | Lanza Locust contra `/predict` (payload inmobiliario). En Grafana muestra el efecto: RPS, latencia p95, errores. |
| 8 | **Cierre** | 9:15–9:45 | Repaso de la tabla RF1–RF10 del README + seguridad (sin credenciales quemadas). |

## Checklist obligatorio del enunciado (§10) — dónde se cubre

- [x] Organización del repositorio → bloque 1
- [x] Arquitectura y comunicación entre componentes → bloque 1
- [x] Workflows de GitHub Actions → bloque 2
- [x] Despliegue mediante Argo CD → bloque 3
- [x] Ejecución del DAG y sus bifurcaciones → bloque 4
- [x] Registro de experimentos y modelos en MLflow → bloque 5
- [x] Caso donde se entrena y se promueve → bloque 5
- [x] Caso donde se entrena pero NO se promueve (o no se entrena) → bloque 5 / bloque 4
- [x] FastAPI y Streamlit para inferencia → bloque 6
- [x] Historial de entrenamiento en Streamlit → bloque 6
- [x] Prueba de carga con Locust y efecto en Grafana → bloque 7

> **Caso "no entrena"**: ocurre cuando el lote no aporta filas nuevas o no hay
> drift/categorías nuevas. Para forzarlo en la demo, re-dispara el DAG cuando la
> API ya no entregue datos nuevos (lote vacío → `t_skip_training`), o muéstralo
> desde el historial si ya ocurrió.

## Comandos útiles

```bash
# Port-forwards
kubectl -n mlops port-forward svc/ui 8501:8501
kubectl -n mlops port-forward svc/api 8000:8000
kubectl -n mlops port-forward svc/mlflow-service 5000:5000
kubectl -n mlops port-forward svc/grafana 3000:3000
kubectl -n argocd port-forward svc/argocd-server 8080:443

# Argo CD: password admin y estado
kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath='{.data.password}' | base64 -d
kubectl get applications -n argocd

# Recargar el modelo (endpoint protegido)
curl -X POST http://localhost:8000/reload-model -H "X-Reload-Token: reload-mlops-2026"
```
