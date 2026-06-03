# Guion del video de sustentación

Proyecto Final — Estimador de precio inmobiliario (MLOps Nivel 4).
Duración objetivo: **10–14 min**. Antes de grabar, deja todo levantado y los
port-forwards abiertos (UI, API/docs, MLflow, Grafana, Airflow, Argo CD).

> Checklist previo: cluster `Running`, `data-api` arriba, un `champion` ya
> promovido, y al menos una fila en `audit.training_history` (corre el DAG una
> vez antes de grabar).

---

## 0. Intro (0:00–1:00)

**Qué decir:** "Es una plataforma MLOps end-to-end sobre Kubernetes que estima
el precio de propiedades. Los datos vienen de una API externa por lotes; el
pipeline decide solo si reentrenar según drift y datos nuevos; se sirve por API
y UI; y todo se opera con CI/CD y GitOps."

**Qué mostrar:** el diagrama de arquitectura del README.

## 1. Fuente de datos — API externa (1:00–1:45)

**Qué mostrar:** `kubectl get pods -n mlops` (pod `data-api`), y un
`curl .../health` y `/data?group_number=1` (estructura del lote).

**Qué decir:** "Es stateful: cada llamada entrega el siguiente lote (70k–230k
filas). El cliente de ingesta es robusto: reintentos, backoff y manejo de fin de
datos." → [pipeline/ingest.py](../pipeline/ingest.py).

## 2. El DAG y la decisión de reentrenar (1:45–4:30)

**Qué mostrar:** Airflow UI → DAG `diabetes_mlops_pipeline`, vista de grafo con
la **bifurcación** (`t_decide → t_branch → t_train/t_skip_training → t_notify`).
Dispara un run y muéstralo en verde.

**Qué decir:** recorre las tareas: ingesta → calidad → preprocess/split
**incrementales** → tres señales (esquema, **categorías nuevas**, **drift PSI**)
→ `t_decide` decide → bifurca. "En este lote detectó drift fuerte, así que
decidió entrenar." → [pipeline/decision.py](../pipeline/decision.py).

**Bonus (autocuración del dato):** muestra `audit.training_history` con la
decisión y el motivo:
```sql
SELECT batch_id, decision, decision_reason, trained, promoted FROM audit.training_history ORDER BY id DESC LIMIT 3;
```

## 3. Experimentación y registro — MLflow (4:30–6:00)

**Qué mostrar:** MLflow UI → experimento `real-estate-price`, los dos candidatos
(Ridge, HistGBR) con MAE/RMSE/MAPE/R², los **artefactos** (plots predicho-vs-real
y residuales), y el Model Registry con el alias `champion`.

**Qué decir:** "Se entrenan dos modelos, se elige el mejor por MAE y solo se
promueve si supera al champion en ≥3% sin empeorar el RMSE." →
[pipeline/promote.py](../pipeline/promote.py).

## 4. Inferencia — API + UI (6:00–8:00)

**Qué mostrar:**
- API `/docs` (Swagger): `POST /predict` con un payload de propiedad → precio.
- `GET /training-history` → JSON del historial.
- Streamlit: pestaña **Predicción** (llenar formulario → precio) y pestaña
  **Historial de entrenamiento** (RF9).

**Qué decir:** "La UI solo habla con la API. Cada predicción (ok o error) se
registra en `inference.predictions`." Muestra:
```sql
SELECT prediction, model_version, status, latency_ms FROM inference.predictions ORDER BY created_at DESC LIMIT 5;
```

## 5. Observabilidad — Grafana (8:00–8:45)

**Qué mostrar:** dashboard con latencias p50/p95/p99, RPS y distribución de
precio. Opcional: lanza Locust unos segundos para ver el efecto en vivo.

## 6. CI/CD — GitHub Actions (8:45–10:00)

**Qué mostrar:** pestaña **Actions** del repo: el workflow **Build & Push** en
verde y las 3 imágenes `dandiazc/mlops-pf-*` publicadas en DockerHub por SHA.
Menciona el `ci.yml` (compilación + paridad de versiones).

**Qué decir:** "No hay builds manuales: el CI construye y publica las imágenes
por SHA del commit."

## 7. GitOps — Argo CD (10:00–12:00)

**Qué mostrar:** UI de Argo CD → la app raíz `mlops-root` y las hijas, todas
**Synced / Healthy** (vista de árbol App-of-Apps).

**Demo de autocuración (selfHeal):** en una terminal escala algo a mano y
muestra cómo Argo lo revierte:
```bash
kubectl scale deploy/ui -n mlops --replicas=2
# en segundos Argo lo devuelve a 1 (lo que dice Git)
kubectl get deploy ui -n mlops -w
```

**Qué decir:** "El estado real sigue a Git: si alguien cambia algo a mano, Argo
lo revierte."

## 8. Cierre (12:00–13:00)

**Qué decir:** repasa el mapeo de requisitos (tabla "Capacidades cubiertas" del
README): ingesta por API, RAW/CLEAN, categorías nuevas, decisión automática de
reentrenar con drift, experimentación + promoción condicional, log de
inferencias, historial en UI, CI/CD y GitOps. Cierra con seguridad: sin
credenciales hardcodeadas; secrets gestionables fuera de git en producción.

---

### Comandos útiles para tener a mano

```bash
# Port-forwards (ejemplos)
kubectl -n mlops port-forward svc/ui 8501:8501
kubectl -n mlops port-forward svc/api 8000:8000
kubectl -n mlops port-forward svc/mlflow-service 5000:5000
kubectl -n mlops port-forward svc/grafana 3000:3000
kubectl -n argocd port-forward svc/argocd-server 8080:443

# Password de Argo CD
kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath='{.data.password}' | base64 -d

# Estado de las Applications
kubectl get applications -n argocd
```
