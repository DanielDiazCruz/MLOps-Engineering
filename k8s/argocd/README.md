# GitOps con Argo CD

Argo CD mantiene el cluster sincronizado con los manifiestos de `k8s/` que viven
en este repositorio. El patrón es **App-of-Apps**: una Application raíz que crea
y gestiona una Application por componente.

## Estructura

```
k8s/argocd/
├── root-app.yaml          # Application raíz (App-of-Apps) — único bootstrap manual
└── apps/                  # una Application por componente, ordenadas por sync-wave
    ├── 00-foundations.yaml  (wave 0)  namespace + postgres + minio
    ├── 10-mlflow.yaml       (wave 1)  MLflow tracking + registry
    ├── 10-data-api.yaml     (wave 1)  API de datos del docente
    ├── 20-prometheus.yaml   (wave 2)  métricas
    ├── 20-grafana.yaml      (wave 2)  dashboards
    ├── 30-api.yaml          (wave 3)  API de inferencia
    ├── 30-ui.yaml           (wave 3)  UI Streamlit
    └── 40-locust.yaml       (wave 4)  pruebas de carga
```

Cada Application tiene `syncPolicy.automated` con `prune` y `selfHeal`: Argo
revierte cualquier cambio manual en el cluster para que el estado real siga
siempre a Git (autocuración / detección de drift).

## Instalación de Argo CD

```bash
kubectl create namespace argocd
kubectl apply --server-side -n argocd \
  -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
```
> `--server-side` es necesario: el CRD `applicationsets` supera el límite de
> 256 KB de la anotación de client-side apply.

## Bootstrap (App-of-Apps)

```bash
kubectl apply -f k8s/argocd/root-app.yaml
```
A partir de aquí Argo lee `k8s/argocd/apps` del repo y despliega todo solo.

## Acceso a la UI

```bash
kubectl -n argocd port-forward svc/argocd-server 8080:443
# usuario: admin
kubectl -n argocd get secret argocd-initial-admin-secret \
  -o jsonpath='{.data.password}' | base64 -d
```

## Alcance y notas

- **Airflow** se despliega con su Helm chart oficial (release `airflow`), fuera
  de Argo CD. Podría gestionarse con una Application de tipo Helm (multi-source
  con el values del repo) como mejora futura.
- **Imágenes**: el CI publica `dandiazc/mlops-pf-{airflow,api,ui}` por SHA. Para
  un GitOps 100% trazable, los tags de `k8s/api` y `k8s/ui` deberían apuntar al
  SHA publicado por el CI (hoy usan un tag de iteración local). El paso de "subir
  el tag tras el build" puede automatizarse en el CI (`kustomize edit set image`
  + commit) o con Argo CD Image Updater.
