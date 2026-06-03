NS := mlops
KBUILD := kustomize build --load-restrictor=LoadRestrictionsNone

define kapply
	$(KBUILD) $(1) | kubectl apply -f -
endef

define kdelete
	$(KBUILD) $(1) | kubectl delete --ignore-not-found -f -
endef

.PHONY: up down forward stop-forward status trigger-dag

up:
	# 1. Namespace y fundaciones (Postgres + MinIO)
	kubectl apply -k k8s/foundations
	kubectl -n $(NS) wait --for=condition=complete job/minio-bootstrap --timeout=180s || true
	# 2. MLflow
	kubectl apply -k k8s/mlflow
	# 3. Airflow (Helm)
	kubectl apply -n $(NS) -f k8s/airflow/secret.yaml
	helm repo add apache-airflow https://airflow.apache.org >/dev/null 2>&1 || true
	helm repo update apache-airflow
	helm upgrade --install airflow apache-airflow/airflow -n $(NS) -f airflow/values/values-local.yaml --timeout 20m
	# 4. API
	kubectl apply -k k8s/api
	# 5. UI
	kubectl apply -k k8s/ui
	# 6. Observabilidad
	kubectl apply -k k8s/prometheus
	kubectl apply -k k8s/grafana
	# 7. Load test
	kubectl apply -k k8s/locust
	# 8. Disparar el DAG (genera y entrena el modelo champion)
	$(MAKE) trigger-dag
	# 9. Port-forwards (Linux/macOS: usa 'make forward'; Windows: .\scripts\port-forward-all.ps1)
	@echo ""
	@echo ">>> Cluster listo. Lanza los port-forwards con: make forward"
	@echo ">>> (Windows PowerShell: .\\scripts\\port-forward-all.ps1)"

trigger-dag:
	@echo ">>> Esperando a que el api-server de Airflow esté listo..."
	kubectl -n $(NS) rollout status deploy/airflow-api-server --timeout=10m
	@echo ">>> Disparando DAG diabetes_mlops_pipeline..."
	kubectl -n $(NS) exec deploy/airflow-api-server -- airflow dags trigger diabetes_mlops_pipeline

down:
	-helm uninstall airflow -n $(NS)
	-$(call kdelete,k8s/locust)
	-$(call kdelete,k8s/grafana)
	-$(call kdelete,k8s/prometheus)
	-$(call kdelete,k8s/ui)
	-$(call kdelete,k8s/api)
	-$(call kdelete,k8s/mlflow)
	-$(call kdelete,k8s/foundations)

forward: stop-forward
	@echo "Airflow    http://localhost:8080"
	@echo "MLflow     http://localhost:5000"
	@echo "MinIO      http://localhost:9001"
	@echo "API        http://localhost:8000/docs"
	@echo "UI         http://localhost:8501"
	@echo "Prometheus http://localhost:9090"
	@echo "Grafana    http://localhost:3000"
	@echo "Locust     http://localhost:8089"
	@kubectl -n $(NS) port-forward svc/airflow-api-server 8080:8080 >/tmp/pf-airflow.log    2>&1 & echo $$! >> /tmp/pf.pids
	@kubectl -n $(NS) port-forward svc/mlflow-service     5000:5000 >/tmp/pf-mlflow.log     2>&1 & echo $$! >> /tmp/pf.pids
	@kubectl -n $(NS) port-forward svc/minio-service      9001:9001 >/tmp/pf-minio-ui.log   2>&1 & echo $$! >> /tmp/pf.pids
	@kubectl -n $(NS) port-forward svc/minio-service      9000:9000 >/tmp/pf-minio-api.log  2>&1 & echo $$! >> /tmp/pf.pids
	@kubectl -n $(NS) port-forward svc/api                8000:8000 >/tmp/pf-api.log        2>&1 & echo $$! >> /tmp/pf.pids
	@kubectl -n $(NS) port-forward svc/ui                 8501:8501 >/tmp/pf-ui.log         2>&1 & echo $$! >> /tmp/pf.pids
	@kubectl -n $(NS) port-forward svc/prometheus         9090:9090 >/tmp/pf-prom.log       2>&1 & echo $$! >> /tmp/pf.pids
	@kubectl -n $(NS) port-forward svc/grafana            3000:3000 >/tmp/pf-grafana.log    2>&1 & echo $$! >> /tmp/pf.pids
	@kubectl -n $(NS) port-forward svc/locust-master      8089:8089 >/tmp/pf-locust.log     2>&1 & echo $$! >> /tmp/pf.pids
	@echo "Forwards corriendo. Para detener: make stop-forward"

stop-forward:
	@[ -f /tmp/pf.pids ] && xargs -r kill < /tmp/pf.pids 2>/dev/null || true
	@rm -f /tmp/pf.pids

status:
	kubectl -n $(NS) get pods,svc
