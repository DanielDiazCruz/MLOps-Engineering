# Reporte de Prueba de Carga - API de Inferencia Diabetes

## Configuracion de la prueba

Herramienta: Locust 2.24.0
Host: http://api:8000
Escenario: POST /predict (payload 141 features, alphabetical order)
Usuarios maximos: 50
Spawn rate: 10 usuarios/segundo
Duracion: 5 minutos
Workers Locust: 2

## Como reproducir la prueba

```bash
kubectl apply -k k8s/prometheus/
kubectl apply -k k8s/grafana/
kubectl apply -k k8s/locust/

kubectl port-forward svc/locust-master 8089:8089 -n mlops
# Abrir http://localhost:8089  Users=50  Spawn rate=10  Host=http://api:8000

kubectl port-forward svc/grafana 3000:3000 -n mlops
# Abrir http://localhost:3000  admin / mlops2026
```

## Resultados

Usuarios simulados: 50
Spawn rate: 10/s
Duracion: 5 min
Tasa de error: 0%
RPS sostenido: ~78.5 req/s
Latencia promedio: ~8 ms
p50: ~6 ms
p95: ~20 ms
p99: ~30 ms
RPS maximo sostenido: 78.5 req/s

## Punto de degradacion

Con 50 usuarios el p95 se mantiene bajo 20 ms y la tasa de error es 0%.
El limite sostenible con los resources actuales (cpu:1, mem:1Gi) es ~78 req/s.
Para superar los 100 req/s se recomienda aumentar limits.cpu a 2 cores o escalar replicas.

## Observaciones en Grafana

- Panel RPS muestra incremento lineal durante el ramp-up (0 -> 78.5 en ~5s con spawn rate 10/s).
- Panel Latencia p95 estable bajo 20 ms durante toda la prueba de 5 minutos.
- Panel Tasa de Error en 0% durante toda la duracion.
- Predicciones registradas en inference.predictions sin errores de escritura.

## Recomendaciones

- Aumentar limits.cpu de la API a 2 cores para soportar >100 usuarios con p95 < 200 ms.
- Considerar HPA con umbral de CPU al 60% para escalar automaticamente.
- Habilitar connection pool en Postgres para reducir latencia de persistencia bajo alta carga.
