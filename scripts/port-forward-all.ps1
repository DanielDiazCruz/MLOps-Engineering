# Port-forward all MLOps services in background jobs.
# Run from repo root:  .\scripts\port-forward-all.ps1
# Stop all:           Get-Job | Stop-Job | Remove-Job

$forwards = @(
    @{ svc = "svc/airflow-api-server"; local = 8080; remote = 8080; label = "Airflow UI" },
    @{ svc = "svc/mlflow-service";     local = 5000; remote = 5000; label = "MLflow" },
    @{ svc = "svc/minio-service";      local = 9001; remote = 9001; label = "MinIO Console" },
    @{ svc = "svc/minio-service";      local = 9000; remote = 9000; label = "MinIO API" },
    @{ svc = "svc/api";                local = 8000; remote = 8000; label = "Inference API" },
    @{ svc = "svc/ui";                 local = 8501; remote = 8501; label = "Streamlit UI" },
    @{ svc = "svc/prometheus";         local = 9090; remote = 9090; label = "Prometheus" },
    @{ svc = "svc/grafana";            local = 3000; remote = 3000; label = "Grafana" },
    @{ svc = "svc/locust-master";      local = 8089; remote = 8089; label = "Locust" },
    @{ svc = "svc/postgres-service";   local = 5432; remote = 5432; label = "PostgreSQL" }
)

Write-Host "`nStarting port-forwards..." -ForegroundColor Cyan

foreach ($f in $forwards) {
    $job = Start-Job -ScriptBlock {
        param($svc, $local, $remote)
        kubectl port-forward -n mlops $svc "${local}:${remote}"
    } -ArgumentList $f.svc, $f.local, $f.remote

    $job | Add-Member -NotePropertyName Label -NotePropertyValue $f.label
    Write-Host ("  [{0,2}] {1,-20} -> http://localhost:{2}" -f $job.Id, $f.label, $f.local) -ForegroundColor Green
}

Write-Host "`nAll forwards running. Press Ctrl+C or run:" -ForegroundColor Yellow
Write-Host "  Get-Job | Stop-Job | Remove-Job`n" -ForegroundColor Yellow

Write-Host "Quick links:" -ForegroundColor Cyan
Write-Host "  Airflow    http://localhost:8080  (admin / admin)"
Write-Host "  MLflow     http://localhost:5000"
Write-Host "  MinIO      http://localhost:9001  (minioadmin / minioadmin123)"
Write-Host "  API        http://localhost:8000/docs"
Write-Host "  UI         http://localhost:8501"
Write-Host "  Prometheus http://localhost:9090"
Write-Host "  Grafana    http://localhost:3000  (admin / mlops2026)"
Write-Host "  Locust     http://localhost:8089"
Write-Host "  PostgreSQL localhost:5432            (mlops_user / mlops_pass_2026 / db=mlops)"

# Keep script alive so jobs don't get orphaned when the terminal closes
try { Wait-Job -Job (Get-Job) | Out-Null }
catch { }
