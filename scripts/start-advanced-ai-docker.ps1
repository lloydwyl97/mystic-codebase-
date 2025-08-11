# Advanced AI Docker Services Startup Script
# Windows Home 11 PowerShell with Docker Support

Write-Host "🧠 Starting Advanced AI Services with Docker" -ForegroundColor Green

# Check Docker
try {
    docker --version | Out-Null
    Write-Host "✅ Docker found" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker not found. Install Docker Desktop first." -ForegroundColor Red
    exit 1
}

# Create directories
$dirs = @("mlflow", "ray", "optuna", "models", "logs", "wandb")
foreach ($dir in $dirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "✅ Created: $dir" -ForegroundColor Green
    }
}

# Start services
Write-Host "🚀 Starting Docker services..." -ForegroundColor Blue
docker-compose -f docker-compose-advanced-ai.yml up -d

Write-Host ""
Write-Host "🎉 Advanced AI Services Started!" -ForegroundColor Green
Write-Host "📊 MLflow: http://localhost:5000" -ForegroundColor Cyan
Write-Host "📈 Streamlit: http://localhost:8501" -ForegroundColor Cyan
Write-Host "📊 Dash: http://localhost:8050" -ForegroundColor Cyan
Write-Host "⚡ Optuna: http://localhost:8080" -ForegroundColor Cyan
Write-Host "📈 TensorBoard: http://localhost:6006" -ForegroundColor Cyan
Write-Host "🌸 Flower: http://localhost:5555" -ForegroundColor Cyan
Write-Host "🎯 Ray: http://localhost:8265" -ForegroundColor Cyan
