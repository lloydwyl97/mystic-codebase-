# Mystic Trading Platform - Streamlit Dashboard Docker Launcher
# This script launches the Streamlit dashboard using Docker

Write-Host "🚀 Starting Mystic Trading Platform Streamlit Dashboard with Docker..." -ForegroundColor Green

# Check if Docker is running
Write-Host "🔍 Checking Docker status..." -ForegroundColor Yellow
try {
    docker version | Out-Null
    Write-Host "✅ Docker is running" -ForegroundColor Green
} catch {
    Write-Host "❌ Error: Docker is not running or not installed" -ForegroundColor Red
    Write-Host "Please start Docker Desktop and try again" -ForegroundColor Yellow
    exit 1
}

# Check if we're in the right directory
if (-not (Test-Path "docker-compose.yml")) {
    Write-Host "❌ Error: docker-compose.yml not found in current directory" -ForegroundColor Red
    Write-Host "Please run this script from the Mystic-Codebase root directory" -ForegroundColor Yellow
    exit 1
}

# Check if backend is running
Write-Host "🔍 Checking if backend is running..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -TimeoutSec 5 -ErrorAction Stop
    if ($response.StatusCode -eq 200) {
        Write-Host "✅ Backend is running" -ForegroundColor Green
    } else {
        Write-Host "⚠️  Backend responded with status: $($response.StatusCode)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "⚠️  Warning: Backend not accessible at http://localhost:8000" -ForegroundColor Yellow
    Write-Host "   Starting backend services..." -ForegroundColor Yellow
    
    # Start backend services
    Write-Host "🔄 Starting backend services..." -ForegroundColor Yellow
    docker-compose up -d backend redis
    
    # Wait for backend to be ready
    Write-Host "⏳ Waiting for backend to be ready..." -ForegroundColor Yellow
    Start-Sleep -Seconds 30
}

# Build and start the dashboard
Write-Host "🔨 Building Streamlit dashboard container..." -ForegroundColor Yellow
docker-compose build dashboard

Write-Host "🚀 Starting Streamlit dashboard..." -ForegroundColor Green
docker-compose up -d dashboard

# Wait for dashboard to be ready
Write-Host "⏳ Waiting for dashboard to be ready..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Check dashboard status
Write-Host "🔍 Checking dashboard status..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8501/_stcore/health" -TimeoutSec 10 -ErrorAction Stop
    if ($response.StatusCode -eq 200) {
        Write-Host "✅ Dashboard is running successfully!" -ForegroundColor Green
    } else {
        Write-Host "⚠️  Dashboard responded with status: $($response.StatusCode)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "⚠️  Dashboard may still be starting up..." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "🌐 Dashboard URLs:" -ForegroundColor Cyan
Write-Host "   Streamlit Dashboard: http://localhost:8501" -ForegroundColor White
Write-Host "   React Frontend: http://localhost:80" -ForegroundColor White
Write-Host "   Backend API: http://localhost:8000" -ForegroundColor White
Write-Host ""
Write-Host "📊 Useful Commands:" -ForegroundColor Cyan
Write-Host "   View logs: docker-compose logs -f dashboard" -ForegroundColor White
Write-Host "   Stop dashboard: docker-compose stop dashboard" -ForegroundColor White
Write-Host "   Restart dashboard: docker-compose restart dashboard" -ForegroundColor White
Write-Host "   Stop all services: docker-compose down" -ForegroundColor White
Write-Host ""
Write-Host "🎉 Dashboard should be accessible at http://localhost:8501" -ForegroundColor Green 