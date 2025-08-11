# launch.ps1 – Run Mystic Trading Platform on Windows 11 Home
# PowerShell script to launch the complete trading platform

Write-Host "🌀 Launching Mystic Trading Platform..." -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

# Check if Docker is running
Write-Host "🔍 Checking Docker status..." -ForegroundColor Yellow
try {
    docker version | Out-Null
    Write-Host "✅ Docker is running" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker is not running. Please start Docker Desktop first." -ForegroundColor Red
    exit 1
}

# Check if we're in the right directory
if (-not (Test-Path "docker-compose.yml")) {
    Write-Host "❌ docker-compose.yml not found. Please run this script from the project root." -ForegroundColor Red
    exit 1
}

# Stop any existing containers
Write-Host "🛑 Stopping existing containers..." -ForegroundColor Yellow
docker-compose down

# Build the containers
Write-Host "🔨 Building containers..." -ForegroundColor Yellow
docker-compose build

# Start the services
Write-Host "🚀 Starting services..." -ForegroundColor Yellow
docker-compose up -d

# Wait for services to start
Write-Host "⏳ Waiting for services to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 30

# Check service status
Write-Host "📊 Checking service status..." -ForegroundColor Yellow
docker-compose ps

# Test the backend
Write-Host "🔍 Testing backend API..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing
    if ($response.StatusCode -eq 200) {
        Write-Host "✅ Backend API is healthy" -ForegroundColor Green
    } else {
        Write-Host "⚠️ Backend API returned status: $($response.StatusCode)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "❌ Backend API is not responding" -ForegroundColor Red
}

# Test the frontend
Write-Host "🔍 Testing frontend..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:80" -UseBasicParsing
    if ($response.StatusCode -eq 200) {
        Write-Host "✅ Frontend is running" -ForegroundColor Green
    } else {
        Write-Host "⚠️ Frontend returned status: $($response.StatusCode)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "❌ Frontend is not responding" -ForegroundColor Red
}

# Display access information
Write-Host ""
Write-Host "🎉 Mystic Trading Platform is running!" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green
Write-Host "🌐 Frontend: http://localhost" -ForegroundColor White
Write-Host "🔧 Backend API: http://localhost:8000" -ForegroundColor White
Write-Host "📚 API Documentation: http://localhost:8000/docs" -ForegroundColor White
Write-Host "🧠 AI Decisions: http://localhost:8000/ai/decisions" -ForegroundColor White
Write-Host "📊 Dashboard: http://localhost:8000/ui/status" -ForegroundColor White
Write-Host ""
Write-Host "💡 Useful commands:" -ForegroundColor Cyan
Write-Host "   View logs: docker-compose logs -f" -ForegroundColor Gray
Write-Host "   Stop services: docker-compose down" -ForegroundColor Gray
Write-Host "   Restart services: docker-compose restart" -ForegroundColor Gray
Write-Host "   View containers: docker-compose ps" -ForegroundColor Gray
Write-Host ""
Write-Host "🔮 Mystic Trading Platform is ready for cosmic analysis!" -ForegroundColor Magenta
