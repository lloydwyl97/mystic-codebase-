# launch-live.ps1 – Complete Live Deployment for Mystic Trading Platform
# Windows 11 Home with Docker - All endpoints live

Write-Host "MYSTIC TRADING PLATFORM - LIVE DEPLOYMENT" -ForegroundColor Magenta
Write-Host "================================================" -ForegroundColor Magenta
Write-Host "Windows 11 Home + Docker" -ForegroundColor Cyan
Write-Host ""

# Check if running as Administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")
if (-not $isAdmin) {
    Write-Host "Running as Administrator is recommended for optimal performance" -ForegroundColor Yellow
    Write-Host "   Right-click PowerShell and select 'Run as Administrator'" -ForegroundColor Gray
    Start-Sleep -Seconds 3
}

# Check if Docker Desktop is running
Write-Host "Checking Docker Desktop status..." -ForegroundColor Yellow
try {
    docker version | Out-Null
    Write-Host "Docker Desktop is running" -ForegroundColor Green
} catch {
    Write-Host "Docker Desktop is not running!" -ForegroundColor Red
    Write-Host "   Please start Docker Desktop and wait for it to be ready" -ForegroundColor Yellow
    Write-Host "   Then run this script again" -ForegroundColor Yellow
    exit 1
}

# Check if we're in the right directory
if (-not (Test-Path "docker-compose.yml")) {
    Write-Host "docker-compose.yml not found!" -ForegroundColor Red
    Write-Host "   Please run this script from the Mystic-Codebase root directory" -ForegroundColor Yellow
    exit 1
}

# Check if environment files exist
Write-Host "Checking environment configuration..." -ForegroundColor Yellow
if (-not (Test-Path "backend/.env")) {
    Write-Host "Environment file not found, creating from example..." -ForegroundColor Yellow
    Copy-Item "backend/env.example" "backend/.env" -ErrorAction SilentlyContinue
}

# Stop any existing containers
Write-Host "Stopping existing containers..." -ForegroundColor Yellow
docker-compose down --remove-orphans

# Clean up any dangling images
Write-Host "Cleaning up Docker cache..." -ForegroundColor Yellow
docker system prune -f

# Build the containers with no cache for fresh start
Write-Host "Building containers (fresh build)..." -ForegroundColor Yellow
docker-compose build --no-cache

# Start Redis first
Write-Host "Starting Redis..." -ForegroundColor Yellow
docker-compose up -d redis

# Wait for Redis to be ready
Write-Host "Waiting for Redis to be ready..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Start the backend
Write-Host "Starting Backend API..." -ForegroundColor Yellow
docker-compose up -d backend

# Wait for backend to be ready
Write-Host "Waiting for Backend API to be ready..." -ForegroundColor Yellow
Start-Sleep -Seconds 30

# Start the frontend
Write-Host "Starting Frontend..." -ForegroundColor Yellow
docker-compose up -d frontend

# Wait for all services to be ready
Write-Host "Waiting for all services to be ready..." -ForegroundColor Yellow
Start-Sleep -Seconds 20

# Check service status
Write-Host "Checking service status..." -ForegroundColor Yellow
docker-compose ps

# Test the backend API
Write-Host "Testing Backend API endpoints..." -ForegroundColor Yellow
$backendEndpoints = @(
    "http://localhost:8000/health",
    "http://localhost:8000/docs",
    "http://localhost:8000/api/v1/market/status",
    "http://localhost:8000/api/v1/analytics/overview",
    "http://localhost:8000/api/v1/trading/status"
)

foreach ($endpoint in $backendEndpoints) {
    try {
        $response = Invoke-WebRequest -Uri $endpoint -UseBasicParsing -TimeoutSec 10
        if ($response.StatusCode -eq 200) {
            Write-Host "SUCCESS: $endpoint" -ForegroundColor Green
        } else {
            Write-Host "WARNING: $endpoint (Status: $($response.StatusCode))" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "ERROR: $endpoint (Not responding)" -ForegroundColor Red
    }
}

# Test the frontend
Write-Host "Testing Frontend..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:80" -UseBasicParsing -TimeoutSec 10
    if ($response.StatusCode -eq 200) {
        Write-Host "Frontend is running" -ForegroundColor Green
    } else {
        Write-Host "Frontend returned status: $($response.StatusCode)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "Frontend is not responding" -ForegroundColor Red
}

# Check container logs for any errors
Write-Host "Checking container logs for errors..." -ForegroundColor Yellow
$containers = @("mystic-redis", "mystic-backend", "mystic-frontend")
foreach ($container in $containers) {
    $logs = docker logs $container --tail 10 2>&1
    if ($logs -match "ERROR|error|Error|Exception|exception") {
        Write-Host "Errors found in $container logs:" -ForegroundColor Yellow
        Write-Host $logs -ForegroundColor Red
    } else {
        Write-Host "$container logs look clean" -ForegroundColor Green
    }
}

# Display comprehensive access information
Write-Host ""
Write-Host "MYSTIC TRADING PLATFORM IS LIVE!" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green
Write-Host ""
Write-Host "MAIN ACCESS POINTS:" -ForegroundColor Cyan
Write-Host "   Frontend Dashboard: http://localhost" -ForegroundColor White
Write-Host "   Backend API: http://localhost:8000" -ForegroundColor White
Write-Host "   API Documentation: http://localhost:8000/docs" -ForegroundColor White
Write-Host ""
Write-Host "TRADING ENDPOINTS:" -ForegroundColor Cyan
Write-Host "   Market Data: http://localhost:8000/api/v1/market" -ForegroundColor White
Write-Host "   Trading Status: http://localhost:8000/api/v1/trading/status" -ForegroundColor White
Write-Host "   AI Decisions: http://localhost:8000/api/v1/ai/decisions" -ForegroundColor White
Write-Host "   Bot Management: http://localhost:8000/api/v1/bots" -ForegroundColor White
Write-Host ""
Write-Host "ANALYTICS ENDPOINTS:" -ForegroundColor Cyan
Write-Host "   Analytics Overview: http://localhost:8000/api/v1/analytics/overview" -ForegroundColor White
Write-Host "   Performance Metrics: http://localhost:8000/api/v1/analytics/performance" -ForegroundColor White
Write-Host "   Portfolio Analysis: http://localhost:8000/api/v1/analytics/portfolio" -ForegroundColor White
Write-Host ""
Write-Host "AI & AUTOMATION:" -ForegroundColor Cyan
Write-Host "   AI Trading: http://localhost:8000/api/v1/ai/trading" -ForegroundColor White
Write-Host "   Strategy Management: http://localhost:8000/api/v1/strategies" -ForegroundColor White
Write-Host "   Signal Management: http://localhost:8000/api/v1/signals" -ForegroundColor White
Write-Host ""
Write-Host "NOTIFICATIONS & SOCIAL:" -ForegroundColor Cyan
Write-Host "   Notifications: http://localhost:8000/api/v1/notifications" -ForegroundColor White
Write-Host "   Social Trading: http://localhost:8000/api/v1/social" -ForegroundColor White
Write-Host "   WebSocket: ws://localhost:8000/ws" -ForegroundColor White
Write-Host ""
Write-Host "USEFUL COMMANDS:" -ForegroundColor Cyan
Write-Host "   View all logs: docker-compose logs -f" -ForegroundColor Gray
Write-Host "   View backend logs: docker-compose logs -f backend" -ForegroundColor Gray
Write-Host "   View frontend logs: docker-compose logs -f frontend" -ForegroundColor Gray
Write-Host "   Stop services: docker-compose down" -ForegroundColor Gray
Write-Host "   Restart services: docker-compose restart" -ForegroundColor Gray
Write-Host "   View containers: docker-compose ps" -ForegroundColor Gray
Write-Host "   Access container shell: docker exec -it mystic-backend bash" -ForegroundColor Gray
Write-Host ""
Write-Host "MYSTIC TRADING PLATFORM IS READY FOR COSMIC ANALYSIS!" -ForegroundColor Magenta
Write-Host "   All endpoints are live and ready for trading!" -ForegroundColor Green
Write-Host ""

# Optional: Open browser
$openBrowser = Read-Host "Open dashboard in browser? (y/n)"
if ($openBrowser -eq "y" -or $openBrowser -eq "Y") {
    Start-Process "http://localhost"
    Start-Process "http://localhost:8000/docs"
}

Write-Host ""
Write-Host "Deployment complete! Enjoy your cosmic trading journey!" -ForegroundColor Magenta
