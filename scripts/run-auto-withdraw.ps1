# run-auto-withdraw.ps1 – Run Auto-Withdraw System with Docker
# Windows 11 Home - Deploy auto-withdraw system

Write-Host "AUTO-WITHDRAW SYSTEM - DOCKER DEPLOYMENT" -ForegroundColor Magenta
Write-Host "================================================" -ForegroundColor Magenta
Write-Host ""

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
if (-not (Test-Path "backend/auto_withdraw.py")) {
    Write-Host "Auto-withdraw system not found!" -ForegroundColor Red
    Write-Host "   Please run this script from the Mystic-Codebase root directory" -ForegroundColor Yellow
    exit 1
}

# Check if .env file exists
if (-not (Test-Path "backend/.env")) {
    Write-Host "Environment file not found! Running setup first..." -ForegroundColor Yellow
    .\setup-auto-withdraw.ps1
}

# Stop any existing auto-withdraw container
Write-Host "Stopping existing auto-withdraw container..." -ForegroundColor Yellow
docker stop mystic-auto-withdraw 2>$null
docker rm mystic-auto-withdraw 2>$null

# Build the auto-withdraw Docker image
Write-Host "Building auto-withdraw Docker image..." -ForegroundColor Yellow
Set-Location backend
docker build -f auto_withdraw.Dockerfile -t mystic-auto-withdraw .
Set-Location ..

# Run the auto-withdraw container
Write-Host "Starting auto-withdraw system..." -ForegroundColor Yellow
docker run -d `
    --name mystic-auto-withdraw `
    --env-file backend/.env `
    -v "${PWD}/backend/logs:/app/logs" `
    --restart unless-stopped `
    mystic-auto-withdraw

# Wait for container to start
Write-Host "Waiting for auto-withdraw system to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# Check container status
Write-Host "Checking auto-withdraw container status..." -ForegroundColor Yellow
docker ps --filter "name=mystic-auto-withdraw"

# Show logs
Write-Host "Auto-withdraw system logs:" -ForegroundColor Yellow
docker logs mystic-auto-withdraw --tail 10

Write-Host ""
Write-Host "AUTO-WITHDRAW SYSTEM IS RUNNING!" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green
Write-Host ""
Write-Host "USEFUL COMMANDS:" -ForegroundColor Cyan
Write-Host "   View logs: docker logs -f mystic-auto-withdraw" -ForegroundColor Gray
Write-Host "   Stop system: docker stop mystic-auto-withdraw" -ForegroundColor Gray
Write-Host "   Restart system: docker restart mystic-auto-withdraw" -ForegroundColor Gray
Write-Host "   View container: docker ps --filter name=mystic-auto-withdraw" -ForegroundColor Gray
Write-Host ""
Write-Host "MONITORING:" -ForegroundColor Cyan
Write-Host "   Log file: backend/logs/auto_withdraw.log" -ForegroundColor White
Write-Host "   Withdrawal history: backend/logs/withdrawals.json" -ForegroundColor White
Write-Host ""
Write-Host "Auto-withdraw system is monitoring your exchange balance!" -ForegroundColor Magenta
Write-Host "It will automatically withdraw to your cold wallet when threshold is reached." -ForegroundColor Green
