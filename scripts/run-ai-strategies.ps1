# run-ai-strategies.ps1 – Run AI Strategy Components
# Windows 11 Home - Deploy AI strategy execution system

Write-Host "AI STRATEGY EXECUTION SYSTEM - DEPLOYMENT" -ForegroundColor Magenta
Write-Host "================================================" -ForegroundColor Magenta
Write-Host ""

# Check if we're in the right directory
if (-not (Test-Path "backend/ai_strategy_execution.py")) {
    Write-Host "AI strategy files not found!" -ForegroundColor Red
    Write-Host "   Please run this script from the Mystic-Codebase root directory" -ForegroundColor Yellow
    exit 1
}

# Check if .env file exists
if (-not (Test-Path "backend/.env")) {
    Write-Host "Environment file not found! Running setup first..." -ForegroundColor Yellow
    .\setup-env.ps1
}

# Create logs directory
if (-not (Test-Path "backend/logs")) {
    New-Item -ItemType Directory -Path "backend/logs" -Force | Out-Null
    Write-Host "Created logs directory" -ForegroundColor Green
}

# Stop any existing AI strategy containers
Write-Host "Stopping existing AI strategy containers..." -ForegroundColor Yellow
docker stop mystic-ai-strategy 2>$null
docker rm mystic-ai-strategy 2>$null
docker stop mystic-ai-leaderboard 2>$null
docker rm mystic-ai-leaderboard 2>$null
docker stop mystic-ai-trade-engine 2>$null
docker rm mystic-ai-trade-engine 2>$null

# Build AI strategy Docker images
Write-Host "Building AI strategy Docker images..." -ForegroundColor Yellow
Set-Location backend

# Build AI Strategy Execution
docker build -f ai_strategy_execution.Dockerfile -t mystic-ai-strategy .

# Build AI Leaderboard Executor
docker build -f ai_leaderboard_executor.Dockerfile -t mystic-ai-leaderboard .

# Build AI Trade Engine
docker build -f ai_trade_engine.Dockerfile -t mystic-ai-trade-engine .

Set-Location ..

# Run AI Strategy Execution
Write-Host "Starting AI Strategy Execution..." -ForegroundColor Yellow
docker run -d `
    --name mystic-ai-strategy `
    --env-file backend/.env `
    -v "${PWD}/backend/logs:/app/logs" `
    --restart unless-stopped `
    mystic-ai-strategy

# Run AI Leaderboard Executor
Write-Host "Starting AI Leaderboard Executor..." -ForegroundColor Yellow
docker run -d `
    --name mystic-ai-leaderboard `
    --env-file backend/.env `
    -v "${PWD}/backend/logs:/app/logs" `
    -v "${PWD}/backend/mutation_leaderboard.json:/app/mutation_leaderboard.json" `
    --restart unless-stopped `
    mystic-ai-leaderboard

# Run AI Trade Engine
Write-Host "Starting AI Trade Engine..." -ForegroundColor Yellow
docker run -d `
    --name mystic-ai-trade-engine `
    --env-file backend/.env `
    -v "${PWD}/backend/logs:/app/logs" `
    --restart unless-stopped `
    mystic-ai-trade-engine

# Wait for containers to start
Write-Host "Waiting for AI strategy containers to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Check container status
Write-Host "Checking AI strategy container status..." -ForegroundColor Yellow
docker ps --filter "name=mystic-ai"

# Show logs
Write-Host "AI Strategy Execution logs:" -ForegroundColor Yellow
docker logs mystic-ai-strategy --tail 5

Write-Host "AI Leaderboard Executor logs:" -ForegroundColor Yellow
docker logs mystic-ai-leaderboard --tail 5

Write-Host "AI Trade Engine logs:" -ForegroundColor Yellow
docker logs mystic-ai-trade-engine --tail 5

Write-Host ""
Write-Host "AI STRATEGY EXECUTION SYSTEM IS RUNNING!" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green
Write-Host ""
Write-Host "COMPONENTS:" -ForegroundColor Cyan
Write-Host "   AI Strategy Execution: mystic-ai-strategy" -ForegroundColor White
Write-Host "   AI Leaderboard Executor: mystic-ai-leaderboard" -ForegroundColor White
Write-Host "   AI Trade Engine: mystic-ai-trade-engine" -ForegroundColor White
Write-Host ""
Write-Host "USEFUL COMMANDS:" -ForegroundColor Cyan
Write-Host "   View all logs: docker logs -f mystic-ai-strategy" -ForegroundColor Gray
Write-Host "   View leaderboard logs: docker logs -f mystic-ai-leaderboard" -ForegroundColor Gray
Write-Host "   View trade engine logs: docker logs -f mystic-ai-trade-engine" -ForegroundColor Gray
Write-Host "   Stop all: docker stop mystic-ai-strategy mystic-ai-leaderboard mystic-ai-trade-engine" -ForegroundColor Gray
Write-Host "   Restart all: docker restart mystic-ai-strategy mystic-ai-leaderboard mystic-ai-trade-engine" -ForegroundColor Gray
Write-Host ""
Write-Host "MONITORING:" -ForegroundColor Cyan
Write-Host "   Strategy logs: backend/logs/ai_strategy_execution.log" -ForegroundColor White
Write-Host "   Leaderboard logs: backend/logs/ai_leaderboard_executor.log" -ForegroundColor White
Write-Host "   Trade engine logs: backend/logs/ai_trade_engine.log" -ForegroundColor White
Write-Host "   Leaderboard data: backend/mutation_leaderboard.json" -ForegroundColor White
Write-Host ""
Write-Host "AI Strategy Execution System is now monitoring and trading!" -ForegroundColor Magenta
Write-Host "It will automatically execute trades based on AI signals and leaderboard rankings." -ForegroundColor Green
