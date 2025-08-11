# Quantum Computing System Launch Script
# Launches all quantum computing services and agents

Write-Host "🚀 Launching Mystic AI Quantum Computing System..." -ForegroundColor Cyan

# Set error action preference
$ErrorActionPreference = "Stop"

# Function to check if Docker is running
function Test-DockerRunning {
    try {
        docker info | Out-Null
        return $true
    }
    catch {
        return $false
    }
}

# Function to check if a container is running
function Test-ContainerRunning {
    param([string]$ContainerName)
    try {
        $container = docker ps --filter "name=$ContainerName" --format "{{.Names}}"
        return $container -eq $ContainerName
    }
    catch {
        return $false
    }
}

# Function to wait for service health
function Wait-ForServiceHealth {
    param([string]$ServiceName, [string]$HealthUrl, [int]$TimeoutSeconds = 60)
    
    Write-Host "⏳ Waiting for $ServiceName to be healthy..." -ForegroundColor Yellow
    
    $startTime = Get-Date
    $timeout = $startTime.AddSeconds($TimeoutSeconds)
    
    while ((Get-Date) -lt $timeout) {
        try {
            $response = Invoke-WebRequest -Uri $HealthUrl -TimeoutSec 5 -UseBasicParsing
            if ($response.StatusCode -eq 200) {
                Write-Host "✅ $ServiceName is healthy" -ForegroundColor Green
                return $true
            }
        }
        catch {
            # Service not ready yet
        }
        
        Start-Sleep -Seconds 2
    }
    
    Write-Host "❌ $ServiceName health check timed out" -ForegroundColor Red
    return $false
}

# Check if Docker is running
if (-not (Test-DockerRunning)) {
    Write-Host "❌ Docker is not running. Please start Docker Desktop first." -ForegroundColor Red
    exit 1
}

# Check if we're in the right directory
if (-not (Test-Path "docker-compose.yml")) {
    Write-Host "❌ docker-compose.yml not found. Please run this script from the project root." -ForegroundColor Red
    exit 1
}

# Stop any existing quantum services
Write-Host "🛑 Stopping existing quantum services..." -ForegroundColor Yellow
docker-compose stop quantum-algorithm-engine quantum-ml-agent quantum-optimization-agent quantum-trading-engine 2>$null

# Start quantum services
Write-Host "🚀 Starting quantum computing services..." -ForegroundColor Green

# Start Quantum Algorithm Engine
Write-Host "🔬 Starting Quantum Algorithm Engine..." -ForegroundColor Cyan
docker-compose up -d quantum-algorithm-engine

# Start Quantum Machine Learning Agent
Write-Host "🧠 Starting Quantum Machine Learning Agent..." -ForegroundColor Cyan
docker-compose up -d quantum-ml-agent

# Start Quantum Optimization Agent
Write-Host "⚡ Starting Quantum Optimization Agent..." -ForegroundColor Cyan
docker-compose up -d quantum-optimization-agent

# Start Quantum Trading Engine
Write-Host "💰 Starting Quantum Trading Engine..." -ForegroundColor Cyan
docker-compose up -d quantum-trading-engine

# Wait for services to start
Write-Host "⏳ Waiting for quantum services to initialize..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Check service status
Write-Host "📊 Checking quantum service status..." -ForegroundColor Cyan

$quantumServices = @(
    @{Name="Quantum Algorithm Engine"; Container="mystic-quantum-algorithm-engine"},
    @{Name="Quantum ML Agent"; Container="mystic-quantum-ml-agent"},
    @{Name="Quantum Optimization Agent"; Container="mystic-quantum-optimization-agent"},
    @{Name="Quantum Trading Engine"; Container="mystic-quantum-trading-engine"}
)

$allHealthy = $true

foreach ($service in $quantumServices) {
    if (Test-ContainerRunning -ContainerName $service.Container) {
        Write-Host "✅ $($service.Name) is running" -ForegroundColor Green
    } else {
        Write-Host "❌ $($service.Name) is not running" -ForegroundColor Red
        $allHealthy = $false
    }
}

# Show logs
Write-Host "📋 Recent quantum service logs:" -ForegroundColor Cyan
Write-Host "----------------------------------------" -ForegroundColor Gray

# Show logs for each service
foreach ($service in $quantumServices) {
    Write-Host "🔍 $($service.Name) logs:" -ForegroundColor Yellow
    docker logs --tail 5 $service.Container 2>$null
    Write-Host ""
}

# Show system status
Write-Host "📊 Quantum System Status:" -ForegroundColor Cyan
Write-Host "----------------------------------------" -ForegroundColor Gray

if ($allHealthy) {
    Write-Host "🎉 All quantum services are running successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "🔗 Quantum System Endpoints:" -ForegroundColor Cyan
    Write-Host "  • Quantum Algorithm Engine: http://localhost:8000/api/quantum/algorithms" -ForegroundColor White
    Write-Host "  • Quantum ML Agent: http://localhost:8000/api/quantum/ml" -ForegroundColor White
    Write-Host "  • Quantum Optimization Agent: http://localhost:8000/api/quantum/optimization" -ForegroundColor White
    Write-Host "  • Quantum Trading Engine: http://localhost:8000/api/quantum/trading" -ForegroundColor White
    Write-Host ""
    Write-Host "📈 Monitor quantum services with:" -ForegroundColor Cyan
    Write-Host "  docker-compose logs -f quantum-algorithm-engine" -ForegroundColor White
    Write-Host "  docker-compose logs -f quantum-ml-agent" -ForegroundColor White
    Write-Host "  docker-compose logs -f quantum-optimization-agent" -ForegroundColor White
    Write-Host "  docker-compose logs -f quantum-trading-engine" -ForegroundColor White
} else {
    Write-Host "⚠️ Some quantum services failed to start. Check logs above for details." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "🎯 Quantum Computing System Launch Complete!" -ForegroundColor Green 