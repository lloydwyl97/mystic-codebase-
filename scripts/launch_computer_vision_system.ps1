# Computer Vision Agent System Launch Script
# Launches all computer vision agents and orchestrator

Write-Host "🚀 Launching Computer Vision Agent System..." -ForegroundColor Green

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

# Function to check if Redis is running
function Test-RedisRunning {
    try {
        $response = docker exec mystic-redis redis-cli ping
        return $response -eq "PONG"
    }
    catch {
        return $false
    }
}

# Function to wait for service to be healthy
function Wait-ForServiceHealth {
    param(
        [string]$ServiceName,
        [int]$TimeoutSeconds = 60
    )
    
    Write-Host "⏳ Waiting for $ServiceName to be healthy..." -ForegroundColor Yellow
    
    $startTime = Get-Date
    $timeout = $startTime.AddSeconds($TimeoutSeconds)
    
    while ((Get-Date) -lt $timeout) {
        try {
            $status = docker ps --filter "name=$ServiceName" --format "{{.Status}}"
            if ($status -match "healthy") {
                Write-Host "✅ $ServiceName is healthy" -ForegroundColor Green
                return $true
            }
            elseif ($status -match "Up") {
                Write-Host "🔄 $ServiceName is running, waiting for health check..." -ForegroundColor Yellow
            }
        }
        catch {
            Write-Host "❌ Error checking $ServiceName status: $_" -ForegroundColor Red
        }
        
        Start-Sleep -Seconds 5
    }
    
    Write-Host "⏰ Timeout waiting for $ServiceName to be healthy" -ForegroundColor Red
    return $false
}

# Function to start a service
function Start-ComputerVisionService {
    param(
        [string]$ServiceName,
        [string]$DisplayName
    )
    
    Write-Host "🚀 Starting $DisplayName..." -ForegroundColor Cyan
    
    try {
        docker-compose up -d $ServiceName
        
        if (Wait-ForServiceHealth $ServiceName) {
            Write-Host "✅ $DisplayName started successfully" -ForegroundColor Green
            return $true
        }
        else {
            Write-Host "❌ Failed to start $DisplayName" -ForegroundColor Red
            return $false
        }
    }
    catch {
        Write-Host "❌ Error starting $DisplayName: $_" -ForegroundColor Red
        return $false
    }
}

# Function to check service logs
function Show-ServiceLogs {
    param(
        [string]$ServiceName,
        [string]$DisplayName,
        [int]$Lines = 10
    )
    
    Write-Host "📋 Recent logs for $DisplayName:" -ForegroundColor Magenta
    try {
        docker-compose logs --tail=$Lines $ServiceName
    }
    catch {
        Write-Host "❌ Error getting logs for $DisplayName: $_" -ForegroundColor Red
    }
    Write-Host ""
}

# Main execution
try {
    # Check if Docker is running
    if (-not (Test-DockerRunning)) {
        Write-Host "❌ Docker is not running. Please start Docker Desktop first." -ForegroundColor Red
        exit 1
    }
    
    Write-Host "✅ Docker is running" -ForegroundColor Green
    
    # Check if Redis is running
    if (-not (Test-RedisRunning)) {
        Write-Host "❌ Redis is not running. Please start the Redis service first." -ForegroundColor Red
        Write-Host "💡 Run: docker-compose up -d redis" -ForegroundColor Yellow
        exit 1
    }
    
    Write-Host "✅ Redis is running" -ForegroundColor Green
    
    # Navigate to project directory
    $projectDir = Split-Path -Parent $PSScriptRoot
    Set-Location $projectDir
    
    Write-Host "📁 Working directory: $projectDir" -ForegroundColor Cyan
    
    # Start Computer Vision services in order
    $services = @(
        @{Name="chart-pattern-agent"; Display="Chart Pattern Agent"},
        @{Name="technical-indicator-agent"; Display="Technical Indicator Agent"},
        @{Name="market-visualization-agent"; Display="Market Visualization Agent"},
        @{Name="computer-vision-orchestrator"; Display="Computer Vision Orchestrator"}
    )
    
    $successCount = 0
    
    foreach ($service in $services) {
        if (Start-ComputerVisionService -ServiceName $service.Name -DisplayName $service.Display) {
            $successCount++
        }
        else {
            Write-Host "❌ Failed to start $($service.Display). Stopping deployment." -ForegroundColor Red
            break
        }
        
        # Wait a bit between services
        Start-Sleep -Seconds 3
    }
    
    if ($successCount -eq $services.Count) {
        Write-Host ""
        Write-Host "🎉 Computer Vision Agent System launched successfully!" -ForegroundColor Green
        Write-Host "📊 Services running: $successCount/$($services.Count)" -ForegroundColor Cyan
        
        # Show service status
        Write-Host ""
        Write-Host "📋 Service Status:" -ForegroundColor Magenta
        docker-compose ps --filter "name=chart-pattern-agent|technical-indicator-agent|market-visualization-agent|computer-vision-orchestrator"
        
        # Show recent logs for each service
        Write-Host ""
        foreach ($service in $services) {
            Show-ServiceLogs -ServiceName $service.Name -DisplayName $service.Display
        }
        
        Write-Host ""
        Write-Host "🔗 Computer Vision System Endpoints:" -ForegroundColor Cyan
        Write-Host "   • Chart Pattern Agent: mystic-chart-pattern-agent" -ForegroundColor White
        Write-Host "   • Technical Indicator Agent: mystic-technical-indicator-agent" -ForegroundColor White
        Write-Host "   • Market Visualization Agent: mystic-market-visualization-agent" -ForegroundColor White
        Write-Host "   • Computer Vision Orchestrator: mystic-computer-vision-orchestrator" -ForegroundColor White
        
        Write-Host ""
        Write-Host "📈 Monitoring Commands:" -ForegroundColor Yellow
        Write-Host "   • View all logs: docker-compose logs -f chart-pattern-agent technical-indicator-agent market-visualization-agent computer-vision-orchestrator" -ForegroundColor Gray
        Write-Host "   • Check service status: docker-compose ps" -ForegroundColor Gray
        Write-Host "   • View Redis data: docker exec mystic-redis redis-cli keys '*cv*'" -ForegroundColor Gray
        Write-Host "   • Stop services: docker-compose stop chart-pattern-agent technical-indicator-agent market-visualization-agent computer-vision-orchestrator" -ForegroundColor Gray
        
        Write-Host ""
        Write-Host "🎯 Computer Vision System Features:" -ForegroundColor Cyan
        Write-Host "   • Chart Pattern Recognition: Head & Shoulders, Triangles, Flags, etc." -ForegroundColor White
        Write-Host "   • Technical Indicator Analysis: RSI, MACD, Bollinger Bands, etc." -ForegroundColor White
        Write-Host "   • Market Visualization: Candlestick, Line, Volume, Heatmap charts" -ForegroundColor White
        Write-Host "   • Unified Coordination: Pattern detection + Indicator signals + Visual analysis" -ForegroundColor White
        
        Write-Host ""
        Write-Host "✅ Computer Vision Agent System is ready for trading analysis!" -ForegroundColor Green
    }
    else {
        Write-Host ""
        Write-Host "❌ Computer Vision Agent System deployment failed!" -ForegroundColor Red
        Write-Host "📊 Services started: $successCount/$($services.Count)" -ForegroundColor Yellow
        
        # Show failed services
        Write-Host ""
        Write-Host "🔍 Troubleshooting:" -ForegroundColor Yellow
        Write-Host "   • Check Docker logs: docker-compose logs" -ForegroundColor Gray
        Write-Host "   • Verify Redis connection: docker exec mystic-redis redis-cli ping" -ForegroundColor Gray
        Write-Host "   • Check system resources: docker stats" -ForegroundColor Gray
        Write-Host "   • Restart failed services: docker-compose restart [service-name]" -ForegroundColor Gray
        
        exit 1
    }
}
catch {
    Write-Host "❌ Unexpected error: $_" -ForegroundColor Red
    Write-Host "🔍 Stack trace: $($_.ScriptStackTrace)" -ForegroundColor Gray
    exit 1
}
finally {
    Write-Host ""
    Write-Host "🏁 Computer Vision System launch script completed" -ForegroundColor Cyan
} 