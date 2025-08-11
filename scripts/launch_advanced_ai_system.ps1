# Advanced AI System Launch Script
# Launches Deep Learning, Reinforcement Learning, AI Model Manager, and Advanced AI Orchestrator

param(
    [switch]$Build,
    [switch]$Logs,
    [switch]$Monitor,
    [switch]$Stop
)

$ErrorActionPreference = "Stop"

# Colors for output
$Red = "Red"
$Green = "Green"
$Yellow = "Yellow"
$Blue = "Blue"
$Cyan = "Cyan"
$White = "White"

# Configuration
$Services = @(
    "deep-learning-agent",
    "reinforcement-learning-agent", 
    "ai-model-manager",
    "advanced-ai-orchestrator"
)

$ServiceNames = @{
    "deep-learning-agent" = "Deep Learning Agent"
    "reinforcement-learning-agent" = "Reinforcement Learning Agent"
    "ai-model-manager" = "AI Model Manager"
    "advanced-ai-orchestrator" = "Advanced AI Orchestrator"
}

$ServicePorts = @{
    "deep-learning-agent" = 8010
    "reinforcement-learning-agent" = 8011
    "ai-model-manager" = 8012
    "advanced-ai-orchestrator" = 8013
}

function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = $White
    )
    Write-Host $Message -ForegroundColor $Color
}

function Write-Header {
    param([string]$Title)
    Write-ColorOutput "`n" $Cyan
    Write-ColorOutput "=" * 60 $Cyan
    Write-ColorOutput "  $Title" $Cyan
    Write-ColorOutput "=" * 60 $Cyan
    Write-ColorOutput "`n" $Cyan
}

function Write-ServiceStatus {
    param(
        [string]$Service,
        [string]$Status,
        [string]$Color = $White
    )
    $serviceName = $ServiceNames[$Service]
    Write-ColorOutput "  [$Status] $serviceName" $Color
}

function Test-DockerCompose {
    try {
        $result = docker-compose --version 2>$null
        if ($LASTEXITCODE -eq 0) {
            return $true
        }
        return $false
    }
    catch {
        return $false
    }
}

function Test-RedisConnection {
    try {
        $result = docker exec mystic-redis redis-cli ping 2>$null
        if ($result -eq "PONG") {
            return $true
        }
        return $false
    }
    catch {
        return $false
    }
}

function Start-AdvancedAIServices {
    Write-Header "Starting Advanced AI System Services"
    
    # Check if Docker Compose is available
    if (-not (Test-DockerCompose)) {
        Write-ColorOutput "❌ Docker Compose is not available. Please install Docker Desktop." $Red
        exit 1
    }
    
    # Check if Redis is running
    if (-not (Test-RedisConnection)) {
        Write-ColorOutput "⚠️  Redis is not running. Starting Redis first..." $Yellow
        docker-compose up -d redis
        Start-Sleep -Seconds 10
    }
    
    # Start services
    foreach ($service in $Services) {
        Write-ColorOutput "Starting $($ServiceNames[$service])..." $Blue
        
        if ($Build) {
            docker-compose build $service
        }
        
        docker-compose up -d $service
        
        if ($LASTEXITCODE -eq 0) {
            Write-ServiceStatus $service "STARTED" $Green
        } else {
            Write-ServiceStatus $service "FAILED" $Red
        }
    }
    
    Write-ColorOutput "`n✅ Advanced AI System services started successfully!" $Green
}

function Stop-AdvancedAIServices {
    Write-Header "Stopping Advanced AI System Services"
    
    foreach ($service in $Services) {
        Write-ColorOutput "Stopping $($ServiceNames[$service])..." $Blue
        docker-compose stop $service
        
        if ($LASTEXITCODE -eq 0) {
            Write-ServiceStatus $service "STOPPED" $Yellow
        } else {
            Write-ServiceStatus $service "ERROR" $Red
        }
    }
    
    Write-ColorOutput "`n✅ Advanced AI System services stopped!" $Green
}

function Show-ServiceLogs {
    Write-Header "Advanced AI System Service Logs"
    
    foreach ($service in $Services) {
        Write-ColorOutput "`n📋 $($ServiceNames[$service]) Logs:" $Cyan
        Write-ColorOutput "-" * 50 $Cyan
        
        try {
            docker-compose logs --tail=20 $service
        }
        catch {
            Write-ColorOutput "❌ Failed to get logs for $service" $Red
        }
    }
}

function Monitor-Services {
    Write-Header "Advanced AI System Service Monitor"
    
    while ($true) {
        Clear-Host
        Write-Header "Advanced AI System Service Status"
        
        $allHealthy = $true
        
        foreach ($service in $Services) {
            $containerName = "mystic-$service"
            
            try {
                $status = docker inspect --format='{{.State.Status}}' $containerName 2>$null
                $health = docker inspect --format='{{.State.Health.Status}}' $containerName 2>$null
                
                if ($status -eq "running") {
                    if ($health -eq "healthy" -or $health -eq "<nil>") {
                        Write-ServiceStatus $service "RUNNING" $Green
                    } else {
                        Write-ServiceStatus $service "UNHEALTHY" $Yellow
                        $allHealthy = $false
                    }
                } else {
                    Write-ServiceStatus $service "STOPPED" $Red
                    $allHealthy = $false
                }
            }
            catch {
                Write-ServiceStatus $service "NOT FOUND" $Red
                $allHealthy = $false
            }
        }
        
        Write-ColorOutput "`n📊 System Status:" $Cyan
        if ($allHealthy) {
            Write-ColorOutput "  ✅ All services are healthy" $Green
        } else {
            Write-ColorOutput "  ⚠️  Some services have issues" $Yellow
        }
        
        Write-ColorOutput "`n🔄 Refreshing in 10 seconds... (Press Ctrl+C to stop)" $Blue
        Start-Sleep -Seconds 10
    }
}

function Test-ServiceHealth {
    Write-Header "Advanced AI System Health Check"
    
    $healthyServices = 0
    $totalServices = $Services.Count
    
    foreach ($service in $Services) {
        $containerName = "mystic-$service"
        
        try {
            # Check if container is running
            $status = docker inspect --format='{{.State.Status}}' $containerName 2>$null
            
            if ($status -eq "running") {
                # Check service-specific health
                $health = await Test-ServiceSpecificHealth $service
                
                if ($health) {
                    Write-ServiceStatus $service "HEALTHY" $Green
                    $healthyServices++
                } else {
                    Write-ServiceStatus $service "UNHEALTHY" $Yellow
                }
            } else {
                Write-ServiceStatus $service "STOPPED" $Red
            }
        }
        catch {
            Write-ServiceStatus $service "NOT FOUND" $Red
        }
    }
    
    Write-ColorOutput "`n📊 Health Summary:" $Cyan
    Write-ColorOutput "  Healthy Services: $healthyServices/$totalServices" $White
    
    if ($healthyServices -eq $totalServices) {
        Write-ColorOutput "  ✅ All services are healthy!" $Green
        return $true
    } else {
        Write-ColorOutput "  ⚠️  Some services need attention" $Yellow
        return $false
    }
}

function Test-ServiceSpecificHealth {
    param([string]$Service)
    
    try {
        switch ($Service) {
            "deep-learning-agent" {
                # Check if deep learning agent is responding
                $result = docker exec mystic-deep-learning-agent python -c "import sys; sys.exit(0)" 2>$null
                return $LASTEXITCODE -eq 0
            }
            "reinforcement-learning-agent" {
                # Check if reinforcement learning agent is responding
                $result = docker exec mystic-reinforcement-learning-agent python -c "import sys; sys.exit(0)" 2>$null
                return $LASTEXITCODE -eq 0
            }
            "ai-model-manager" {
                # Check if AI model manager is responding
                $result = docker exec mystic-ai-model-manager python -c "import sys; sys.exit(0)" 2>$null
                return $LASTEXITCODE -eq 0
            }
            "advanced-ai-orchestrator" {
                # Check if advanced AI orchestrator is responding
                $result = docker exec mystic-advanced-ai-orchestrator python -c "import sys; sys.exit(0)" 2>$null
                return $LASTEXITCODE -eq 0
            }
            default {
                return $true
            }
        }
    }
    catch {
        return $false
    }
}

function Show-ServiceInfo {
    Write-Header "Advanced AI System Information"
    
    Write-ColorOutput "🤖 Advanced AI Components:" $Cyan
    Write-ColorOutput "  • Deep Learning Agent - Neural networks for price prediction" $White
    Write-ColorOutput "  • Reinforcement Learning Agent - Strategy optimization" $White
    Write-ColorOutput "  • AI Model Manager - Model versioning and deployment" $White
    Write-ColorOutput "  • Advanced AI Orchestrator - Multi-agent coordination" $White
    
    Write-ColorOutput "`n🔧 Key Features:" $Cyan
    Write-ColorOutput "  • LSTM and CNN models for time series prediction" $White
    Write-ColorOutput "  • Q-learning and Actor-Critic algorithms" $White
    Write-ColorOutput "  • Automated model versioning and deployment" $White
    Write-ColorOutput "  • Cross-agent strategy coordination" $White
    Write-ColorOutput "  • Real-time performance monitoring" $White
    
    Write-ColorOutput "`n📊 Monitoring:" $Cyan
    Write-ColorOutput "  • Service health checks" $White
    Write-ColorOutput "  • Performance metrics collection" $White
    Write-ColorOutput "  • Model training progress" $White
    Write-ColorOutput "  • Strategy generation status" $White
    
    Write-ColorOutput "`n🌐 Integration:" $Cyan
    Write-ColorOutput "  • Redis for inter-agent communication" $White
    Write-ColorOutput "  • Docker containerization" $White
    Write-ColorOutput "  • RESTful API endpoints" $White
    Write-ColorOutput "  • Real-time data streaming" $White
}

function Show-Usage {
    Write-Header "Advanced AI System Usage"
    
    Write-ColorOutput "Usage: .\launch_advanced_ai_system.ps1 [OPTIONS]" $White
    Write-ColorOutput "`nOptions:" $Cyan
    Write-ColorOutput "  -Build          Build Docker images before starting" $White
    Write-ColorOutput "  -Logs           Show service logs" $White
    Write-ColorOutput "  -Monitor        Start real-time monitoring" $White
    Write-ColorOutput "  -Stop           Stop all services" $White
    Write-ColorOutput "  -Help           Show this help message" $White
    
    Write-ColorOutput "`nExamples:" $Cyan
    Write-ColorOutput "  .\launch_advanced_ai_system.ps1" $White
    Write-ColorOutput "  .\launch_advanced_ai_system.ps1 -Build" $White
    Write-ColorOutput "  .\launch_advanced_ai_system.ps1 -Logs" $White
    Write-ColorOutput "  .\launch_advanced_ai_system.ps1 -Monitor" $White
    Write-ColorOutput "  .\launch_advanced_ai_system.ps1 -Stop" $White
}

# Main execution
try {
    if ($Stop) {
        Stop-AdvancedAIServices
    }
    elseif ($Logs) {
        Show-ServiceLogs
    }
    elseif ($Monitor) {
        Monitor-Services
    }
    elseif ($Help -or $args -contains "-h" -or $args -contains "--help") {
        Show-Usage
    }
    else {
        Show-ServiceInfo
        Start-AdvancedAIServices
        
        Write-ColorOutput "`n🔍 Running health check..." $Blue
        Start-Sleep -Seconds 5
        Test-ServiceHealth
        
        Write-ColorOutput "`n📋 Next steps:" $Cyan
        Write-ColorOutput "  • Use -Logs to view service logs" $White
        Write-ColorOutput "  • Use -Monitor for real-time monitoring" $White
        Write-ColorOutput "  • Use -Stop to stop all services" $White
        Write-ColorOutput "  • Check the dashboard for AI insights" $White
    }
}
catch {
    Write-ColorOutput "❌ Error: $($_.Exception.Message)" $Red
    exit 1
} 