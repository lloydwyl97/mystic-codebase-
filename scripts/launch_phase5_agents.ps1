# Phase 5: Interdimensional Signal Processing & Bio-Quantum Harmonization
# Launch script for all Phase 5 agents

param(
    [switch]$SkipHealthCheck,
    [switch]$SkipLogs,
    [switch]$ForceRestart
)

Write-Host "🔮 Phase 5: Interdimensional Signal Processing & Bio-Quantum Harmonization" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan

# Configuration
$Services = @(
    @{
        Name = "interdimensional_signal_decoder"
        DisplayName = "Interdimensional Signal Decoder"
        Description = "Decodes non-linear market signatures and fractal patterns"
        Icon = "🔮"
    },
    @{
        Name = "neuro_synchronization_engine"
        DisplayName = "Neuro-Synchronization Engine"
        Description = "Links brainwave profiles to system parameters"
        Icon = "🧠"
    },
    @{
        Name = "cosmic_pattern_recognizer"
        DisplayName = "Cosmic Pattern Recognizer"
        Description = "Finds archetypal patterns in cosmic data"
        Icon = "🌌"
    },
    @{
        Name = "auranet_channel"
        DisplayName = "AuraNet Channel Interface"
        Description = "Biofeedback/EEG API for energy tuning"
        Icon = "🌀"
    }
)

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
        docker exec mystic_redis redis-cli ping | Out-Null
        return $true
    }
    catch {
        return $false
    }
}

# Function to start a service
function Start-Phase5Service {
    param(
        [string]$ServiceName,
        [string]$DisplayName,
        [string]$Description,
        [string]$Icon
    )
    
    Write-Host "$Icon Starting $DisplayName..." -ForegroundColor Yellow
    Write-Host "   Description: $Description" -ForegroundColor Gray
    
    try {
        # Check if service is already running
        $containerName = "mystic_$ServiceName"
        $existingContainer = docker ps -q -f "name=$containerName"
        
        if ($existingContainer -and -not $ForceRestart) {
            Write-Host "   ⚠️  Service already running (use -ForceRestart to restart)" -ForegroundColor Yellow
            return $true
        }
        
        # Stop existing container if force restart
        if ($existingContainer -and $ForceRestart) {
            Write-Host "   🔄 Stopping existing container..." -ForegroundColor Yellow
            docker stop $containerName | Out-Null
            docker rm $containerName | Out-Null
        }
        
        # Start the service
        docker-compose -f backend/docker-compose.yml up -d $ServiceName
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "   ✅ $DisplayName started successfully" -ForegroundColor Green
            return $true
        } else {
            Write-Host "   ❌ Failed to start $DisplayName" -ForegroundColor Red
            return $false
        }
    }
    catch {
        Write-Host "   ❌ Error starting $DisplayName: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

# Function to check service health
function Test-ServiceHealth {
    param(
        [string]$ServiceName,
        [string]$DisplayName,
        [string]$Icon
    )
    
    Write-Host "$Icon Checking health of $DisplayName..." -ForegroundColor Blue
    
    try {
        $containerName = "mystic_$ServiceName"
        $healthStatus = docker inspect --format='{{.State.Health.Status}}' $containerName 2>$null
        
        if ($healthStatus -eq "healthy") {
            Write-Host "   ✅ $DisplayName is healthy" -ForegroundColor Green
            return $true
        } elseif ($healthStatus -eq "starting") {
            Write-Host "   ⏳ $DisplayName is starting..." -ForegroundColor Yellow
            return $false
        } else {
            Write-Host "   ❌ $DisplayName health check failed: $healthStatus" -ForegroundColor Red
            return $false
        }
    }
    catch {
        Write-Host "   ❌ Error checking health of $DisplayName: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

# Function to show service logs
function Show-ServiceLogs {
    param(
        [string]$ServiceName,
        [string]$DisplayName,
        [string]$Icon
    )
    
    Write-Host "$Icon Recent logs for $DisplayName:" -ForegroundColor Magenta
    
    try {
        $containerName = "mystic_$ServiceName"
        docker logs --tail 10 $containerName 2>$null
        Write-Host ""
    }
    catch {
        Write-Host "   ❌ Error retrieving logs for $DisplayName: $($_.Exception.Message)" -ForegroundColor Red
    }
}

# Function to wait for service to be ready
function Wait-ServiceReady {
    param(
        [string]$ServiceName,
        [string]$DisplayName,
        [int]$TimeoutSeconds = 120
    )
    
    Write-Host "⏳ Waiting for $DisplayName to be ready (timeout: ${TimeoutSeconds}s)..." -ForegroundColor Yellow
    
    $startTime = Get-Date
    $timeout = $startTime.AddSeconds($TimeoutSeconds)
    
    while ((Get-Date) -lt $timeout) {
        if (Test-ServiceHealth -ServiceName $ServiceName -DisplayName $DisplayName -Icon "🔍") {
            return $true
        }
        Start-Sleep -Seconds 5
    }
    
    Write-Host "   ⏰ Timeout waiting for $DisplayName to be ready" -ForegroundColor Red
    return $false
}

# Main execution
try {
    # Check prerequisites
    Write-Host "🔍 Checking prerequisites..." -ForegroundColor Blue
    
    if (-not (Test-DockerRunning)) {
        Write-Host "❌ Docker is not running. Please start Docker and try again." -ForegroundColor Red
        exit 1
    }
    
    if (-not (Test-RedisRunning)) {
        Write-Host "❌ Redis is not running. Please start the Redis service first." -ForegroundColor Red
        Write-Host "   Run: docker-compose -f backend/docker-compose.yml up -d redis" -ForegroundColor Yellow
        exit 1
    }
    
    Write-Host "✅ Prerequisites check passed" -ForegroundColor Green
    Write-Host ""
    
    # Start services
    $successCount = 0
    $totalServices = $Services.Count
    
    foreach ($service in $Services) {
        $success = Start-Phase5Service -ServiceName $service.Name -DisplayName $service.DisplayName -Description $service.Description -Icon $service.Icon
        
        if ($success) {
            $successCount++
        }
        
        Write-Host ""
    }
    
    # Wait for services to be ready
    if (-not $SkipHealthCheck) {
        Write-Host "🔍 Waiting for services to be ready..." -ForegroundColor Blue
        Write-Host ""
        
        foreach ($service in $Services) {
            Wait-ServiceReady -ServiceName $service.Name -DisplayName $service.DisplayName
            Write-Host ""
        }
    }
    
    # Show logs if requested
    if (-not $SkipLogs) {
        Write-Host "📋 Recent logs for all Phase 5 services:" -ForegroundColor Cyan
        Write-Host "=========================================" -ForegroundColor Cyan
        Write-Host ""
        
        foreach ($service in $Services) {
            Show-ServiceLogs -ServiceName $service.Name -DisplayName $service.DisplayName -Icon $service.Icon
        }
    }
    
    # Summary
    Write-Host "📊 Phase 5 Launch Summary:" -ForegroundColor Cyan
    Write-Host "=========================" -ForegroundColor Cyan
    Write-Host "✅ Successfully started: $successCount/$totalServices services" -ForegroundColor Green
    
    if ($successCount -eq $totalServices) {
        Write-Host ""
        Write-Host "🎉 All Phase 5 services are running!" -ForegroundColor Green
        Write-Host ""
        Write-Host "🔮 Interdimensional Signal Processing & Bio-Quantum Harmonization" -ForegroundColor Cyan
        Write-Host "   - Non-linear market signature decoding" -ForegroundColor Gray
        Write-Host "   - Brainwave-to-system parameter mapping" -ForegroundColor Gray
        Write-Host "   - Cosmic pattern recognition" -ForegroundColor Gray
        Write-Host "   - Biofeedback energy tuning" -ForegroundColor Gray
        Write-Host ""
        Write-Host "📊 Monitor services:" -ForegroundColor Yellow
        Write-Host "   docker-compose -f backend/docker-compose.yml logs -f" -ForegroundColor Gray
        Write-Host ""
        Write-Host "🛑 Stop services:" -ForegroundColor Yellow
        Write-Host "   docker-compose -f backend/docker-compose.yml stop" -ForegroundColor Gray
    } else {
        Write-Host "⚠️  Some services failed to start. Check logs for details." -ForegroundColor Yellow
        exit 1
    }
}
catch {
    Write-Host "❌ Error during Phase 5 launch: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
} 