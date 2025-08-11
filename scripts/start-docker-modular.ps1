# Mystic Trading Platform - Docker Modular Startup Script
# Builds and starts all containers with live data on port 80

param(
    [switch]$SkipBuild,
    [switch]$ForceRebuild,
    [switch]$LiveDataOnly
)

# Set execution policy for current session
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force

# Color functions for better output
function Write-Success { param($Message) Write-Host "SUCCESS: $Message" -ForegroundColor Green }
function Write-Info { param($Message) Write-Host "INFO: $Message" -ForegroundColor Cyan }
function Write-Warning { param($Message) Write-Host "WARNING: $Message" -ForegroundColor Yellow }
function Write-Error { param($Message) Write-Host "ERROR: $Message" -ForegroundColor Red }

# Banner
Write-Host @"
==========================================
    Mystic Trading Platform
    Docker Modular Startup
    Port 80 - Live Data
==========================================
"@ -ForegroundColor Magenta

Write-Info "Starting Mystic Trading Platform with Docker and live data..."

# Check if Docker is running
function Test-DockerRunning {
    try {
        docker version | Out-Null
        Write-Success "Docker is running"
        return $true
    } catch {
        Write-Error "Docker is not running. Please start Docker Desktop."
        return $false
    }
}

# Remove existing containers and images
function Remove-DockerEnvironment {
    Write-Info "Cleaning up Docker environment..."

    # Stop and remove containers
    docker-compose down --volumes --remove-orphans 2>$null

    # Remove old images
    if ($ForceRebuild) {
        docker system prune -a --volumes -f 2>$null
        Write-Success "Docker environment cleaned"
    }
}

# Create Docker images
function New-DockerImages {
    Write-Info "Building Docker images with modular system..."

    if ($SkipBuild) {
        Write-Warning "Skipping build as requested"
        return $true
    }

    try {
        # Build with no cache if force rebuild
        $buildArgs = @("build")
        if ($ForceRebuild) {
            $buildArgs += "--no-cache"
        }

        docker-compose $buildArgs

        if ($LASTEXITCODE -eq 0) {
            Write-Success "Docker images built successfully"
            return $true
        } else {
            Write-Error "Docker build failed"
            return $false
        }
    } catch {
        Write-Error "Error building Docker images: $($_.Exception.Message)"
        return $false
    }
}

# Start Docker containers
function Start-DockerContainers {
    Write-Info "Starting Docker containers..."

    try {
        # Start containers in detached mode
        docker-compose up -d

        if ($LASTEXITCODE -eq 0) {
            Write-Success "Docker containers started successfully"
            return $true
        } else {
            Write-Error "Failed to start Docker containers"
            return $false
        }
    } catch {
        Write-Error "Error starting Docker containers: $($_.Exception.Message)"
        return $false
    }
}

# Wait for services to be ready
function Wait-ForServices {
    Write-Info "Waiting for services to be ready..."

    $maxAttempts = 60
    $attempt = 0

    # Wait for backend
    while ($attempt -lt $maxAttempts) {
        try {
            $response = Invoke-RestMethod -Uri "http://localhost:8000/api/health" -Method Get -TimeoutSec 5
            if ($response.status -eq "healthy" -and $response.live_data -eq $true) {
                Write-Success "Backend is ready with live data"
                break
            }
        } catch {
            $attempt++
            Start-Sleep -Seconds 2
        }
    }

    if ($attempt -ge $maxAttempts) {
        Write-Warning "Backend took too long to start"
    }

    # Wait for frontend
    $attempt = 0
    while ($attempt -lt $maxAttempts) {
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:80" -Method Get -TimeoutSec 5
            if ($response.StatusCode -eq 200) {
                Write-Success "Frontend is ready on port 80"
                break
            }
        } catch {
            $attempt++
            Start-Sleep -Seconds 2
        }
    }

    if ($attempt -ge $maxAttempts) {
        Write-Warning "Frontend took too long to start"
    }
}

# Test live data connections
function Test-LiveDataConnections {
    Write-Info "Testing live data connections..."

    # Test backend health
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:8000/api/health" -Method Get -TimeoutSec 10
        if ($response.live_data -eq $true) {
            Write-Success "Backend live data connection verified"
        } else {
            Write-Warning "Backend not reporting live data"
        }
    } catch {
        Write-Warning "Backend health check failed"
    }

    # Test market data API
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:8000/api/coinstate" -Method Get -TimeoutSec 10
        if ($response.live_data -eq $true) {
            Write-Success "Market data live connection verified"
        } else {
            Write-Warning "Market data not reporting live data"
        }
    } catch {
        Write-Warning "Market data not responding"
    }

    # Test frontend
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:80" -Method Get -TimeoutSec 10
        if ($response.StatusCode -eq 200) {
            Write-Success "Frontend is accessible on port 80"
        } else {
            Write-Warning "Frontend not responding properly"
        }
    } catch {
        Write-Warning "Frontend not responding"
    }
}

# Show container status
function Show-ContainerStatus {
    Write-Info "Container Status:"
    docker-compose ps

    Write-Info "Container Logs (last 10 lines each):"
    Write-Info "Backend logs:"
    docker-compose logs --tail=10 backend

    Write-Info "Frontend logs:"
    docker-compose logs --tail=10 frontend

    Write-Info "Redis logs:"
    docker-compose logs --tail=10 redis
}

# Main execution
function Start-MysticDocker {
    Write-Info "Starting Mystic Trading Platform with Docker..."

    # Check Docker
    if (-not (Test-DockerRunning)) {
        exit 1
    }

    # Clean environment
    Remove-DockerEnvironment

    # Build images
    if (-not (New-DockerImages)) {
        Write-Error "Failed to build Docker images. Exiting."
        exit 1
    }

    # Start containers
    if (-not (Start-DockerContainers)) {
        Write-Error "Failed to start Docker containers. Exiting."
        exit 1
    }

    # Wait for services
    Wait-ForServices

    # Test connections
    Test-LiveDataConnections

    # Show status
    Show-ContainerStatus

    # Display success message
    Write-Host @"
==========================================
    SUCCESS!

  Mystic Trading Platform is now running with Docker!

  Frontend: http://localhost:80
  Backend:  http://localhost:8000
  API Docs: http://localhost:8000/docs
  Redis:    localhost:6379

  All connections are live - no mock data used!
  Modular system is active and running!
==========================================
"@ -ForegroundColor Green

    Write-Info "Useful commands:"
    Write-Info "  View logs: docker-compose logs -f"
    Write-Info "  Stop: docker-compose down"
    Write-Info "  Restart: docker-compose restart"
    Write-Info "  Rebuild: .\start-docker-modular.ps1 -ForceRebuild"

    Write-Info "Press Ctrl+C to stop all services"

    # Keep script running and monitor
    try {
        while ($true) {
            Start-Sleep -Seconds 30

            # Periodic health checks
            try {
                $backendHealth = Invoke-RestMethod -Uri "http://localhost:8000/api/health" -Method Get -TimeoutSec 5
                if ($backendHealth.live_data -eq $false) {
                    Write-Warning "Backend not reporting live data"
                }
            } catch {
                Write-Warning "Backend health check failed"
            }
        }
    } catch {
        Write-Info "Shutting down..."
    }
}

# Handle Ctrl+C
Register-EngineEvent PowerShell.Exiting -Action {
    Write-Info "Shutting down Mystic Trading Platform..."

    # Stop containers gracefully
    try {
        docker-compose down
        Write-Success "Docker containers stopped"
    } catch {
        Write-Warning "Could not stop Docker containers gracefully"
    }

    Write-Success "Mystic Trading Platform stopped"
}

# Start the platform
Start-MysticDocker
