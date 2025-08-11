# Mystic Trading Platform - Backend Docker Management Script
# Windows 11 Home PowerShell Script

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("build", "start", "stop", "restart", "logs", "shell", "clean", "status", "health")]
    [string]$Action = "status"
)

# Colors for output
$Green = "Green"
$Yellow = "Yellow"
$Red = "Red"
$Cyan = "Cyan"
$White = "White"

# Function to write colored output
function Write-ColorOutput {
    param([string]$Message, [string]$Color = $White)
    Write-Host $Message -ForegroundColor $Color
}

# Function to check if Docker is running
function Test-DockerRunning {
    try {
        docker version | Out-Null
        return $true
    }
    catch {
        return $false
    }
}

# Function to check if container exists
function Test-ContainerExists {
    param([string]$ContainerName)
    $containers = docker ps -a --format "table {{.Names}}" | Select-String $ContainerName
    return $containers.Count -gt 0
}

# Function to check if container is running
function Test-ContainerRunning {
    param([string]$ContainerName)
    $running = docker ps --format "table {{.Names}}" | Select-String $ContainerName
    return $running.Count -gt 0
}

# Function to get container status
function Get-ContainerStatus {
    param([string]$ContainerName)
    if (Test-ContainerExists $ContainerName) {
        if (Test-ContainerRunning $ContainerName) {
            return "Running"
        } else {
            return "Stopped"
        }
    } else {
        return "Not Found"
    }
}

# Main script logic
Write-ColorOutput "🐳 Mystic Trading Platform - Backend Docker Management" $Cyan
Write-ColorOutput "=====================================================" $Yellow
Write-ColorOutput ""

# Check if Docker is running
if (-not (Test-DockerRunning)) {
    Write-ColorOutput "❌ Docker is not running. Please start Docker Desktop first." $Red
    Write-ColorOutput "   You can download Docker Desktop from: https://www.docker.com/products/docker-desktop/" $Yellow
    exit 1
}

Write-ColorOutput "✅ Docker is running" $Green

# Navigate to backend directory
$backendPath = Join-Path $PSScriptRoot "backend"
if (-not (Test-Path $backendPath)) {
    Write-ColorOutput "❌ Backend directory not found at: $backendPath" $Red
    exit 1
}

Set-Location $backendPath
Write-ColorOutput "📁 Working directory: $(Get-Location)" $Cyan

# Container name
$containerName = "mystic-backend"

switch ($Action.ToLower()) {
    "build" {
        Write-ColorOutput "🔨 Building backend Docker image..." $Yellow
        Write-ColorOutput "   This may take several minutes for the first build..." $Yellow

        # Build the image
        docker build -t mystic-backend:latest .

        if ($LASTEXITCODE -eq 0) {
            Write-ColorOutput "✅ Backend image built successfully!" $Green
        } else {
            Write-ColorOutput "❌ Failed to build backend image" $Red
            exit 1
        }
    }

    "start" {
        Write-ColorOutput "🚀 Starting backend container..." $Yellow

        if (Test-ContainerRunning $containerName) {
            Write-ColorOutput "⚠️  Container is already running" $Yellow
        } else {
            # Start the container
            docker run -d `
                --name $containerName `
                --network mystic-network `
                -p 8000:8000 `
                -e ENVIRONMENT=production `
                -e REDIS_URL=redis://redis:6379 `
                -e DATABASE_URL=sqlite:///./mystic_trading.db `
                -e SECRET_KEY=vbdueie85kfgmxklso `
                -e LOG_LEVEL=INFO `
                -e HOST=0.0.0.0 `
                -e PORT=8000 `
                -e USE_MOCK_DATA=false `
                -e TRADING_ENABLED=false `
                -v "${PWD}/logs:/app/logs" `
                -v "${PWD}/mystic_trading.db:/app/mystic_trading.db" `
                --restart unless-stopped `
                mystic-backend:latest

            if ($LASTEXITCODE -eq 0) {
                Write-ColorOutput "✅ Backend container started successfully!" $Green
                Write-ColorOutput "🌐 API will be available at: http://localhost:8000" $Cyan
                Write-ColorOutput "📊 Dashboard: http://localhost:8000/mutation/dashboard" $Cyan
                Write-ColorOutput "📚 API Docs: http://localhost:8000/docs" $Cyan
            } else {
                Write-ColorOutput "❌ Failed to start backend container" $Red
                exit 1
            }
        }
    }

    "stop" {
        Write-ColorOutput "🛑 Stopping backend container..." $Yellow

        if (Test-ContainerRunning $containerName) {
            docker stop $containerName
            if ($LASTEXITCODE -eq 0) {
                Write-ColorOutput "✅ Backend container stopped" $Green
            } else {
                Write-ColorOutput "❌ Failed to stop backend container" $Red
            }
        } else {
            Write-ColorOutput "⚠️  Container is not running" $Yellow
        }
    }

    "restart" {
        Write-ColorOutput "🔄 Restarting backend container..." $Yellow

        if (Test-ContainerExists $containerName) {
            docker restart $containerName
            if ($LASTEXITCODE -eq 0) {
                Write-ColorOutput "✅ Backend container restarted" $Green
                Write-ColorOutput "🌐 API will be available at: http://localhost:8000" $Cyan
            } else {
                Write-ColorOutput "❌ Failed to restart backend container" $Red
            }
        } else {
            Write-ColorOutput "⚠️  Container does not exist. Use 'start' to create it." $Yellow
        }
    }

    "logs" {
        Write-ColorOutput "📋 Backend container logs:" $Yellow
        Write-ColorOutput "   Press Ctrl+C to exit logs" $Cyan
        Write-ColorOutput ""

        if (Test-ContainerExists $containerName) {
            docker logs -f $containerName
        } else {
            Write-ColorOutput "❌ Container does not exist" $Red
        }
    }

    "shell" {
        Write-ColorOutput "🐚 Opening shell in backend container..." $Yellow

        if (Test-ContainerRunning $containerName) {
            docker exec -it $containerName /bin/bash
        } else {
            Write-ColorOutput "❌ Container is not running. Start it first with 'start'" $Red
        }
    }

    "clean" {
        Write-ColorOutput "🧹 Cleaning up backend containers and images..." $Yellow

        # Stop and remove container
        if (Test-ContainerExists $containerName) {
            docker stop $containerName 2>$null
            docker rm $containerName 2>$null
            Write-ColorOutput "✅ Container removed" $Green
        }

        # Remove image
        docker rmi mystic-backend:latest 2>$null
        Write-ColorOutput "✅ Image removed" $Green

        Write-ColorOutput "🧹 Cleanup completed" $Green
    }

    "status" {
        Write-ColorOutput "📊 Backend Container Status:" $Yellow
        Write-ColorOutput "   Container: $containerName" $White
        Write-ColorOutput "   Status: $(Get-ContainerStatus $containerName)" $White

        if (Test-ContainerRunning $containerName) {
            Write-ColorOutput ""
            Write-ColorOutput "🌐 Service URLs:" $Cyan
            Write-ColorOutput "   API: http://localhost:8000" $White
            Write-ColorOutput "   Dashboard: http://localhost:8000/mutation/dashboard" $White
            Write-ColorOutput "   API Docs: http://localhost:8000/docs" $White
            Write-ColorOutput "   Health: http://localhost:8000/health" $White

            Write-ColorOutput ""
            Write-ColorOutput "📋 Recent logs (last 10 lines):" $Cyan
            docker logs --tail 10 $containerName
        }
    }

    "health" {
        Write-ColorOutput "🏥 Checking backend health..." $Yellow

        if (Test-ContainerRunning $containerName) {
            try {
                $response = Invoke-RestMethod -Uri "http://localhost:8000/health" -Method GET -TimeoutSec 10
                Write-ColorOutput "✅ Backend is healthy: $($response.status)" $Green
            }
            catch {
                Write-ColorOutput "❌ Backend health check failed: $($_.Exception.Message)" $Red
            }
        } else {
            Write-ColorOutput "❌ Container is not running" $Red
        }
    }

    default {
        Write-ColorOutput "❌ Invalid action: $Action" $Red
        Write-ColorOutput ""
        Write-ColorOutput "Available actions:" $Yellow
        Write-ColorOutput "  build   - Build the backend Docker image" $White
        Write-ColorOutput "  start   - Start the backend container" $White
        Write-ColorOutput "  stop    - Stop the backend container" $White
        Write-ColorOutput "  restart - Restart the backend container" $White
        Write-ColorOutput "  logs    - Show container logs" $White
        Write-ColorOutput "  shell   - Open shell in container" $White
        Write-ColorOutput "  clean   - Remove container and image" $White
        Write-ColorOutput "  status  - Show container status" $White
        Write-ColorOutput "  health  - Check backend health" $White
    }
}

Write-ColorOutput ""
Write-ColorOutput "🎯 Backend Docker management completed!" $Green
