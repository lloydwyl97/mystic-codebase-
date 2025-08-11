# Mystic Trading Platform - Frontend Docker Management Script
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
Write-ColorOutput "🐳 Mystic Trading Platform - Frontend Docker Management" $Cyan
Write-ColorOutput "======================================================" $Yellow
Write-ColorOutput ""

# Check if Docker is running
if (-not (Test-DockerRunning)) {
    Write-ColorOutput "❌ Docker is not running. Please start Docker Desktop first." $Red
    Write-ColorOutput "   You can download Docker Desktop from: https://www.docker.com/products/docker-desktop/" $Yellow
    exit 1
}

Write-ColorOutput "✅ Docker is running" $Green

# Navigate to frontend directory
$frontendPath = Join-Path $PSScriptRoot "frontend"
if (-not (Test-Path $frontendPath)) {
    Write-ColorOutput "❌ Frontend directory not found at: $frontendPath" $Red
    exit 1
}

Set-Location $frontendPath
Write-ColorOutput "📁 Working directory: $(Get-Location)" $Cyan

# Container name
$containerName = "mystic-frontend"

switch ($Action.ToLower()) {
    "build" {
        Write-ColorOutput "🔨 Building frontend Docker image..." $Yellow
        Write-ColorOutput "   This may take several minutes for the first build..." $Yellow

        # Check if package.json exists
        if (-not (Test-Path "package.json")) {
            Write-ColorOutput "❌ package.json not found. Are you in the correct directory?" $Red
            exit 1
        }

        # Build the image
        docker build -t mystic-frontend:latest .

        if ($LASTEXITCODE -eq 0) {
            Write-ColorOutput "✅ Frontend image built successfully!" $Green
        } else {
            Write-ColorOutput "❌ Failed to build frontend image" $Red
            exit 1
        }
    }

    "start" {
        Write-ColorOutput "🚀 Starting frontend container..." $Yellow

        if (Test-ContainerRunning $containerName) {
            Write-ColorOutput "⚠️  Container is already running" $Yellow
        } else {
            # Start the container
            docker run -d `
                --name $containerName `
                --network mystic-network `
                -p 80:80 `
                -e VITE_API_URL=http://localhost:8000 `
                --restart unless-stopped `
                mystic-frontend:latest

            if ($LASTEXITCODE -eq 0) {
                Write-ColorOutput "✅ Frontend container started successfully!" $Green
                Write-ColorOutput "🌐 Frontend will be available at: http://localhost" $Cyan
                Write-ColorOutput "📱 Web App: http://localhost" $Cyan
                Write-ColorOutput "🔗 API Backend: http://localhost:8000" $Cyan
            } else {
                Write-ColorOutput "❌ Failed to start frontend container" $Red
                exit 1
            }
        }
    }

    "stop" {
        Write-ColorOutput "🛑 Stopping frontend container..." $Yellow

        if (Test-ContainerRunning $containerName) {
            docker stop $containerName
            if ($LASTEXITCODE -eq 0) {
                Write-ColorOutput "✅ Frontend container stopped" $Green
            } else {
                Write-ColorOutput "❌ Failed to stop frontend container" $Red
            }
        } else {
            Write-ColorOutput "⚠️  Container is not running" $Yellow
        }
    }

    "restart" {
        Write-ColorOutput "🔄 Restarting frontend container..." $Yellow

        if (Test-ContainerExists $containerName) {
            docker restart $containerName
            if ($LASTEXITCODE -eq 0) {
                Write-ColorOutput "✅ Frontend container restarted" $Green
                Write-ColorOutput "🌐 Frontend will be available at: http://localhost" $Cyan
            } else {
                Write-ColorOutput "❌ Failed to restart frontend container" $Red
            }
        } else {
            Write-ColorOutput "⚠️  Container does not exist. Use 'start' to create it." $Yellow
        }
    }

    "logs" {
        Write-ColorOutput "📋 Frontend container logs:" $Yellow
        Write-ColorOutput "   Press Ctrl+C to exit logs" $Cyan
        Write-ColorOutput ""

        if (Test-ContainerExists $containerName) {
            docker logs -f $containerName
        } else {
            Write-ColorOutput "❌ Container does not exist" $Red
        }
    }

    "shell" {
        Write-ColorOutput "🐚 Opening shell in frontend container..." $Yellow

        if (Test-ContainerRunning $containerName) {
            docker exec -it $containerName /bin/sh
        } else {
            Write-ColorOutput "❌ Container is not running. Start it first with 'start'" $Red
        }
    }

    "clean" {
        Write-ColorOutput "🧹 Cleaning up frontend containers and images..." $Yellow

        # Stop and remove container
        if (Test-ContainerExists $containerName) {
            docker stop $containerName 2>$null
            docker rm $containerName 2>$null
            Write-ColorOutput "✅ Container removed" $Green
        }

        # Remove image
        docker rmi mystic-frontend:latest 2>$null
        Write-ColorOutput "✅ Image removed" $Green

        Write-ColorOutput "🧹 Cleanup completed" $Green
    }

    "status" {
        Write-ColorOutput "📊 Frontend Container Status:" $Yellow
        Write-ColorOutput "   Container: $containerName" $White
        Write-ColorOutput "   Status: $(Get-ContainerStatus $containerName)" $White

        if (Test-ContainerRunning $containerName) {
            Write-ColorOutput ""
            Write-ColorOutput "🌐 Service URLs:" $Cyan
            Write-ColorOutput "   Frontend: http://localhost" $White
            Write-ColorOutput "   Web App: http://localhost" $White
            Write-ColorOutput "   API Backend: http://localhost:8000" $White

            Write-ColorOutput ""
            Write-ColorOutput "📋 Recent logs (last 10 lines):" $Cyan
            docker logs --tail 10 $containerName
        }
    }

    "health" {
        Write-ColorOutput "🏥 Checking frontend health..." $Yellow

        if (Test-ContainerRunning $containerName) {
            try {
                $response = Invoke-WebRequest -Uri "http://localhost" -Method GET -TimeoutSec 10
                if ($response.StatusCode -eq 200) {
                    Write-ColorOutput "✅ Frontend is healthy: Status $($response.StatusCode)" $Green
                } else {
                    Write-ColorOutput "⚠️  Frontend responded with status: $($response.StatusCode)" $Yellow
                }
            }
            catch {
                Write-ColorOutput "❌ Frontend health check failed: $($_.Exception.Message)" $Red
            }
        } else {
            Write-ColorOutput "❌ Container is not running" $Red
        }
    }

    default {
        Write-ColorOutput "❌ Invalid action: $Action" $Red
        Write-ColorOutput ""
        Write-ColorOutput "Available actions:" $Yellow
        Write-ColorOutput "  build   - Build the frontend Docker image" $White
        Write-ColorOutput "  start   - Start the frontend container" $White
        Write-ColorOutput "  stop    - Stop the frontend container" $White
        Write-ColorOutput "  restart - Restart the frontend container" $White
        Write-ColorOutput "  logs    - Show container logs" $White
        Write-ColorOutput "  shell   - Open shell in container" $White
        Write-ColorOutput "  clean   - Remove container and image" $White
        Write-ColorOutput "  status  - Show container status" $White
        Write-ColorOutput "  health  - Check frontend health" $White
    }
}

Write-ColorOutput ""
Write-ColorOutput "🎯 Frontend Docker management completed!" $Green
