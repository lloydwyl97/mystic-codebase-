# Mystic Trading Platform - Full Docker Compose Management Script
# Windows 11 Home PowerShell Script

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("up", "down", "build", "restart", "logs", "status", "health", "clean", "update", "backup")]
    [string]$Action = "status"
)

# Colors for output
$Green = "Green"
$Yellow = "Yellow"
$Red = "Red"
$Cyan = "Cyan"
$White = "White"
$Magenta = "Magenta"

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

# Function to check if Docker Compose is available
function Test-DockerCompose {
    try {
        docker-compose version | Out-Null
        return $true
    }
    catch {
        try {
            docker compose version | Out-Null
            return $true
        }
        catch {
            return $false
        }
    }
}

# Function to get Docker Compose command
function Get-DockerComposeCmd {
    try {
        docker-compose version | Out-Null
        return "docker-compose"
    }
    catch {
        return "docker compose"
    }
}

# Function to create network if it doesn't exist
function New-MysticNetwork {
    $networkExists = docker network ls --format "table {{.Name}}" | Select-String "mystic-network"
    if ($networkExists.Count -eq 0) {
        Write-ColorOutput "🌐 Creating mystic-network..." $Yellow
        docker network create mystic-network
        if ($LASTEXITCODE -eq 0) {
            Write-ColorOutput "✅ Network created successfully" $Green
        } else {
            Write-ColorOutput "❌ Failed to create network" $Red
        }
    } else {
        Write-ColorOutput "✅ Network already exists" $Green
    }
}

# Function to check service health
function Test-ServiceHealth {
    param([string]$ServiceName, [string]$Url)

    try {
        $response = Invoke-WebRequest -Uri $Url -Method GET -TimeoutSec 10
        if ($response.StatusCode -eq 200) {
            Write-ColorOutput "✅ $ServiceName is healthy" $Green
            return $true
        } else {
            Write-ColorOutput "⚠️  $ServiceName responded with status: $($response.StatusCode)" $Yellow
            return $false
        }
    }
    catch {
        Write-ColorOutput "❌ $ServiceName health check failed: $($_.Exception.Message)" $Red
        return $false
    }
}

# Main script logic
Write-ColorOutput "🐳 Mystic Trading Platform - Full Docker Compose Management" $Cyan
Write-ColorOutput "=========================================================" $Yellow
Write-ColorOutput ""

# Check if Docker is running
if (-not (Test-DockerRunning)) {
    Write-ColorOutput "❌ Docker is not running. Please start Docker Desktop first." $Red
    Write-ColorOutput "   You can download Docker Desktop from: https://www.docker.com/products/docker-desktop/" $Yellow
    exit 1
}

Write-ColorOutput "✅ Docker is running" $Green

# Check if Docker Compose is available
if (-not (Test-DockerCompose)) {
    Write-ColorOutput "❌ Docker Compose is not available." $Red
    Write-ColorOutput "   Please install Docker Compose or use Docker Desktop which includes it." $Yellow
    exit 1
}

$composeCmd = Get-DockerComposeCmd
Write-ColorOutput "✅ Docker Compose is available ($composeCmd)" $Green

# Check if docker-compose.yml exists
if (-not (Test-Path "docker-compose.yml")) {
    Write-ColorOutput "❌ docker-compose.yml not found in current directory" $Red
    exit 1
}

Write-ColorOutput "✅ docker-compose.yml found" $Green

switch ($Action.ToLower()) {
    "up" {
        Write-ColorOutput "🚀 Starting full Mystic Trading Platform stack..." $Yellow

        # Create network first
        New-MysticNetwork

        # Start all services
        Write-ColorOutput "📦 Starting services (this may take a few minutes)..." $Yellow
        & $composeCmd up -d

        if ($LASTEXITCODE -eq 0) {
            Write-ColorOutput "✅ All services started successfully!" $Green
            Write-ColorOutput ""
            Write-ColorOutput "🌐 Service URLs:" $Cyan
            Write-ColorOutput "   Frontend: http://localhost" $White
            Write-ColorOutput "   Backend API: http://localhost:8000" $White
            Write-ColorOutput "   API Documentation: http://localhost:8000/docs" $White
            Write-ColorOutput "   AI Dashboard: http://localhost:8000/mutation/dashboard" $White
            Write-ColorOutput "   Redis: localhost:6379" $White
            Write-ColorOutput ""
            Write-ColorOutput "⏳ Services are starting up. Check status in 30 seconds..." $Yellow
        } else {
            Write-ColorOutput "❌ Failed to start services" $Red
            exit 1
        }
    }

    "down" {
        Write-ColorOutput "🛑 Stopping all services..." $Yellow

        & $composeCmd down

        if ($LASTEXITCODE -eq 0) {
            Write-ColorOutput "✅ All services stopped" $Green
        } else {
            Write-ColorOutput "❌ Failed to stop services" $Red
        }
    }

    "build" {
        Write-ColorOutput "🔨 Building all Docker images..." $Yellow
        Write-ColorOutput "   This may take several minutes for the first build..." $Yellow

        # Build all services
        & $composeCmd build --no-cache

        if ($LASTEXITCODE -eq 0) {
            Write-ColorOutput "✅ All images built successfully!" $Green
        } else {
            Write-ColorOutput "❌ Failed to build images" $Red
            exit 1
        }
    }

    "restart" {
        Write-ColorOutput "🔄 Restarting all services..." $Yellow

        & $composeCmd restart

        if ($LASTEXITCODE -eq 0) {
            Write-ColorOutput "✅ All services restarted" $Green
            Write-ColorOutput "🌐 Services will be available at:" $Cyan
            Write-ColorOutput "   Frontend: http://localhost" $White
            Write-ColorOutput "   Backend: http://localhost:8000" $White
        } else {
            Write-ColorOutput "❌ Failed to restart services" $Red
        }
    }

    "logs" {
        Write-ColorOutput "📋 Service logs (all services):" $Yellow
        Write-ColorOutput "   Press Ctrl+C to exit logs" $Cyan
        Write-ColorOutput ""

        & $composeCmd logs -f
    }

    "status" {
        Write-ColorOutput "📊 Full Stack Status:" $Yellow
        Write-ColorOutput ""

        # Show compose status
        & $composeCmd ps

        Write-ColorOutput ""
        Write-ColorOutput "🌐 Service URLs:" $Cyan
        Write-ColorOutput "   Frontend: http://localhost" $White
        Write-ColorOutput "   Backend API: http://localhost:8000" $White
        Write-ColorOutput "   API Documentation: http://localhost:8000/docs" $White
        Write-ColorOutput "   AI Dashboard: http://localhost:8000/mutation/dashboard" $White
        Write-ColorOutput "   Redis: localhost:6379" $White

        Write-ColorOutput ""
        Write-ColorOutput "📋 Recent logs (last 5 lines per service):" $Cyan
        & $composeCmd logs --tail 5
    }

    "health" {
        Write-ColorOutput "🏥 Checking service health..." $Yellow
        Write-ColorOutput ""

        # Check each service
        $services = @(
            @{Name="Frontend"; Url="http://localhost"},
            @{Name="Backend API"; Url="http://localhost:8000/health"},
            @{Name="API Documentation"; Url="http://localhost:8000/docs"}
        )

        foreach ($service in $services) {
            Test-ServiceHealth -ServiceName $service.Name -Url $service.Url
        }

        Write-ColorOutput ""
        Write-ColorOutput "🎯 Health check completed!" $Green
    }

    "clean" {
        Write-ColorOutput "🧹 Cleaning up all containers, images, and volumes..." $Yellow
        Write-ColorOutput "   This will remove ALL data. Are you sure? (y/N)" $Red

        $response = Read-Host
        if ($response -eq "y" -or $response -eq "Y") {
            Write-ColorOutput "🧹 Stopping and removing containers..." $Yellow
            & $composeCmd down -v

            Write-ColorOutput "🧹 Removing images..." $Yellow
            docker rmi mystic-backend:latest mystic-frontend:latest 2>$null

            Write-ColorOutput "🧹 Removing network..." $Yellow
            docker network rm mystic-network 2>$null

            Write-ColorOutput "✅ Cleanup completed" $Green
        } else {
            Write-ColorOutput "❌ Cleanup cancelled" $Yellow
        }
    }

    "update" {
        Write-ColorOutput "🔄 Updating services..." $Yellow

        # Pull latest images
        Write-ColorOutput "📥 Pulling latest images..." $Yellow
        & $composeCmd pull

        # Rebuild and restart
        Write-ColorOutput "🔨 Rebuilding services..." $Yellow
        & $composeCmd up -d --build

        if ($LASTEXITCODE -eq 0) {
            Write-ColorOutput "✅ Services updated successfully!" $Green
        } else {
            Write-ColorOutput "❌ Failed to update services" $Red
        }
    }

    "backup" {
        Write-ColorOutput "💾 Creating backup..." $Yellow

        $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
        $backupDir = "backup_$timestamp"

        # Create backup directory
        New-Item -ItemType Directory -Path $backupDir -Force | Out-Null

        # Backup database
        if (Test-Path "backend/mystic_trading.db") {
            Copy-Item "backend/mystic_trading.db" "$backupDir/"
            Write-ColorOutput "✅ Database backed up" $Green
        }

        # Backup logs
        if (Test-Path "backend/logs") {
            Copy-Item "backend/logs" "$backupDir/" -Recurse
            Write-ColorOutput "✅ Logs backed up" $Green
        }

        # Backup strategies
        if (Test-Path "backend/strategies") {
            Copy-Item "backend/strategies" "$backupDir/" -Recurse
            Write-ColorOutput "✅ Strategies backed up" $Green
        }

        Write-ColorOutput "✅ Backup completed: $backupDir" $Green
    }

    default {
        Write-ColorOutput "❌ Invalid action: $Action" $Red
        Write-ColorOutput ""
        Write-ColorOutput "Available actions:" $Yellow
        Write-ColorOutput "  up      - Start all services" $White
        Write-ColorOutput "  down    - Stop all services" $White
        Write-ColorOutput "  build   - Build all images" $White
        Write-ColorOutput "  restart - Restart all services" $White
        Write-ColorOutput "  logs    - Show all service logs" $White
        Write-ColorOutput "  status  - Show service status" $White
        Write-ColorOutput "  health  - Check service health" $White
        Write-ColorOutput "  clean   - Remove all containers/images/volumes" $White
        Write-ColorOutput "  update  - Update and rebuild services" $White
        Write-ColorOutput "  backup  - Create backup of data" $White
    }
}

Write-ColorOutput ""
Write-ColorOutput "🎯 Docker Compose management completed!" $Green
