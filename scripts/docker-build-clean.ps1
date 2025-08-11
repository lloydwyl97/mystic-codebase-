# Mystic Trading Platform - Docker Build and Clean Script
param(
    [switch]$Build,
    [switch]$Clean,
    [switch]$FullClean,
    [switch]$Start,
    [switch]$Stop,
    [switch]$Restart,
    [switch]$Logs,
    [switch]$Status,
    [switch]$Help
)

Write-Host "Docker Management for Mystic Trading Platform" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green

# Function to check if Docker is running
function Test-Docker {
    try {
        docker version | Out-Null
        return $true
    } catch {
        Write-Host "Docker is not running or not installed!" -ForegroundColor Red
        Write-Host "Please start Docker Desktop and try again." -ForegroundColor Yellow
        return $false
    }
}

# Function to clean Docker resources
function Clear-DockerResources {
    Write-Host "Cleaning Docker resources..." -ForegroundColor Yellow
    
    # Stop and remove containers
    Write-Host "Stopping containers..." -ForegroundColor Yellow
    docker-compose -f ./docker-compose.yml down --remove-orphans 2>$null
    
    # Remove containers
    Write-Host "Removing containers..." -ForegroundColor Yellow
    docker container prune -f 2>$null
    
    # Remove images
    Write-Host "Removing images..." -ForegroundColor Yellow
    docker image prune -f 2>$null
    
    # Remove volumes
    Write-Host "Removing volumes..." -ForegroundColor Yellow
    docker volume prune -f 2>$null
    
    # Remove networks
    Write-Host "Removing networks..." -ForegroundColor Yellow
    docker network prune -f 2>$null
    
    Write-Host "Docker cleanup complete!" -ForegroundColor Green
}

# Function to full clean (including build cache)
function Clear-DockerFull {
    Write-Host "Full Docker cleanup (including build cache)..." -ForegroundColor Yellow
    
    # Stop all containers
    docker stop $(docker ps -aq) 2>$null
    
    # Remove all containers
    docker rm $(docker ps -aq) 2>$null
    
    # Remove all images
    docker rmi $(docker images -q) 2>$null
    
    # Remove all volumes
    docker volume rm $(docker volume ls -q) 2>$null
    
    # Remove all networks
    docker network rm $(docker network ls -q) 2>$null
    
    # Clean build cache
    docker builder prune -a -f 2>$null
    
    # System prune
    docker system prune -a -f 2>$null
    
    Write-Host "Full Docker cleanup complete!" -ForegroundColor Green
}

# Function to build Docker images
function New-DockerImages {
    Write-Host "Building Docker images..." -ForegroundColor Yellow
    
    # Check if .env file exists
    if (-not (Test-Path ".env")) {
        Write-Host ".env file not found!" -ForegroundColor Red
        Write-Host "Please ensure your .env file is in the root directory." -ForegroundColor Yellow
        return
    }
    
    # Build backend image
    Write-Host "Building backend image..." -ForegroundColor Yellow
    docker build -t mystic-backend:latest ./backend
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Backend build failed!" -ForegroundColor Red
        return
    }
    
    # Build frontend image
    Write-Host "Building frontend image..." -ForegroundColor Yellow
    docker build -t mystic-frontend:latest ./frontend
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Frontend build failed!" -ForegroundColor Red
        return
    }
    
    Write-Host "Docker images built successfully!" -ForegroundColor Green
}

# Function to start Docker services
function Start-DockerServices {
    Write-Host "Starting Docker services..." -ForegroundColor Yellow
    
    # Start services with docker-compose
    docker-compose -f ./docker-compose.yml up -d
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Docker services started successfully!" -ForegroundColor Green
        Write-Host ""
        Write-Host "Services available at:" -ForegroundColor Cyan
        Write-Host "Frontend: http://localhost" -ForegroundColor White
        Write-Host "Backend API: http://localhost:8000" -ForegroundColor White
        Write-Host "API Docs: http://localhost:8000/docs" -ForegroundColor White
        Write-Host "Redis: localhost:6379" -ForegroundColor White
    } else {
        Write-Host "Failed to start Docker services!" -ForegroundColor Red
    }
}

# Function to stop Docker services
function Stop-DockerServices {
    Write-Host "Stopping Docker services..." -ForegroundColor Yellow
    
    docker-compose -f ./docker-compose.yml down
    
    Write-Host "Docker services stopped!" -ForegroundColor Green
}

# Function to show Docker status
function Get-DockerStatus {
    Write-Host "Docker Status:" -ForegroundColor Yellow
    
    # Show running containers
    Write-Host "Running containers:" -ForegroundColor Cyan
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    
    Write-Host ""
    
    # Show images
    Write-Host "Available images:" -ForegroundColor Cyan
    docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"
    
    Write-Host ""
    
    # Show volumes
    Write-Host "Volumes:" -ForegroundColor Cyan
    docker volume ls
}

# Function to show logs
function Get-DockerLogs {
    Write-Host "Docker Logs:" -ForegroundColor Yellow
    
    # Show logs for all services
    docker-compose -f ./docker-compose.yml logs --tail=50
}

# Function to show help
function Get-DockerHelp {
    Write-Host "Usage: .\scripts\docker-build-clean.ps1 [options]" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Options:" -ForegroundColor Cyan
    Write-Host "  -Build      Build Docker images" -ForegroundColor White
    Write-Host "  -Clean      Clean Docker resources (containers, images, volumes)" -ForegroundColor White
    Write-Host "  -FullClean  Full cleanup including build cache" -ForegroundColor White
    Write-Host "  -Start      Start Docker services" -ForegroundColor White
    Write-Host "  -Stop       Stop Docker services" -ForegroundColor White
    Write-Host "  -Restart    Restart Docker services" -ForegroundColor White
    Write-Host "  -Logs       Show Docker logs" -ForegroundColor White
    Write-Host "  -Status     Show Docker status" -ForegroundColor White
    Write-Host "  -Help       Show this help message" -ForegroundColor White
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Cyan
    Write-Host "  .\scripts\docker-build-clean.ps1 -Build -Start" -ForegroundColor White
    Write-Host "  .\scripts\docker-build-clean.ps1 -Clean" -ForegroundColor White
    Write-Host "  .\scripts\docker-build-clean.ps1 -Status" -ForegroundColor White
}

# Main execution
if ($Help) {
    Get-DockerHelp
    exit 0
}

# Check if Docker is running
if (-not (Test-Docker)) {
    exit 1
}

# Execute requested actions
if ($Clean) {
    Clear-DockerResources
}

if ($FullClean) {
    Clear-DockerFull
}

if ($Build) {
    New-DockerImages
}

if ($Stop) {
    Stop-DockerServices
}

if ($Start) {
    Start-DockerServices
}

if ($Restart) {
    Stop-DockerServices
    Start-Sleep -Seconds 2
    Start-DockerServices
}

if ($Status) {
    Get-DockerStatus
}

if ($Logs) {
    Get-DockerLogs
}

# If no specific action is requested, show help
if (-not ($Clean -or $FullClean -or $Build -or $Start -or $Stop -or $Restart -or $Status -or $Logs)) {
    Get-DockerHelp
} 