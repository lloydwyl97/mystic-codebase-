# ===== BUILD MYSTIC BASE IMAGE SCRIPT =====
# This script builds the shared base image for all Mystic services

Write-Host "Building Mystic Base Image..." -ForegroundColor Green

# Check if Docker is running
try {
    docker version | Out-Null
} catch {
    Write-Host "Docker is not running. Please start Docker Desktop first." -ForegroundColor Red
    exit 1
}

# Build the base image
Write-Host "Building mystic_base:latest..." -ForegroundColor Yellow
docker build -f Dockerfile.base -t mystic_base:latest .

if ($LASTEXITCODE -eq 0) {
    Write-Host "Base image built successfully!" -ForegroundColor Green
    Write-Host "Image size:" -ForegroundColor Cyan
    docker images mystic_base:latest --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"
} else {
    Write-Host "Failed to build base image" -ForegroundColor Red
    exit 1
}

Write-Host "Base image ready for service inheritance!" -ForegroundColor Green 