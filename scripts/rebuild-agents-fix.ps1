#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Rebuild Docker images with fixes for missing dependencies

.DESCRIPTION
    This script rebuilds the Docker images to fix the missing qiskit, torch, 
    ai_strategy_generator, and db_logger modules, and Redis connection issues.
#>

Write-Host "🔧 Rebuilding Docker images with dependency fixes..." -ForegroundColor Blue

# Stop all running containers
Write-Host "🛑 Stopping all containers..." -ForegroundColor Yellow
docker-compose down

# Remove old images
Write-Host "🗑️ Removing old images..." -ForegroundColor Yellow
docker rmi mystic-agents:latest 2>$null
docker rmi mystic-visualization:latest 2>$null
docker rmi mystic-backend-optimized:latest 2>$null
docker rmi mystic-ai-optimized:latest 2>$null

# Build base image first
Write-Host "🏗️ Building base image..." -ForegroundColor Green
docker build -f Dockerfile.base -t mystic-base:latest .

# Build backend optimized image
Write-Host "🏗️ Building backend optimized image..." -ForegroundColor Green
docker build -f backend/Dockerfile.backend-optimized -t mystic-backend-optimized:latest .

# Build AI optimized image
Write-Host "🏗️ Building AI optimized image..." -ForegroundColor Green
docker build -f ai/Dockerfile.optimized -t mystic-ai-optimized:latest .

# Build agents image with fixes
Write-Host "🏗️ Building agents image with dependency fixes..." -ForegroundColor Green
docker build -f backend/Dockerfile.agents -t mystic-agents:latest .

# Build visualization image with fixes
Write-Host "🏗️ Building visualization image with db_logger fix..." -ForegroundColor Green
docker build -f services/visualization/Dockerfile -t mystic-visualization:latest .

# Build other service images
Write-Host "🏗️ Building other service images..." -ForegroundColor Green
docker build -f middleware/Dockerfile -t mystic-middleware:latest .
docker build -f alerts/Dockerfile -t mystic-alerts:latest .

# Start services
Write-Host "🚀 Starting services..." -ForegroundColor Green
docker-compose up -d

# Wait for services to start
Write-Host "⏳ Waiting for services to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 30

# Check service status
Write-Host "📊 Checking service status..." -ForegroundColor Green
docker-compose ps

Write-Host "✅ Rebuild complete! Check the logs for any remaining issues." -ForegroundColor Green
Write-Host "📋 To view logs: docker-compose logs -f" -ForegroundColor Cyan 