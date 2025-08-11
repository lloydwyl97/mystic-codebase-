
# Mystic AI Trading Platform - Backend Server Startup Script
# This script ensures the backend server always starts from the correct directory

Write-Host "Starting Mystic AI Trading Platform Backend Server..." -ForegroundColor Green

# Change to backend directory
Set-Location -Path "backend"

# Check if main.py exists
if (-not (Test-Path "main.py")) {
    Write-Host "ERROR: main.py not found in backend directory!" -ForegroundColor Red
    Write-Host "Current directory: $(Get-Location)" -ForegroundColor Yellow
    exit 1
}

Write-Host "Starting uvicorn server from: $(Get-Location)" -ForegroundColor Cyan
Write-Host "Server will be available at: http://localhost:9000" -ForegroundColor Cyan

# Start the uvicorn server
uvicorn main:app --reload --host 0.0.0.0 --port 9000 