# Mystic AI Trading Platform - Windows Startup Script
# Run this script to start the complete AI trading system

Write-Host "🚀 Starting Mystic AI Trading Platform..." -ForegroundColor Green

# Check if Python is installed
try {
    $pythonVersion = python --version
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found. Please install Python 3.11+" -ForegroundColor Red
    exit 1
}

# Check if virtual environment exists
if (-not (Test-Path "venv")) {
    Write-Host "📦 Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
}

# Activate virtual environment
Write-Host "🔧 Activating virtual environment..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"

# Install dependencies
Write-Host "📚 Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

# Create necessary directories
Write-Host "📁 Creating directories..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "backups"
New-Item -ItemType Directory -Force -Path "model_versions"

# Test the system
Write-Host "🧪 Testing AI system..." -ForegroundColor Yellow
python test_ai_system.py

# Start the AI trading platform
Write-Host "🤖 Starting AI Trading Platform..." -ForegroundColor Green
Write-Host "   - Dashboard: http://localhost:8000/ai/dashboard" -ForegroundColor Cyan
Write-Host "   - Health: http://localhost:8000/health" -ForegroundColor Cyan
Write-Host "   - Press Ctrl+C to stop" -ForegroundColor Yellow

# Start the main application
python start_ai_trading.py
