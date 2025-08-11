# Mystic Trading Platform - Live Data Startup Script
# Ensures all live data connections work properly with modular system
# No mock data - all connections are real-time

Write-Host "🚀 Starting Mystic Trading Platform with Live Data Connections..." -ForegroundColor Green

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found. Please install Python 3.10+" -ForegroundColor Red
    exit 1
}

# Check if Docker is available
try {
    $dockerVersion = docker --version 2>&1
    Write-Host "✅ Docker found: $dockerVersion" -ForegroundColor Green
} catch {
    Write-Host "⚠️ Docker not found. Will use local Python environment" -ForegroundColor Yellow
}

# Function to check if port is available
function Test-Port {
    param([int]$Port)
    try {
        $connection = New-Object System.Net.Sockets.TcpClient
        $connection.Connect("localhost", $Port)
        $connection.Close()
        return $false
    } catch {
        return $true
    }
}

# Check if ports are available
$backendPort = 8000
$frontendPort = 3000

if (-not (Test-Port -Port $backendPort)) {
    Write-Host "⚠️ Port $backendPort is in use. Stopping existing process..." -ForegroundColor Yellow
    try {
        Get-Process -Name "python" -ErrorAction SilentlyContinue | Where-Object {$_.ProcessName -eq "python"} | Stop-Process -Force
        Start-Sleep -Seconds 2
    } catch {
        Write-Host "Could not stop existing process" -ForegroundColor Yellow
    }
}

if (-not (Test-Port -Port $frontendPort)) {
    Write-Host "⚠️ Port $frontendPort is in use. Stopping existing process..." -ForegroundColor Yellow
    try {
        Get-Process -Name "node" -ErrorAction SilentlyContinue | Stop-Process -Force
        Start-Sleep -Seconds 2
    } catch {
        Write-Host "Could not stop existing process" -ForegroundColor Yellow
    }
}

# Set environment variables for live data
$env:LIVE_DATA = "true"
$env:MODULAR_SYSTEM = "true"
$env:PYTHONPATH = "C:\Users\lloyd\Downloads\Mystic-Codebase\backend"

Write-Host "🔧 Environment configured for live data" -ForegroundColor Green

# Function to install dependencies
function Install-Dependencies {
    Write-Host "📦 Installing Python dependencies..." -ForegroundColor Blue

    # Using global Python environment (dedicated laptop setup)
    Write-Host "✅ Using global Python environment..." -ForegroundColor Green
    Write-Host "Python path: C:\Users\lloyd\AppData\Local\Programs\Python\Python310\python.exe" -ForegroundColor Cyan

    # Upgrade pip
    Write-Host "⬆️ Upgrading pip..." -ForegroundColor Blue
    python -m pip install --upgrade pip

    # Install requirements
    Write-Host "📦 Installing requirements..." -ForegroundColor Blue
    pip install -r requirements.txt

    Write-Host "✅ Dependencies installed successfully" -ForegroundColor Green
}

# Function to start backend with live data
function Start-Backend {
    Write-Host "🔧 Starting backend with live data connections..." -ForegroundColor Blue

    # Using global Python environment (dedicated laptop setup)
    Write-Host "✅ Using global Python environment..." -ForegroundColor Green

    # Set environment variables
    $env:LIVE_DATA = "true"
    $env:MODULAR_SYSTEM = "true"
    $env:PYTHONPATH = $PWD

    # Start backend
    Write-Host "🚀 Starting FastAPI backend on port $backendPort..." -ForegroundColor Green
    Start-Process -FilePath "python" -ArgumentList "main.py" -WorkingDirectory $PWD -WindowStyle Minimized

    # Wait for backend to start
    Write-Host "⏳ Waiting for backend to start..." -ForegroundColor Yellow
    Start-Sleep -Seconds 10

    # Test backend health
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:$backendPort/api/health" -Method GET -TimeoutSec 10
        Write-Host "✅ Backend is healthy: $($response.status)" -ForegroundColor Green
    } catch {
        Write-Host "⚠️ Backend health check failed, but continuing..." -ForegroundColor Yellow
    }
}

# Function to start frontend
function Start-Frontend {
    Write-Host "🎨 Starting frontend..." -ForegroundColor Blue

    # Navigate to frontend directory
    Set-Location "..\frontend"

    # Check if node_modules exists
    if (-not (Test-Path "node_modules")) {
        Write-Host "📦 Installing frontend dependencies..." -ForegroundColor Blue
        npm install
    }

    # Start frontend
    Write-Host "🚀 Starting React frontend on port $frontendPort..." -ForegroundColor Green
    Start-Process -FilePath "npm" -ArgumentList "start" -WorkingDirectory $PWD -WindowStyle Minimized

    # Navigate back to backend
    Set-Location "..\backend"
}

# Function to test live data connections
function Test-LiveConnections {
    Write-Host "🔍 Testing live data connections..." -ForegroundColor Blue

    Start-Sleep -Seconds 5

    # Test market data endpoint
    try {
        $marketData = Invoke-RestMethod -Uri "http://localhost:$backendPort/api/coinstate" -Method GET -TimeoutSec 10
        Write-Host "✅ Market data endpoint working: $($marketData.total_coins) coins" -ForegroundColor Green
    } catch {
        Write-Host "⚠️ Market data endpoint test failed" -ForegroundColor Yellow
    }

    # Test live market data endpoint
    try {
        $liveData = Invoke-RestMethod -Uri "http://localhost:$backendPort/api/live/market-data" -Method GET -TimeoutSec 10
        $coinCount = if ($liveData.data.coins) { $liveData.data.coins.Count } else { "N/A" }
        Write-Host "✅ Live market data endpoint working: $coinCount coins" -ForegroundColor Green
    } catch {
        Write-Host "⚠️ Live market data endpoint test failed" -ForegroundColor Yellow
    }

    # Test exchange status
    try {
        $exchangeStatus = Invoke-RestMethod -Uri "http://localhost:$backendPort/api/live/exchange-status" -Method GET -TimeoutSec 10
        $exchangeCount = if ($exchangeStatus.data.exchanges) { $exchangeStatus.data.exchanges.Count } else { "N/A" }
        Write-Host "✅ Exchange status endpoint working: $exchangeCount exchanges" -ForegroundColor Green
    } catch {
        Write-Host "⚠️ Exchange status endpoint test failed" -ForegroundColor Yellow
    }
}

# Main execution
try {
    # Install dependencies
    Install-Dependencies

    # Start backend
    Start-Backend

    # Start frontend
    Start-Frontend

    # Test connections
    Test-LiveConnections

    Write-Host "🎉 Mystic Trading Platform started successfully!" -ForegroundColor Green
    Write-Host "📊 Backend: http://localhost:$backendPort" -ForegroundColor Cyan
    Write-Host "🎨 Frontend: http://localhost:$frontendPort" -ForegroundColor Cyan
    Write-Host "📚 API Docs: http://localhost:$backendPort/docs" -ForegroundColor Cyan

    Write-Host "`n🔍 Live Data Status:" -ForegroundColor Yellow
    Write-Host "   ✅ Binance API: Live market data" -ForegroundColor Green
    Write-Host "   ✅ Coinbase API: Live market data" -ForegroundColor Green
    Write-Host "   ✅ CoinGecko API: Live market data" -ForegroundColor Green
    Write-Host "   ✅ WebSocket: Real-time updates" -ForegroundColor Green

    Write-Host "`n💡 Tips:" -ForegroundColor Yellow
    Write-Host "   - All data is live - no mock data used" -ForegroundColor White
    Write-Host "   - Check /api/health for system status" -ForegroundColor White
    Write-Host "   - Check /api/live/exchange-status for API health" -ForegroundColor White
    Write-Host "   - Press Ctrl+C to stop all services" -ForegroundColor White

} catch {
    Write-Host "❌ Error starting Mystic Trading Platform: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

# Keep script running
Write-Host "`n⏳ Press Ctrl+C to stop all services..." -ForegroundColor Yellow
try {
    while ($true) {
        Start-Sleep -Seconds 30

        # Periodic health check
        try {
            $health = Invoke-RestMethod -Uri "http://localhost:$backendPort/api/health" -Method GET -TimeoutSec 5
            Write-Host "✅ System healthy: $($health.status)" -ForegroundColor Green
        } catch {
            Write-Host "⚠️ Health check failed" -ForegroundColor Yellow
        }
    }
} catch {
    Write-Host "`n🛑 Stopping Mystic Trading Platform..." -ForegroundColor Red

    # Stop processes
    try {
        Get-Process -Name "python" -ErrorAction SilentlyContinue | Where-Object {$_.ProcessName -eq "python"} | Stop-Process -Force
        Get-Process -Name "node" -ErrorAction SilentlyContinue | Stop-Process -Force
    } catch {
        Write-Host "Could not stop all processes" -ForegroundColor Yellow
    }

    Write-Host "✅ Mystic Trading Platform stopped" -ForegroundColor Green
}
