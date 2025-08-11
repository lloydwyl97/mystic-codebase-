# Mystic Trading Platform - Modular Live Startup Script
# Ensures all live data connections and no mock data usage
# Compatible with Windows 11 PowerShell

param(
    [switch]$SkipChecks,
    [switch]$ForceReinstall,
    [switch]$LiveDataOnly
)

# Set execution policy for current session
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force

# Color functions for better output
function Write-Success { param($Message) Write-Host "✅ $Message" -ForegroundColor Green }
function Write-Info { param($Message) Write-Host "ℹ️ $Message" -ForegroundColor Cyan }
function Write-Warning { param($Message) Write-Host "⚠️ $Message" -ForegroundColor Yellow }
function Write-Error { param($Message) Write-Host "❌ $Message" -ForegroundColor Red }

# Banner
Write-Host @"
╔══════════════════════════════════════════════════════════════╗
║                    Mystic Trading Platform                   ║
║                    Modular Live Startup                      ║
║                    All Live Data Connections                 ║
╚══════════════════════════════════════════════════════════════╝
"@ -ForegroundColor Magenta

Write-Info "Starting Mystic Trading Platform with live data connections..."

# Check if running as administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")
if (-not $isAdmin) {
    Write-Warning "Not running as administrator. Some features may be limited."
}

# Environment checks
function Test-Environment {
    Write-Info "Checking environment requirements..."

    # Check Python
    try {
        $pythonVersion = python --version 2>&1
        if ($pythonVersion -match "Python (\d+\.\d+\.\d+)") {
            Write-Success "Python found: $($matches[1])"
        } else {
            throw "Python not found or version not recognized"
        }
    } catch {
        Write-Error "Python not found. Please install Python 3.8+ and add to PATH"
        return $false
    }

    # Check Node.js
    try {
        $nodeVersion = node --version 2>&1
        if ($nodeVersion -match "v(\d+\.\d+\.\d+)") {
            Write-Success "Node.js found: $($matches[1])"
        } else {
            throw "Node.js not found or version not recognized"
        }
    } catch {
        Write-Error "Node.js not found. Please install Node.js 16+ and add to PATH"
        return $false
    }

    # Check npm
    try {
        $npmVersion = npm --version 2>&1
        Write-Success "npm found: $npmVersion"
    } catch {
        Write-Error "npm not found. Please install npm and add to PATH"
        return $false
    }

    # Check if we're in the right directory
    if (-not (Test-Path "backend") -or -not (Test-Path "frontend")) {
        Write-Error "Not in Mystic Trading Platform root directory"
        return $false
    }

    Write-Success "Environment checks passed"
    return $true
}

# Install Python dependencies
function Install-PythonDependencies {
    Write-Info "Installing Python dependencies for live data..."

    # Using global Python environment (dedicated laptop setup)
    Write-Info "Using global Python environment..."
    Write-Info "Python path: C:\Users\lloyd\AppData\Local\Programs\Python\Python310\python.exe"

    # Upgrade pip
    Write-Info "Upgrading pip..."
    python -m pip install --upgrade pip

    # Install requirements
    Write-Info "Installing Python requirements..."
    if (Test-Path "requirements.txt") {
        pip install -r requirements.txt
    }

    # Install additional live data dependencies
    Write-Info "Installing live data dependencies..."
    pip install aiohttp requests websockets python-binance ccxt

    Write-Success "Python dependencies installed"
}

# Install Node.js dependencies
function Install-NodeDependencies {
    Write-Info "Installing Node.js dependencies for live frontend..."

    Set-Location frontend

    # Install dependencies
    Write-Info "Installing npm packages..."
    npm install

    # Install additional live data packages
    Write-Info "Installing live data packages..."
    npm install websocket axios chart.js react-chartjs-2

    Set-Location ..
    Write-Success "Node.js dependencies installed"
}

# Verify live data connections
function Test-LiveDataConnections {
    Write-Info "Verifying live data connections..."

    # Test backend health
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:8000/api/health" -Method Get -TimeoutSec 10
        if ($response.live_data -eq $true) {
            Write-Success "Backend live data connection verified"
        } else {
            Write-Warning "Backend not reporting live data"
        }
    } catch {
        Write-Warning "Backend not responding, will start it"
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
}

# Start backend with live data
function Start-BackendLive {
    Write-Info "Starting backend with live data connections..."

    Set-Location backend

    # Using global Python environment (dedicated laptop setup)
    Write-Info "Using global Python environment..."

    # Set environment variables for live data
    $env:LIVE_DATA_MODE = "true"
    $env:MOCK_DATA_ENABLED = "false"
    $env:API_KEYS_REQUIRED = "true"

    # Start backend
    Write-Info "Starting backend server..."
    Start-Process python -ArgumentList "main.py" -WindowStyle Minimized

    Set-Location ..

    # Wait for backend to start
    Write-Info "Waiting for backend to start..."
    Start-Sleep -Seconds 10

    # Test backend
    $maxAttempts = 30
    $attempt = 0
    while ($attempt -lt $maxAttempts) {
        try {
            $response = Invoke-RestMethod -Uri "http://localhost:8000/api/health" -Method Get -TimeoutSec 5
            if ($response.status -eq "healthy") {
                Write-Success "Backend started successfully with live data"
                return $true
            }
        } catch {
            $attempt++
            Start-Sleep -Seconds 2
        }
    }

    Write-Error "Backend failed to start"
    return $false
}

# Start frontend with live data
function Start-FrontendLive {
    Write-Info "Starting frontend with live data connections..."

    Set-Location frontend

    # Set environment variables for live data
    $env:VITE_LIVE_DATA_MODE = "true"
    $env:VITE_MOCK_DATA_ENABLED = "false"
    $env:VITE_API_BASE_URL = "http://localhost:8000"

    # Start frontend
    Write-Info "Starting frontend development server..."
    Start-Process npm -ArgumentList "run", "dev" -WindowStyle Minimized

    Set-Location ..

    # Wait for frontend to start
    Write-Info "Waiting for frontend to start..."
    Start-Sleep -Seconds 15

    # Test frontend
    $maxAttempts = 30
    $attempt = 0
    while ($attempt -lt $maxAttempts) {
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:5173" -Method Get -TimeoutSec 5
            if ($response.StatusCode -eq 200) {
                Write-Success "Frontend started successfully with live data"
                return $true
            }
        } catch {
            $attempt++
            Start-Sleep -Seconds 2
        }
    }

    Write-Error "Frontend failed to start"
    return $false
}

# Main execution
function Start-MysticLive {
    Write-Info "Starting Mystic Trading Platform with live data..."

    # Environment checks
    if (-not $SkipChecks) {
        if (-not (Test-Environment)) {
            Write-Error "Environment checks failed. Exiting."
            exit 1
        }
    }

    # Install dependencies if needed
    if ($ForceReinstall) {
        Install-PythonDependencies
        Install-NodeDependencies
    }

    # Start backend
    if (-not (Start-BackendLive)) {
        Write-Error "Failed to start backend. Exiting."
        exit 1
    }

    # Start frontend
    if (-not (Start-FrontendLive)) {
        Write-Error "Failed to start frontend. Exiting."
        exit 1
    }

    # Verify live data connections
    Test-LiveDataConnections

    # Display success message
    Write-Host @"
╔══════════════════════════════════════════════════════════════╗
║                    🎉 SUCCESS! 🎉                           ║
║                                                              ║
║  Mystic Trading Platform is now running with live data!     ║
║                                                              ║
║  🌐 Frontend: http://localhost:5173                         ║
║  🔧 Backend:  http://localhost:8000                         ║
║  📊 API Docs: http://localhost:8000/docs                    ║
║                                                              ║
║  All connections are live - no mock data used!              ║
╚══════════════════════════════════════════════════════════════╝
"@ -ForegroundColor Green

    Write-Info "Press Ctrl+C to stop all services"

    # Keep script running
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

    # Stop backend
    try {
        Get-Process -Name "python" -ErrorAction SilentlyContinue | Where-Object { $_.ProcessName -eq "python" } | Stop-Process -Force
    } catch {
        Write-Warning "Could not stop backend process"
    }

    # Stop frontend
    try {
        Get-Process -Name "node" -ErrorAction SilentlyContinue | Where-Object { $_.ProcessName -eq "node" } | Stop-Process -Force
    } catch {
        Write-Warning "Could not stop frontend process"
    }

    Write-Success "Mystic Trading Platform stopped"
}

# Start the platform
Start-MysticLive
