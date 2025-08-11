# Mystic Trading Platform - Development Setup Script
param(
    [switch]$SkipPython,
    [switch]$SkipNode,
    [switch]$SkipRedis,
    [switch]$SkipEnv,
    [switch]$Force
)

Write-Host "🚀 Mystic Trading Platform - Development Setup" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green

# Function to check if command exists
function Test-Command($cmdname) {
    return [bool](Get-Command -Name $cmdname -ErrorAction SilentlyContinue)
}

# Function to initialize Python environment
function Initialize-PythonEnvironment {
    Write-Host "🐍 Setting up Python environment..." -ForegroundColor Yellow
    
    if (Test-Path "venv") {
        if ($Force) {
            Write-Host "Removing existing virtual environment..." -ForegroundColor Yellow
            Remove-Item -Recurse -Force "venv"
        } else {
            Write-Host "Virtual environment already exists. Use -Force to recreate." -ForegroundColor Yellow
            return
        }
    }
    
    # Create virtual environment
    Write-Host "Creating Python virtual environment..." -ForegroundColor Yellow
    python -m venv venv
    
    # Activate virtual environment
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    & "venv\Scripts\Activate.ps1"
    
    # Upgrade pip
    Write-Host "Upgrading pip..." -ForegroundColor Yellow
    python -m pip install --upgrade pip
    
    # Install requirements
    Write-Host "Installing Python dependencies..." -ForegroundColor Yellow
    pip install -r requirements.txt
    
    # Install development dependencies
    Write-Host "Installing development dependencies..." -ForegroundColor Yellow
    pip install -r backend/requirements-dev.txt
    
    Write-Host "✅ Python environment setup complete!" -ForegroundColor Green
}

# Function to initialize Node.js environment
function Initialize-NodeEnvironment {
    Write-Host "📦 Setting up Node.js environment..." -ForegroundColor Yellow
    
    if (-not (Test-Command "node")) {
        Write-Host "❌ Node.js not found. Please install Node.js first." -ForegroundColor Red
        Write-Host "Download from: https://nodejs.org/" -ForegroundColor Yellow
        return
    }
    
    if (-not (Test-Command "npm")) {
        Write-Host "❌ npm not found. Please install npm first." -ForegroundColor Red
        return
    }
    
    # Check if frontend directory exists
    if (-not (Test-Path "frontend")) {
        Write-Host "❌ Frontend directory not found." -ForegroundColor Red
        return
    }
    
    # Navigate to frontend and install dependencies
    Push-Location "frontend"
    
    if (Test-Path "node_modules") {
        if ($Force) {
            Write-Host "Removing existing node_modules..." -ForegroundColor Yellow
            Remove-Item -Recurse -Force "node_modules"
        } else {
            Write-Host "node_modules already exists. Use -Force to reinstall." -ForegroundColor Yellow
            Pop-Location
            return
        }
    }
    
    Write-Host "Installing Node.js dependencies..." -ForegroundColor Yellow
    npm install
    
    Pop-Location
    Write-Host "✅ Node.js environment setup complete!" -ForegroundColor Green
}

# Function to initialize Redis
function Initialize-Redis {
    Write-Host "🔴 Setting up Redis..." -ForegroundColor Yellow
    
    if (Test-Path "redis-server") {
        Write-Host "Redis server directory found." -ForegroundColor Yellow
    } else {
        Write-Host "❌ Redis server directory not found." -ForegroundColor Red
        Write-Host "Please ensure Redis is installed and configured." -ForegroundColor Yellow
        return
    }
    
    Write-Host "✅ Redis setup complete!" -ForegroundColor Green
}

# Function to initialize environment file
function Initialize-EnvironmentFile {
    Write-Host "⚙️ Setting up environment configuration..." -ForegroundColor Yellow
    
    if (Test-Path ".env") {
        if ($Force) {
            Write-Host "Removing existing .env file..." -ForegroundColor Yellow
            Remove-Item ".env" -Force
        } else {
            Write-Host ".env file already exists. Use -Force to recreate." -ForegroundColor Yellow
            return
        }
    }
    
    # Create .env file with template
    $envContent = @"
# Mystic Trading Platform - Environment Configuration

# Database Configuration
DATABASE_URL=sqlite:///./mystic_trading.db

# Redis Configuration
REDIS_URL=redis://localhost:6379

# API Keys (Replace with your actual keys)
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_SECRET_KEY=your_binance_secret_key_here
COINBASE_API_KEY=your_coinbase_api_key_here
COINBASE_SECRET_KEY=your_coinbase_secret_key_here

# AI Services
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Security
SECRET_KEY=your_secret_key_here
JWT_SECRET=your_jwt_secret_here

# Application Settings
DEBUG=true
ENVIRONMENT=development
LOG_LEVEL=INFO
"@
    
    $envContent | Out-File -FilePath ".env" -Encoding UTF8
    Write-Host "✅ Environment file created!" -ForegroundColor Green
    Write-Host "⚠️  Please update .env with your actual API keys!" -ForegroundColor Yellow
}

# Function to run quality checks
function Test-QualityChecks {
    Write-Host "🔍 Running quality checks..." -ForegroundColor Yellow
    
    if (Test-Path "backend/run_quality_checks.py") {
        Push-Location "backend"
        python run_quality_checks.py
        Pop-Location
    } else {
        Write-Host "Quality checks script not found." -ForegroundColor Yellow
    }
}

# Main setup process
# Check Python
if (-not $SkipPython) {
    if (-not (Test-Command "python")) {
        Write-Host "❌ Python not found. Please install Python 3.10 first." -ForegroundColor Red
        Write-Host "Download from: https://www.python.org/downloads/" -ForegroundColor Yellow
        exit 1
    }
    
    $pythonVersion = python --version
    Write-Host "Found: $pythonVersion" -ForegroundColor Green
    
    Initialize-PythonEnvironment
    
    if (-not $SkipEnv) {
        Initialize-EnvironmentFile
    } else {
        Write-Host "⏭️  Skipping environment file creation (using existing .env)" -ForegroundColor Yellow
    }
}

# Check Node.js
if (-not $SkipNode) {
    Initialize-NodeEnvironment
}

# Check Redis
if (-not $SkipRedis) {
    Initialize-Redis
}

# Run quality checks
Test-QualityChecks

Write-Host ""
Write-Host "🎉 Setup complete!" -ForegroundColor Green
Write-Host "==================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Your .env file is already configured" -ForegroundColor White
Write-Host "2. Start Redis server: scripts/start-redis.bat" -ForegroundColor White
Write-Host "3. Start backend: scripts/start-backend.bat" -ForegroundColor White
Write-Host "4. Start frontend: scripts/start-frontend.bat" -ForegroundColor White
Write-Host "5. Or start all: scripts/start-all.bat" -ForegroundColor White
Write-Host ""
Write-Host "Documentation: docs/" -ForegroundColor Cyan
Write-Host "Scripts: scripts/" -ForegroundColor Cyan 