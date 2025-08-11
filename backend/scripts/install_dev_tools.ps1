# Install Development Tools for Mystic Trading Platform
# Run this script to install all code quality tools

Write-Host "🔧 Installing Development Tools for Code Quality" -ForegroundColor Green
Write-Host "=" * 60

# Check if Python is available
try {
    $pythonVersion = python --version
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found. Please install Python 3.10+ first." -ForegroundColor Red
    exit 1
}

# Install development requirements
Write-Host "📦 Installing development dependencies..." -ForegroundColor Yellow
try {
    pip install -r requirements-dev.txt
    Write-Host "✅ Development dependencies installed successfully" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to install development dependencies" -ForegroundColor Red
    Write-Host "Trying individual installations..." -ForegroundColor Yellow

    # Install tools individually
    $tools = @(
        "flake8==6.1.0",
        "black==23.12.1",
        "isort==5.13.2",
        "mypy==1.8.0",
        "bandit==1.7.5",
        "pytest==7.4.4",
        "pytest-cov==4.1.0",
        "radon==6.0.1",
        "vulture==2.10",
        "pre-commit==3.6.0"
    )

    foreach ($tool in $tools) {
        Write-Host "Installing $tool..." -ForegroundColor Yellow
        pip install $tool
    }
}

# Install pre-commit hooks
Write-Host "🔗 Installing pre-commit hooks..." -ForegroundColor Yellow
try {
    pre-commit install
    Write-Host "✅ Pre-commit hooks installed successfully" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Pre-commit installation failed (optional)" -ForegroundColor Yellow
}

# Create necessary directories
Write-Host "📁 Creating necessary directories..." -ForegroundColor Yellow
$dirs = @("logs", "reports", "coverage")
foreach ($dir in $dirs) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir
        Write-Host "  Created: $dir" -ForegroundColor Green
    }
}

# Test installations
Write-Host "🧪 Testing installations..." -ForegroundColor Yellow
$tools = @(
    @{Name="flake8"; Command="flake8 --version"},
    @{Name="black"; Command="black --version"},
    @{Name="isort"; Command="isort --version"},
    @{Name="mypy"; Command="mypy --version"},
    @{Name="bandit"; Command="bandit --version"},
    @{Name="pytest"; Command="pytest --version"},
    @{Name="radon"; Command="radon --version"},
    @{Name="vulture"; Command="vulture --version"}
)

foreach ($tool in $tools) {
    try {
        $output = Invoke-Expression $tool.Command 2>&1
        Write-Host "  ✅ $($tool.Name): Working" -ForegroundColor Green
    } catch {
        Write-Host "  ❌ $($tool.Name): Failed" -ForegroundColor Red
    }
}

Write-Host "`n🎉 Development tools installation complete!" -ForegroundColor Green
Write-Host "`n📋 Available commands:" -ForegroundColor Cyan
Write-Host "  python scripts/code_quality.py    - Run all quality checks" -ForegroundColor White
Write-Host "  black .                           - Format code" -ForegroundColor White
Write-Host "  isort .                          - Sort imports" -ForegroundColor White
Write-Host "  flake8 .                         - Check code style" -ForegroundColor White
Write-Host "  mypy .                           - Check types" -ForegroundColor White
Write-Host "  bandit -r .                      - Security scan" -ForegroundColor White
Write-Host "  pytest                           - Run tests" -ForegroundColor White
Write-Host "  pre-commit run --all-files       - Run all pre-commit hooks" -ForegroundColor White
