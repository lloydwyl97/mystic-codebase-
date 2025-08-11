# PowerShell script to run comprehensive code quality checks
# Mystic Trading Platform - Professional Code Quality System

Write-Host "🚀 Starting Comprehensive Code Quality Checks..." -ForegroundColor Green
Write-Host "📁 Working directory: $(Get-Location)" -ForegroundColor Cyan

# Check if we're in the right directory
if (-not (Test-Path "requirements-dev.txt")) {
    Write-Host "❌ Error: requirements-dev.txt not found. Please run this script from the backend directory." -ForegroundColor Red
    exit 1
}

# Activate virtual environment if it exists
if (Test-Path "venv\Scripts\Activate.ps1") {
    Write-Host "🔧 Activating virtual environment..." -ForegroundColor Yellow
    & "venv\Scripts\Activate.ps1"
}

# Run the Python quality checker
Write-Host "🐍 Running Python quality checks..." -ForegroundColor Yellow
python run_quality_checks.py

# Check exit code
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ All quality checks completed successfully!" -ForegroundColor Green
} else {
    Write-Host "❌ Some quality checks failed. Please review the report." -ForegroundColor Red
}

exit $LASTEXITCODE
