# Setup Python 3.10 Environment for Mystic Trading Platform
# This script helps you set up Python 3.10 for full compatibility

Write-Host "Setting up Python 3.10 for Mystic Trading Platform..." -ForegroundColor Green

# Check if Python 3.10 is installed
Write-Host "Checking for Python 3.10..." -ForegroundColor Yellow
$python310 = Get-Command python3.10 -ErrorAction SilentlyContinue
$python310Alt = Get-Command py -ErrorAction SilentlyContinue

if ($python310) {
    Write-Host "Python 3.10 found at: $($python310.Source)" -ForegroundColor Green
    $pythonCmd = "python3.10"
} elseif ($python310Alt) {
    Write-Host "Python launcher found, checking for Python 3.10..." -ForegroundColor Yellow
    $version = & py -3.10 --version 2>$null
    if ($version) {
        Write-Host "Python 3.10 found via py launcher" -ForegroundColor Green
        $pythonCmd = "py -3.10"
    } else {
        Write-Host "Python 3.10 not found via py launcher" -ForegroundColor Red
    }
} else {
    Write-Host "Python 3.10 not found in PATH" -ForegroundColor Red
}

if (-not $pythonCmd) {
    Write-Host "`nPython 3.10 is not installed. Please install it first:" -ForegroundColor Red
    Write-Host "1. Download Python 3.10 from: https://www.python.org/downloads/release/python-31011/" -ForegroundColor Yellow
    Write-Host "2. Run the installer and make sure to check 'Add Python to PATH'" -ForegroundColor Yellow
    Write-Host "3. Restart PowerShell and run this script again" -ForegroundColor Yellow
    exit 1
}

# Create new virtual environment with Python 3.10
Write-Host "`nCreating new virtual environment with Python 3.10..." -ForegroundColor Cyan
if (Test-Path "venv310") {
    Write-Host "Removing existing venv310 directory..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force "venv310"
}

& $pythonCmd -m venv venv310

if (Test-Path "venv310\Scripts\Activate.ps1") {
    Write-Host "Activating Python 3.10 virtual environment..." -ForegroundColor Green
    & "venv310\Scripts\Activate.ps1"

    # Upgrade pip
    Write-Host "Upgrading pip..." -ForegroundColor Cyan
    python -m pip install --upgrade pip

    Write-Host "`nPython 3.10 environment is ready!" -ForegroundColor Green
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host "1. Run: .\scripts\install-requirements.ps1" -ForegroundColor Cyan
    Write-Host "2. All packages should install successfully" -ForegroundColor Cyan
    Write-Host "3. Your Mystic Trading Platform will have full functionality" -ForegroundColor Cyan

} else {
    Write-Host "Failed to create virtual environment" -ForegroundColor Red
    exit 1
} 