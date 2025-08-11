# Setup Python 3.11 Environment for Mystic Trading Platform
# This script helps you set up Python 3.11 for full compatibility

Write-Host "Setting up Python 3.11 for Mystic Trading Platform..." -ForegroundColor Green

# Check if Python 3.11 is installed
Write-Host "Checking for Python 3.11..." -ForegroundColor Yellow
$python311 = Get-Command python3.11 -ErrorAction SilentlyContinue
$python311Alt = Get-Command py -ErrorAction SilentlyContinue

if ($python311) {
    Write-Host "Python 3.11 found at: $($python311.Source)" -ForegroundColor Green
    $pythonCmd = "python3.11"
} elseif ($python311Alt) {
    Write-Host "Python launcher found, checking for Python 3.11..." -ForegroundColor Yellow
    $version = & py -3.11 --version 2>$null
    if ($version) {
        Write-Host "Python 3.11 found via py launcher" -ForegroundColor Green
        $pythonCmd = "py -3.11"
    } else {
        Write-Host "Python 3.11 not found via py launcher" -ForegroundColor Red
    }
} else {
    Write-Host "Python 3.11 not found in PATH" -ForegroundColor Red
}

if (-not $pythonCmd) {
    Write-Host "`nPython 3.11 is not installed. Please install it first:" -ForegroundColor Red
    Write-Host "1. Download Python 3.11 from: https://www.python.org/downloads/release/python-3119/" -ForegroundColor Yellow
    Write-Host "2. Run the installer and make sure to check 'Add Python to PATH'" -ForegroundColor Yellow
    Write-Host "3. Restart PowerShell and run this script again" -ForegroundColor Yellow
    exit 1
}

# Create new virtual environment with Python 3.11
Write-Host "`nCreating new virtual environment with Python 3.11..." -ForegroundColor Cyan
if (Test-Path "venv311") {
    Write-Host "Removing existing venv311 directory..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force "venv311"
}

& $pythonCmd -m venv venv311

if (Test-Path "venv311\Scripts\Activate.ps1") {
    Write-Host "Activating Python 3.11 virtual environment..." -ForegroundColor Green
    & "venv311\Scripts\Activate.ps1"

    # Upgrade pip
    Write-Host "Upgrading pip..." -ForegroundColor Cyan
    python -m pip install --upgrade pip

    Write-Host "`nPython 3.11 environment is ready!" -ForegroundColor Green
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host "1. Run: .\install-requirements.ps1" -ForegroundColor Cyan
    Write-Host "2. All packages should install successfully" -ForegroundColor Cyan
    Write-Host "3. Your Mystic Trading Platform will have full functionality" -ForegroundColor Cyan

} else {
    Write-Host "Failed to create virtual environment" -ForegroundColor Red
    exit 1
}
