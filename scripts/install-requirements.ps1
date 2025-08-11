# Simple Requirements Installation Script
# This script installs packages from requirements.txt files

Write-Host "Installing packages from requirements files..." -ForegroundColor Green

# Using global Python environment (dedicated laptop setup)
Write-Host "Using global Python environment..." -ForegroundColor Yellow
Write-Host "Python path: C:\Users\lloyd\AppData\Local\Programs\Python\Python310\python.exe" -ForegroundColor Cyan

# Install from main requirements.txt
Write-Host "Installing from main requirements.txt..." -ForegroundColor Cyan
pip install -r requirements.txt

# Install from backend requirements.txt
Write-Host "Installing from backend requirements.txt..." -ForegroundColor Cyan
pip install -r backend/requirements.txt

Write-Host "Requirements installation completed!" -ForegroundColor Green
Write-Host "Checking for any installation errors..." -ForegroundColor Yellow

# Check for any failed installations
pip check

Write-Host "Installation process completed!" -ForegroundColor Green
Write-Host "Your Mystic Trading Platform should now have all features available!" -ForegroundColor Green
