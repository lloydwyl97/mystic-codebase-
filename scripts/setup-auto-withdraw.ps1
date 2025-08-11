# setup-auto-withdraw.ps1 – Auto-Withdraw System Setup for Windows 11 Home
# Configure and deploy the Mystic Auto-Withdraw system

Write-Host "AUTO-WITHDRAW SYSTEM SETUP" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Check if we're in the right directory
if (-not (Test-Path "backend/auto_withdraw.py")) {
    Write-Host "Auto-withdraw system not found!" -ForegroundColor Red
    Write-Host "   Please run this script from the Mystic-Codebase root directory" -ForegroundColor Yellow
    exit 1
}

# Check if .env file exists
if (-not (Test-Path "backend/.env")) {
    Write-Host "Environment file not found! Running setup-env.ps1 first..." -ForegroundColor Yellow
    .\setup-env.ps1
}

Write-Host "Configuring auto-withdraw system..." -ForegroundColor Yellow

# Read current .env file
$envContent = Get-Content "backend/.env" -Raw

# Add auto-withdraw specific configuration if not present
$autoWithdrawConfig = @"

# ===== AUTO-WITHDRAW CONFIGURATION =====
EXCHANGE=binance
CHECK_INTERVAL=60
"@

if ($envContent -notmatch "AUTO-WITHDRAW CONFIGURATION") {
    $envContent += $autoWithdrawConfig
    $envContent | Out-File -FilePath "backend/.env" -Encoding UTF8
    Write-Host "Added auto-withdraw configuration to .env" -ForegroundColor Green
}

# Create auto-withdraw logs directory
if (-not (Test-Path "backend/logs")) {
    New-Item -ItemType Directory -Path "backend/logs" -Force | Out-Null
    Write-Host "Created logs directory" -ForegroundColor Green
}

Write-Host ""
Write-Host "AUTO-WITHDRAW CONFIGURATION COMPLETE!" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green
Write-Host ""
Write-Host "NEXT STEPS:" -ForegroundColor Cyan
Write-Host "   1. Edit backend/.env and configure:" -ForegroundColor White
Write-Host "      - EXCHANGE=binance (or coinbase)" -ForegroundColor White
Write-Host "      - COLD_WALLET_ADDRESS=your_cold_wallet_address" -ForegroundColor White
Write-Host "      - COLD_WALLET_THRESHOLD=250.00" -ForegroundColor White
Write-Host "      - BINANCE_API_KEY and BINANCE_API_SECRET" -ForegroundColor White
Write-Host "      - COINBASE_API_KEY, COINBASE_API_SECRET, COINBASE_PASSPHRASE" -ForegroundColor White
Write-Host "      - DISCORD_WEBHOOK and/or TELEGRAM_TOKEN/TELEGRAM_CHAT_ID" -ForegroundColor White
Write-Host ""
Write-Host "   2. Run the auto-withdraw system:" -ForegroundColor White
Write-Host "      - Local: cd backend && python auto_withdraw.py" -ForegroundColor White
Write-Host "      - Docker: .\run-auto-withdraw.ps1" -ForegroundColor White
Write-Host ""
Write-Host "IMPORTANT:" -ForegroundColor Yellow
Write-Host "   - Ensure your API keys have withdrawal permissions" -ForegroundColor White
Write-Host "   - Test with small amounts first" -ForegroundColor White
Write-Host "   - Monitor the logs/auto_withdraw.log file" -ForegroundColor White
Write-Host ""
Write-Host "Auto-withdraw system is ready for configuration!" -ForegroundColor Magenta
