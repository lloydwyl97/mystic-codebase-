# Binance US Autobuy System Startup Script
Write-Host "Starting Binance US Autobuy System..." -ForegroundColor Green
Write-Host ""
Write-Host "Trading Pairs: SOLUSDT, BTCUSDT, ETHUSDT, AVAXUSDT" -ForegroundColor Yellow
Write-Host "Dashboard: http://localhost:8080" -ForegroundColor Yellow
Write-Host ""

try {
    python launch_autobuy.py
} catch {
    Write-Host "Error starting system: $_" -ForegroundColor Red
}

Read-Host "Press Enter to exit"
