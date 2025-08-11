# Check missing services from logs
Write-Host "=== Missing Services Analysis ===" -ForegroundColor Green

$missingServices = @(
    "cosmic_fetcher",
    "trade_engine", 
    "unified_signal_manager",
    "shared_cache",
    "auto_trading_manager",
    "signal_manager",
    "social_trading_manager",
    "connection_manager",
    "notification_service",
    "health_monitor",
    "metrics_collector",
    "audit_logger",
    "enhanced_logger",
    "signature_manager",
    "advanced_trading",
    "ai_strategies"
)

Write-Host "`nMissing Services Found in Logs:" -ForegroundColor Yellow
foreach ($service in $missingServices) {
    $filePath = "backend/services/$service.py"
    if (Test-Path $filePath) {
        Write-Host "✅ $service`: EXISTS" -ForegroundColor Green
    } else {
        Write-Host "❌ $service`: MISSING" -ForegroundColor Red
    }
}

Write-Host "`n=== Service Check Complete ===" -ForegroundColor Green 