# Test Live Data Connections for Mystic Trading Platform
# Verifies all live data endpoints are working correctly

Write-Host "🔍 Testing Live Data Connections..." -ForegroundColor Blue

$backendPort = 8000
$baseUrl = "http://localhost:$backendPort"

# Function to test endpoint
function Test-Endpoint {
    param(
        [string]$Endpoint,
        [string]$Description
    )

    try {
        $response = Invoke-RestMethod -Uri "$baseUrl$Endpoint" -Method GET -TimeoutSec 10
        $status = if ($response.status) { $response.status } else { "OK" }
        Write-Host "✅ $Description - Working (Status: $status, Response: $($response | ConvertTo-Json -Compress))" -ForegroundColor Green
        return $true
    } catch {
        Write-Host "❌ $Description - Failed: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

# Test basic health
Write-Host "`n🏥 Testing Basic Health..." -ForegroundColor Yellow
Test-Endpoint -Endpoint "/api/health" -Description "Health Check"
Test-Endpoint -Endpoint "/api/version" -Description "Version Info"

# Test market data endpoints
Write-Host "`n📊 Testing Market Data..." -ForegroundColor Yellow
Test-Endpoint -Endpoint "/api/coinstate" -Description "Coin State"
Test-Endpoint -Endpoint "/api/live/market-data" -Description "Live Market Data"
Test-Endpoint -Endpoint "/api/live/exchange-status" -Description "Exchange Status"

# Test specific coin data
Write-Host "`n🪙 Testing Specific Coin Data..." -ForegroundColor Yellow
Test-Endpoint -Endpoint "/api/live/market-data/BTC" -Description "Bitcoin Data"
Test-Endpoint -Endpoint "/api/live/price-comparison/BTC" -Description "Price Comparison"

# Test global data
Write-Host "`n🌍 Testing Global Data..." -ForegroundColor Yellow
Test-Endpoint -Endpoint "/api/live/global" -Description "Global Market Data"

# Test WebSocket connection
Write-Host "`n🔌 Testing WebSocket..." -ForegroundColor Yellow
try {
    $ws = New-Object System.Net.WebSockets.ClientWebSocket
    $cancellationToken = New-Object System.Threading.CancellationToken
    $task = $ws.ConnectAsync("ws://localhost:$backendPort/ws/feed", $cancellationToken)
    $task.Wait(5000)

    if ($ws.State -eq "Open") {
        Write-Host "✅ WebSocket Connection - Working" -ForegroundColor Green
        $ws.CloseAsync([System.Net.WebSockets.WebSocketCloseStatus]::NormalClosure, "Test complete", $cancellationToken)
    } else {
        Write-Host "❌ WebSocket Connection - Failed: $($ws.State)" -ForegroundColor Red
    }
} catch {
    Write-Host "❌ WebSocket Connection - Failed: $($_.Exception.Message)" -ForegroundColor Red
}

# Test modular API endpoints
Write-Host "`n🔧 Testing Modular API..." -ForegroundColor Yellow
Test-Endpoint -Endpoint "/api/market-data" -Description "Modular Market Data"
Test-Endpoint -Endpoint "/api/signals" -Description "Trading Signals"
Test-Endpoint -Endpoint "/api/portfolio" -Description "Portfolio Data"

Write-Host "`n🎯 Live Data Connection Test Complete!" -ForegroundColor Green
Write-Host "💡 All endpoints should return live data from Binance, Coinbase, and CoinGecko" -ForegroundColor Cyan
