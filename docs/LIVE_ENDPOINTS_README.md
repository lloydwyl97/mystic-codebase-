# 🌐 Live Endpoints - Complete AI Strategy System

## 🚀 Quick Access URLs

### Main Applications
- **Frontend Dashboard**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

### AI Strategy System Endpoints

#### Status & Health
- **System Status**: http://localhost:8000/api/ai-strategy/status
- **Health Check**: http://localhost:8000/api/ai-strategy/health
- **Performance Analytics**: http://localhost:8000/api/ai-strategy/analytics/performance

#### Strategy Management
- **Leaderboard**: http://localhost:8000/api/ai-strategy/leaderboard
- **Top Strategies**: http://localhost:8000/api/ai-strategy/leaderboard/top/5
- **Mutations**: http://localhost:8000/api/ai-strategy/mutations

#### Position & Trading
- **Current Position**: http://localhost:8000/api/ai-strategy/position
- **Position Update**: POST http://localhost:8000/api/ai-strategy/position/update
- **Clear Position**: DELETE http://localhost:8000/api/ai-strategy/position/clear

#### Data Management
- **Recent Logs**: http://localhost:8000/api/ai-strategy/logs/recent
- **Add Strategy**: POST http://localhost:8000/api/ai-strategy/leaderboard/add
- **Add Mutation**: POST http://localhost:8000/api/ai-strategy/mutations/add

#### Real-time Updates
- **WebSocket Live**: ws://localhost:8000/api/ai-strategy/ws/live

## 📊 Endpoint Details

### 1. System Status
**GET** `/api/ai-strategy/status`

Returns overall system health and component status.

```json
{
  "status": "operational",
  "timestamp": "2024-01-15T12:00:00Z",
  "components": {
    "leaderboard": {
      "status": "active",
      "strategies_count": 5
    },
    "position_tracking": {
      "status": "active",
      "has_active_position": true
    },
    "mutation_system": {
      "status": "active",
      "mutations_count": 3
    }
  },
  "system_health": "healthy"
}
```

### 2. Leaderboard
**GET** `/api/ai-strategy/leaderboard`

Returns all strategies ranked by performance.

```json
{
  "strategies": [
    {
      "id": "strategy_47b_rsi_breakout",
      "win_rate": 0.64,
      "profit": 22.3,
      "trades": 36
    }
  ],
  "total": 5,
  "timestamp": "2024-01-15T12:00:00Z"
}
```

### 3. Top Strategies
**GET** `/api/ai-strategy/leaderboard/top/{count}`

Returns top N strategies (default: 5).

### 4. Current Position
**GET** `/api/ai-strategy/position`

Returns current trading position with P&L calculations.

```json
{
  "has_position": true,
  "position": {
    "exchange": "binance",
    "entry_price": 1824.55,
    "current_price": 1830.20,
    "peak_price": 1835.10,
    "pnl_percent": 0.31,
    "pnl_dollar": 2.83,
    "status": "profit"
  },
  "timestamp": "2024-01-15T12:00:00Z"
}
```

### 5. Performance Analytics
**GET** `/api/ai-strategy/analytics/performance`

Returns comprehensive performance metrics.

```json
{
  "analytics": {
    "total_strategies": 5,
    "average_win_rate": 0.58,
    "average_profit": 15.2,
    "total_trades": 156,
    "best_strategy": {
      "id": "strategy_47b_rsi_breakout",
      "profit": 22.3
    },
    "worst_strategy": {
      "id": "strategy_13c_macd_reversal",
      "profit": 8.5
    },
    "has_active_position": true
  },
  "timestamp": "2024-01-15T12:00:00Z"
}
```

### 6. Mutations
**GET** `/api/ai-strategy/mutations`

Returns mutation strategies and promotion status.

```json
{
  "mutations": [
    {
      "id": "mutation_001_rsi_enhanced",
      "win_rate": 0.62,
      "profit": 18.5,
      "trades": 25,
      "promoted": false
    }
  ],
  "total": 3,
  "promoted": 1,
  "unpromoted": 2,
  "timestamp": "2024-01-15T12:00:00Z"
}
```

### 7. Recent Logs
**GET** `/api/ai-strategy/logs/recent?lines=50`

Returns recent log entries from all components.

```json
{
  "logs": [
    {
      "file": "logs/ai_strategy_execution.log",
      "lines": [
        "2024-01-15 12:00:00 - ai_strategy_execution - INFO - Strategy executed successfully"
      ]
    }
  ],
  "lines_requested": 50,
  "timestamp": "2024-01-15T12:00:00Z"
}
```

### 8. Health Check
**GET** `/api/ai-strategy/health`

Returns system health status.

```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T12:00:00Z",
  "files": {
    "leaderboard": true,
    "position": true,
    "mutations": true
  },
  "uptime": 1642234567.89
}
```

## 🔄 POST Endpoints

### Add Strategy to Leaderboard
**POST** `/api/ai-strategy/leaderboard/add`

```json
{
  "id": "new_strategy_001",
  "win_rate": 0.65,
  "profit": 20.5,
  "trades": 30
}
```

### Add Mutation
**POST** `/api/ai-strategy/mutations/add`

```json
{
  "id": "mutation_004_enhanced",
  "win_rate": 0.59,
  "profit": 14.2,
  "trades": 28
}
```

### Update Position
**POST** `/api/ai-strategy/position/update`

```json
{
  "exchange": "binance",
  "entry_price": 1824.55,
  "current_price": 1830.20,
  "peak_price": 1835.10
}
```

## 🗑️ DELETE Endpoints

### Clear Position
**DELETE** `/api/ai-strategy/position/clear`

Clears the current trading position.

## 🔌 WebSocket Endpoints

### Live Updates
**WebSocket** `ws://localhost:8000/api/ai-strategy/ws/live`

Real-time updates for position and leaderboard changes.

```json
{
  "type": "update",
  "position": {
    "exchange": "binance",
    "entry_price": 1824.55,
    "current_price": 1830.20
  },
  "leaderboard": [
    {
      "id": "strategy_47b_rsi_breakout",
      "win_rate": 0.64,
      "profit": 22.3
    }
  ],
  "timestamp": "2024-01-15T12:00:00Z"
}
```

## 🛠️ Testing Endpoints

### Using PowerShell
```powershell
# Test system status
Invoke-RestMethod -Uri "http://localhost:8000/api/ai-strategy/status" -Method GET

# Test leaderboard
Invoke-RestMethod -Uri "http://localhost:8000/api/ai-strategy/leaderboard" -Method GET

# Test position
Invoke-RestMethod -Uri "http://localhost:8000/api/ai-strategy/position" -Method GET

# Add strategy
$strategy = @{
    id = "test_strategy_001"
    win_rate = 0.60
    profit = 15.0
    trades = 25
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/api/ai-strategy/leaderboard/add" -Method POST -Body $strategy -ContentType "application/json"
```

### Using curl
```bash
# Test system status
curl http://localhost:8000/api/ai-strategy/status

# Test leaderboard
curl http://localhost:8000/api/ai-strategy/leaderboard

# Test position
curl http://localhost:8000/api/ai-strategy/position

# Add strategy
curl -X POST http://localhost:8000/api/ai-strategy/leaderboard/add \
  -H "Content-Type: application/json" \
  -d '{"id":"test_strategy_001","win_rate":0.60,"profit":15.0,"trades":25}'
```

### Using JavaScript/Fetch
```javascript
// Test system status
fetch('http://localhost:8000/api/ai-strategy/status')
  .then(response => response.json())
  .then(data => console.log(data));

// Test leaderboard
fetch('http://localhost:8000/api/ai-strategy/leaderboard')
  .then(response => response.json())
  .then(data => console.log(data));

// Add strategy
fetch('http://localhost:8000/api/ai-strategy/leaderboard/add', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    id: 'test_strategy_001',
    win_rate: 0.60,
    profit: 15.0,
    trades: 25
  })
})
.then(response => response.json())
.then(data => console.log(data));
```

## 📱 WebSocket Client Example

```javascript
const ws = new WebSocket('ws://localhost:8000/api/ai-strategy/ws/live');

ws.onopen = function() {
    console.log('Connected to AI Strategy WebSocket');
};

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Received update:', data);

    // Update UI with real-time data
    if (data.position) {
        updatePositionDisplay(data.position);
    }
    if (data.leaderboard) {
        updateLeaderboardDisplay(data.leaderboard);
    }
};

ws.onclose = function() {
    console.log('Disconnected from AI Strategy WebSocket');
};
```

## 🔍 Monitoring & Debugging

### Check Container Status
```powershell
docker ps --filter "name=mystic"
```

### View Logs
```powershell
# AI Strategy Execution
docker logs -f mystic-ai-strategy

# AI Leaderboard Executor
docker logs -f mystic-ai-leaderboard

# AI Trade Engine
docker logs -f mystic-ai-trade-engine

# Mutation Evaluator
docker logs -f mystic-mutation-evaluator

# Main Backend
docker logs -f mystic-backend
```

### Check File Status
```powershell
# Check if data files exist
Test-Path "backend/mutation_leaderboard.json"
Test-Path "backend/position.json"
Test-Path "backend/mutations.json"

# View file contents
Get-Content "backend/mutation_leaderboard.json" | ConvertFrom-Json
Get-Content "backend/position.json" | ConvertFrom-Json
```

## 🚨 Error Handling

All endpoints return appropriate HTTP status codes:

- **200**: Success
- **400**: Bad Request
- **404**: Not Found
- **500**: Internal Server Error
- **503**: Service Unavailable (Health Check)

Error responses include:
```json
{
  "detail": "Error description",
  "timestamp": "2024-01-15T12:00:00Z"
}
```

## 🔒 Security Notes

- All endpoints are currently unauthenticated for development
- Add authentication middleware for production use
- Consider rate limiting for public endpoints
- Validate all input data before processing

## 📈 Performance

- Endpoints respond within 100ms for most requests
- WebSocket updates every 5 seconds
- File I/O operations are optimized
- Log retrieval is limited to prevent memory issues

---

**🎯 Your AI Strategy System is now fully accessible via live endpoints!**

Access the web interface at http://localhost:3000 for a complete monitoring experience.
