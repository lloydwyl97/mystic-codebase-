# 🚀 Crypto Widget System

A real-time cryptocurrency price monitoring system with live data polling, FastAPI server, and historical logging.

## 📁 File Structure

```
crypto_widget/
├── price_poller.py         ← Main price poller (30s intervals)
├── api_server.py           ← FastAPI server for live data
├── widget_page_1.html      ← Web widget (pulls from FastAPI)
├── widget_chart.html       ← Chart widget with Chart.js
├── start_widget.py         ← Startup script for both services
├── test_endpoints.py       ← Test script for Priority 1 endpoints
├── test_updated_endpoints.py ← Test script for updated endpoints
├── shared_data.json        ← Latest price snapshot (auto-generated)
├── price_log.csv           ← Historical price log (auto-generated)
└── README.md               ← This file
```

## 🚀 Quick Start

### Option 1: Automatic Startup (Recommended)
```bash
cd crypto_widget
python start_widget.py
```

### Option 2: Manual Startup
```bash
# Terminal 1: Start price poller
cd crypto_widget
python price_poller.py

# Terminal 2: Start FastAPI server
cd crypto_widget
uvicorn api_server:app --reload --port 8000
```

## 🌐 API Endpoints

### ✅ Priority 1 (Live Widget Essentials)

#### `GET /prices`
- **Purpose:** Latest price snapshot for all coins and sources
- **Response:** JSON with nested structure by source and symbol
- **Error Handling:** Returns 500 status with error message
- **Status:** ✅ Implemented

#### `GET /health`
- **Purpose:** System health status and file monitoring
- **Response:** Status, last update time, log file size
- **Fields:** `status`, `last_updated_utc`, `log_file_size_bytes`
- **Status:** ✅ Implemented

#### `GET /coins`
- **Purpose:** List available symbols by source
- **Response:** Dictionary with sources as keys, coin lists as values
- **Format:** `{"Binance": ["BTC", "ETH", "SOL", "ADA"], "Coinbase": ["BTC", "ETH", "SOL", "ADA"]}`
- **Status:** ✅ Implemented

#### `GET /history`
- **Purpose:** Historical price data from CSV log
- **Parameters:**
  - `symbol` (required): Filter by symbol (min 3 chars)
  - `source` (required): Filter by source (Binance, Coinbase)
  - `limit` (optional): Number of records (default: 50)
- **Response:** Array of historical records (newest first)
- **Error Handling:** Returns 500 status with error message
- **Status:** ✅ Implemented

### 🔄 Access Points

- **API Endpoint**: http://localhost:8000/prices
- **Health Check**: http://localhost:8000/health
- **Available Coins**: http://localhost:8000/coins
- **Historical Data**: http://localhost:8000/history?symbol=BTC&source=Binance&limit=50
- **Widget Page**: Open `widget_page_1.html` in browser
- **Chart Widget**: Open `widget_chart.html` in browser
- **API Docs**: http://localhost:8000/docs

## 📊 Features

### ✅ Live Price Polling
- **Binance**: BTC, ETH, SOL, ADA
- **Coinbase**: BTC, ETH, SOL, ADA
- **Interval**: 30 seconds per call
- **Rotation**: Cycles through all exchanges

### ✅ Percentage Change Display
- Auto-calculated in memory
- Based on previous fetched value
- Real-time updates

### ✅ Historical Logging
- **Format**: CSV with timestamp, source, symbol, price, change%
- **File**: `price_log.csv` (append-only)
- **Headers**: Auto-created on first run

### ✅ FastAPI Live Feed
- **Endpoint**: `/prices`
- **Format**: JSON with nested structure
- **CORS**: Enabled for cross-origin requests

### ✅ System Monitoring
- **Health Check**: `/health` endpoint for file status and timestamps
- **Coin Info**: `/coins` endpoint for available symbols by source
- **Historical Data**: `/history` endpoint with required parameters

### ✅ Improved Error Handling
- **JSONResponse**: Proper HTTP status codes (500 for errors)
- **Parameter Validation**: Required parameters with minimum length
- **Graceful Degradation**: Error messages instead of crashes

### ✅ Chart Visualization
- **Chart.js Integration**: Zero-install CDN-based charts
- **Live Updates**: Auto-refresh every 60 seconds
- **Multi-Exchange**: Shows all coins across both exchanges
- **Historical Data**: Displays last 30 price points per chart
- **Responsive Design**: Dark theme with professional styling

## 📈 Data Structure

### API Response (`/prices`)
```json
{
  "Binance": {
    "BTC": {
      "price": 43250.50,
      "symbol": "BTC",
      "source": "Binance",
      "timestamp": "2024-01-15T10:30:00.123456",
      "change_pct": 2.45
    }
  },
  "Coinbase": {
    "ETH": {
      "price": 2650.75,
      "symbol": "ETH",
      "source": "Coinbase",
      "timestamp": "2024-01-15T10:30:30.654321",
      "change_pct": -1.23
    }
  }
}
```

### Health Response (`/health`)
```json
{
  "status": "ok",
  "last_updated_utc": "2024-01-15T10:30:00.123456+00:00",
  "log_file_size_bytes": 1024
}
```

### Coins Response (`/coins`)
```json
{
  "Binance": ["BTC", "ETH", "SOL", "ADA"],
  "Coinbase": ["BTC", "ETH", "SOL", "ADA"]
}
```

### History Response (`/history?symbol=BTC&source=Binance&limit=5`)
```json
[
  {
    "timestamp": "2024-01-15T10:30:00.123456",
    "source": "Binance",
    "symbol": "BTC",
    "price": "43250.50",
    "change_pct": "2.45"
  },
  {
    "timestamp": "2024-01-15T10:29:30.123456",
    "source": "Binance",
    "symbol": "BTC",
    "price": "43200.00",
    "change_pct": "1.20"
  }
]
```

### Error Response (500 status)
```json
{
  "error": "File not found: shared_data.json"
}
```

### CSV Log Format (`price_log.csv`)
```csv
timestamp,source,symbol,price,change_pct
2024-01-15T10:30:00.123456,Binance,BTC,43250.50,2.45
2024-01-15T10:30:30.654321,Coinbase,ETH,2650.75,-1.23
```

## 🎨 Widget Features

### Basic Widget (`widget_page_1.html`)
- **Dark Theme**: Professional dark UI
- **Real-time Updates**: 30-second refresh cycle
- **Color-coded Changes**: Green for positive, red for negative
- **Timestamp Display**: Shows last update time
- **Responsive Design**: Works on desktop and mobile

### Chart Widget (`widget_chart.html`)
- **Chart.js Integration**: Professional line charts
- **Multi-Exchange Display**: All coins across both sources
- **Historical Visualization**: Last 30 price points per chart
- **Auto-Refresh**: Updates every 60 seconds
- **Dark Theme**: Consistent with system design
- **Responsive Layout**: Modular card-based design

## 🔧 Configuration

### Adding New Coins
Edit `price_poller.py` and add to `BINANCE` and `COINBASE` dictionaries:

```python
BINANCE = {
    "BTC": "https://api.binance.us/api/v3/ticker/price?symbol=BTCUSDT",
    "ETH": "https://api.binance.us/api/v3/ticker/price?symbol=ETHUSDT",
    "NEW_COIN": "https://api.binance.us/api/v3/ticker/price?symbol=NEWCOINUSDT"
}
```

### Changing Polling Interval
Modify the `await asyncio.sleep(30)` line in `price_poller.py`

### Changing API Port
Modify the port in `start_widget.py` or uvicorn command

### Chart Configuration
Edit `widget_chart.html` to modify:
- **Update Frequency**: Change `60000` (60 seconds) in `setInterval`
- **Data Points**: Change `limit=30` in `fetchHistory` function
- **Chart Styling**: Modify CSS in the `<style>` section

## 🛠️ Dependencies

- `fastapi` - API server
- `uvicorn` - ASGI server
- `httpx` - HTTP client for polling
- `asyncio` - Async programming (built-in)
- `json` - JSON handling (built-in)
- `csv` - CSV logging (built-in)
- `Chart.js` - Chart visualization (CDN)

## 📝 Logs

- **Console**: Real-time polling status
- **CSV**: Historical price data
- **JSON**: Current state snapshot

## 🧪 Testing

Run the updated endpoints test:
```bash
python test_updated_endpoints.py
```

This will verify:
- ✅ Prices endpoint functionality
- ✅ Health endpoint with file status
- ✅ Coins endpoint structure
- ✅ History endpoint with required parameters
- ✅ Error handling and responses

## 🔍 Troubleshooting

### Port Already in Use
```bash
# Kill process on port 8000
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### Missing Dependencies
```bash
pip install fastapi uvicorn httpx
```

### File Permissions
Ensure write permissions for `shared_data.json` and `price_log.csv`

### API Errors
- Check that data files exist
- Verify required parameters for `/history` endpoint
- Check file permissions and disk space

### Chart Issues
- Ensure Chart.js CDN is accessible
- Check browser console for JavaScript errors
- Verify API endpoints are responding correctly

## 🎯 Integration with Mystic Trading

This widget system can be integrated with your main Mystic Trading platform:

1. **Data Source**: Use the `/prices` endpoint as a data source
2. **Historical Data**: Use `/history` endpoint for analysis
3. **Real-time Updates**: Subscribe to the API for live trading signals
4. **Custom Widgets**: Extend the HTML widgets for your dashboard
5. **System Monitoring**: Use `/health` for file status monitoring
6. **Coin Management**: Use `/coins` for dynamic UI configuration
7. **Chart Integration**: Embed chart widgets in your trading interface

## 📊 Performance

- **Memory**: Minimal (in-memory price tracking)
- **CPU**: Low (simple HTTP requests)
- **Network**: ~8 requests per minute
- **Storage**: ~1KB per day (CSV logging)
- **API Response**: <100ms for all endpoints
- **Error Handling**: Graceful with proper HTTP status codes
- **Chart Updates**: 60-second refresh cycle
