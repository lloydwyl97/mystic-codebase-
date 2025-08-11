#!/usr/bin/env python3
"""
Binance US Autobuy Dashboard
Real-time monitoring and control for the autobuy system
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

from endpoints.autobuy.autobuy_config import get_config
from binance_us_autobuy import autobuy_system

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="Binance US Autobuy Dashboard", version="1.0.0")

# WebSocket connections
websocket_connections: List[WebSocket] = []


@dataclass
class DashboardStats:
    """Dashboard statistics"""

    total_trades: int
    successful_trades: int
    failed_trades: int
    total_volume: float
    active_trades: int
    success_rate: float
    avg_trade_amount: float
    last_trade_time: Optional[str]
    system_uptime: float
    trading_enabled: bool
    emergency_stop: bool


class AutobuyDashboard:
    """Dashboard for monitoring and controlling the autobuy system"""

    def __init__(self):
        self.config = get_config()
        self.start_time = time.time()
        self.last_update = datetime.now(timezone.utc)

    def get_system_stats(self) -> DashboardStats:
        """Get current system statistics"""
        total_trades = autobuy_system.total_trades
        successful_trades = autobuy_system.successful_trades
        failed_trades = autobuy_system.failed_trades
        total_volume = autobuy_system.total_volume
        active_trades = len(autobuy_system.active_trades)

        success_rate = (successful_trades / total_trades * 100) if total_trades > 0 else 0
        avg_trade_amount = total_volume / total_trades if total_trades > 0 else 0

        # Get last trade time
        last_trade_time = None
        if autobuy_system.trade_history:
            last_trade = autobuy_system.trade_history[-1]
            last_trade_time = last_trade.get("timestamp")

        system_uptime = time.time() - self.start_time

        return DashboardStats(
            total_trades=total_trades,
            successful_trades=successful_trades,
            failed_trades=failed_trades,
            total_volume=total_volume,
            active_trades=active_trades,
            success_rate=success_rate,
            avg_trade_amount=avg_trade_amount,
            last_trade_time=last_trade_time,
            system_uptime=system_uptime,
            trading_enabled=self.config.trading_enabled,
            emergency_stop=self.config.emergency_stop,
        )

    def get_trading_pairs_status(self) -> Dict[str, Any]:
        """Get status of all trading pairs"""
        pairs_status = {}

        for symbol in self.config.get_enabled_pairs():
            pair_config = self.config.get_pair_config(symbol)
            if not pair_config:
                continue

            # Get recent signals for this pair
            recent_signals = []
            if symbol in autobuy_system.signal_history:
                recent_signals = autobuy_system.signal_history[symbol][-5:]  # Last 5 signals

            # Check if there's an active trade
            active_trade = autobuy_system.active_trades.get(symbol)

            # Get last signal time
            last_signal_time = autobuy_system.last_signal_time.get(symbol, 0)

            pairs_status[symbol] = {
                "name": pair_config.name,
                "enabled": pair_config.enabled,
                "min_trade_amount": pair_config.min_trade_amount,
                "max_trade_amount": pair_config.max_trade_amount,
                "target_frequency": pair_config.target_frequency,
                "active_trade": active_trade is not None,
                "last_signal_time": last_signal_time,
                "recent_signals": recent_signals,
                "total_signals": len(autobuy_system.signal_history.get(symbol, [])),
            }

        return pairs_status

    def get_recent_trades(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent trade history"""
        return autobuy_system.trade_history[-limit:] if autobuy_system.trade_history else []

    def get_system_config(self) -> Dict[str, Any]:
        """Get current system configuration"""
        return self.config.to_dict()


# Global dashboard instance
dashboard = AutobuyDashboard()


@app.get("/")
async def get_dashboard_html():
    """Get dashboard HTML page"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Binance US Autobuy Dashboard</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; }
            .header { background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
            .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 20px; }
            .stat-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .stat-value { font-size: 2em; font-weight: bold; color: #2c3e50; }
            .stat-label { color: #7f8c8d; margin-top: 5px; }
            .pairs-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 20px; }
            .pair-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .pair-name { font-size: 1.2em; font-weight: bold; color: #2c3e50; }
            .status-indicator { display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; }
            .status-active { background: #27ae60; }
            .status-inactive { background: #e74c3c; }
            .trades-table { background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); overflow: hidden; }
            .trades-table table { width: 100%; border-collapse: collapse; }
            .trades-table th, .trades-table td { padding: 12px; text-align: left; border-bottom: 1px solid #ecf0f1; }
            .trades-table th { background: #34495e; color: white; }
            .controls { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }
            .btn { padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; margin-right: 10px; }
            .btn-primary { background: #3498db; color: white; }
            .btn-danger { background: #e74c3c; color: white; }
            .btn-success { background: #27ae60; color: white; }
            .btn-warning { background: #f39c12; color: white; }
            .refresh-btn { position: fixed; top: 20px; right: 20px; z-index: 1000; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚀 Binance US Autobuy Dashboard</h1>
                <p>Real-time monitoring for SOLUSDT, BTCUSDT, ETHUSDT, AVAXUSDT</p>
            </div>

            <div class="controls">
                <h3>System Controls</h3>
                <button class="btn btn-primary" onclick="refreshData()">🔄 Refresh Data</button>
                <button class="btn btn-success" onclick="startSystem()">▶️ Start System</button>
                <button class="btn btn-warning" onclick="stopSystem()">⏹️ Stop System</button>
                <button class="btn btn-danger" onclick="emergencyStop()">🚨 Emergency Stop</button>
            </div>

            <div class="stats-grid" id="stats-grid">
                <!-- Stats will be populated by JavaScript -->
            </div>

            <div class="pairs-grid" id="pairs-grid">
                <!-- Trading pairs will be populated by JavaScript -->
            </div>

            <div class="trades-table">
                <h3>Recent Trades</h3>
                <table id="trades-table">
                    <thead>
                        <tr>
                            <th>Time</th>
                            <th>Symbol</th>
                            <th>Amount</th>
                            <th>Price</th>
                            <th>Confidence</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody id="trades-body">
                        <!-- Trades will be populated by JavaScript -->
                    </tbody>
                </table>
            </div>
        </div>

        <script>
            function refreshData() {
                fetch('/api/stats')
                    .then(response => response.json())
                    .then(data => updateStats(data));

                fetch('/api/pairs')
                    .then(response => response.json())
                    .then(data => updatePairs(data));

                fetch('/api/trades')
                    .then(response => response.json())
                    .then(data => updateTrades(data));
            }

            function updateStats(stats) {
                const statsGrid = document.getElementById('stats-grid');
                statsGrid.innerHTML = `
                    <div class="stat-card">
                        <div class="stat-value">${stats.total_trades}</div>
                        <div class="stat-label">Total Trades</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">${stats.success_rate.toFixed(1)}%</div>
                        <div class="stat-label">Success Rate</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">$${stats.total_volume.toFixed(2)}</div>
                        <div class="stat-label">Total Volume</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">${stats.active_trades}</div>
                        <div class="stat-label">Active Trades</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">${stats.trading_enabled ? '✅' : '❌'}</div>
                        <div class="stat-label">Trading Enabled</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">${stats.emergency_stop ? '🚨' : '✅'}</div>
                        <div class="stat-label">Emergency Stop</div>
                    </div>
                `;
            }

            function updatePairs(pairs) {
                const pairsGrid = document.getElementById('pairs-grid');
                pairsGrid.innerHTML = '';

                Object.entries(pairs).forEach(([symbol, data]) => {
                    const statusClass = data.active_trade ? 'status-active' : 'status-inactive';
                    const statusText = data.active_trade ? 'Active' : 'Inactive';

                    pairsGrid.innerHTML += `
                        <div class="pair-card">
                            <div class="pair-name">${data.name} (${symbol})</div>
                            <div><span class="status-indicator ${statusClass}"></span>${statusText}</div>
                            <div>Min Trade: $${data.min_trade_amount}</div>
                            <div>Max Trade: $${data.max_trade_amount}</div>
                            <div>Frequency: ${data.target_frequency} min</div>
                            <div>Total Signals: ${data.total_signals}</div>
                        </div>
                    `;
                });
            }

            function updateTrades(trades) {
                const tradesBody = document.getElementById('trades-body');
                tradesBody.innerHTML = '';

                trades.forEach(trade => {
                    const time = new Date(trade.timestamp).toLocaleString();
                    tradesBody.innerHTML += `
                        <tr>
                            <td>${time}</td>
                            <td>${trade.symbol}</td>
                            <td>$${trade.amount_usd}</td>
                            <td>$${trade.price}</td>
                            <td>${trade.confidence}%</td>
                            <td>${trade.status}</td>
                        </tr>
                    `;
                });
            }

            function startSystem() {
                fetch('/api/control/start', { method: 'POST' })
                    .then(() => refreshData());
            }

            function stopSystem() {
                fetch('/api/control/stop', { method: 'POST' })
                    .then(() => refreshData());
            }

            function emergencyStop() {
                if (confirm('Are you sure you want to emergency stop the system?')) {
                    fetch('/api/control/emergency-stop', { method: 'POST' })
                        .then(() => refreshData());
                }
            }

            // Auto-refresh every 30 seconds
            setInterval(refreshData, 30000);

            // Initial load
            refreshData();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/api/stats")
async def get_stats():
    """Get system statistics"""
    try:
        stats = dashboard.get_system_stats()
        return asdict(stats)
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/pairs")
async def get_pairs():
    """Get trading pairs status"""
    try:
        pairs = dashboard.get_trading_pairs_status()
        return pairs
    except Exception as e:
        logger.error(f"Error getting pairs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/trades")
async def get_trades(limit: int = 20):
    """Get recent trades"""
    try:
        trades = dashboard.get_recent_trades(limit)
        return trades
    except Exception as e:
        logger.error(f"Error getting trades: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/config")
async def get_system_config():
    """Get system configuration"""
    try:
        dashboard = AutobuyDashboard()
        config = dashboard.get_system_config()
        return {"config": config, "timestamp": int(time.time())}
    except Exception as e:
        logger.error(f"Error getting system config: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/control/start")
async def start_system():
    """Start the autobuy system"""
    try:
        await autobuy_system.start()
        logger.info("Autobuy system started via API.")
        return {"status": "success", "message": "System started."}
    except Exception as e:
        logger.error(f"Error starting system: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/control/stop")
async def stop_system():
    """Stop the autobuy system"""
    try:
        await autobuy_system.stop()
        logger.info("Autobuy system stopped via API.")
        return {"status": "success", "message": "System stopped."}
    except Exception as e:
        logger.error(f"Error stopping system: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/control/emergency-stop")
async def emergency_stop():
    """Emergency stop the autobuy system"""
    try:
        await autobuy_system.emergency_stop()
        logger.warning("EMERGENCY STOP triggered via API!")
        return {"status": "success", "message": "Emergency stop activated."}
    except Exception as e:
        logger.error(f"Error emergency stopping system: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    websocket_connections.append(websocket)

    try:
        while True:
            # Send periodic updates
            stats = dashboard.get_system_stats()
            await websocket.send_text(json.dumps({"type": "stats_update", "data": asdict(stats)}))
            await asyncio.sleep(30)  # Update every 30 seconds
    except WebSocketDisconnect:
        websocket_connections.remove(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in websocket_connections:
            websocket_connections.remove(websocket)


async def broadcast_update(update_type: str, data: Dict[str, Any]):
    """Broadcast update to all WebSocket connections"""
    message = json.dumps({"type": update_type, "data": data})
    disconnected = []

    for websocket in websocket_connections:
        try:
            await websocket.send_text(message)
        except Exception:
            disconnected.append(websocket)

    # Remove disconnected websockets
    for websocket in disconnected:
        websocket_connections.remove(websocket)


if __name__ == "__main__":
    logger.info("🚀 Starting Binance US Autobuy Dashboard...")
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
