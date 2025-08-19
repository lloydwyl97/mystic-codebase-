# dashboard_api.py
"""
FastAPI Dashboard Backend
Real-time monitoring and control interface for the AI trading system.
"""

import os
from datetime import datetime, timezone

import uvicorn
from capital_allocator import CapitalAllocator
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from hyper_tuner import HyperparameterTuner
from position_sizer import PositionSizer

# Import your modules
from strategy_leaderboard import get_strategy_leaderboard
from watchdog import TradingWatchdog
from yield_rotator import YieldRotator

# Initialize FastAPI app
app = FastAPI(
    title="Mystic AI Trading Dashboard",
    description="Real-time monitoring and control for AI crypto trading system",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
position_sizer = PositionSizer()
capital_allocator = CapitalAllocator()
yield_rotator = YieldRotator()
watchdog = TradingWatchdog()
hyper_tuner = HyperparameterTuner()


@app.get("/")
async def root():
    """Root endpoint with basic info."""
    return {
        "message": "Mystic AI Trading Dashboard",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "services": {
            "dashboard": "running",
            "trade_logger": "running",
            "strategy_leaderboard": "running",
        },
    }


@app.get("/api/leaderboard")
async def get_leaderboard(hours_back: int = 24):
    """Get strategy leaderboard."""
    try:
        leaderboard = get_strategy_leaderboard(hours_back)
        return {
            "leaderboard": leaderboard,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "period_hours": hours_back,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching leaderboard: {str(e)}")


@app.get("/api/trades")
async def get_recent_trades(limit: int = 50):
    """Get recent trades."""
    try:
        import sqlite3

        db_file = os.getenv("TRADE_LOG_DB", "trades.db")

        if not os.path.exists(db_file):
            return {"trades": [], "message": "No trade database found"}

        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT timestamp, symbol, strategy, direction, entry_price,
                   exit_price, quantity, profit_usd, duration_sec
            FROM trades
            ORDER BY timestamp DESC
            LIMIT ?
        """,
            (limit,),
        )

        rows = cursor.fetchall()
        conn.close()

        trades = []
        for row in rows:
            trades.append(
                {
                    "timestamp": row[0],
                    "symbol": row[1],
                    "strategy": row[2],
                    "direction": row[3],
                    "entry_price": row[4],
                    "exit_price": row[5],
                    "quantity": row[6],
                    "profit_usd": row[7],
                    "duration_sec": row[8],
                }
            )

        return {
            "trades": trades,
            "count": len(trades),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching trades: {str(e)}")


@app.get("/api/stats")
async def get_system_stats():
    """Get system statistics."""
    try:
        # Get leaderboard for stats
        leaderboard = get_strategy_leaderboard(24)

        # Calculate stats
        total_profit = sum(s["total_profit"] for s in leaderboard)
        total_trades = sum(s["trades"] for s in leaderboard)
        avg_win_rate = (
            sum(s["win_rate"] for s in leaderboard) / len(leaderboard) if leaderboard else 0
        )

        # Get position sizing stats
        position_history = position_sizer.get_position_history(10)

        # Get capital allocation stats
        portfolio_summary = capital_allocator.get_portfolio_summary()

        # Get yield rotation stats
        yield_summary = yield_rotator.get_parked_capital_summary()

        # Get system health
        system_health = watchdog.get_system_summary()

        return {
            "trading_stats": {
                "total_profit_24h": round(total_profit, 2),
                "total_trades_24h": total_trades,
                "average_win_rate": round(avg_win_rate, 3),
                "active_strategies": len(leaderboard),
            },
            "position_sizing": {
                "recent_positions": len(position_history),
                "avg_position_size": round(
                    sum(p.get("position_size_usd", 0) for p in position_history)
                    / max(len(position_history), 1),
                    2,
                ),
            },
            "capital_allocation": portfolio_summary,
            "yield_rotation": yield_summary,
            "system_health": system_health,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching stats: {str(e)}")


@app.post("/api/optimize")
async def optimize_strategy(
    strategy_type: str = "rsi_ema_breakout",
    method: str = "genetic",
    rounds: int = 50,
    background_tasks: BackgroundTasks = None,
):
    """Start hyperparameter optimization."""
    try:
        # Run optimization in background
        if background_tasks:
            background_tasks.add_task(
                hyper_tuner.optimize_strategy,
                strategy_type,
                method,
                rounds=rounds,
            )
            return {
                "message": (f"Optimization started for {strategy_type} using {method} method"),
                "strategy_type": strategy_type,
                "method": method,
                "rounds": rounds,
                "status": "started",
            }
        else:
            # Run synchronously
            result = hyper_tuner.optimize_strategy(strategy_type, method, rounds=rounds)
            return {
                "message": "Optimization completed",
                "result": result,
                "status": "completed",
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting optimization: {str(e)}")


@app.post("/api/allocate-capital")
async def allocate_capital(method: str = "performance", total_capital: float = 10000):
    """Allocate capital using specified method."""
    try:
        allocations = (
            capital_allocator.allocate_by_performance()
            if method == "performance"
            else (
                capital_allocator.allocate_by_risk_parity()
                if method == "risk_parity"
                else (
                    capital_allocator.allocate_by_equal_weight()
                    if method == "equal_weight"
                    else (
                        capital_allocator.allocate_by_kelly_criterion()
                        if method == "kelly"
                        else (
                            capital_allocator.allocate_by_momentum()
                            if method == "momentum"
                            else capital_allocator.allocate_by_performance()
                        )
                    )
                )
            )
        )

        return {
            "allocations": allocations,
            "method": method,
            "total_capital": total_capital,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error allocating capital: {str(e)}")


@app.post("/api/park-capital")
async def park_capital(amount: float, protocol_id: str = None, lock_period: int = 30):
    """Park capital in yield protocol."""
    try:
        result = yield_rotator.park_capital(amount, protocol_id, lock_period)
        return {
            "result": result,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error parking capital: {str(e)}")


@app.post("/api/withdraw-capital")
async def withdraw_capital(parking_id: str, force: bool = False):
    """Withdraw capital from yield protocol."""
    try:
        result = yield_rotator.withdraw_capital(parking_id, force)
        return {
            "result": result,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error withdrawing capital: {str(e)}")


@app.get("/api/system-health")
async def get_system_health():
    """Get detailed system health information."""
    try:
        # Monitor all services
        monitoring_results = watchdog.monitor_all_services()

        return {
            "monitoring_results": monitoring_results,
            "system_summary": watchdog.get_system_summary(),
            "health_history": watchdog.get_health_history(10),
            "restart_history": watchdog.get_restart_history(10),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching system health: {str(e)}")


@app.post("/api/restart-service")
async def restart_service(service_name: str):
    """Restart a specific service."""
    try:
        result = watchdog.restart_service(service_name)
        return {
            "result": result,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error restarting service: {str(e)}")


@app.get("/api/position-sizing")
async def get_position_sizing_info(capital: float = 10000, strategy_name: str = "RSI_Breakout_V1"):
    """Get position sizing information."""
    try:
        # Get strategy performance
        leaderboard = get_strategy_leaderboard(24)
        strategy_perf = next((s for s in leaderboard if s["strategy"] == strategy_name), None)

        if not strategy_perf:
            return {"error": f"Strategy {strategy_name} not found"}

        # Calculate position size
        position_result = position_sizer.calculate_optimal_position_size(
            capital,
            strategy_name,
            "BTC/USDT",
            50000,
            method="volatility_adjusted",
        )

        return {
            "strategy_performance": strategy_perf,
            "position_sizing": position_result,
            "position_history": position_sizer.get_position_history(10),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error calculating position size: {str(e)}",
        )


@app.get("/api/yield-summary")
async def get_yield_summary():
    """Get yield rotation summary."""
    try:
        return {
            "parked_capital": yield_rotator.get_parked_capital_summary(),
            "yield_history": yield_rotator.get_yield_history(20),
            "available_protocols": yield_rotator.yield_protocols,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching yield summary: {str(e)}")


@app.get("/api/capital-allocation")
async def get_capital_allocation():
    """Get capital allocation information."""
    try:
        return {
            "current_allocations": capital_allocator.current_allocations,
            "portfolio_summary": capital_allocator.get_portfolio_summary(),
            "allocation_history": capital_allocator.get_allocation_history(10),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching capital allocation: {str(e)}",
        )


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_html():
    """Simple HTML dashboard."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset=\"utf-8\">
        <title>Mystic AI Trading Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; }
            .header { background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
            .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 20px; }
            .stat-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .stat-value { font-size: 24px; font-weight: bold; color: #2c3e50; }
            .stat-label { color: #7f8c8d; margin-top: 5px; }
            .section { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .refresh-btn { background: #3498db; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; }
            .refresh-btn:hover { background: #2980b9; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>ðŸš€ Mystic AI Trading Dashboard</h1>
                <p>Real-time monitoring and control for AI crypto trading system</p>
            </div>

            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value" id="total-profit">$0.00</div>
                    <div class="stat-label">24h Total Profit</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="win-rate">0%</div>
                    <div class="stat-label">Average Win Rate</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="active-strategies">0</div>
                    <div class="stat-label">Active Strategies</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="system-health">Unknown</div>
                    <div class="stat-label">System Health</div>
                </div>
            </div>

            <div class="section">
                <h2>ðŸ“Š Strategy Leaderboard</h2>
                <button class="refresh-btn" onclick="loadLeaderboard()">Refresh</button>
                <div id="leaderboard">Loading...</div>
            </div>

            <div class="section">
                <h2>ðŸ’° Recent Trades</h2>
                <button class="refresh-btn" onclick="loadTrades()">Refresh</button>
                <div id="trades">Loading...</div>
            </div>

            <div class="section">
                <h2>ðŸ›¡ï¸ System Health</h2>
                <button class="refresh-btn" onclick="loadSystemHealth()">Refresh</button>
                <div id="system-status">Loading...</div>
            </div>
        </div>

        <script>
            async function loadStats() {
                try {
                    const response = await fetch('/api/stats');
                    const data = await response.json();

                    document.getElementById('total-profit').textContent = '$' + data.trading_stats.total_profit_24h.toFixed(2);
                    document.getElementById('win-rate').textContent = (data.trading_stats.average_win_rate * 100).toFixed(1) + '%';
                    document.getElementById('active-strategies').textContent = data.trading_stats.active_strategies;
                    document.getElementById('system-health').textContent = data.system_health.overall_health;
                } catch (error) {
                    console.error('Error loading stats:', error);
                }
            }

            async function loadLeaderboard() {
                try {
                    const response = await fetch('/api/leaderboard');
                    const data = await response.json();

                    let html = '<table style="width: 100%; border-collapse: collapse;">';
                    html += '<tr><th>Strategy</th><th>Profit</th><th>Win Rate</th><th>Trades</th></tr>';

                    data.leaderboard.forEach(strategy => {
                        html += `<tr>
                            <td>${strategy.strategy}</td>
                            <td>$${strategy.total_profit.toFixed(2)}</td>
                            <td>${(strategy.win_rate * 100).toFixed(1)}%</td>
                            <td>${strategy.trades}</td>
                        </tr>`;
                    });

                    html += '</table>';
                    document.getElementById('leaderboard').innerHTML = html;
                } catch (error) {
                    document.getElementById('leaderboard').innerHTML = 'Error loading leaderboard';
                }
            }

            async function loadTrades() {
                try {
                    const response = await fetch('/api/trades?limit=10');
                    const data = await response.json();

                    let html = '<table style="width: 100%; border-collapse: collapse;">';
                    html += '<tr><th>Time</th><th>Symbol</th><th>Strategy</th><th>Profit</th></tr>';

                    data.trades.forEach(trade => {
                        html += `<tr>
                            <td>${trade.timestamp}</td>
                            <td>${trade.symbol}</td>
                            <td>${trade.strategy}</td>
                            <td>$${trade.profit_usd.toFixed(2)}</td>
                        </tr>`;
                    });

                    html += '</table>';
                    document.getElementById('trades').innerHTML = html;
                } catch (error) {
                    document.getElementById('trades').innerHTML = 'Error loading trades';
                }
            }

            async function loadSystemHealth() {
                try {
                    const response = await fetch('/api/system-health');
                    const data = await response.json();

                    let html = '<p><strong>Overall Health:</strong> ' + data.system_summary.overall_health + '</p>';
                    html += '<p><strong>Healthy Services:</strong> ' + data.system_summary.healthy_services + '/' + data.system_summary.total_services + '</p>';
                    html += '<p><strong>Health Percentage:</strong> ' + data.system_summary.health_percentage + '%</p>';

                    document.getElementById('system-status').innerHTML = html;
                } catch (error) {
                    document.getElementById('system-status').innerHTML = 'Error loading system health';
                }
            }

            // Load data on page load
            loadStats();
            loadLeaderboard();
            loadTrades();
            loadSystemHealth();

            // Auto-refresh every 30 seconds
            setInterval(() => {
                loadStats();
                loadLeaderboard();
                loadTrades();
                loadSystemHealth();
            }, 30000);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


