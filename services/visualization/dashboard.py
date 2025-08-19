# dashboard.py
import logging

import pandas as pd
import plotly.graph_objs as go
import uvicorn
from db_logger import get_active_strategies, get_recent_trades
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from plotly.subplots import make_subplots
from reward_engine import get_top_performers

# Configure logging
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="Mystic AI Trading Dashboard", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_trade_data(limit: int = 100) -> pd.DataFrame:
    """Load recent trade data from SQLite"""
    try:
        trades = get_recent_trades(limit=limit)
        if trades:
            df = pd.DataFrame(trades)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading trade data: {e}")
        return pd.DataFrame()


def create_profit_chart(df: pd.DataFrame) -> go.Figure:
    """Create profit over time chart"""
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No trade data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    # Filter completed trades
    completed_trades = df[df["exit_price"].notna()].copy()

    if completed_trades.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No completed trades available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    # Calculate cumulative profit
    completed_trades = completed_trades.sort_values("timestamp")
    completed_trades["cumulative_profit"] = completed_trades["profit"].cumsum()

    fig = go.Figure()

    # Add profit line
    fig.add_trace(
        go.Scatter(
            x=completed_trades["timestamp"],
            y=completed_trades["cumulative_profit"],
            mode="lines+markers",
            name="Cumulative Profit",
            line=dict(color="#00ff00", width=2),
            marker=dict(size=6),
        )
    )

    # Add individual trade points
    fig.add_trace(
        go.Scatter(
            x=completed_trades["timestamp"],
            y=completed_trades["profit"],
            mode="markers",
            name="Individual Trades",
            marker=dict(
                color=completed_trades["profit"].apply(
                    lambda x: "green" if x > 0 else "red"
                ),
                size=8,
                symbol="circle",
            ),
            yaxis="y2",
        )
    )

    fig.update_layout(
        title="Trading Performance Over Time",
        xaxis_title="Time",
        yaxis_title="Cumulative Profit",
        yaxis2=dict(title="Individual Trade Profit", overlaying="y", side="right"),
        hovermode="x unified",
        height=500,
    )

    return fig


def create_strategy_performance_chart() -> go.Figure:
    """Create strategy performance comparison chart"""
    try:
        strategies = get_active_strategies()
        if not strategies:
            fig = go.Figure()
            fig.add_annotation(
                text="No active strategies available",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
            return fig

        # Prepare data
        strategy_names = [s["name"] for s in strategies]
        win_rates = [s["win_rate"] for s in strategies]
        avg_profits = [s["avg_profit"] for s in strategies]
        trade_counts = [s["trades_executed"] for s in strategies]

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Win Rate by Strategy",
                "Average Profit by Strategy",
                "Total Trades by Strategy",
                "Strategy Overview",
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
            ],
        )

        # Win Rate
        fig.add_trace(
            go.Bar(
                x=strategy_names,
                y=win_rates,
                name="Win Rate",
                marker_color="blue",
            ),
            row=1,
            col=1,
        )

        # Average Profit
        colors = ["green" if p > 0 else "red" for p in avg_profits]
        fig.add_trace(
            go.Bar(
                x=strategy_names,
                y=avg_profits,
                name="Avg Profit",
                marker_color=colors,
            ),
            row=1,
            col=2,
        )

        # Trade Count
        fig.add_trace(
            go.Bar(
                x=strategy_names,
                y=trade_counts,
                name="Trade Count",
                marker_color="orange",
            ),
            row=2,
            col=1,
        )

        # Scatter plot of win rate vs avg profit
        fig.add_trace(
            go.Scatter(
                x=win_rates,
                y=avg_profits,
                mode="markers+text",
                text=strategy_names,
                textposition="top center",
                marker=dict(size=10, color=trade_counts, colorscale="Viridis"),
                name="Win Rate vs Profit",
            ),
            row=2,
            col=2,
        )

        fig.update_layout(
            title="Strategy Performance Overview", height=800, showlegend=False
        )

        return fig

    except Exception as e:
        logger.error(f"Error creating strategy performance chart: {e}")
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error loading strategy data: {e}",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig


def create_recent_trades_table(df: pd.DataFrame) -> go.Figure:
    """Create recent trades table"""
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No recent trades available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    # Prepare table data
    table_data = df.head(20).copy()  # Show last 20 trades

    # Format data for table
    table_data["timestamp"] = table_data["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    table_data["profit"] = table_data["profit"].apply(
        lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
    )
    table_data["profit_percentage"] = table_data["profit_percentage"].apply(
        lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A"
    )
    table_data["success"] = table_data["success"].apply(
        lambda x: "✅" if x else "❌" if pd.notna(x) else "N/A"
    )

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=[
                        "Time",
                        "Coin",
                        "Strategy",
                        "Entry",
                        "Exit",
                        "Profit",
                        "P%",
                        "Success",
                    ],
                    fill_color="paleturquoise",
                    align="left",
                    font=dict(size=12),
                ),
                cells=dict(
                    values=[
                        table_data["timestamp"],
                        table_data["coin"],
                        table_data["strategy_name"],
                        table_data["entry_price"].apply(lambda x: f"{x:.2f}"),
                        table_data["exit_price"].apply(
                            lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
                        ),
                        table_data["profit"],
                        table_data["profit_percentage"],
                        table_data["success"],
                    ],
                    fill_color="lavender",
                    align="left",
                    font=dict(size=11),
                ),
            )
        ]
    )

    fig.update_layout(title="Recent Trades", height=400)

    return fig


@app.get("/", response_class=HTMLResponse)
async def dashboard_home():
    """Main dashboard page"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Mystic AI Trading Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
            .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                     color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
            .container { max-width: 1400px; margin: 0 auto; }
            .chart-container { background: white; padding: 20px; margin: 20px 0; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }
            .stat-card { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); text-align: center; }
            .stat-value { font-size: 2em; font-weight: bold; color: #667eea; }
            .stat-label { color: #666; margin-top: 5px; }
            .refresh-btn { background: #667eea; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; margin: 10px 0; }
            .refresh-btn:hover { background: #5a6fd8; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚀 Mystic AI Trading Dashboard</h1>
                <p>Real-time trading performance and strategy analysis</p>
                <button class="refresh-btn" onclick="refreshDashboard()">🔄 Refresh Data</button>
            </div>
            
            <div class="stats-grid" id="stats-grid">
                <!-- Stats will be loaded here -->
            </div>
            
            <div class="chart-container">
                <h2>📈 Trading Performance</h2>
                <div id="profit-chart"></div>
            </div>
            
            <div class="chart-container">
                <h2>🎯 Strategy Performance</h2>
                <div id="strategy-chart"></div>
            </div>
            
            <div class="chart-container">
                <h2>📊 Recent Trades</h2>
                <div id="trades-table"></div>
            </div>
        </div>
        
        <script>
            function refreshDashboard() {
                loadStats();
                loadCharts();
            }
            
            function loadStats() {
                fetch('/api/stats')
                    .then(response => response.json())
                    .then(data => {
                        const statsGrid = document.getElementById('stats-grid');
                        statsGrid.innerHTML = `
                            <div class="stat-card">
                                <div class="stat-value">${data.total_trades}</div>
                                <div class="stat-label">Total Trades</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value">${(data.win_rate * 100).toFixed(1)}%</div>
                                <div class="stat-label">Win Rate</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value">$${data.total_profit.toFixed(2)}</div>
                                <div class="stat-label">Total Profit</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value">${data.active_strategies}</div>
                                <div class="stat-label">Active Strategies</div>
                            </div>
                        `;
                    });
            }
            
            function loadCharts() {
                // Load profit chart
                fetch('/api/profit-chart')
                    .then(response => response.json())
                    .then(data => {
                        Plotly.newPlot('profit-chart', data.data, data.layout);
                    });
                
                // Load strategy chart
                fetch('/api/strategy-chart')
                    .then(response => response.json())
                    .then(data => {
                        Plotly.newPlot('strategy-chart', data.data, data.layout);
                    });
                
                // Load trades table
                fetch('/api/trades-table')
                    .then(response => response.json())
                    .then(data => {
                        Plotly.newPlot('trades-table', data.data, data.layout);
                    });
            }
            
            // Load data on page load
            document.addEventListener('DOMContentLoaded', function() {
                loadStats();
                loadCharts();
            });
            
            // Auto-refresh every 30 seconds
            setInterval(refreshDashboard, 30000);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/api/stats")
async def get_stats():
    """Get dashboard statistics"""
    try:
        trades = get_recent_trades(limit=1000)
        strategies = get_active_strategies()

        if not trades:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "total_profit": 0.0,
                "active_strategies": len(strategies),
            }

        # Calculate stats
        completed_trades = [t for t in trades if t["exit_price"] is not None]
        total_trades = len(completed_trades)

        if total_trades == 0:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "total_profit": 0.0,
                "active_strategies": len(strategies),
            }

        winning_trades = sum(1 for t in completed_trades if t["success"])
        win_rate = winning_trades / total_trades
        total_profit = sum(
            t["profit"] for t in completed_trades if t["profit"] is not None
        )

        return {
            "total_trades": total_trades,
            "win_rate": win_rate,
            "total_profit": total_profit,
            "active_strategies": len(strategies),
        }

    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "total_profit": 0.0,
            "active_strategies": 0,
        }


@app.get("/api/profit-chart")
async def get_profit_chart():
    """Get profit chart data"""
    try:
        df = load_trade_data(limit=100)
        fig = create_profit_chart(df)
        return fig.to_dict()
    except Exception as e:
        logger.error(f"Error creating profit chart: {e}")
        return {"data": [], "layout": {"title": "Error loading chart"}}


@app.get("/api/strategy-chart")
async def get_strategy_chart():
    """Get strategy performance chart data"""
    try:
        fig = create_strategy_performance_chart()
        return fig.to_dict()
    except Exception as e:
        logger.error(f"Error creating strategy chart: {e}")
        return {"data": [], "layout": {"title": "Error loading chart"}}


@app.get("/api/trades-table")
async def get_trades_table():
    """Get recent trades table data"""
    try:
        df = load_trade_data(limit=50)
        fig = create_recent_trades_table(df)
        return fig.to_dict()
    except Exception as e:
        logger.error(f"Error creating trades table: {e}")
        return {"data": [], "layout": {"title": "Error loading table"}}


@app.get("/api/strategies")
async def get_strategies():
    """Get all active strategies"""
    try:
        strategies = get_active_strategies()
        return {"strategies": strategies}
    except Exception as e:
        logger.error(f"Error getting strategies: {e}")
        return {"strategies": []}


@app.get("/api/top-performers")
async def get_top_performers_api():
    """Get top performing strategies"""
    try:
        top = get_top_performers(top_n=5, min_trades=5)
        return {"top_performers": top}
    except Exception as e:
        logger.error(f"Error getting top performers: {e}")
        return {"top_performers": []}


if __name__ == "__main__":
    uvicorn.run("dashboard:app", host="0.0.0.0", port=8080, reload=True)
