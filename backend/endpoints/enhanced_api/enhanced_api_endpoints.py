"""
Enhanced API Endpoints for Mystic Trader
Comprehensive endpoints for backtest analysis, live trading, market sentiment, and risk management
"""

import asyncio
import json
import os
from datetime import datetime
from typing import Any, Dict

import numpy as np
import pandas as pd
import redis
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

router = APIRouter(prefix="/api", tags=["Enhanced API"])

# Redis connection
try:
    redis_client = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        db=int(os.getenv("REDIS_DB", 0)),
        decode_responses=True,
    )
    redis_client.ping()
except Exception as e:
    print(f"Redis connection failed: {e}")
    redis_client = None


# ===== TECHNICAL ANALYSIS FUNCTIONS (NO TA-LIB) =====
def calculate_sma(data: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average"""
    return data.rolling(window=period).mean()


def calculate_ema(data: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average"""
    return data.ewm(span=period).mean()


def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    """Calculate MACD (Moving Average Convergence Divergence)"""
    ema_fast = data.ewm(span=fast).mean()
    ema_slow = data.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_bollinger_bands(data: pd.Series, period: int = 20, std_dev: float = 2) -> tuple:
    """Calculate Bollinger Bands"""
    sma = data.rolling(window=period).mean()
    std = data.rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, sma, lower_band


# ===== BACKTEST ENDPOINTS =====
# Import best available live data sources
try:
    from backend.ai_mutation.backtester import get_backtest_results as get_live_backtest_results
except ImportError:
    get_live_backtest_results = None
try:
    from strategy_backups import get_backtest_data as get_backup_backtest_data
except ImportError:
    get_backup_backtest_data = None
try:
    from backend.ai.trade_tracker import get_trade_history as get_live_trade_history
except ImportError:
    get_live_trade_history = None
try:
    from ai_strategy_executor_service import AIStrategyExecutorService

    ai_strategy_executor_service = AIStrategyExecutorService()
except ImportError:
    ai_strategy_executor_service = None


@router.get("/backtest/results")
async def get_backtest_results():
    """Get historical backtest results (live)"""
    try:
        results = []
        if get_live_backtest_results:
            results = get_live_backtest_results()
        elif get_backup_backtest_data:
            results = get_backup_backtest_data()
        elif get_live_trade_history:
            results = get_live_trade_history()
        return {"backtests": results, "total": len(results)}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching backtest results: {str(e)}",
        )


@router.post("/backtest/run")
async def run_backtest(
    strategy_type: str,
    symbol: str,
    start_date: str,
    end_date: str,
    parameters: Dict[str, Any] = {},
):
    """Run a new backtest (live)"""
    try:
        if ai_strategy_executor_service:
            request_data = {
                "strategy_type": strategy_type,
                "symbol": symbol,
                "start_date": start_date,
                "end_date": end_date,
                "parameters": parameters,
            }
            result = await ai_strategy_executor_service.execute_strategy_request(request_data)
            return result
        else:
            raise HTTPException(
                status_code=503, detail="No live backtest execution service available."
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running backtest: {str(e)}")


def simulate_trading_strategy(data: pd.DataFrame, strategy_type: str) -> pd.DataFrame:
    """Simulate trading strategy and return signals"""
    try:
        signals = pd.DataFrame(index=data.index)
        signals["Close"] = data["Close"]

        if strategy_type == "sma_crossover":
            signals["SMA_20"] = calculate_sma(data["Close"], 20)
            signals["SMA_50"] = calculate_sma(data["Close"], 50)
            signals["Signal"] = 0
            signals.loc[signals["SMA_20"] > signals["SMA_50"], "Signal"] = 1
            signals["Position"] = signals["Signal"].diff()

        elif strategy_type == "rsi_strategy":
            signals["RSI"] = calculate_rsi(data["Close"], 14)
            signals["Signal"] = 0
            signals.loc[signals["RSI"] < 30, "Signal"] = 1
            signals.loc[signals["RSI"] > 70, "Signal"] = -1
            signals["Position"] = signals["Signal"].diff()

        elif strategy_type == "macd_strategy":
            macd_line, signal_line, histogram = calculate_macd(data["Close"])
            signals["MACD"] = macd_line
            signals["Signal_Line"] = signal_line
            signals["Histogram"] = histogram
            signals["Signal"] = 0
            signals.loc[signals["MACD"] > signals["Signal_Line"], "Signal"] = 1
            signals["Position"] = signals["Signal"].diff()

        elif strategy_type == "bollinger_bands":
            upper, middle, lower = calculate_bollinger_bands(data["Close"])
            signals["Upper_Band"] = upper
            signals["Middle_Band"] = middle
            signals["Lower_Band"] = lower
            signals["Signal"] = 0
            signals.loc[data["Close"] < lower, "Signal"] = 1
            signals.loc[data["Close"] > upper, "Signal"] = -1
            signals["Position"] = signals["Signal"].diff()

        signals["Returns"] = data["Close"].pct_change()
        signals["Strategy_Returns"] = signals["Position"].shift(1) * signals["Returns"]
        signals["Cumulative_Returns"] = (1 + signals["Strategy_Returns"]).cumprod()

        return signals

    except Exception as e:
        print(f"Error simulating trading strategy: {e}")
        return pd.DataFrame()


def calculate_backtest_metrics(returns: pd.Series) -> Dict[str, float]:
    """Calculate comprehensive backtest metrics"""
    try:
        metrics = {}

        # Basic metrics
        metrics["total_return"] = (1 + returns).prod() - 1
        metrics["annual_return"] = metrics["total_return"] * (252 / len(returns))
        metrics["volatility"] = returns.std() * np.sqrt(252)
        metrics["sharpe_ratio"] = (
            metrics["annual_return"] / metrics["volatility"] if metrics["volatility"] > 0 else 0
        )

        # Drawdown calculation
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        metrics["max_drawdown"] = drawdown.min()

        # Win/Loss metrics
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        metrics["win_rate"] = len(positive_returns) / len(returns) if len(returns) > 0 else 0
        metrics["avg_win"] = positive_returns.mean() if len(positive_returns) > 0 else 0
        metrics["avg_loss"] = negative_returns.mean() if len(negative_returns) > 0 else 0
        metrics["profit_factor"] = (
            abs(positive_returns.sum() / negative_returns.sum())
            if negative_returns.sum() != 0
            else float("inf")
        )

        return metrics
    except Exception as e:
        print(f"Error calculating backtest metrics: {e}")
        return {}


# ===== LIVE TRADING ENDPOINTS =====
# Import best available live data sources for analytics endpoints
try:
    from backend.ai.trade_tracker import get_active_trades, get_trade_history
except ImportError:
    get_active_trades = None
    get_trade_history = None
try:
    from backend.services.analytics_service import analytics_service
except ImportError:
    analytics_service = None
try:
    from ai_strategy_endpoints import get_strategy_performance as get_live_strategy_performance
except ImportError:
    get_live_strategy_performance = None
try:
    from backend.endpoints.market.market_data_endpoints import market_data_service
except ImportError:
    market_data_service = None
try:
    from backend.endpoints.trading.portfolio_endpoints import portfolio_manager
except ImportError:
    portfolio_manager = None
try:
    from backend.ai.persistent_cache import get_persistent_cache
except ImportError:
    get_persistent_cache = None


@router.get("/trading/live-trades")
async def get_live_trades():
    """Get live trades (live)"""
    try:
        if get_active_trades:
            trades = get_active_trades()
        elif get_trade_history:
            trades = get_trade_history()
        else:
            raise HTTPException(status_code=503, detail="No live trade data source available.")
        return {"live_trades": trades, "total": len(trades)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching live trades: {str(e)}")


# ===== MARKET SENTIMENT ENDPOINTS =====
@router.get("/analytics/sentiment")
async def get_market_sentiment():
    """Get market sentiment analysis (live)"""
    try:
        if analytics_service and hasattr(analytics_service, "get_market_sentiment"):
            sentiment_data = await analytics_service.get_market_sentiment()
        else:
            raise HTTPException(
                status_code=503, detail="No live market sentiment data source available."
            )
        return sentiment_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching market sentiment: {str(e)}")


# ===== RISK MANAGEMENT ENDPOINTS =====
@router.get("/risk/metrics")
async def get_risk_metrics():
    """Get risk management metrics (live)"""
    try:
        if analytics_service and hasattr(analytics_service, "get_risk_metrics"):
            risk_data = await analytics_service.get_risk_metrics()
        elif portfolio_manager and hasattr(portfolio_manager, "get_risk_metrics"):
            risk_data = await portfolio_manager.get_risk_metrics()
        else:
            raise HTTPException(
                status_code=503, detail="No live risk metrics data source available."
            )
        return risk_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching risk metrics: {str(e)}")


# ===== STRATEGY PERFORMANCE ENDPOINTS =====
@router.get("/strategies/performance")
async def get_strategy_performance():
    """Get strategy performance data (live)"""
    try:
        if get_live_strategy_performance:
            strategies = await get_live_strategy_performance()
        else:
            raise HTTPException(
                status_code=503, detail="No live strategy performance data source available."
            )
        return {"strategies": strategies, "total": len(strategies)}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error fetching strategy performance: {str(e)}"
        )


# ===== ENHANCED MARKET DATA ENDPOINTS =====
@router.get("/markets/enhanced")
async def get_enhanced_market_data():
    """Get enhanced market data with technical indicators (live)"""
    try:
        if market_data_service and hasattr(market_data_service, "get_live_data"):
            markets = await market_data_service.get_live_data()
        elif get_persistent_cache:
            cache = get_persistent_cache()
            markets = cache.get_latest_prices()
        else:
            raise HTTPException(status_code=503, detail="No live market data source available.")
        return {"markets": markets, "total": len(markets)}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error fetching enhanced market data: {str(e)}"
        )


# ===== PORTFOLIO ANALYTICS ENDPOINTS =====
@router.get("/portfolio/analytics")
async def get_portfolio_analytics():
    """Get comprehensive portfolio analytics (live)"""
    try:
        if portfolio_manager and hasattr(portfolio_manager, "get_performance_metrics"):
            analytics = await portfolio_manager.get_performance_metrics()
        elif portfolio_manager and hasattr(portfolio_manager, "get_live_portfolio_data"):
            analytics = await portfolio_manager.get_live_portfolio_data()
        else:
            raise HTTPException(
                status_code=503, detail="No live portfolio analytics data source available."
            )
        return analytics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching portfolio analytics: {str(e)}")


# ===== SYSTEM HEALTH ENDPOINTS =====
# Import best available live data sources for system health and export
try:
    import psutil
except ImportError:
    psutil = None
try:
    from backend.services.health_monitor_service import HealthMonitorService

    health_monitor = HealthMonitorService()
except ImportError:
    health_monitor = None
try:
    from backend.services.performance_monitor import PerformanceMonitor

    performance_monitor = PerformanceMonitor()
except ImportError:
    performance_monitor = None
try:
    from backend.endpoints.core.system_endpoints import SystemMonitor

    system_monitor = SystemMonitor()
except ImportError:
    system_monitor = None
try:
    from backend.ai_mutation.backtester import export_backtest_results
except ImportError:
    export_backtest_results = None


@router.get("/system/health")
async def get_system_health():
    """Get comprehensive system health status (live)"""
    try:
        health_data = {}
        # Prefer HealthMonitorService if available
        if health_monitor:
            health_data = await health_monitor.get_system_health()
        elif psutil:
            health_data = {
                "overall_status": "HEALTHY" if psutil.cpu_percent() < 90 else "UNHEALTHY",
                "services": {
                    "api_server": "HEALTHY",
                    "trading_engine": "HEALTHY",
                    "data_feed": "HEALTHY",
                    "redis_cache": (
                        "HEALTHY" if redis_client and redis_client.ping() else "UNHEALTHY"
                    ),
                    "database": "HEALTHY",
                },
                "performance": {
                    "cpu_usage": psutil.cpu_percent(),
                    "memory_usage": psutil.virtual_memory().percent,
                    "disk_usage": psutil.disk_usage("/").percent,
                    "network_latency": None,
                },
                "alerts": [],
                "last_updated": datetime.now().isoformat(),
            }
        else:
            raise HTTPException(
                status_code=503, detail="No live system health data source available."
            )
        return health_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching system health: {str(e)}")


@router.get("/export/backtest/{backtest_id}")
async def export_backtest(backtest_id: str, format: str = "csv"):
    """Export backtest results (live)"""
    try:
        if export_backtest_results:
            export_data = await export_backtest_results(backtest_id, format)
            return export_data
        else:
            raise HTTPException(
                status_code=503, detail="No live backtest export service available."
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exporting backtest: {str(e)}")


@router.websocket("/ws/live-data")
async def websocket_live_data(websocket):
    """WebSocket endpoint for real-time data (live)"""
    try:
        await websocket.accept()
        while True:
            if market_data_service and hasattr(market_data_service, "get_live_data"):
                markets = await market_data_service.get_live_data()
            elif get_persistent_cache:
                cache = get_persistent_cache()
                markets = cache.get_latest_prices()
            else:
                await websocket.send_text(
                    json.dumps({"error": "No live market data source available."})
                )
                await asyncio.sleep(5)
                continue
            live_data = {
                "type": "live_data",
                "timestamp": datetime.now().isoformat(),
                "markets": markets,
            }
            await websocket.send_text(json.dumps(live_data))
            await asyncio.sleep(5)
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()


@router.get("/health")
async def health_check():
    """Health check endpoint (live)"""
    try:
        health = {}
        if health_monitor:
            health = await health_monitor.get_system_health()
        elif psutil:
            health = {
                "status": "healthy" if psutil.cpu_percent() < 90 else "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "version": "2.0.0",
                "services": {
                    "api": "healthy",
                    "redis": ("healthy" if redis_client and redis_client.ping() else "unhealthy"),
                },
            }
        else:
            raise HTTPException(
                status_code=503, detail="No live health check data source available."
            )
        return health
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in health check: {str(e)}")



