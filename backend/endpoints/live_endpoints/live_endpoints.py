# -*- coding: utf-8 -*-
"""
Live Endpoints Module - All Live Data, No Mock Data

This module provides all endpoints with live data from real APIs.
All endpoints use live data sources - no mock data.
"""

import logging
from datetime import datetime, timezone

import httpx
from fastapi import APIRouter, HTTPException

# Import service manager
try:
    from ...services.service_manager import service_manager
except ImportError:
    service_manager = None

# Import best available live data sources for live endpoints
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
    from ai_strategy_endpoints import get_trading_signals as get_live_trading_signals
except ImportError:
    get_live_trading_signals = None
try:
    from backend.endpoints.trading.portfolio_endpoints import portfolio_manager
except ImportError:
    portfolio_manager = None
try:
    from backend.ai.ai_predictions import get_ai_predictions
except ImportError:
    get_ai_predictions = None

# Import best available live data sources for remaining live endpoints
try:
    from backend.ai.social_trading import get_leaderboard
except ImportError:
    get_leaderboard = None
try:
    from bot_manager import get_bots_status
except ImportError:
    get_bots_status = None
try:
    from backend.modules.notifications.alert_manager import AlertManager

    alert_manager = AlertManager()
except ImportError:
    alert_manager = None
try:
    from backend.services.analytics_service import analytics_service
except ImportError:
    analytics_service = None
try:
    from backend.ai.trade_tracker import get_orders, get_trade_history
except ImportError:
    get_orders = None
    get_trade_history = None
try:
    from ai_strategy_endpoints import get_live_strategies
except ImportError:
    get_live_strategies = None
try:
    from backend.endpoints.core.system_endpoints import SystemMonitor

    system_monitor = SystemMonitor()
except ImportError:
    system_monitor = None

logger = logging.getLogger(__name__)
router = APIRouter()

# ============================================================================
# LIVE MARKET DATA ENDPOINTS
# ============================================================================


@router.get("/api/live/market-data")
async def get_live_market_data():
    """Get live market data from CoinGecko API"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.coingecko.com/api/v3/coins/markets",
                params={
                    "vs_currency": "usd",
                    "order": "market_cap_desc",
                    "per_page": 100,
                    "page": 1,
                    "sparkline": False,
                    "price_change_percentage": "24h,7d,30d",
                },
                timeout=10.0,
            )
            if response.status_code == 200:
                data = response.json()
                return {
                    "status": "success",
                    "data": {
                        "coins": data,
                        "total_coins": len(data),
                        "last_updated": datetime.now(timezone.utc).isoformat(),
                        "source": "coingecko",
                    },
                }
            else:
                raise HTTPException(
                    status_code=response.status_code,
                    detail="CoinGecko API error",
                )
    except Exception as e:
        logger.error(f"Error fetching live market data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching market data: {str(e)}")


@router.get("/api/live/price/{symbol}")
async def get_live_price(symbol: str):
    """Get live price for a specific cryptocurrency"""
    try:
        if not service_manager or not service_manager.market_data_service:
            logger.warning("Service manager not available, using direct API call")
            # Fallback to direct API call
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://api.coingecko.com/api/v3/simple/price",
                    params={
                        "ids": symbol.lower(),
                        "vs_currencies": "usd",
                        "include_24hr_change": True,
                        "include_market_cap": True,
                    },
                    timeout=10.0,
                )
                if response.status_code == 200:
                    data = response.json()
                    return {
                        "status": "success",
                        "data": {
                            "symbol": symbol,
                            "price_data": data,
                            "timestamp": (datetime.now(timezone.utc).isoformat()),
                            "source": "coingecko",
                        },
                    }
                else:
                    raise HTTPException(
                        status_code=response.status_code,
                        detail="CoinGecko API error",
                    )

        cached_data = await service_manager.market_data_service.get_cached_data(symbol.upper())
        logger.info(f"Cached data for {symbol}: {cached_data}")

        if cached_data and cached_data.get("price", 0) > 0:
            return {
                "status": "success",
                "data": {
                    "symbol": symbol.upper(),
                    "price": cached_data["price"],
                    "change_24h": cached_data.get("change_24h", 0),
                    "volume_24h": cached_data.get("volume_24h", 0),
                    "high_24h": cached_data.get("high_24h", 0),
                    "low_24h": cached_data.get("low_24h", 0),
                    "timestamp": cached_data.get(
                        "timestamp", datetime.now(timezone.utc).isoformat()
                    ),
                    "source": cached_data.get("api_source", "unknown"),
                },
            }
        else:
            # Fallback to direct API call if no cached data
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://api.coingecko.com/api/v3/simple/price",
                    params={
                        "ids": symbol.lower(),
                        "vs_currencies": "usd",
                        "include_24hr_change": True,
                        "include_market_cap": True,
                    },
                    timeout=10.0,
                )
                if response.status_code == 200:
                    data = response.json()
                    return {
                        "status": "success",
                        "data": {
                            "symbol": symbol,
                            "price_data": data,
                            "timestamp": (datetime.now(timezone.utc).isoformat()),
                            "source": "coingecko",
                        },
                    }
                else:
                    raise HTTPException(
                        status_code=response.status_code,
                        detail="CoinGecko API error",
                    )
    except Exception as e:
        logger.error(f"Error fetching live price for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching price: {str(e)}")


# ============================================================================
# LIVE TRADING SIGNALS
# ============================================================================


@router.get("/api/live/trading/signals")
async def get_live_trading_signals_endpoint():
    """Get live trading signals from market data analysis (live)"""
    try:
        if get_live_trading_signals:
            signals = await get_live_trading_signals()
        elif analytics_service and hasattr(analytics_service, "get_trading_signals"):
            signals = await analytics_service.get_trading_signals()
        else:
            raise HTTPException(
                status_code=503, detail="No live trading signals data source available."
            )
        return {
            "status": "success",
            "data": {
                "signals": signals,
                "total_signals": len(signals),
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "source": "market_analysis",
            },
        }
    except Exception as e:
        logger.error(f"Error getting live trading signals: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting signals: {str(e)}")


# ============================================================================
# LIVE PORTFOLIO
# ============================================================================


@router.get("/api/live/portfolio/positions")
async def get_live_portfolio_positions():
    """Get live portfolio positions from exchange APIs (live)"""
    try:
        if portfolio_manager and hasattr(portfolio_manager, "get_live_portfolio_positions"):
            positions = await portfolio_manager.get_live_portfolio_positions()
        elif portfolio_manager and hasattr(portfolio_manager, "get_live_portfolio_data"):
            data = await portfolio_manager.get_live_portfolio_data()
            positions = data.get("positions", [])
        else:
            raise HTTPException(
                status_code=503, detail="No live portfolio positions data source available."
            )
        return {
            "status": "success",
            "data": {
                "positions": positions,
                "total_positions": len(positions),
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "source": "exchange_api",
            },
        }
    except Exception as e:
        logger.error(f"Error getting live portfolio positions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting positions: {str(e)}")


@router.get("/api/live/portfolio/summary")
async def get_live_portfolio_summary():
    """Get live portfolio summary from exchange APIs (live)"""
    try:
        if portfolio_manager and hasattr(portfolio_manager, "get_portfolio_summary"):
            summary = await portfolio_manager.get_portfolio_summary()
        elif portfolio_manager and hasattr(portfolio_manager, "get_live_portfolio_data"):
            summary = await portfolio_manager.get_live_portfolio_data()
        else:
            raise HTTPException(
                status_code=503, detail="No live portfolio summary data source available."
            )
        return {
            "status": "success",
            "data": summary,
        }
    except Exception as e:
        logger.error(f"Error getting live portfolio summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting summary: {str(e)}")


# ============================================================================
# LIVE AI PREDICTIONS
# ============================================================================


@router.get("/api/live/ai/predictions")
async def get_live_ai_predictions():
    """Get live AI predictions from market data analysis (live)"""
    try:
        if get_ai_predictions:
            predictions = await get_ai_predictions()
        elif analytics_service and hasattr(analytics_service, "get_ai_predictions"):
            predictions = await analytics_service.get_ai_predictions()
        else:
            raise HTTPException(
                status_code=503, detail="No live AI predictions data source available."
            )
        return {
            "status": "success",
            "data": {
                "predictions": predictions,
                "total_predictions": len(predictions),
                "model_version": None,
            },
        }
    except Exception as e:
        logger.error(f"Error getting live AI predictions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting predictions: {str(e)}")


# ============================================================================
# LIVE SOCIAL TRADING
# ============================================================================


@router.get("/api/live/social/leaderboard")
async def get_live_social_leaderboard():
    """Get live social trading leaderboard (live)"""
    try:
        if get_leaderboard:
            leaderboard = await get_leaderboard()
        else:
            raise HTTPException(
                status_code=503, detail="No live social leaderboard data source available."
            )
        return {
            "status": "success",
            "data": {
                "leaderboard": leaderboard,
                "total_traders": len(leaderboard),
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "source": "social_platforms",
            },
        }
    except Exception as e:
        logger.error(f"Error getting live social leaderboard: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting leaderboard: {str(e)}")


# ============================================================================
# LIVE BOT STATUS
# ============================================================================


@router.get("/api/live/bots/status")
async def get_live_bot_status():
    """Get live bot status and performance (live)"""
    try:
        if get_bots_status:
            bots = await get_bots_status()
        else:
            raise HTTPException(status_code=503, detail="No live bot status data source available.")
        return {
            "status": "success",
            "data": bots,
        }
    except Exception as e:
        logger.error(f"Error getting live bot status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting bot status: {str(e)}")


# ============================================================================
# LIVE NOTIFICATIONS (removed alias: use /api/alerts/recent)
# ============================================================================


# ============================================================================
# LIVE ANALYTICS
# ============================================================================


@router.get("/api/live/analytics/performance")
async def get_live_analytics_performance():
    """Get live performance analytics (live)"""
    try:
        if analytics_service and hasattr(analytics_service, "get_performance_metrics"):
            performance_metrics = await analytics_service.get_performance_metrics()
        else:
            raise HTTPException(
                status_code=503, detail="No live analytics performance data source available."
            )
        return {
            "status": "success",
            "data": performance_metrics,
        }
    except Exception as e:
        logger.error(f"Error getting live analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting analytics: {str(e)}")


# ============================================================================
# LIVE ORDERS & TRADE HISTORY
# ============================================================================


@router.get("/api/live/orders")
async def get_live_orders():
    """Get live orders from exchange APIs (live)"""
    try:
        if get_orders:
            orders = await get_orders()
        else:
            raise HTTPException(status_code=503, detail="No live orders data source available.")
        return {
            "status": "success",
            "data": orders,
        }
    except Exception as e:
        logger.error(f"Error getting live orders: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting orders: {str(e)}")


@router.get("/api/live/trades/history")
async def get_live_trade_history():
    """Get live trade history from exchange APIs (live)"""
    try:
        if get_trade_history:
            trades = await get_trade_history()
        else:
            raise HTTPException(
                status_code=503, detail="No live trade history data source available."
            )
        return {
            "status": "success",
            "data": trades,
        }
    except Exception as e:
        logger.error(f"Error getting live trade history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting trade history: {str(e)}")


# ============================================================================
# LIVE STRATEGY ENGINE
# ============================================================================


@router.get("/api/live/strategies")
async def get_live_strategies_endpoint():
    """Get live strategy performance (live)"""
    try:
        if get_live_strategies:
            strategies = await get_live_strategies()
        else:
            raise HTTPException(status_code=503, detail="No live strategies data source available.")
        return {
            "status": "success",
            "data": strategies,
        }
    except Exception as e:
        logger.error(f"Error getting live strategies: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting strategies: {str(e)}")


# ============================================================================
# SYSTEM STATUS & SOURCES
# ============================================================================


@router.get("/api/live/system/status")
async def get_live_system_status():
    """Get live system status and health (live)"""
    try:
        if system_monitor:
            status = await system_monitor.get_system_status()
        else:
            raise HTTPException(
                status_code=503, detail="No live system status data source available."
            )
        return {
            "status": "success",
            "data": status,
        }
    except Exception as e:
        logger.error(f"Error getting live system status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting system status: {str(e)}")


@router.get("/api/live/sources")
async def get_live_data_sources():
    """Get available live data sources and their status (live)"""
    try:
        sources = []
        if service_manager:
            # Get live source status from service manager
            sources = await service_manager.get_data_source_status()
        elif system_monitor:
            # Get live source status from system monitor
            sources = await system_monitor.get_data_source_status()
        else:
            raise HTTPException(status_code=503, detail="No live data sources status available.")
        return {
            "status": "success",
            "data": {
                "sources": sources,
                "total_sources": len(sources),
                "active_sources": len([s for s in sources if s.get("status") == "active"]),
                "last_updated": datetime.now(timezone.utc).isoformat(),
            },
        }
    except Exception as e:
        logger.error(f"Error getting live data sources: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting data sources: {str(e)}")



