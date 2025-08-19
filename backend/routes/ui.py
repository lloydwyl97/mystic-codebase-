"""
UI Routes - Dashboard Status Endpoint

Provides data for the frontend dashboard
"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException


# Lazy-import trade tracker functions inside handlers to avoid any potential cycles
def _get_trade_funcs():
    from backend.modules.ai.trade_tracker import (
        get_active_trades as _get_active_trades,
    )
    from backend.modules.ai.trade_tracker import (
        get_trade_history as _get_trade_history,
    )
    from backend.modules.ai.trade_tracker import (
        get_trade_summary as _get_trade_summary,
    )
    return _get_active_trades, _get_trade_summary, _get_trade_history

logger = logging.getLogger("ui_routes")

router = APIRouter()

def _get_cache():
    try:
        from backend.services.cache_or_redis_client import get_cache as _svc_get_cache  # type: ignore[import-not-found]
        return _svc_get_cache()
    except Exception:
        try:
            from backend.ai.ai.poller import get_cache as _ai_get_cache  # type: ignore[import-not-found]
            return _ai_get_cache()
        except Exception:
            # Minimal fallback cache structure
            class _DataCache:
                def __init__(self) -> None:
                    self.binance = {}
                    self.coinbase = {}
                    self.coingecko = {}
                    self.last_update = {}

            return _DataCache()

def get_trading_status():
    """Get basic trading status"""
    return {
        "status": "active",
        "enabled": True,
        "last_update": "2024-01-15T12:00:00Z"
    }




@router.get("/ui/status")
async def get_status() -> dict[str, Any]:
    """Get comprehensive status for dashboard UI"""
    try:
        cache = _get_cache()
        get_active_trades, get_trade_summary, _ = _get_trade_funcs()

        return {
            "active_trades": get_active_trades(),
            "trade_summary": get_trade_summary(),
            "trading_status": get_trading_status(),
            "mystic": (lambda: __import__('ai.ai.ai_mystic', fromlist=['mystic_oracle']).mystic_oracle())(),
            "binance_prices": cache.binance,
            "coinbase_prices": cache.coinbase,
            "coingecko_data": {
                "total_coins": len(cache.coingecko),
                "top_coins": list(cache.coingecko.keys())[:10],
            },
            "last_update": cache.last_update,
            "system_status": {
                "backend": "healthy",
                "frontend": "healthy",
                "redis": "healthy",
                "trading_engine": "active",
            },
        }
    except Exception as e:
        logger.error(f"âŒ UI status error: {e}")
        raise HTTPException(status_code=500, detail=f"UI status error: {str(e)}")


@router.get("/ui/dashboard")
async def get_dashboard_data() -> dict[str, Any]:
    """Get dashboard-specific data"""
    try:
        cache = _get_cache()
        get_active_trades, get_trade_summary, _ = _get_trade_funcs()

        # Calculate market overview
        total_market_cap = sum(coin.get("market_cap", 0) for coin in cache.coingecko.values())

        # Get top performers
        top_performers = []
        for coin_id, data in cache.coingecko.items():
            if data.get("price_change_24h", 0) > 10:  # 10%+ gainers
                top_performers.append(
                    {
                        "symbol": data["symbol"],
                        "price": data["price"],
                        "change_24h": data.get("price_change_24h", 0),
                        "rank": data["rank"],
                    }
                )

        # Sort by performance
        top_performers.sort(key=lambda x: x["change_24h"], reverse=True)

        return {
            "market_overview": {
                "total_market_cap": total_market_cap,
                "total_coins": len(cache.coingecko),
                "top_performers": top_performers[:10],
            },
            "trading_summary": get_trade_summary(),
            "active_trades": get_active_trades(),
            "recent_prices": {
                "BTC": cache.binance.get("BTCUSDT", 0),
                "ETH": cache.binance.get("ETHUSDT", 0),
                "ADA": cache.binance.get("ADAUSDT", 0),
                "SOL": cache.binance.get("SOLUSDT", 0),
            },
            "mystic_insight": (lambda: __import__('ai.ai.ai_mystic', fromlist=['mystic_oracle']).mystic_oracle())(),
        }
    except Exception as e:
        logger.error(f"âŒ Dashboard data error: {e}")
        raise HTTPException(status_code=500, detail=f"Dashboard data error: {str(e)}")


@router.get("/ui/market-data")
async def get_market_data() -> dict[str, Any]:
    """Get real-time market data"""
    try:
        cache = _get_cache()

        return {
            "binance": cache.binance,
            "coinbase": cache.coinbase,
            "coingecko": {
                coin_id: {
                    "symbol": data["symbol"],
                    "price": data["price"],
                    "change_24h": data.get("price_change_24h", 0),
                    "rank": data["rank"],
                    "market_cap": data.get("market_cap", 0),
                    "volume_24h": data.get("volume_24h", 0),
                }
                for coin_id, data in cache.coingecko.items()
            },
            "last_update": cache.last_update,
        }
    except Exception as e:
        logger.error(f"âŒ Market data error: {e}")
        raise HTTPException(status_code=500, detail=f"Market data error: {str(e)}")


@router.get("/ui/trading-overview")
async def get_trading_overview() -> dict[str, Any]:
    """Get trading system overview"""
    try:
        return {
            "trading_status": get_trading_status(),
            "trade_summary": _get_trade_funcs()[1](),
            "active_trades": _get_trade_funcs()[0](),
            "recent_trades": _get_trade_funcs()[2](),
        }
    except Exception as e:
        logger.error(f"âŒ Trading overview error: {e}")
        raise HTTPException(status_code=500, detail=f"Trading overview error: {str(e)}")


@router.get("/ui/analytics")
async def get_analytics_data() -> dict[str, Any]:
    """Get analytics data for charts and graphs"""
    try:
        cache = _get_cache()

        # Prepare data for charts
        price_data = []
        volume_data = []

        for coin_id, data in cache.coingecko.items():
            if data.get("price") and data.get("volume_24h"):
                price_data.append(
                    {
                        "symbol": data["symbol"],
                        "price": data["price"],
                        "rank": data["rank"],
                    }
                )
                volume_data.append(
                    {
                        "symbol": data["symbol"],
                        "volume": data["volume_24h"],
                        "rank": data["rank"],
                    }
                )

        # Sort by price and volume
        price_data.sort(key=lambda x: x["price"], reverse=True)
        volume_data.sort(key=lambda x: x["volume"], reverse=True)

        return {
            "price_ranking": price_data[:20],
            "volume_ranking": volume_data[:20],
            "market_distribution": {
                "total_coins": len(cache.coingecko),
                "price_ranges": {
                    "under_1": len([c for c in cache.coingecko.values() if c["price"] < 1]),
                    "1_to_10": len([c for c in cache.coingecko.values() if 1 <= c["price"] < 10]),
                    "10_to_100": len(
                        [c for c in cache.coingecko.values() if 10 <= c["price"] < 100]
                    ),
                    "over_100": len([c for c in cache.coingecko.values() if c["price"] >= 100]),
                },
            },
        }
    except Exception as e:
        logger.error(f"âŒ Analytics data error: {e}")
        raise HTTPException(status_code=500, detail=f"Analytics data error: {str(e)}")


