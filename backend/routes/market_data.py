import logging
import os
import sys
from typing import Any

from fastapi import APIRouter, HTTPException

# Add the parent directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.services.market_data import MarketDataService
from backend.services.notification import get_notification_service
from backend.services.service_manager import service_manager

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize market data service and notification service
market_data_service = MarketDataService()
notification_service = get_notification_service(
    None
)  # Will be properly initialized with redis_client later


@router.get("/api/markets")
async def get_all_markets() -> dict[str, Any]:
    """Get all available markets (bundled, deduped, cached, rotated)"""
    try:
        # Check service manager health
        service_health = service_manager.get_health_status()

        # Get markets data using the correct method
        markets_data: dict[str, Any] = await market_data_service.get_markets()

        # Extract markets from the response
        markets: dict[str, dict[str, Any]] = markets_data.get("markets", {})

        # Bundle and dedupe by symbol
        bundled: dict[str, dict[str, Any]] = {}
        for symbol, market_data in markets.items():
            if symbol not in bundled:
                bundled[symbol] = {
                    "id": symbol,
                    "name": symbol,
                    "symbol": symbol,
                    "price": market_data.get("price", 0.0),
                    "change_24h": market_data.get("change_24h", 0.0),
                }

        # Get API health status for monitoring
        api_health: dict[str, bool] = markets_data.get("api_health", {})

        return {
            "markets": list(bundled.values()),
            "source": "market_data_service",
            "timestamp": markets_data.get("timestamp"),
            "api_health": api_health,
            "service_health": service_health,
            "status": "success",
        }
    except Exception as e:
        logger.error(f"Error in /api/markets: {str(e)}")
        # Send notification for critical errors
        try:
            await notification_service.send_notification(
                "Market Data Error",
                f"Failed to fetch markets: {str(e)}",
                "error",
            )
        except (ConnectionError, TimeoutError, ValueError, Exception) as notify_error:
            logger.warning(f"Failed to send notification: {notify_error}")
            pass  # Don't fail if notification fails
        raise HTTPException(status_code=500, detail=f"Failed to fetch markets: {str(e)}")


@router.get("/api/market/{symbol}")
async def get_market(symbol: str) -> dict[str, Any]:
    """Get the latest price and change for a specific symbol (cached, rotated)"""
    try:
        # Get market data using the correct method
        market_data: dict[str, Any] | None = await market_data_service.get_market_data(symbol)
        if not market_data:
            raise HTTPException(status_code=404, detail=f"Market data for {symbol} not found")

        # Get cache info for monitoring
        cache_info: dict[str, bool | int | Any] = {
            "hit": symbol in market_data_service.cache,
            "duration": 300,  # Default cache duration
            "entries": len(market_data_service.cache),
        }

        return {
            "id": symbol,
            "price": market_data.get("price", 0.0),
            "change_24h": market_data.get("change_24h", 0.0),
            "timestamp": market_data.get("timestamp"),
            "source": market_data.get("api_source", "unknown"),
            "cache_info": cache_info,
            "status": "success",
        }
    except Exception as e:
        logger.error(f"Error in /api/market/{symbol}: {str(e)}")
        # Send notification for critical errors
        try:
            await notification_service.send_notification(
                "Market Data Error",
                f"Failed to fetch market data for {symbol}: {str(e)}",
                "error",
            )
        except (ConnectionError, TimeoutError, ValueError, Exception) as notify_error:
            logger.warning(f"Failed to send notification: {notify_error}")
            pass  # Don't fail if notification fails
        raise HTTPException(status_code=500, detail=f"Failed to fetch market data: {str(e)}")


@router.get("/api/market-data/status")
async def get_market_data_status() -> dict[str, Any]:
    """Get the status of all market data APIs and rate limits"""
    try:
        # Get markets data to access API health
        markets_data: dict[str, Any] = await market_data_service.get_markets()

        # Get API health status
        api_health: dict[str, dict[str, Any]] = markets_data.get("api_health", {})

        # Get cache stats
        cache_stats: dict[str, int | Any] = {
            "entries": len(market_data_service.cache),
            "duration_seconds": 300,  # Default cache duration
        }

        return {
            "current_api": "market_data_service",
            "api_health": api_health,
            "cache": cache_stats,
            "last_update": markets_data.get("timestamp"),
            "status": "success",
        }
    except Exception as e:
        logger.error(f"Error in /api/market-data/status: {str(e)}")
        # Send notification for critical errors
        try:
            await notification_service.send_notification(
                "Market Data Error",
                f"Failed to fetch market data status: {str(e)}",
                "error",
            )
        except (ConnectionError, TimeoutError, ValueError, Exception) as notify_error:
            logger.warning(f"Failed to send notification: {notify_error}")
            pass  # Don't fail if notification fails
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch market data status: {str(e)}",
        )


