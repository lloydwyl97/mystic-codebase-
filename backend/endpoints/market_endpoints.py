"""
Market Data Endpoints

Handles all market-related API endpoints including prices, indicators, and cosmic signals.
"""

import logging
import time
from typing import Any

from fastapi import APIRouter, HTTPException

logger = logging.getLogger("market_endpoints")

# Global service references (will be set by main.py)
price_fetcher = None
indicators_fetcher = None
cosmic_fetcher = None


def set_services(pf, ind_f, cf):
    """Set service references from main.py"""
    global price_fetcher, indicators_fetcher, cosmic_fetcher
    price_fetcher = pf
    indicators_fetcher = ind_f
    cosmic_fetcher = cf


router = APIRouter()


@router.get("/prices")
async def get_market_prices() -> dict[str, Any]:
    """Get current market prices for all supported coins"""
    try:
        if price_fetcher and hasattr(price_fetcher, "get_market_data"):
            data = await price_fetcher.get_market_data()
            return {"market_data": data}
        else:
            raise HTTPException(status_code=503, detail="Price fetcher not available")
    except Exception as e:
        logger.error(f"Error fetching market prices: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch market prices")


@router.get("/indicators/{symbol}")
async def get_technical_indicators(symbol: str) -> dict[str, Any]:
    """Get technical indicators for a specific symbol"""
    try:
        if indicators_fetcher and hasattr(indicators_fetcher, "get_indicators"):
            indicators = await indicators_fetcher.get_indicators(symbol)
            return {
                "symbol": symbol,
                "indicators": indicators,
                "timestamp": time.time(),
            }
        else:
            raise HTTPException(status_code=503, detail="Indicators fetcher not available")
    except Exception as e:
        logger.error(f"Error fetching indicators for {symbol}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch indicators")


@router.get("/cosmic-signals")
async def get_cosmic_signals() -> dict[str, Any]:
    """Get cosmic analysis signals"""
    try:
        if cosmic_fetcher and hasattr(cosmic_fetcher, "get_global_signals"):
            signals = await cosmic_fetcher.get_global_signals()
            return {"signals": signals, "timestamp": time.time()}
        else:
            raise HTTPException(status_code=503, detail="Cosmic fetcher not available")
    except Exception as e:
        logger.error(f"Error fetching cosmic signals: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch cosmic signals")


@router.get("/supported-coins")
async def get_supported_coins() -> dict[str, Any]:
    """Get list of supported coins"""
    try:
        if price_fetcher and hasattr(price_fetcher, "get_supported_coins"):
            coins = await price_fetcher.get_supported_coins()
            return {"coins": coins, "count": len(coins) if coins else 0}
        else:
            raise HTTPException(status_code=503, detail="Price fetcher not available")
    except Exception as e:
        logger.error(f"Error fetching supported coins: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch supported coins")


@router.get("/coinstate")
async def get_coin_state() -> dict[str, Any]:
    """Get current state of all coins"""
    try:
        if price_fetcher and hasattr(price_fetcher, "get_all_prices"):
            prices = await price_fetcher.get_all_prices()
            if prices and isinstance(prices, dict):
                return {
                    "coins": [
                        {
                            "symbol": str(coin),
                            "price": (float(price) if price is not None else 0.0),
                            "status": "active",
                        }
                        for coin, price in prices.items()
                    ],
                    "timestamp": time.time(),
                }
            else:
                return {"coins": [], "timestamp": time.time()}
        else:
            raise HTTPException(status_code=503, detail="Price fetcher not available")
    except Exception as e:
        logger.error(f"Error fetching coin state: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch coin state")



