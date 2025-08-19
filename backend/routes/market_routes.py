"""
Market Routes

API endpoints for market data and signals.
"""

import asyncio
import logging
import random
from datetime import timezone, datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

from backend.services.market_data import MarketDataService
from backend.services.notification import get_notification_service
from backend.services.service_manager import service_manager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["market"])

# Initialize services for health checks and notifications
market_data_service = MarketDataService()
notification_service = get_notification_service(None)

# Simulated live data
live_market_data: Dict[str, Dict[str, Any]] = {
    "BTC/USD": {"price": 45000.0, "change": 2.5, "volume": 1000000},
    "ETH/USD": {"price": 3200.0, "change": -1.2, "volume": 800000},
    "ADA/USD": {"price": 1.20, "change": 5.8, "volume": 500000},
    "SOL/USD": {"price": 150.0, "change": 8.3, "volume": 300000},
    "DOT/USD": {"price": 25.0, "change": -0.5, "volume": 200000},
}

live_portfolio: Dict[str, Any] = {
    "total_value": 50000.0,
    "total_change": 12.5,
    "positions": [
        {"symbol": "BTC/USD", "amount": 0.5, "value": 22500.0, "change": 2.5},
        {"symbol": "ETH/USD", "amount": 8.0, "value": 25600.0, "change": -1.2},
        {
            "symbol": "ADA/USD",
            "amount": 2000.0,
            "value": 2400.0,
            "change": 5.8,
        },
    ],
}

live_trades: List[Dict[str, Any]] = []


@router.get("/coins/all")
async def get_all_coins() -> Dict[str, Any]:
    """Get all available coins"""
    try:
        if not service_manager.market_data_service:
            raise HTTPException(status_code=503, detail="Market data service not available")

        cached_data = {}
        try:
            cached_data = await service_manager.market_data_service.get_all_cached_data()
        except Exception as e:
            logger.warning(f"Failed to get cached data: {e}")

        coin_config = {}
        # Safely access coin_config attribute
        try:
            coin_config = getattr(service_manager.market_data_service, "coin_config", {})
        except Exception as e:
            logger.warning(f"Failed to get coin config: {str(e)}")
            coin_config = {}

        # Check service health
        service_health = hasattr(market_data_service, "get_all_cached_data")

        return {
            "coins": cached_data,
            "coin_config": coin_config,
            "total_coins": len(cached_data),
            "service_health": service_health,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"Error getting all coins: {e}")
        # Send notification for critical errors
        try:
            await notification_service.send_notification(
                "Market Data Error",
                f"Failed to get all coins: {str(e)}",
                "error",
            )
        except Exception as notification_error:
            logger.error(f"Failed to send notification: {notification_error}")
            pass
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/markets")
async def get_markets() -> Dict[str, Any]:
    """Get market overview"""
    try:
        # Use live market data service instead of service_manager
        from backend.services.live_market_data import live_market_data_service

        # Get market data from live service
        market_data = await live_market_data_service.get_market_data("usd", 50)
        coins = market_data.get("coins", [])

        # Format markets data
        markets = {}
        for coin in coins:
            symbol = coin.get("symbol", "").upper()
            markets[symbol] = {
                "symbol": symbol,
                "name": coin.get("name", ""),
                "price": coin.get("current_price", 0),
                "change_24h": coin.get("price_change_percentage_24h", 0),
                "volume": coin.get("total_volume", 0),
                "market_cap": coin.get("market_cap", 0),
                "rank": coin.get("market_cap_rank", 0),
            }

        return {
            "markets": markets,
            "count": len(markets),
            "service_health": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "live_market_data_service",
        }
    except Exception as e:
        logger.error(f"Error getting markets: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting markets: {str(e)}")


@router.get("/markets/test")
async def test_markets() -> Dict[str, Any]:
    """Test endpoint for markets"""
    return {
        "message": "Markets test endpoint working",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "success",
    }


@router.get("/market/{symbol}")
async def get_market(symbol: str) -> Dict[str, Any]:
    """Get specific market data"""
    try:
        if not service_manager.market_data_service:
            raise HTTPException(status_code=503, detail="Market data service not available")

        market_data = await service_manager.market_data_service.get_market_data(symbol)

        # Check service health
        service_health = hasattr(market_data_service, "get_market_data")

        return {
            "symbol": symbol,
            "data": market_data,
            "service_health": service_health,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"Error getting market data for {symbol}: {e}")
        # Send notification for critical errors
        try:
            await notification_service.send_notification(
                "Market Data Error",
                f"Failed to get market data for {symbol}: {str(e)}",
                "error",
            )
        except Exception as notification_error:
            logger.error(f"Failed to send notification: {notification_error}")
            pass
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/signals/live")
async def get_live_signals(
    symbol: str = Query("all", description="Asset symbol to analyze")
) -> Dict[str, Any]:
    """Get live trading signals"""
    try:
        if not service_manager.market_data_service:
            raise HTTPException(status_code=503, detail="Market data service not available")

        signals = await service_manager.market_data_service.get_live_signals(symbol)

        # Check service health
        service_health = hasattr(market_data_service, "get_live_signals")

        return {
            "signals": signals,
            "symbol": symbol,
            "service_health": service_health,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"Error getting live signals: {e}")
        # Send notification for critical errors
        try:
            await notification_service.send_notification(
                "Market Data Error",
                f"Failed to get live signals: {str(e)}",
                "error",
            )
        except Exception as notification_error:
            logger.error(f"Failed to send notification: {notification_error}")
            pass
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/signals/unified")
async def get_unified_signals(symbol: Optional[str] = None) -> Dict[str, Any]:
    """Get unified signals"""
    try:
        if not service_manager.unified_signal_manager:
            raise HTTPException(status_code=503, detail="Signal system not available")

        signals = await service_manager.unified_signal_manager.get_unified_signals(symbol)
        return {
            "unified_signals": signals,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"Error getting unified signals: {e}")
        # Send notification for critical errors
        try:
            await notification_service.send_notification(
                "Market Data Error",
                f"Failed to get unified signals: {str(e)}",
                "error",
            )
        except Exception as notification_error:
            logger.error(f"Failed to send notification: {notification_error}")
            pass
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/signals/summary")
async def get_signal_summary() -> Dict[str, Any]:
    """Get signal summary"""
    try:
        if not service_manager.unified_signal_manager:
            raise HTTPException(status_code=503, detail="Signal system not available")

        summary = await service_manager.unified_signal_manager.get_signal_summary()
        return summary
    except Exception as e:
        logger.error(f"Error getting signal summary: {e}")
        # Send notification for critical errors
        try:
            await notification_service.send_notification(
                "Market Data Error",
                f"Failed to get signal summary: {str(e)}",
                "error",
            )
        except Exception as notification_error:
            logger.error(f"Failed to send notification: {notification_error}")
            pass
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/live/market-data")
async def get_live_market_data() -> Dict[str, Any]:
    """Get live market data for testing"""
    # Simulate price updates
    for symbol in live_market_data:
        current = live_market_data[symbol]
        # Random price movement
        change = random.uniform(-5, 5)
        current["price"] = round(current["price"] * (1 + change / 100), 2)
        current["change"] = round(change, 2)
        current["volume"] = int(current["volume"] * random.uniform(0.8, 1.2))
        current["timestamp"] = datetime.now().isoformat()

    return {
        "status": "success",
        "data": live_market_data,
        "timestamp": datetime.now().isoformat(),
    }


@router.get("/live/portfolio")
async def get_live_portfolio() -> Dict[str, Any]:
    """Get live portfolio data for testing"""
    # Update portfolio based on market data
    total_value = 0.0
    for position in live_portfolio["positions"]:
        symbol = position["symbol"]
        if symbol in live_market_data:
            market_price = live_market_data[symbol]["price"]
            position["value"] = round(position["amount"] * market_price, 2)
            position["change"] = live_market_data[symbol]["change"]
            total_value += position["value"]

    live_portfolio["total_value"] = round(total_value, 2)
    live_portfolio["total_change"] = round((total_value - 50000.0) / 50000.0 * 100, 2)

    return {
        "status": "success",
        "data": live_portfolio,
        "timestamp": datetime.now().isoformat(),
    }


@router.get("/live/trades")
async def get_live_trades() -> Dict[str, Any]:
    """Get live trading activity for testing"""
    # Simulate new trades
    if random.random() < 0.3:  # 30% chance of new trade
        symbols = list(live_market_data.keys())
        symbol = random.choice(symbols)
        trade_type = random.choice(["buy", "sell"])
        amount = round(random.uniform(0.1, 2.0), 4)
        price = live_market_data[symbol]["price"]

        new_trade: Dict[str, Any] = {
            "id": len(live_trades) + 1,
            "symbol": symbol,
            "type": trade_type,
            "amount": amount,
            "price": price,
            "value": round(amount * price, 2),
            "timestamp": datetime.now().isoformat(),
        }

        live_trades.append(new_trade)

        # Keep only last 50 trades
        if len(live_trades) > 50:
            live_trades.pop(0)

    return {
        "status": "success",
        "data": live_trades[-10:],  # Return last 10 trades
        "timestamp": datetime.now().isoformat(),
    }


@router.get("/live/system-status")
async def get_system_status() -> Dict[str, Any]:
    """Get live system status for testing"""
    return {
        "status": "success",
        "data": {
            "system": "online",
            "websocket": "connected",
            "database": "healthy",
            "api_latency": random.randint(10, 100),
            "active_connections": random.randint(5, 50),
            "uptime": "2 days, 14 hours, 32 minutes",
            "last_update": datetime.now().isoformat(),
        },
        "timestamp": datetime.now().isoformat(),
    }


@router.get("/market-data")
async def get_market_data() -> Dict[str, Any]:
    """Get market data endpoint"""
    try:
        # Import persistent cache
        from backend.modules.ai.persistent_cache import get_persistent_cache

        cache = get_persistent_cache()
        market_data = {}

        # Process Binance data
        binance_data = cache.get_binance()
        for symbol, price in binance_data.items():
            base_symbol = symbol.replace("USDT", "")
            market_data[base_symbol] = {
                "symbol": base_symbol,
                "price": float(price),
                "volume": 1000.0,  # Default volume
                "change_24h": 2.5,  # Default change
                "high_24h": float(price) * 1.02,  # Estimate high
                "low_24h": float(price) * 0.98,  # Estimate low
                "timestamp": datetime.now().timestamp(),
                "exchange": "binance",
            }

        # Process Coinbase data (if not already in market_data)
        coinbase_data = cache.get_coinbase()
        for symbol, price in coinbase_data.items():
            base_symbol = symbol.replace("-USD", "")
            if base_symbol not in market_data:
                market_data[base_symbol] = {
                    "symbol": base_symbol,
                    "price": float(price),
                    "volume": 800.0,  # Default volume
                    "change_24h": 1.8,  # Default change
                    "high_24h": float(price) * 1.02,  # Estimate high
                    "low_24h": float(price) * 0.98,  # Estimate low
                    "timestamp": datetime.now().timestamp(),
                    "exchange": "coinbase",
                }

        # Process CoinGecko data (if not already in market_data)
        coingecko_data = cache.get_coingecko()
        for coin_id, coin_data in coingecko_data.items():
            symbol = coin_data.get("symbol", "").upper()
            if symbol and symbol not in market_data:
                market_data[symbol] = {
                    "symbol": symbol,
                    "price": float(coin_data.get("price", 0)),
                    "volume": float(coin_data.get("volume_24h", 0)),
                    "change_24h": float(coin_data.get("price_change_24h", 0)),
                    "high_24h": (float(coin_data.get("price", 0)) * 1.02),  # Estimate high
                    "low_24h": (float(coin_data.get("price", 0)) * 0.98),  # Estimate low
                    "timestamp": datetime.now().timestamp(),
                    "exchange": "coingecko",
                }

        # If no data in cache, raise error
        if not market_data:
            logger.warning("No data in persistent cache")
            raise HTTPException(status_code=503, detail="Market data not available")

        return market_data

    except Exception as e:
        logger.error(f"Error getting market data from persistent cache: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting market data: {str(e)}")


@router.get("/market-updates")
async def get_market_updates() -> Dict[str, Any]:
    """Get market updates endpoint"""
    updates = []
    for symbol, data in live_market_data.items():
        updates.append(
            {
                "symbol": symbol,
                "price": data["price"],
                "change": data["change"],
                "volume": data["volume"],
                "timestamp": data.get("timestamp", datetime.now().isoformat()),
            }
        )

    return {
        "status": "success",
        "data": updates,
        "timestamp": datetime.now().isoformat(),
    }


@router.get("/portfolio")
async def get_portfolio() -> Dict[str, Any]:
    """Get portfolio data endpoint"""
    return {
        "status": "success",
        "data": live_portfolio,
        "timestamp": datetime.now().isoformat(),
    }


@router.get("/auto-trades")
async def get_auto_trades() -> Dict[str, Any]:
    """Get auto trades endpoint"""
    auto_trades = [
        {
            "symbol": "BTC/USD",
            "strategy": "RSI + MACD",
            "status": "active",
            "profit": 2.5,
            "lastTrade": "2024-02-20T10:30:00Z",
        },
        {
            "symbol": "ETH/USD",
            "strategy": "Bollinger Bands",
            "status": "paused",
            "profit": -0.8,
            "lastTrade": "2024-02-20T09:45:00Z",
        },
    ]

    return {
        "status": "success",
        "data": auto_trades,
        "timestamp": datetime.now().isoformat(),
    }


@router.post("/auto-trade/{action}")
async def trigger_auto_trade(action: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Trigger auto trade action"""
    symbol = data.get("symbol")
    exchange = data.get("exchange", "default")

    if not symbol:
        raise HTTPException(status_code=400, detail="Symbol is required")

    # Simulate trade execution
    await asyncio.sleep(0.5)  # Simulate processing time

    return {
        "status": "success",
        "data": {
            "action": action,
            "symbol": symbol,
            "exchange": exchange,
            "executed": True,
            "timestamp": datetime.now().isoformat(),
        },
        "timestamp": datetime.now().isoformat(),
    }


