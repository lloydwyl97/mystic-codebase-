"""
Market Data Service
Handles live market data fetching and caching
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from services.market_data_sources import (
    fetch_from_binance,
    fetch_from_coinbase,
    is_supported,
)
from services.websocket_manager import websocket_manager

logger = logging.getLogger(__name__)


class MarketDataService:
    """Service for managing live market data"""

    def __init__(self):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.background_tasks: List[asyncio.Task] = []
        self.is_running = False
        logger.info("✅ MarketDataService initialized")

    async def initialize(self):
        """Initialize the service and start background updates"""
        logger.info("Initializing MarketDataService...")
        self.is_running = True
        await self.start_background_updates()
        logger.info("✅ MarketDataService initialized")

    async def close(self):
        """Close the service and stop background tasks"""
        logger.info("Closing MarketDataService...")
        self.is_running = False

        for task in self.background_tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        self.background_tasks.clear()
        logger.info("MarketDataService closed")

    async def start_background_updates(self):
        """Start background update tasks"""
        logger.info("Starting background market data updates...")

        # Start high priority updates (BTC, ETH, USDT)
        high_priority_task = asyncio.create_task(self._update_high_priority_coins())
        self.background_tasks.append(high_priority_task)

        # Start normal priority updates
        normal_priority_task = asyncio.create_task(self._update_normal_priority_coins())
        self.background_tasks.append(normal_priority_task)

        logger.info(f"✅ Background tasks started: {len(self.background_tasks)} tasks")

    async def _update_high_priority_coins(self):
        """Update high priority coins every 30 seconds"""
        high_priority = ["BTC", "ETH", "USDT"]
        while self.is_running:
            try:
                await self._batch_update_coins(high_priority, "high")
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"High priority update error: {e}")
                await asyncio.sleep(30)

    async def _update_normal_priority_coins(self):
        """Update normal priority coins every 60 seconds"""
        normal_priority = ["ADA", "DOT", "LINK", "SOL", "MATIC"]
        while self.is_running:
            try:
                await self._batch_update_coins(normal_priority, "normal")
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Normal priority update error: {e}")
                await asyncio.sleep(60)

    async def _batch_update_coins(self, coins: List[str], priority: str):
        """Process a batch of coins"""
        logger.info(f"Processing {len(coins)} coins for {priority} priority: {coins}")
        tasks = []

        for coin in coins:
            if is_supported(coin):
                task = asyncio.create_task(self._fetch_coin_data(coin))
                tasks.append(task)
            else:
                logger.warning(f"❌ {coin} is not supported")

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for coin, result in zip(coins, results):
                if isinstance(result, Exception):
                    logger.warning(f"Failed to fetch {coin}: {result}")
                elif isinstance(result, dict):
                    logger.info(f"✅ Successfully fetched {coin}: {result.get('price', 'N/A')}")
                    await self._update_cache(coin, result)
                else:
                    logger.warning(f"No data returned for {coin}")
        else:
            logger.warning("No tasks created - no supported coins found")

    async def _fetch_coin_data(self, coin: str) -> Optional[Dict[str, Any]]:
        """Fetch data for a single coin"""
        try:
            # Try Binance first
            data = fetch_from_binance(coin)
            if data:
                data["api_source"] = "binance"
                data["timestamp"] = datetime.now(timezone.timezone.utc).isoformat()
                return data

            # Try Coinbase as fallback
            data = fetch_from_coinbase(coin)
            if data:
                data["api_source"] = "coinbase"
                data["timestamp"] = datetime.now(timezone.timezone.utc).isoformat()
                return data

        except Exception as e:
            logger.error(f"Error fetching data for {coin}: {e}")

        return None

    async def _update_cache(self, coin: str, data: Dict[str, Any]):
        """Update the cache with new data"""
        self.cache[coin] = data
        logger.debug(f"Updated cache for {coin}")

        # Broadcast market data update
        try:
            await websocket_manager.broadcast_json(
                {
                    "type": "market_data_update",
                    "data": {
                        "symbol": coin,
                        "price": data.get("price", 0.0),
                        "api_source": data.get("api_source", "unknown"),
                        "timestamp": data.get(
                            "timestamp",
                            datetime.now(timezone.timezone.utc).isoformat(),
                        ),
                    },
                }
            )
        except Exception as e:
            logger.error(f"Error broadcasting update: {e}")

    async def get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get market data for a specific symbol"""
        # Try cache first
        cached_data = self.cache.get(symbol)
        if cached_data:
            return cached_data

        # If not in cache, fetch fresh data
        data = await self._fetch_coin_data(symbol)
        if data:
            await self._update_cache(symbol, data)
            return data

        return None

    async def get_cached_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get cached data for a specific symbol"""
        return self.cache.get(symbol)

    async def get_all_cached_data(self) -> Dict[str, Dict[str, Any]]:
        """Get all cached data"""
        return self.cache.copy()

    async def get_markets(self) -> Dict[str, Any]:
        """Get markets overview with current data"""
        try:
            cached_data = await self.get_all_cached_data()
            markets_data = {}

            for symbol, data in cached_data.items():
                markets_data[symbol] = {
                    "price": data.get("price", 0),
                    "api_source": data.get("api_source", "unknown"),
                    "timestamp": data.get(
                        "timestamp",
                        datetime.now(timezone.timezone.utc).isoformat(),
                    ),
                }

            return {
                "markets": markets_data,
                "count": len(markets_data),
                "timestamp": datetime.now(timezone.timezone.utc).isoformat(),
            }
        except Exception as e:
            logger.error(f"Error in get_markets: {e}")
            return {
                "markets": {},
                "count": 0,
                "error": str(e),
                "timestamp": datetime.now(timezone.timezone.utc).isoformat(),
            }

    async def get_live_signals(self, symbol: str = "all") -> Dict[str, Any]:
        """Get live trading signals for symbols"""
        try:
            if symbol.lower() == "all":
                # Return signals for all coins
                cached_data = await self.get_all_cached_data()
                signals = {}

                for symbol, data in cached_data.items():
                    price = data.get("price", 0)
                    if price > 0:
                        signals[symbol] = {
                            "price": price,
                            "signal": "neutral",
                            "strength": 0.5,
                            "timestamp": data.get(
                                "timestamp",
                                datetime.now(timezone.timezone.utc).isoformat(),
                            ),
                        }

                return {
                    "signals": signals,
                    "total_signals": len(signals),
                    "timestamp": (datetime.now(timezone.timezone.utc).isoformat()),
                }
            else:
                # Return signal for specific symbol
                data = await self.get_market_data(symbol)
                if not data:
                    return {
                        "signals": {},
                        "error": f"Symbol {symbol} not found",
                        "timestamp": (datetime.now(timezone.timezone.utc).isoformat()),
                    }

                price = data.get("price", 0)
                signals = {
                    symbol: {
                        "price": price,
                        "signal": "neutral",
                        "strength": 0.5,
                        "timestamp": data.get(
                            "timestamp",
                            datetime.now(timezone.timezone.utc).isoformat(),
                        ),
                    }
                }

                return {
                    "signals": signals,
                    "total_signals": 1,
                    "timestamp": (datetime.now(timezone.timezone.utc).isoformat()),
                }
        except Exception as e:
            logger.error(f"Error getting live signals: {e}")
            return {
                "signals": {},
                "error": str(e),
                "timestamp": datetime.now(timezone.timezone.utc).isoformat(),
            }


# Global instance
market_data_service = MarketDataService()
