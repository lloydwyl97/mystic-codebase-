"""
Optimized Market Data Service for Mystic Trading Platform

Combines database optimization and API throttling for high-performance
market data operations with intelligent caching and rate limiting.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

from api_throttler import api_throttler
from database_optimized import optimized_db_manager

from backend.utils.exceptions import MarketDataException, handle_exception

logger = logging.getLogger(__name__)


@dataclass
class OptimizedMarketData:
    """Optimized market data structure with caching"""

    symbol: str
    price: float
    volume: float
    change_24h: float
    high_24h: float
    low_24h: float
    timestamp: float
    exchange: str = "binance"
    cache_key: str = ""
    cache_ttl: int = 30  # 30 seconds cache

    def __post_init__(self):
        self.cache_key = f"market_data:{self.symbol}:{self.exchange}"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return {
            "symbol": self.symbol,
            "price": self.price,
            "volume": self.volume,
            "change_24h": self.change_24h,
            "high_24h": self.high_24h,
            "low_24h": self.low_24h,
            "timestamp": self.timestamp,
            "exchange": self.exchange,
            "cache_key": self.cache_key,
        }


class OptimizedMarketDataService:
    """High-performance market data service with optimizations"""

    def __init__(self):
        self.supported_symbols = ["BTC", "ETH", "ADA", "DOT", "SOL", "MATIC"]
        self.data_cache: dict[str, OptimizedMarketData] = {}
        self.cache_timestamps: dict[str, float] = {}
        self.cache_ttl = 30  # seconds

        # Performance tracking
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "api_calls": 0,
            "database_queries": 0,
            "average_response_time": 0.0,
        }

        # Initialize with conservative throttling
        api_throttler.set_throttle_level(api_throttler.throttle_level)

        logger.info("âœ… OptimizedMarketDataService initialized")

    def _is_cache_valid(self, symbol: str) -> bool:
        """Check if cached data is still valid"""
        if symbol not in self.cache_timestamps:
            return False

        return time.time() - self.cache_timestamps[symbol] < self.cache_ttl

    def _get_cached_data(self, symbol: str) -> OptimizedMarketData | None:
        """Get cached market data"""
        if symbol in self.data_cache and self._is_cache_valid(symbol):
            self.stats["cache_hits"] += 1
            return self.data_cache[symbol]

        self.stats["cache_misses"] += 1
        return None

    def _cache_data(self, symbol: str, data: OptimizedMarketData):
        """Cache market data"""
        self.data_cache[symbol] = data
        self.cache_timestamps[symbol] = time.time()

    @handle_exception("Failed to get optimized market data", MarketDataException)
    async def get_market_data(self, symbol: str) -> OptimizedMarketData | None:
        """Get market data with optimizations"""
        start_time = time.time()
        self.stats["total_requests"] += 1

        try:
            # Check cache first
            cached_data = self._get_cached_data(symbol)
            if cached_data:
                return cached_data

            # Try database first (faster than API)
            db_data = await self._get_from_database(symbol)
            if db_data:
                self._cache_data(symbol, db_data)
                return db_data

            # Fallback to API with throttling
            api_data = await self._get_from_api(symbol)
            if api_data:
                # Store in database for future use
                await self._store_in_database(api_data)
                self._cache_data(symbol, api_data)
                return api_data

            return None

        finally:
            # Update response time stats
            response_time = time.time() - start_time
            if self.stats["total_requests"] > 0:
                self.stats["average_response_time"] = (
                    self.stats["average_response_time"] * (self.stats["total_requests"] - 1)
                    + response_time
                ) / self.stats["total_requests"]

    async def _get_from_database(self, symbol: str) -> OptimizedMarketData | None:
        """Get market data from optimized database"""
        try:
            self.stats["database_queries"] += 1

            # Use optimized query with caching
            query = """
                SELECT symbol, price, volume, timestamp
                FROM market_data
                WHERE symbol = ?
                ORDER BY timestamp DESC
                LIMIT 1
            """

            result = optimized_db_manager.execute_query(query, (symbol,))

            if result:
                row = result[0]
                return OptimizedMarketData(
                    symbol=row["symbol"],
                    price=row["price"],
                    volume=row["volume"] or 0.0,
                    change_24h=0.0,  # Not stored in DB
                    high_24h=0.0,  # Not stored in DB
                    low_24h=0.0,  # Not stored in DB
                    timestamp=float(row["timestamp"]),
                    exchange="database",
                )

            return None

        except Exception as e:
            logger.error(f"Database query failed for {symbol}: {e}")
            return None

    async def _get_from_api(self, symbol: str) -> OptimizedMarketData | None:
        """Get market data from API with throttling"""
        try:
            self.stats["api_calls"] += 1

            # Use throttled API call
            result = await api_throttler.throttle_request(
                endpoint=f"market_data_{symbol}",
                method="GET",
                func=self._fetch_api_data,
                symbol=symbol,
            )

            return result

        except Exception as e:
            logger.error(f"API call failed for {symbol}: {e}")
            return None

    async def _fetch_api_data(self, symbol: str) -> OptimizedMarketData | None:
        """Fetch data from external API (simulated)"""
        # Simulate API call with delay
        await asyncio.sleep(0.1)

        # Get real data from live market data service
        from .modules.data.market_data import market_data_manager

        try:
            real_data = await market_data_manager.get_market_data(symbol)
            if real_data:
                return OptimizedMarketData(
                    symbol=symbol,
                    price=real_data.price,
                    volume=real_data.volume,
                    change_24h=real_data.change_24h,
                    high_24h=real_data.high_24h,
                    low_24h=real_data.low_24h,
                    timestamp=real_data.timestamp,
                    exchange=real_data.exchange,
                )
        except Exception as e:
            logger.error(f"Failed to get real market data for {symbol}: {e}")

        return None

    async def _store_in_database(self, data: OptimizedMarketData):
        """Store market data in optimized database"""
        try:
            query = """
                INSERT INTO market_data (symbol, price, volume, timestamp)
                VALUES (?, ?, ?, ?)
            """

            optimized_db_manager.execute_query(
                query,
                (data.symbol, data.price, data.volume, data.timestamp),
                use_cache=False,
            )

        except Exception as e:
            logger.error(f"Failed to store data in database: {e}")

    @handle_exception("Failed to get all optimized market data", MarketDataException)
    async def get_all_market_data(self) -> dict[str, OptimizedMarketData]:
        """Get market data for all supported symbols with optimizations"""
        try:
            tasks = [self.get_market_data(symbol) for symbol in self.supported_symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            market_data: dict[str, OptimizedMarketData] = {}
            for i, result in enumerate(results):
                if isinstance(result, OptimizedMarketData):
                    market_data[self.supported_symbols[i]] = result
                else:
                    logger.warning(f"Failed to get data for {self.supported_symbols[i]}: {result}")

            return market_data

        except Exception as e:
            logger.error(f"Failed to get all market data: {e}")
            return {}

    @handle_exception("Failed to get optimized market summary", MarketDataException)
    async def get_market_summary(self) -> dict[str, Any]:
        """Get optimized market summary"""
        try:
            market_data = await self.get_all_market_data()

            if not market_data:
                return {
                    "symbols": [],
                    "total_symbols": 0,
                    "total_volume": 0,
                    "average_change_24h": 0,
                    "timestamp": time.time(),
                    "optimized": True,
                    "cache_stats": self.stats,
                }

            total_volume = sum(data.volume for data in market_data.values())
            avg_change = sum(data.change_24h for data in market_data.values()) / len(market_data)

            return {
                "symbols": [data.to_dict() for data in market_data.values()],
                "total_symbols": len(market_data),
                "total_volume": total_volume,
                "average_change_24h": avg_change,
                "timestamp": time.time(),
                "optimized": True,
                "cache_stats": self.stats,
                "performance_stats": {
                    "database_stats": (optimized_db_manager.get_performance_stats()),
                    "api_stats": api_throttler.get_performance_stats(),
                },
            }

        except Exception as e:
            logger.error(f"Failed to get market summary: {e}")
            return {
                "symbols": [],
                "total_symbols": 0,
                "total_volume": 0,
                "average_change_24h": 0,
                "timestamp": time.time(),
                "optimized": True,
                "error": str(e),
                "cache_stats": self.stats,
            }

    def get_performance_stats(self) -> dict[str, Any]:
        """Get comprehensive performance statistics"""
        return {
            "market_data_stats": self.stats.copy(),
            "database_stats": optimized_db_manager.get_performance_stats(),
            "api_stats": api_throttler.get_performance_stats(),
            "cache_efficiency": {
                "hit_rate": (self.stats["cache_hits"] / max(self.stats["total_requests"], 1)),
                "miss_rate": (self.stats["cache_misses"] / max(self.stats["total_requests"], 1)),
                "cache_size": len(self.data_cache),
            },
        }

    def clear_cache(self):
        """Clear all cached data"""
        self.data_cache.clear()
        self.cache_timestamps.clear()
        optimized_db_manager.clear_cache()
        logger.info("âœ… All caches cleared")

    def increase_throttling(self):
        """Increase API throttling"""
        api_throttler.increase_throttling()

    def decrease_throttling(self):
        """Decrease API throttling"""
        api_throttler.decrease_throttling()

    def optimize_database(self):
        """Run database optimization"""
        optimized_db_manager.optimize_database()


# Global optimized market data service instance
optimized_market_service = OptimizedMarketDataService()


