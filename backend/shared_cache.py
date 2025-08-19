"""
Shared Cache System for Mystic AI Trading Platform
Provides a centralized cache for market data, cosmic data, and other shared information.
"""

import json
import logging
from datetime import datetime
from typing import Any

import redis

logger = logging.getLogger(__name__)


class SharedCache:
    """Centralized cache for shared data across the platform"""

    def __init__(self, redis_client: redis.Redis | None = None):
        self.redis_client = redis_client
        self.memory_cache = {
            "market_data": {},
            "cosmic_data": {},
            "indicator_data": {},
            "volume_data": {},
            "last_update": {},
            "created_at": datetime.now().isoformat(),
        }

        # Initialize Redis connection if not provided
        if not self.redis_client:
            try:
                # Try localhost first, then Docker hostname as fallback
                redis_hosts = ["localhost", "redis"]
                redis_port = 6379

                for host in redis_hosts:
                    try:
                        self.redis_client = redis.Redis(
                            host=host,
                            port=redis_port,
                            db=0,
                            decode_responses=True,
                            socket_connect_timeout=5,
                            socket_timeout=5,
                        )
                        # Test connection
                        self.redis_client.ping()
                        logger.info(
                            f"âœ… Redis connection established for SharedCache using {host}:{redis_port}"
                        )
                        break
                    except Exception as e:
                        logger.warning(f"âŒ Redis connection failed for {host}:{redis_port} - {e}")
                        continue

                if not self.redis_client:
                    logger.warning(
                        "âŒ All Redis connection attempts failed - using memory cache only"
                    )
                    self.redis_client = None

            except Exception as e:
                logger.warning(f"âŒ Redis connection failed for SharedCache: {e}")
                self.redis_client = None

    def update_market_data(self, data: dict[str, Any]) -> None:
        """Update market data in cache"""
        try:
            self.memory_cache["market_data"] = data
            self.memory_cache["last_update"]["market_data"] = datetime.now().isoformat()

            if self.redis_client:
                self.redis_client.setex("market_data", 300, json.dumps(data))  # 5 minutes expiry

            logger.debug("âœ… Market data updated in SharedCache")
        except Exception as e:
            logger.error(f"âŒ Error updating market data in SharedCache: {e}")

    def update_price(self, symbol: str, price: float) -> None:
        """Update price for a specific symbol"""
        try:
            if "prices" not in self.memory_cache:
                self.memory_cache["prices"] = {}

            self.memory_cache["prices"][symbol] = {
                "price": price,
                "timestamp": datetime.now().isoformat()
            }

            if self.redis_client:
                self.redis_client.setex(f"price:{symbol}", 300, json.dumps({
                    "price": price,
                    "timestamp": datetime.now().isoformat()
                }))

            logger.debug(f"âœ… Price updated for {symbol}: {price}")
        except Exception as e:
            logger.error(f"âŒ Error updating price for {symbol}: {e}")

    def should_update_indicators(self, symbol: str) -> bool:
        """Check if indicators should be updated for a symbol"""
        try:
            # Check if we have recent price data
            if "prices" in self.memory_cache and symbol in self.memory_cache["prices"]:
                price_data = self.memory_cache["prices"][symbol]
                timestamp = datetime.fromisoformat(price_data["timestamp"])
                age = (datetime.now() - timestamp).total_seconds()
                return age < 300  # Update if price is less than 5 minutes old
            return False
        except Exception as e:
            logger.error(f"âŒ Error checking indicator update for {symbol}: {e}")
            return False

    def should_update_volume(self, symbol: str) -> bool:
        """Check if volume should be updated for a symbol"""
        try:
            # Check if we have recent volume data
            if "volumes" in self.memory_cache and symbol in self.memory_cache["volumes"]:
                volume_data = self.memory_cache["volumes"][symbol]
                timestamp = datetime.fromisoformat(volume_data["timestamp"])
                age = (datetime.now() - timestamp).total_seconds()
                return age > 180  # Update if volume is older than 3 minutes
            return True  # Update if no volume data exists
        except Exception as e:
            logger.error(f"âŒ Error checking volume update for {symbol}: {e}")
            return True

    def update_volume(self, symbol: str, volume: float) -> None:
        """Update volume for a specific symbol"""
        try:
            if "volumes" not in self.memory_cache:
                self.memory_cache["volumes"] = {}

            self.memory_cache["volumes"][symbol] = {
                "volume": volume,
                "timestamp": datetime.now().isoformat()
            }

            if self.redis_client:
                self.redis_client.setex(f"volume:{symbol}", 300, json.dumps({
                    "volume": volume,
                    "timestamp": datetime.now().isoformat()
                }))

            logger.debug(f"âœ… Volume updated for {symbol}: {volume}")
        except Exception as e:
            logger.error(f"âŒ Error updating volume for {symbol}: {e}")

    def get_coin_cache(self, symbol: str) -> dict[str, Any]:
        """Get coin cache data for a specific symbol"""
        try:
            coin_data = {}

            # Get price data
            if "prices" in self.memory_cache and symbol in self.memory_cache["prices"]:
                coin_data["price"] = self.memory_cache["prices"][symbol]

            # Get volume data
            if "volumes" in self.memory_cache and symbol in self.memory_cache["volumes"]:
                coin_data["volume"] = self.memory_cache["volumes"][symbol]

            # Get indicator data
            if "indicator_data" in self.memory_cache and symbol in self.memory_cache["indicator_data"]:
                coin_data["indicators"] = self.memory_cache["indicator_data"][symbol]

            return coin_data
        except Exception as e:
            logger.error(f"âŒ Error getting coin cache for {symbol}: {e}")
            return {}

    def update_indicators(self, symbol: str, rsi: float, macd: dict[str, float], volatility: float) -> None:
        """Update indicators for a specific symbol"""
        try:
            if "indicator_data" not in self.memory_cache:
                self.memory_cache["indicator_data"] = {}

            self.memory_cache["indicator_data"][symbol] = {
                "rsi": rsi,
                "macd": macd,
                "volatility": volatility,
                "timestamp": datetime.now().isoformat()
            }

            if self.redis_client:
                self.redis_client.setex(f"indicators:{symbol}", 300, json.dumps({
                    "rsi": rsi,
                    "macd": macd,
                    "volatility": volatility,
                    "timestamp": datetime.now().isoformat()
                }))

            logger.debug(f"âœ… Indicators updated for {symbol}: RSI={rsi}, MACD={macd}, Vol={volatility}")
        except Exception as e:
            logger.error(f"âŒ Error updating indicators for {symbol}: {e}")

    def get_market_data(self) -> dict[str, Any]:
        """Get market data from cache"""
        try:
            # Try Redis first
            if self.redis_client:
                cached_data = self.redis_client.get("market_data")
                if cached_data:
                    return json.loads(cached_data)

            # Fallback to memory cache
            return self.memory_cache.get("market_data", {})
        except Exception as e:
            logger.error(f"âŒ Error getting market data from SharedCache: {e}")
            return {}

    def update_cosmic_data(self, data: dict[str, Any]) -> None:
        """Update cosmic data in cache"""
        try:
            self.memory_cache["cosmic_data"] = data
            self.memory_cache["last_update"]["cosmic_data"] = datetime.now().isoformat()

            if self.redis_client:
                self.redis_client.setex("cosmic_data", 3600, json.dumps(data))  # 1 hour expiry

            logger.debug("âœ… Cosmic data updated in SharedCache")
        except Exception as e:
            logger.error(f"âŒ Error updating cosmic data in SharedCache: {e}")

    def get_cosmic_data(self) -> dict[str, Any]:
        """Get cosmic data from cache"""
        try:
            # Try Redis first
            if self.redis_client:
                cached_data = self.redis_client.get("cosmic_data")
                if cached_data:
                    return json.loads(cached_data)

            # Fallback to memory cache
            return self.memory_cache.get("cosmic_data", {})
        except Exception as e:
            logger.error(f"âŒ Error getting cosmic data from SharedCache: {e}")
            return {}

    def update_indicator_data(self, symbol: str, data: dict[str, Any]) -> None:
        """Update indicator data for a specific symbol"""
        try:
            if "indicator_data" not in self.memory_cache:
                self.memory_cache["indicator_data"] = {}

            self.memory_cache["indicator_data"][symbol] = data
            self.memory_cache["last_update"]["indicator_data"] = datetime.now().isoformat()

            if self.redis_client:
                self.redis_client.setex(
                    f"indicator_data:{symbol}", 300, json.dumps(data)
                )  # 5 minutes expiry

            logger.debug(f"âœ… Indicator data updated for {symbol} in SharedCache")
        except Exception as e:
            logger.error(f"âŒ Error updating indicator data for {symbol} in SharedCache: {e}")

    def get_indicator_data(self, symbol: str) -> dict[str, Any]:
        """Get indicator data for a specific symbol"""
        try:
            # Try Redis first
            if self.redis_client:
                cached_data = self.redis_client.get(f"indicator_data:{symbol}")
                if cached_data:
                    return json.loads(cached_data)

            # Fallback to memory cache
            return self.memory_cache.get("indicator_data", {}).get(symbol, {})
        except Exception as e:
            logger.error(f"âŒ Error getting indicator data for {symbol} from SharedCache: {e}")
            return {}

    def update_volume_data(self, symbol: str, data: dict[str, Any]) -> None:
        """Update volume data for a specific symbol"""
        try:
            if "volume_data" not in self.memory_cache:
                self.memory_cache["volume_data"] = {}

            self.memory_cache["volume_data"][symbol] = data
            self.memory_cache["last_update"]["volume_data"] = datetime.now().isoformat()

            if self.redis_client:
                self.redis_client.setex(
                    f"volume_data:{symbol}", 300, json.dumps(data)
                )  # 5 minutes expiry

            logger.debug(f"âœ… Volume data updated for {symbol} in SharedCache")
        except Exception as e:
            logger.error(f"âŒ Error updating volume data for {symbol} in SharedCache: {e}")

    def get_volume_data(self, symbol: str) -> dict[str, Any]:
        """Get volume data for a specific symbol"""
        try:
            # Try Redis first
            if self.redis_client:
                cached_data = self.redis_client.get(f"volume_data:{symbol}")
                if cached_data:
                    return json.loads(cached_data)

            # Fallback to memory cache
            return self.memory_cache.get("volume_data", {}).get(symbol, {})
        except Exception as e:
            logger.error(f"âŒ Error getting volume data for {symbol} from SharedCache: {e}")
            return {}

    def get_cache_status(self) -> dict[str, Any]:
        """Get status of all cache data"""
        try:
            status = {
                "memory_cache_size": len(self.memory_cache),
                "redis_connected": self.redis_client is not None,
                "last_updates": self.memory_cache.get("last_update", {}),
                "created_at": self.memory_cache.get("created_at"),
                "timestamp": datetime.now().isoformat(),
            }

            # Add Redis status if available
            if self.redis_client:
                try:
                    redis_info = self.redis_client.info()
                    status["redis_info"] = {
                        "used_memory": redis_info.get("used_memory_human", "unknown"),
                        "connected_clients": redis_info.get("connected_clients", 0),
                        "uptime": redis_info.get("uptime_in_seconds", 0),
                    }
                except Exception as e:
                    status["redis_info"] = {"error": str(e)}

            return status
        except Exception as e:
            logger.error(f"âŒ Error getting cache status: {e}")
            return {"error": str(e)}

    def clear_cache(self, cache_type: str | None = None) -> None:
        """Clear cache data"""
        try:
            if cache_type:
                if cache_type in self.memory_cache:
                    self.memory_cache[cache_type] = {}
                    logger.info(f"âœ… Cleared {cache_type} cache")
            else:
                self.memory_cache = {
                    "market_data": {},
                    "cosmic_data": {},
                    "indicator_data": {},
                    "volume_data": {},
                    "last_update": {},
                    "created_at": datetime.now().isoformat(),
                }
                logger.info("âœ… Cleared all cache data")
        except Exception as e:
            logger.error(f"âŒ Error clearing cache: {e}")

    def is_fresh(self, cache_type: str, max_age_seconds: int = 300) -> bool:
        """Check if cache data is fresh (updated within max_age_seconds)"""
        try:
            last_update = self.memory_cache.get("last_update", {}).get(cache_type)
            if not last_update:
                return False

            update_time = datetime.fromisoformat(last_update)
            age = (datetime.now() - update_time).total_seconds()
            return age < max_age_seconds
        except Exception as e:
            logger.error(f"âŒ Error checking cache freshness: {e}")
            return False


class CoinCache:
    """Specialized cache for coin-specific data"""

    def __init__(self, shared_cache: SharedCache):
        self.shared_cache = shared_cache

    def update_coin_data(self, symbol: str, data: dict[str, Any]) -> None:
        """Update data for a specific coin"""
        try:
            # Update market data
            market_data = self.shared_cache.get_market_data()
            market_data[symbol] = data
            self.shared_cache.update_market_data(market_data)

            logger.debug(f"âœ… Coin data updated for {symbol}")
        except Exception as e:
            logger.error(f"âŒ Error updating coin data for {symbol}: {e}")

    def get_coin_data(self, symbol: str) -> dict[str, Any]:
        """Get data for a specific coin"""
        try:
            market_data = self.shared_cache.get_market_data()
            return market_data.get(symbol, {})
        except Exception as e:
            logger.error(f"âŒ Error getting coin data for {symbol}: {e}")
            return {}

    def get_all_coins(self) -> dict[str, Any]:
        """Get data for all coins"""
        try:
            return self.shared_cache.get_market_data()
        except Exception as e:
            logger.error(f"âŒ Error getting all coins data: {e}")
            return {}


