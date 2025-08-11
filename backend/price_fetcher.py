#!/usr/bin/env python3
"""
Tier 1: Real-Time Price Fetcher
Handles real-time signals every 10-15 seconds for autobuy/autosell decisions
Optimized for 10 Binance + 10 Coinbase coins (20 total)
"""

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import timezone, datetime
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class PriceSignal:
    symbol: str
    price: float
    change_1m: float
    volume_1m: float
    timestamp: str
    api_source: str
    orderbook_depth: Optional[Dict[str, Any]] = None


class PriceFetcher:
    def __init__(self, redis_client: Any):
        self.redis_client = redis_client
        self.session: Optional[aiohttp.ClientSession] = None
        self.is_running = False

        # Tier 1 Configuration - OPTIMIZED FOR 20 COINS
        self.config = {
            "price_fetch_interval": 10,  # Every 10 seconds per coin
            "momentum_fetch_interval": 15,  # Every 15 seconds globally
            "orderbook_fetch_interval": 15,  # Every 15 seconds
            "cache_ttl": 30,  # seconds
            "max_retries": 3,
            "retry_delay": 1,
        }

        # API endpoints
        self.binance_base_url = "https://api.binance.us/api/v3"
        self.coinbase_base_url = "https://api.pro.coinbase.us"

        # Track last fetch times for throttling
        self.last_fetch_times: Dict[str, float] = {}
        self.last_momentum_fetch = 0

        # OPTIMIZED COIN LISTS - Updated with user specified pairs
        self.binance_coins = [
            "BTCUSDT",
            "ETHUSDT",
            "ADAUSDT",
            "SOLUSDT",
            "DOTUSDT",
            "LINKUSDT",
            "MATICUSDT",
            "AVAXUSDT",
            "UNIUSDT",
            "ATOMUSDT",
        ]
        self.coinbase_coins = [
            "BTC-USD",
            "ETH-USD",
            "ADA-USD",
            "SOL-USD",
            "DOT-USD",
            "LINK-USD",
            "MATIC-USD",
            "AVAX-USD",
            "UNI-USD",
            "ATOM-USD",
        ]

        # Price history for momentum calculations
        self.price_history: Dict[str, List[Dict[str, Any]]] = {}

        logger.info(
            f"Price Fetcher initialized with {len(self.binance_coins)} Binance + {len(self.coinbase_coins)} Coinbase coins"
        )

    async def initialize(self):
        """Initialize the price fetcher"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=10)
            self.session = aiohttp.ClientSession(timeout=timeout)
        logger.info("Price Fetcher initialized")

    async def close(self):
        """Close the price fetcher"""
        if self.session:
            await self.session.close()
        self.is_running = False
        logger.info("Price Fetcher closed")

    def _should_fetch_price(self, symbol: str) -> bool:
        """Check if we should fetch price for specific coin (10 second interval)"""
        now = time.time()
        key = f"price_{symbol}"

        if key not in self.last_fetch_times:
            return True

        last_fetch = self.last_fetch_times[key]
        return (now - last_fetch) >= self.config["price_fetch_interval"]

    def _should_fetch_momentum(self) -> bool:
        """Check if we should fetch momentum globally (15 second interval)"""
        now = time.time()
        return (now - self.last_momentum_fetch) >= self.config["momentum_fetch_interval"]

    def _update_fetch_time(self, fetch_type: str, symbol: Optional[str] = None):
        """Update the last fetch time for throttling"""
        if fetch_type == "momentum":
            self.last_momentum_fetch = time.time()
        else:
            key = f"{fetch_type}_{symbol}"
            self.last_fetch_times[key] = time.time()

    async def fetch_price(self, symbol: str, exchange: str) -> Optional[PriceSignal]:
        """Fetch real-time price data (10 second frequency per coin)"""
        if not self._should_fetch_price(symbol):
            return None

        try:
            if exchange == "binance":
                data = await self._fetch_binance_price(symbol)
            elif exchange == "coinbase":
                data = await self._fetch_coinbase_price(symbol)
            else:
                logger.error(f"Unsupported exchange: {exchange}")
                return None

            if data:
                self._update_fetch_time("price", symbol)
                await self._cache_price_signal(data)
                await self._update_price_history(symbol, data.price)
                return data

        except Exception as e:
            logger.error(f"Error fetching price for {symbol} from {exchange}: {e}")

        return None

    async def fetch_momentum_signals(self) -> Dict[str, float]:
        """Fetch momentum signals globally (15 second frequency)"""
        if not self._should_fetch_momentum():
            return {}

        try:
            momentum_data: Dict[str, float] = {}

            # Calculate 1-minute change for all coins
            all_coins = self.binance_coins + self.coinbase_coins

            for symbol in all_coins:
                change_1m = await self._calculate_1m_change(symbol)
                if change_1m is not None:
                    momentum_data[symbol] = change_1m

            self._update_fetch_time("momentum")
            await self._cache_momentum_data(momentum_data)

            return momentum_data

        except Exception as e:
            logger.error(f"Error calculating momentum signals: {e}")
            return {}

    async def _calculate_1m_change(self, symbol: str) -> Optional[float]:
        """Calculate 1-minute price change for a symbol"""
        try:
            # Get current price from cache
            cached_data = await self._get_cached_price_signal(symbol)
            if not cached_data:
                return None

            current_price = cached_data["price"]

            # Get price history for this symbol
            if symbol not in self.price_history:
                return None

            history = self.price_history[symbol]
            if len(history) < 2:
                return None

            # Find price from 1 minute ago (assuming 10-second intervals)
            # We need price from 6 intervals ago (60 seconds / 10 seconds = 6)
            if len(history) >= 6:
                old_price = history[-6]["price"]
                change_1m = ((current_price - old_price) / old_price) * 100
                return round(change_1m, 4)

            return None

        except Exception as e:
            logger.error(f"Error calculating 1m change for {symbol}: {e}")
            return None

    async def _fetch_binance_price(self, symbol: str) -> Optional[PriceSignal]:
        """Fetch price from Binance"""
        if not self.session:
            return None

        try:
            url = f"{self.binance_base_url}/ticker/price"
            params = {"symbol": symbol}

            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return PriceSignal(
                        symbol=symbol,
                        price=float(data["price"]),
                        change_1m=0.0,  # Will be calculated separately
                        volume_1m=0.0,  # Will be calculated separately
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        api_source="binance",
                    )
                else:
                    logger.error(f"Binance price API error: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error fetching Binance price: {e}")
            return None

    async def _fetch_coinbase_price(self, symbol: str) -> Optional[PriceSignal]:
        """Fetch price from Coinbase"""
        if not self.session:
            return None

        try:
            url = f"{self.coinbase_base_url}/products/{symbol}/ticker"

            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return PriceSignal(
                        symbol=symbol,
                        price=float(data["price"]),
                        change_1m=0.0,  # Will be calculated separately
                        volume_1m=float(data.get("volume", 0.0)),
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        api_source="coinbase",
                    )
                else:
                    logger.error(f"Coinbase price API error: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error fetching Coinbase price: {e}")
            return None

    async def _update_price_history(self, symbol: str, price: float):
        """Update price history for momentum calculations"""
        if symbol not in self.price_history:
            self.price_history[symbol] = []

        self.price_history[symbol].append(
            {
                "price": price,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        # Keep only last 60 data points (10 minutes at 10-second intervals)
        if len(self.price_history[symbol]) > 60:
            self.price_history[symbol] = self.price_history[symbol][-60:]

    async def _cache_price_signal(self, signal: PriceSignal):
        """Cache price signal in Redis"""
        try:
            key = f"price_signal_{signal.symbol}"
            self.redis_client.setex(key, self.config["cache_ttl"], json.dumps(asdict(signal)))
        except Exception as e:
            logger.error(f"Error caching price signal: {e}")

    async def _cache_momentum_data(self, data: Dict[str, float]):
        """Cache momentum data in Redis"""
        try:
            key = "momentum_signals"
            self.redis_client.setex(key, self.config["cache_ttl"], json.dumps(data))
        except Exception as e:
            logger.error(f"Error caching momentum data: {e}")

    async def _get_cached_price_signal(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get cached price signal from Redis"""
        try:
            key = f"price_signal_{symbol}"
            data = self.redis_client.get(key)
            return json.loads(data) if data else None
        except Exception as e:
            logger.error(f"Error getting cached price signal: {e}")
            return None

    async def fetch_all_tier1_signals(self) -> Dict[str, Any]:
        """Fetch all Tier 1 signals for all 20 coins"""
        results: Dict[str, Any] = {
            "prices": {},
            "momentum": {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Fetch Binance prices (10 coins)
        for symbol in self.binance_coins:
            price_signal = await self.fetch_price(symbol, "binance")
            if price_signal:
                results["prices"][symbol] = asdict(price_signal)

        # Fetch Coinbase prices (10 coins)
        for symbol in self.coinbase_coins:
            price_signal = await self.fetch_price(symbol, "coinbase")
            if price_signal:
                results["prices"][symbol] = asdict(price_signal)

        # Fetch momentum signals globally
        momentum_data = await self.fetch_momentum_signals()
        results["momentum"] = momentum_data

        # Update momentum data in price signals
        for symbol, momentum in momentum_data.items():
            if symbol in results["prices"]:
                results["prices"][symbol]["change_1m"] = momentum

        # Cache the complete Tier 1 data
        await self._cache_tier1_data(results)

        return results

    async def _cache_tier1_data(self, data: Dict[str, Any]):
        """Cache complete Tier 1 data"""
        try:
            self.redis_client.setex("tier1_signals", self.config["cache_ttl"], json.dumps(data))
        except Exception as e:
            logger.error(f"Error caching Tier 1 data: {e}")

    async def run(self):
        """Main price fetcher loop - OPTIMIZED FOR 20 COINS"""
        logger.info("Starting Tier 1 Price Fetcher (20 coins)...")
        self.is_running = True

        try:
            await self.initialize()

            while self.is_running:
                try:
                    # Fetch all Tier 1 signals
                    signals = await self.fetch_all_tier1_signals()

                    logger.debug(f"Fetched Tier 1 signals for {len(signals['prices'])} coins")

                    # Wait for next cycle (10 seconds)
                    await asyncio.sleep(self.config["price_fetch_interval"])

                except Exception as e:
                    logger.error(f"Error in price fetcher loop: {e}")
                    await asyncio.sleep(self.config["retry_delay"])

        except Exception as e:
            logger.error(f"Fatal error in price fetcher: {e}")
        finally:
            await self.close()

    def get_status(self) -> Dict[str, Any]:
        """Get price fetcher status"""
        return {
            "status": "running" if self.is_running else "stopped",
            "config": self.config,
            "last_fetch_times": self.last_fetch_times,
            "last_momentum_fetch": self.last_momentum_fetch,
            "supported_coins": {
                "binance": self.binance_coins,
                "coinbase": self.coinbase_coins,
                "total": len(self.binance_coins) + len(self.coinbase_coins),
            },
            "price_history_count": len(self.price_history),
        }
