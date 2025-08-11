#!/usr/bin/env python3
"""
Data Fetchers for Mystic Trading Platform

Handles fetching market data from various sources.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import aiohttp
import redis

# Use absolute imports
from crypto_autoengine_config import get_config
from shared_cache import SharedCache

logger = logging.getLogger(__name__)


class PriceFetcher:
    """Bulk price fetcher - runs every 10 seconds"""

    def __init__(self, cache: SharedCache) -> None:
        self.cache = cache
        self.config = get_config()
        self.session: Optional[aiohttp.ClientSession] = None
        self.last_fetch: float = 0

    async def start(self) -> None:
        """Start the price fetcher"""
        self.session = aiohttp.ClientSession()
        logger.info("Price fetcher started")

        while True:
            try:
                await self.fetch_all_prices()
                await asyncio.sleep(self.config.fetcher_config.price_fetch_interval)
            except Exception as e:
                logger.error(f"Error in price fetcher: {e}")
                await asyncio.sleep(5)

    async def fetch_all_prices(self) -> None:
        """Bulk fetch prices for all coins"""
        current_time = time.time()

        # Check throttling
        if current_time - self.last_fetch < self.config.throttling_rules["price"]["min_interval"]:
            return

        self.last_fetch = current_time

        # Fetch Coinbase prices
        await self._fetch_coinbase_prices()

        # Fetch Binance prices
        await self._fetch_binance_prices()

        # Fetch CoinGecko prices
        await self._fetch_coingecko_prices()

        # Fetch Kraken prices
        await self._fetch_kraken_prices()

        logger.debug("Completed bulk price fetch")

    async def _fetch_coinbase_prices(self) -> None:
        """Fetch prices for all Coinbase coins"""
        coinbase_coins = self.config.get_coins_by_exchange("coinbase")
        if not coinbase_coins:
            return

        try:
            # Fetch all Coinbase prices in batches
            batch_size = int(self.config.throttling_rules["price"]["bundle_size"])
            for i in range(0, len(coinbase_coins), batch_size):
                batch = coinbase_coins[i:i + batch_size]
                await self._fetch_coinbase_batch(batch)
                await asyncio.sleep(0.1)  # Small delay between batches

        except Exception as e:
            logger.error(f"Error fetching Coinbase prices: {e}")

    async def _fetch_coinbase_batch(self, coins: List[Any]) -> None:
        """Fetch comprehensive data for a batch of Coinbase coins"""
        for coin_config in coins:
            symbol = coin_config.symbol
            try:
                # Use the public Coinbase API endpoints that don't require authentication
                # For public data, we use the exchange API
                product_url = f"https://api.exchange.coinbase.com/products/{symbol}"
                ticker_url = f"https://api.exchange.coinbase.com/products/{symbol}/ticker"

                if self.session:
                    # Get product details
                    async with self.session.get(
                        product_url, timeout=aiohttp.ClientTimeout(total=5)
                    ) as response:
                        if response.status == 200:
                            product_data = await response.json()

                            # Get ticker data for price
                            async with self.session.get(
                                ticker_url, timeout=aiohttp.ClientTimeout(total=5)
                            ) as ticker_response:
                                if ticker_response.status == 200:
                                    ticker_data = await ticker_response.json()

                                    # Extract comprehensive data
                                    price = float(ticker_data.get("price", 0))
                                    volume = float(ticker_data.get("size", 0))  # size is volume in Coinbase
                                    bid = float(ticker_data.get("bid", 0))
                                    ask = float(ticker_data.get("ask", 0))

                                    # Update cache with comprehensive data
                                    self.cache.update_price(symbol, price)
                                    self.cache.update_volume(symbol, volume)

                                    # Store additional market data
                                    market_data = {
                                        "price": price,
                                        "volume": volume,
                                        "bid": bid,
                                        "ask": ask,
                                        "product_id": product_data.get("id", symbol),
                                        "base_currency": product_data.get("base_currency", ""),
                                        "quote_currency": product_data.get("quote_currency", ""),
                                        "status": product_data.get("status", ""),
                                        "trading_enabled": product_data.get("trading_enabled", True),
                                        "timestamp": time.time()
                                    }

                                    # Update market data cache
                                    if hasattr(self.cache, 'update_market_data'):
                                        self.cache.update_market_data({symbol: market_data})

                                else:
                                    logger.warning(f"Failed to fetch ticker for {symbol}: {ticker_response.status}")
                        else:
                            logger.warning(f"Failed to fetch product data for {symbol}: {response.status}")
            except Exception as e:
                logger.error(f"Error fetching comprehensive data for {symbol}: {e}")

    async def _fetch_binance_prices(self) -> None:
        """Fetch prices for all Binance coins"""
        binance_coins = self.config.get_coins_by_exchange("binance")
        if not binance_coins:
            return

        try:
            # Fetch all Binance prices in batches
            batch_size = int(self.config.throttling_rules["price"]["bundle_size"])
            for i in range(0, len(binance_coins), batch_size):
                batch = binance_coins[i:i + batch_size]
                await self._fetch_binance_batch(batch)
                await asyncio.sleep(0.1)  # Small delay between batches

        except Exception as e:
            logger.error(f"Error fetching Binance prices: {e}")

    async def _fetch_binance_batch(self, coins: List[Any]) -> None:
        """Fetch prices for a batch of Binance coins"""
        for coin_config in coins:
            symbol = coin_config.symbol
            try:
                url = f"{self.config.api_config.binance_base_url}/ticker/24hr"
                params = {"symbol": symbol}

                if self.session:
                    async with self.session.get(
                        url,
                        params=params,
                        timeout=aiohttp.ClientTimeout(total=5),
                    ) as response:
                        if response.status == 200:
                            ticker = await response.json()
                            price = float(ticker["lastPrice"])
                            self.cache.update_price(symbol, price)
                        else:
                            logger.warning(f"Failed to fetch price for {symbol}: {response.status}")
            except Exception as e:
                logger.error(f"Error fetching Binance price for {symbol}: {e}")

    async def _fetch_coingecko_prices(self) -> None:
        """Fetch prices from CoinGecko API"""
        try:
            # CoinGecko uses different symbols (e.g., 'bitcoin', 'ethereum')
            coingecko_coins = {
                'bitcoin': 'BTC',
                'ethereum': 'ETH',
                'cardano': 'ADA',
                'solana': 'SOL',
                'polkadot': 'DOT',
                'chainlink': 'LINK',
                'litecoin': 'LTC',
                'stellar': 'XLM',
                'uniswap': 'UNI',
                'avalanche': 'AVAX'
            }

            if self.session:
                for coin_id, symbol in coingecko_coins.items():
                    try:
                        url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd&include_24hr_change=true"
                        async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                            if response.status == 200:
                                data = await response.json()
                                if coin_id in data and 'usd' in data[coin_id]:
                                    price = data[coin_id]['usd']
                                    change_24h = data[coin_id].get('usd_24h_change', 0)

                                    # Store in cache with CoinGecko prefix
                                    cache_key = f"coingecko_{symbol}_USD"
                                    self.cache.update_price(cache_key, price)
                                    self.cache.update_volume(cache_key, 0)  # CoinGecko doesn't provide volume in simple price

                                    logger.debug(f"CoinGecko: {symbol} = ${price:.2f} ({change_24h:+.2f}%)")

                        await asyncio.sleep(0.1)  # Rate limiting for CoinGecko

                    except Exception as e:
                        logger.warning(f"Failed to fetch CoinGecko price for {symbol}: {e}")

        except Exception as e:
            logger.error(f"Error fetching CoinGecko prices: {e}")

    async def _fetch_kraken_prices(self) -> None:
        """Fetch prices from Kraken API"""
        try:
            # Kraken uses different symbols (e.g., 'XBTUSD', 'ETHUSD')
            kraken_pairs = {
                'XBTUSD': 'BTC-USD',
                'ETHUSD': 'ETH-USD',
                'ADAUSD': 'ADA-USD',
                'SOLUSD': 'SOL-USD',
                'DOTUSD': 'DOT-USD',
                'LINKUSD': 'LINK-USD',
                'LTCUSD': 'LTC-USD',
                'XLMUSD': 'XLM-USD',
                'UNIUSD': 'UNI-USD',
                'AVAXUSD': 'AVAX-USD'
            }

            if self.session:
                for kraken_pair, standard_pair in kraken_pairs.items():
                    try:
                        url = f"https://api.kraken.com/0/public/Ticker?pair={kraken_pair}"
                        async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                            if response.status == 200:
                                data = await response.json()
                                if 'result' in data and kraken_pair in data['result']:
                                    ticker = data['result'][kraken_pair]
                                    price = float(ticker['c'][0])  # Current price
                                    volume_24h = float(ticker['v'][1])  # 24h volume

                                    # Store in cache with Kraken prefix
                                    cache_key = f"kraken_{standard_pair}"
                                    self.cache.update_price(cache_key, price)
                                    self.cache.update_volume(cache_key, volume_24h)

                                    logger.debug(f"Kraken: {standard_pair} = ${price:.2f} (vol: {volume_24h:.0f})")

                        await asyncio.sleep(0.1)  # Rate limiting for Kraken

                    except Exception as e:
                        logger.warning(f"Failed to fetch Kraken price for {kraken_pair}: {e}")

        except Exception as e:
            logger.error(f"Error fetching Kraken prices: {e}")


class VolumeFetcher:
    """Volume fetcher - runs every 2-3 minutes"""

    def __init__(self, cache: SharedCache) -> None:
        self.cache = cache
        self.config = get_config()
        self.session: Optional[aiohttp.ClientSession] = None
        self.last_fetch: float = 0

    async def start(self) -> None:
        """Start the volume fetcher"""
        self.session = aiohttp.ClientSession()
        logger.info("Volume fetcher started")

        while True:
            try:
                await self.fetch_all_volumes()
                await asyncio.sleep(self.config.fetcher_config.volume_fetch_interval)
            except Exception as e:
                logger.error(f"Error in volume fetcher: {e}")
                await asyncio.sleep(30)

    async def fetch_all_volumes(self) -> None:
        """Fetch volumes for coins that need updating"""
        current_time = time.time()

        # Check throttling
        if current_time - self.last_fetch < self.config.throttling_rules["volume"]["min_interval"]:
            return

        self.last_fetch = current_time

        # Check which coins need volume updates
        for coin_config in self.config.get_enabled_coins():
            symbol = coin_config.symbol
            if self.cache.should_update_volume(symbol):
                await self._fetch_coin_volume(symbol, coin_config.exchange)
                await asyncio.sleep(0.1)  # Small delay between requests

        logger.debug("Completed volume fetch")

    async def _fetch_coin_volume(self, symbol: str, exchange: str) -> None:
        """Fetch volume for a specific coin"""
        try:
            if exchange == "coinbase":
                await self._fetch_coinbase_volume(symbol)
            elif exchange == "binance":
                await self._fetch_binance_volume(symbol)
        except Exception as e:
            logger.error(f"Error fetching volume for {symbol}: {e}")

    async def _fetch_coinbase_volume(self, symbol: str) -> None:
        """Fetch volume for Coinbase coin"""
        try:
            if not self.session:
                return
            # Use the public exchange API like the price fetcher does
            url = f"https://api.exchange.coinbase.com/products/{symbol}/stats"
            async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status == 200:
                    stats = await response.json()
                    volume = float(stats["volume"])
                    self.cache.update_volume(symbol, volume)
                else:
                    logger.warning(f"Failed to fetch volume for {symbol}: {response.status}")
        except Exception as e:
            logger.error(f"Error fetching Coinbase volume for {symbol}: {e}")

    async def _fetch_binance_volume(self, symbol: str) -> None:
        """Fetch volume data from Binance"""
        try:
            if not self.session:
                return
            url = f"{self.config.api_config.binance_base_url}/ticker/24hr"
            params = {"symbol": symbol}
            async with self.session.get(
                url, params=params, timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                if response.status == 200:
                    ticker = await response.json()
                    volume = float(ticker["volume"])
                    self.cache.update_volume(symbol, volume)
                else:
                    logger.warning(f"Failed to fetch volume for {symbol}: {response.status}")
        except Exception as e:
            logger.error(f"Error fetching Binance volume for {symbol}: {e}")


class IndicatorCalculator:
    """Indicator calculator - runs every 1-3 minutes"""

    def __init__(self, cache: SharedCache) -> None:
        self.cache = cache
        self.config = get_config()

    async def start(self) -> None:
        """Start the indicator calculator"""
        logger.info("Indicator calculator started")

        while True:
            try:
                await self.calculate_all_indicators()
                await asyncio.sleep(self.config.fetcher_config.indicator_calc_interval)
            except Exception as e:
                logger.error(f"Error in indicator calculator: {e}")
                await asyncio.sleep(30)

    async def calculate_all_indicators(self) -> None:
        """Calculate indicators for coins that need updating"""
        for coin_config in self.config.get_enabled_coins():
            symbol = coin_config.symbol
            if self.cache.should_update_indicators(symbol):
                await self._calculate_coin_indicators(symbol)

        logger.debug("Completed indicator calculations")

    async def _calculate_coin_indicators(self, symbol: str):
        """Calculate indicators for a specific coin"""
        coin_data = self.cache.get_coin_cache(symbol)
        if not coin_data:
            return

        try:
            # Get price data from the cache
            price_data = coin_data.get("price", {})
            if not price_data:
                return

            # For now, use a simple approach with current price
            # In a real implementation, you'd want to build price history over time
            current_price = price_data.get("price", 0)
            if current_price <= 0:
                return

            # Create a simple price history for demonstration
            # In production, you'd want to store and retrieve actual price history

            # Calculate RSI (simplified for single price point)
            rsi = 50.0  # Default RSI when we don't have enough price history

            # Calculate MACD (simplified)
            macd = {"macd_line": 0.0, "signal_line": 0.0, "histogram": 0.0}

            # Calculate volatility (simplified)
            volatility = 0.0  # Default volatility when we don't have enough data

            # Update cache
            self.cache.update_indicators(symbol, rsi, macd, volatility)

        except Exception as e:
            logger.error(f"Error calculating indicators for {symbol}: {e}")

    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50.0

        gains: List[float] = []
        losses: List[float] = []

        for i in range(1, len(prices)):
            change = prices[i] - prices[i - 1]
            if change > 0:
                gains.append(change)
                losses.append(0.0)
            else:
                gains.append(0.0)
                losses.append(abs(change))

        if len(gains) < period:
            return 50.0

        # Calculate average gains and losses
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(self, prices: List[float]) -> Dict[str, float]:
        """Calculate MACD"""
        if len(prices) < 26:
            return {"macd_line": 0.0, "signal_line": 0.0, "histogram": 0.0}

        # Calculate EMAs
        ema12 = self._calculate_ema(prices, 12)
        ema26 = self._calculate_ema(prices, 26)

        macd_line = ema12 - ema26

        # Calculate signal line
        if len(prices) >= 35:
            macd_values: List[float] = []
            for i in range(26, len(prices)):
                ema12_i = self._calculate_ema(prices[: i + 1], 12)
                ema26_i = self._calculate_ema(prices[: i + 1], 26)
                macd_values.append(ema12_i - ema26_i)

            if len(macd_values) >= 9:
                signal_line = self._calculate_ema(macd_values, 9)
            else:
                signal_line = macd_line
        else:
            signal_line = macd_line

        histogram = macd_line - signal_line

        return {
            "macd_line": macd_line,
            "signal_line": signal_line,
            "histogram": histogram,
        }

    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """Calculate EMA"""
        if len(prices) < period:
            return prices[-1] if prices else 0.0

        multiplier = 2 / (period + 1)
        ema = sum(prices[:period]) / period

        for price in prices[period:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))

        return ema

    def _calculate_volatility(self, prices: List[float], period: int = 20) -> float:
        """Calculate volatility"""
        if len(prices) < period + 1:
            return 0.0

        returns: List[float] = []
        for i in range(1, len(prices)):
            if prices[i - 1] != 0:
                returns.append((prices[i] - prices[i - 1]) / prices[i - 1])

        if len(returns) < period:
            return 0.0

        # Calculate standard deviation of returns
        recent_returns = returns[-period:]
        mean_return = sum(recent_returns) / len(recent_returns)

        variance = sum((r - mean_return) ** 2 for r in recent_returns) / len(recent_returns)
        volatility = (variance**0.5) * 100  # Convert to percentage

        return volatility


class MysticFetcher:
    """Mystic data fetcher - calculates cosmic and lunar influences"""

    def __init__(self, cache: SharedCache):
        self.cache = cache
        self.config = get_config()

        # Fix Redis connection to use localhost first
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
                        f"✅ Redis connection established for MysticFetcher using {host}:{redis_port}"
                    )
                    break
                except Exception as e:
                    logger.warning(f"❌ Redis connection failed for {host}:{redis_port} - {e}")
                    continue

            if not hasattr(self, "redis_client") or self.redis_client is None:
                logger.warning("❌ All Redis connection attempts failed for MysticFetcher")
                self.redis_client = None

        except Exception as e:
            logger.error(f"❌ Redis connection failed for MysticFetcher: {e}")
            self.redis_client = None

        self.last_fetch = 0

    async def start(self):
        """Start the mystic fetcher"""
        logger.info("Mystic fetcher started")

        while True:
            try:
                await self.fetch_mystic_data()
                await asyncio.sleep(self.config.fetcher_config.mystic_fetch_interval)
            except Exception as e:
                logger.error(f"Error in mystic fetcher: {e}")
                await asyncio.sleep(300)  # 5 minutes on error

    async def fetch_mystic_data(self):
        """Fetch and calculate mystic data"""
        current_time = time.time()

        # Check if we have recent mystic data in Redis
        cached_data = self.redis_client.get("mystic_data")
        if cached_data:
            try:
                import json

                data = json.loads(str(cached_data))
                cache_time = data.get("timestamp", 0)
                if current_time - cache_time < 3600:  # 1 hour cache
                    logger.debug("Using cached mystic data")
                    return data
            except Exception as e:
                logger.warning(f"Error parsing cached mystic data: {e}")

        # Calculate new mystic data
        mystic_data = {
            "solar_activity": self._calculate_solar_activity(),
            "schumann_resonance": self._calculate_schumann_resonance(),
            "cosmic_alignment": self._calculate_cosmic_alignment(),
            "lunar_phase": self._calculate_lunar_phase(datetime.now()),
            "cosmic_energy": self._calculate_cosmic_energy(datetime.now()),
            "timestamp": current_time,
        }

        # Cache in Redis with timedelta-based expiration
        try:
            import json

            cache_expiry = timedelta(hours=1).total_seconds()
            self.redis_client.setex("mystic_data", int(cache_expiry), json.dumps(mystic_data))
        except Exception as e:
            logger.warning(f"Error caching mystic data: {e}")

        # Update cache
        self.cache.update_cosmic_data(mystic_data)
        self.last_fetch = current_time

        logger.debug("Updated mystic data")

    def _calculate_solar_activity(self) -> float:
        """Calculate solar activity index (simplified)"""
        # Simplified solar activity based on time of day
        hour = datetime.now().hour
        if 6 <= hour <= 18:  # Daytime
            return 0.8
        else:  # Nighttime
            return 0.2

    def _calculate_schumann_resonance(self) -> Dict[str, Any]:
        """Fetch Schumann resonance data"""
        # This is a simplified version - real implementation would need actual API
        return {
            "frequency": 7.83,  # Base frequency
            "amplitude": 0.5,  # Relative amplitude
            "timestamp": datetime.now().isoformat(),
            "harmonic": 1,
        }

    def _calculate_cosmic_alignment(self) -> Dict[str, Any]:
        """Calculate cosmic alignment factors"""
        now = datetime.now()

        return {
            "lunar_phase": self._calculate_lunar_phase(now),
            "solar_activity": self._calculate_solar_activity(),
            "cosmic_energy_index": self._calculate_cosmic_energy(now),
            "timestamp": now.isoformat(),
        }

    def _calculate_lunar_phase(self, dt: datetime) -> str:
        """Calculate lunar phase (simplified)"""
        # Simplified lunar phase calculation
        days_since_new_moon = (dt.day + dt.month * 30) % 29.5
        if days_since_new_moon < 7.4:
            return "new_moon"
        elif days_since_new_moon < 14.8:
            return "waxing_crescent"
        elif days_since_new_moon < 22.1:
            return "full_moon"
        else:
            return "waning_crescent"

    def _calculate_cosmic_energy(self, dt: datetime) -> float:
        """Calculate cosmic energy index (simplified)"""
        # Simplified cosmic energy calculation
        return 0.6  # Base level


class DataFetcherManager:
    """Manages all data fetchers"""

    def __init__(self, cache: SharedCache):
        self.cache = cache
        self.price_fetcher = PriceFetcher(cache)
        self.volume_fetcher = VolumeFetcher(cache)
        self.indicator_calculator = IndicatorCalculator(cache)
        self.mystic_fetcher = MysticFetcher(cache)

        self.tasks = []

    async def start_all(self):
        """Start all data fetchers"""
        logger.info("Starting all data fetchers...")

        # Start all fetchers as tasks
        self.tasks = [
            asyncio.create_task(self.price_fetcher.start()),
            asyncio.create_task(self.volume_fetcher.start()),
            asyncio.create_task(self.indicator_calculator.start()),
            asyncio.create_task(self.mystic_fetcher.start()),
        ]

        logger.info("All data fetchers started")

    async def stop_all(self):
        """Stop all data fetchers"""
        logger.info("Stopping all data fetchers...")

        for task in self.tasks:
            task.cancel()

        # Wait for all tasks to complete
        await asyncio.gather(*self.tasks, return_exceptions=True)

        logger.info("All data fetchers stopped")

    def get_status(self) -> Dict[str, Any]:
        """Get status of all fetchers"""
        return {
            "price_fetcher": {
                "running": not self.tasks[0].done() if self.tasks else False,
                "last_fetch": self.price_fetcher.last_fetch,
            },
            "volume_fetcher": {
                "running": (not self.tasks[1].done() if len(self.tasks) > 1 else False),
                "last_fetch": self.volume_fetcher.last_fetch,
            },
            "indicator_calculator": {
                "running": (not self.tasks[2].done() if len(self.tasks) > 2 else False)
            },
            "mystic_fetcher": {
                "running": (not self.tasks[3].done() if len(self.tasks) > 3 else False),
                "last_fetch": self.mystic_fetcher.last_fetch,
            },
        }
