"""
Unified Data Fetcher for Mystic AI Trading Platform
Fetches live prices from multiple exchanges with rate limiting and stores in persistent cache.
"""

import asyncio
import aiohttp
import json
import time
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import sys
import os

# Add backend to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backend.modules.ai.persistent_cache import PersistentCache

logger = logging.getLogger(__name__)


class UnifiedFetcher:
    def __init__(self):
        """Initialize unified fetcher with rate limits and configuration"""
        self.cache = PersistentCache()
        self.session = None
        self.rate_limits = {
            'coinbase_us': {'requests': 3, 'window': 1},  # 3 req/sec
            'binance_us': {'requests': 10, 'window': 1},  # 10 req/sec
            'kraken_us': {'requests': 1, 'window': 1},   # 1 req/sec
            'coingecko': {'requests': 50, 'window': 60}  # 50 req/min
        }
        self.last_request = {exchange: 0 for exchange in self.rate_limits.keys()}
        self._load_config()

    def _load_config(self):
        """Load symbols configuration"""
        try:
            config_path = Path(__file__).parent.parent / 'config' / 'symbols_config.json'
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            logger.info("âœ… Loaded symbols configuration")
        except Exception as e:
            logger.error(f"âŒ Failed to load symbols config: {e}")
            raise

    async def _rate_limit(self, exchange: str):
        """Implement rate limiting for exchanges"""
        if exchange not in self.rate_limits:
            return

        limit = self.rate_limits[exchange]
        current_time = time.time()
        time_since_last = current_time - self.last_request[exchange]

        if time_since_last < limit['window'] / limit['requests']:
            sleep_time = (limit['window'] / limit['requests']) - time_since_last
            await asyncio.sleep(sleep_time)

        self.last_request[exchange] = time.time()

    async def _make_request(self, session: aiohttp.ClientSession, url: str,
                          headers: Optional[Dict] = None, timeout: int = 10) -> Optional[Dict]:
        """Make HTTP request with retry logic"""
        for attempt in range(3):
            try:
                async with session.get(url, headers=headers, timeout=timeout) as response:
                    if response.status == 429:
                        retry_after = int(response.headers.get('Retry-After', 60))
                        logger.warning(f"Rate limited, waiting {retry_after} seconds")
                        await asyncio.sleep(retry_after)
                        continue

                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.error(f"HTTP {response.status} for {url}")
                        return None

            except asyncio.TimeoutError:
                logger.warning(f"Timeout on attempt {attempt + 1} for {url}")
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                continue
            except Exception as e:
                logger.error(f"Request failed for {url}: {e}")
                return None

        return None

    async def fetch_coinbase_prices(self, session: aiohttp.ClientSession) -> Dict[str, float]:
        """Fetch prices from Coinbase US"""
        prices = {}
        symbols = self.config['exchanges']['coinbase_us']['symbols']

        for symbol in symbols:
            await self._rate_limit('coinbase_us')
            url = f"https://api.pro.coinbase.com/products/{symbol}/ticker"

            data = await self._make_request(session, url)
            if data and 'price' in data:
                price = float(data['price'])
                prices[symbol] = price
                self.cache.set_price('coinbase_us', symbol, price,
                                   float(data.get('volume', 0)))
                logger.debug(f"Coinbase {symbol}: ${price}")

        return prices

    async def fetch_binance_prices(self, session: aiohttp.ClientSession) -> Dict[str, float]:
        """Fetch prices from Binance US"""
        prices = {}
        symbols = self.config['exchanges']['binance_us']['symbols']

        for symbol in symbols:
            await self._rate_limit('binance_us')
            url = f"https://api.binance.us/api/v3/ticker/price?symbol={symbol}"

            data = await self._make_request(session, url)
            if data and 'price' in data:
                price = float(data['price'])
                prices[symbol] = price
                self.cache.set_price('binance_us', symbol, price)
                logger.debug(f"Binance US {symbol}: ${price}")

        return prices

    async def fetch_kraken_prices(self, session: aiohttp.ClientSession) -> Dict[str, float]:
        """Fetch prices from Kraken US"""
        prices = {}
        symbols = self.config['exchanges']['kraken_us']['symbols']

        for symbol in symbols:
            await self._rate_limit('kraken_us')
            url = f"https://api.kraken.com/0/public/Ticker?pair={symbol}"

            data = await self._make_request(session, url)
            if data and 'result' in data and symbol in data['result']:
                ticker_data = data['result'][symbol]
                price = float(ticker_data['c'][0])  # Current price
                prices[symbol] = price
                self.cache.set_price('kraken_us', symbol, price,
                                   float(ticker_data.get('v', [0])[0]))
                logger.debug(f"Kraken {symbol}: ${price}")

        return prices

    async def fetch_coingecko_prices(self, session: aiohttp.ClientSession) -> Dict[str, float]:
        """Fetch prices from CoinGecko"""
        prices = {}
        symbols = self.config['exchanges']['coingecko']['symbols']

        # CoinGecko allows batch requests
        await self._rate_limit('coingecko')
        symbol_ids = ','.join(symbols)
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={symbol_ids}&vs_currencies=usd&include_24hr_vol=true"

        data = await self._make_request(session, url)
        if data:
            for symbol_id in symbols:
                if symbol_id in data and 'usd' in data[symbol_id]:
                    price = float(data[symbol_id]['usd'])
                    volume = float(data[symbol_id].get('usd_24h_vol', 0))

                    # Map CoinGecko ID to symbol
                    symbol_mapping = {
                        'bitcoin': 'BTC-USD',
                        'ethereum': 'ETH-USD',
                        'cardano': 'ADA-USD',
                        'solana': 'SOL-USD',
                        'polkadot': 'DOT-USD',
                        'chainlink': 'LINK-USD',
                        'matic-network': 'MATIC-USD',
                        'avalanche-2': 'AVAX-USD',
                        'uniswap': 'UNI-USD',
                        'cosmos': 'ATOM-USD'
                    }

                    symbol = symbol_mapping.get(symbol_id, symbol_id)
                    prices[symbol] = price
                    self.cache.set_price('coingecko', symbol, price, volume)
                    logger.debug(f"CoinGecko {symbol}: ${price}")

        return prices

    async def fetch_all_exchanges(self) -> Dict[str, Dict[str, float]]:
        """Fetch prices from all exchanges"""
        async with aiohttp.ClientSession() as session:
            tasks = [
                self.fetch_coinbase_prices(session),
                self.fetch_binance_prices(session),
                self.fetch_kraken_prices(session),
                self.fetch_coingecko_prices(session)
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            exchanges = ['coinbase_us', 'binance_us', 'kraken_us', 'coingecko']
            all_prices = {}

            for exchange, result in zip(exchanges, results):
                if isinstance(result, Exception):
                    logger.error(f"âŒ {exchange} fetch failed: {result}")
                    all_prices[exchange] = {}
                else:
                    all_prices[exchange] = result
                    logger.info(f"âœ… {exchange}: {len(result)} prices fetched")

            return all_prices

    def aggregate_prices(self, all_prices: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Aggregate prices from multiple exchanges using weighted average"""
        aggregated = {}
        symbol_mapping = self.config['symbol_mapping']

        for base_symbol in symbol_mapping.keys():
            prices = []
            weights = []

            for exchange, prices_dict in all_prices.items():
                if exchange in symbol_mapping[base_symbol]:
                    exchange_symbol = symbol_mapping[base_symbol][exchange]
                    if exchange_symbol in prices_dict:
                        prices.append(prices_dict[exchange_symbol])
                        # Weight by exchange priority (lower number = higher priority)
                        priority = self.config['exchanges'][exchange]['priority']
                        weights.append(1.0 / priority)

            if prices:
                # Calculate weighted average
                total_weight = sum(weights)
                weighted_price = sum(p * w for p, w in zip(prices, weights)) / total_weight
                aggregated[base_symbol] = weighted_price
                logger.info(f"Aggregated {base_symbol}: ${weighted_price:.2f} from {len(prices)} exchanges")

        return aggregated

    async def run_all(self) -> Dict[str, Any]:
        """Run the complete unified fetching process"""
        try:
            logger.info("ðŸš€ Starting unified data fetch from all exchanges...")
            start_time = time.time()

            # Fetch from all exchanges
            all_prices = await self.fetch_all_exchanges()

            # Aggregate prices
            aggregated_prices = self.aggregate_prices(all_prices)

            # Store aggregated prices in cache
            for symbol, price in aggregated_prices.items():
                self.cache.set_price('aggregated', symbol, price)

            # Get cache statistics
            stats = self.cache.get_cache_stats()

            end_time = time.time()
            duration = end_time - start_time

            result = {
                "success": True,
                "duration": duration,
                "exchanges_fetched": len([p for p in all_prices.values() if p]),
                "total_symbols": len(aggregated_prices),
                "cache_stats": stats,
                "aggregated_prices": aggregated_prices
            }

            logger.info(f"âœ… Unified fetch completed in {duration:.2f}s")
            logger.info(f"ðŸ“Š Fetched {len(aggregated_prices)} symbols from {result['exchanges_fetched']} exchanges")

            return result

        except Exception as e:
            logger.error(f"âŒ Unified fetch failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "duration": time.time() - start_time if 'start_time' in locals() else 0
            }


# Global fetcher instance
unified_fetcher = UnifiedFetcher()


async def run_all() -> Dict[str, Any]:
    """Run the unified fetcher (async wrapper)"""
    return await unified_fetcher.run_all()


def run_all_sync() -> Dict[str, Any]:
    """Run the unified fetcher (synchronous wrapper)"""
    return asyncio.run(run_all())


if __name__ == "__main__":
    # Test the fetcher
    result = run_all_sync()
    print(json.dumps(result, indent=2))


