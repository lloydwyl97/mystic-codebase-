"""
Market Data Manager Module for Mystic Trading Platform

Extracted from data_fetchers.py and cosmic_fetcher.py to improve modularity.
Handles live market data fetching, processing, and caching with standardized error handling.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

from backend.utils.exceptions import MarketDataException, handle_async_exception

logger = logging.getLogger(__name__)


@dataclass
class MarketData:
    """Market data structure"""

    symbol: str
    price: float
    volume: float
    change_24h: float
    high_24h: float
    low_24h: float
    timestamp: float
    exchange: str = "binance"

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
        }


class MarketDataManager:
    """Manages live market data fetching and processing"""

    def __init__(self):
        self.market_data: dict[str, MarketData] = {}
        self.last_update: dict[str, float] = {}
        self.update_interval = 30  # seconds
        self.supported_symbols = ["BTC", "ETH", "ADA", "DOT", "SOL", "MATIC"]
        logger.info("âœ… Live MarketDataManager initialized")

    @handle_async_exception("Failed to get market data for symbol", MarketDataException)
    async def get_market_data(self, symbol: str) -> MarketData | None:
        """Get live market data for a symbol"""
        try:
            # Check if data is fresh
            if self._is_data_fresh(symbol):
                return self.market_data.get(symbol)

            # Fetch live data from exchange APIs
            data = await self._fetch_live_market_data(symbol)
            if data:
                self.market_data[symbol] = data
                self.last_update[symbol] = time.time()
                return data

            return None
        except Exception as e:
            logger.error(f"âŒ Error getting live market data for {symbol}: {str(e)}")
            raise MarketDataException(
                message=f"Failed to get market data for {symbol}",
                details={"symbol": symbol, "error": str(e)},
            )

    @handle_async_exception("Failed to get all market data", MarketDataException)
    async def get_all_market_data(self) -> dict[str, MarketData]:
        """Get live market data for all supported symbols"""
        try:
            tasks = [self.get_market_data(symbol) for symbol in self.supported_symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            market_data: dict[str, MarketData] = {}
            for i, result in enumerate(results):
                if isinstance(result, MarketData):
                    market_data[self.supported_symbols[i]] = result
                else:
                    logger.warning(
                        f"Failed to get live data for {self.supported_symbols[i]}: {result}"
                    )

            return market_data
        except Exception as e:
            logger.error(f"âŒ Error getting all live market data: {str(e)}")
            raise MarketDataException(
                message="Failed to get all market data",
                details={"error": str(e)},
            )

    @handle_async_exception("Failed to get market summary", MarketDataException)
    async def get_market_summary(self) -> dict[str, Any]:
        """Get live market summary for all symbols"""
        try:
            market_data = await self.get_all_market_data()

            total_volume = sum(data.volume for data in market_data.values())
            avg_change = (
                sum(data.change_24h for data in market_data.values()) / len(market_data)
                if market_data
                else 0
            )

            return {
                "symbols": [data.to_dict() for data in market_data.values()],
                "total_symbols": len(market_data),
                "total_volume": total_volume,
                "average_change_24h": avg_change,
                "timestamp": time.time(),
                "live_data": True,
            }
        except Exception as e:
            logger.error(f"âŒ Error getting live market summary: {str(e)}")
            return {
                "symbols": [],
                "total_symbols": 0,
                "total_volume": 0,
                "average_change_24h": 0,
                "timestamp": time.time(),
                "live_data": False,
                "error": str(e),
            }

    def _is_data_fresh(self, symbol: str) -> bool:
        """Check if market data is fresh"""
        if symbol not in self.last_update:
            return False

        return time.time() - self.last_update[symbol] < self.update_interval

    @handle_async_exception("Failed to fetch live market data", MarketDataException)
    async def _fetch_live_market_data(self, symbol: str) -> MarketData | None:
        """Fetch live market data from exchange APIs"""
        try:
            # Try multiple live data sources
            data_sources = [
                self._fetch_binance_live_data,
                self._fetch_coinbase_live_data,
                self._fallback_live_data,
            ]

            for source in data_sources:
                try:
                    data = await source(symbol)
                    if data:
                        logger.debug(
                            f"âœ… Fetched live market data for {symbol} from {source.__name__}"
                        )
                        return data
                except Exception as e:
                    logger.debug(f"Source {source.__name__} failed for {symbol}: {e}")
                    continue

            logger.warning(f"No live data sources available for {symbol}")
            return None

        except Exception as e:
            logger.error(f"âŒ Error fetching live market data for {symbol}: {str(e)}")
            raise MarketDataException(
                message=f"Failed to fetch live market data for {symbol}",
                details={"symbol": symbol, "error": str(e)},
            )

    @handle_async_exception("Failed to fetch Binance live data", MarketDataException)
    async def _fetch_binance_live_data(self, symbol: str) -> MarketData | None:
        """Fetch live data from Binance US API"""
        try:
            # Import Binance client
            try:
                from binance.client import Client  # type: ignore
            except ImportError:
                logger.warning("Binance client not available")
                return None

            # Initialize Binance US client (no API key needed for public data)
            client = Client("", "", tld="us")

            # Get 24hr ticker
            ticker = client.get_ticker(symbol=f"{symbol}USDT")  # type: ignore

            data = MarketData(
                symbol=symbol,
                price=float(ticker["lastPrice"]),
                volume=float(ticker["volume"]),
                change_24h=float(ticker["priceChangePercent"]),
                high_24h=float(ticker["highPrice"]),
                low_24h=float(ticker["lowPrice"]),
                timestamp=time.time(),
                exchange="binance_us",
            )

            return data

        except Exception as e:
            logger.debug(f"Binance US API failed for {symbol}: {e}")
            return None

    @handle_async_exception("Failed to fetch Coinbase live data", MarketDataException)
    async def _fetch_coinbase_live_data(self, symbol: str) -> MarketData | None:
        """Fetch live data from Coinbase API"""
        try:
            import aiohttp

            # Coinbase Pro API endpoint
            url = f"https://api.pro.coinbase.us/products/{symbol}-USD/ticker"

            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        ticker = await response.json()

                        data = MarketData(
                            symbol=symbol,
                            price=float(ticker["price"]),
                            volume=float(ticker["volume"]),
                            change_24h=0.0,  # Coinbase doesn't provide this in ticker
                            high_24h=0.0,  # Would need separate API call
                            low_24h=0.0,  # Would need separate API call
                            timestamp=time.time(),
                            exchange="coinbase",
                        )

                        return data

            return None

        except Exception as e:
            logger.debug(f"Coinbase API failed for {symbol}: {e}")
            return None

    @handle_async_exception("Failed to fetch fallback live data", MarketDataException)
    async def _fallback_live_data(self, symbol: str) -> MarketData | None:
        """Fallback method for live data when primary sources fail"""
        try:
            import aiohttp

            # Try a public crypto API as fallback
            url = f"https://api.coingecko.com/api/v3/simple/price?ids={symbol.lower()}&vs_currencies=usd&include_24hr_vol=true&include_24hr_change=true"

            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        if symbol.lower() in data:
                            symbol_data = data[symbol.lower()]

                            market_data = MarketData(
                                symbol=symbol,
                                price=symbol_data.get("usd", 0.0),
                                volume=symbol_data.get("usd_24h_vol", 0.0),
                                change_24h=symbol_data.get("usd_24h_change", 0.0),
                                high_24h=0.0,  # Not provided by this API
                                low_24h=0.0,  # Not provided by this API
                                timestamp=time.time(),
                                exchange="coingecko",
                            )

                            return market_data

            return None

        except Exception as e:
            logger.error(f"Fallback live data failed for {symbol}: {e}")
            return None

    def get_supported_symbols(self) -> list[str]:
        """Get list of supported symbols"""
        return self.supported_symbols.copy()

    def add_symbol(self, symbol: str) -> bool:
        """Add a new symbol to supported list"""
        try:
            if symbol not in self.supported_symbols:
                self.supported_symbols.append(symbol)
                logger.info(f"âœ… Added symbol: {symbol}")
                return True
            return False
        except Exception as e:
            logger.error(f"âŒ Error adding symbol {symbol}: {str(e)}")
            return False

    def remove_symbol(self, symbol: str) -> bool:
        """Remove a symbol from supported list"""
        try:
            if symbol in self.supported_symbols:
                self.supported_symbols.remove(symbol)
                if symbol in self.market_data:
                    del self.market_data[symbol]
                if symbol in self.last_update:
                    del self.last_update[symbol]
                logger.info(f"âœ… Removed symbol: {symbol}")
                return True
            return False
        except Exception as e:
            logger.error(f"âŒ Error removing symbol {symbol}: {str(e)}")
            return False

    def get_statistics(self) -> dict[str, Any]:
        """Get market data statistics"""
        try:
            return {
                "supported_symbols": len(self.supported_symbols),
                "cached_symbols": len(self.market_data),
                "last_update": self.last_update,
                "update_interval": self.update_interval,
                "live_data": True,
            }
        except Exception as e:
            logger.error(f"âŒ Error getting market data statistics: {str(e)}")
            return {}


# Global market data manager instance
market_data_manager = MarketDataManager()


