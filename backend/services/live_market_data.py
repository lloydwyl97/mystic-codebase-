"""
Live Market Data Service
Connects to real APIs for live market data
"""

import asyncio
import aiohttp
import ccxt
import yfinance as yf
from typing import Dict, Any, List, Optional
import os
from datetime import datetime
import logging
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LiveMarketDataService:
    """Service for fetching live market data from real APIs"""

    def __init__(self):
        self.binance_api_key = os.getenv("BINANCE_API_KEY")
        self.binance_secret = os.getenv("BINANCE_SECRET")
        self.alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        self.polygon_key = os.getenv("POLYGON_API_KEY")

        # Initialize exchange connections
        self.binance = (
            ccxt.binance(
                {
                    "apiKey": self.binance_api_key,
                    "secret": self.binance_secret,
                    "sandbox": False,
                }
            )
            if self.binance_api_key
            else None
        )

        self.coinbase = ccxt.coinbase({"sandbox": False})

        # Cache for rate limiting
        self.cache = {}
        self.cache_timeout = 30  # seconds

    async def get_live_prices(self, symbols: List[str]) -> Dict[str, Any]:
        """Get live prices for multiple symbols"""
        try:
            results = {}

            for symbol in symbols:
                # Try multiple data sources
                price_data = await self._get_symbol_price(symbol)
                if price_data:
                    results[symbol] = price_data

            return {
                "status": "success",
                "data": results,
                "timestamp": datetime.now().isoformat(),
                "source": "live_apis",
            }

        except Exception as e:
            logger.error(f"Error fetching live prices: {e}")
            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    async def _get_symbol_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get price for a single symbol from multiple sources"""
        try:
            # Try Binance first (most reliable for crypto)
            if self.binance and "/" in symbol:
                ticker = await asyncio.to_thread(self.binance.fetch_ticker, symbol)
                return {
                    "price": float(ticker["last"]),
                    "change_24h": float(ticker["percentage"]),
                    "volume": float(ticker["baseVolume"]),
                    "high_24h": float(ticker["high"]),
                    "low_24h": float(ticker["low"]),
                    "source": "binance",
                }

            # Try Coinbase for crypto
            elif "/" in symbol:
                ticker = await asyncio.to_thread(self.coinbase.fetch_ticker, symbol)
                return {
                    "price": float(ticker["last"]),
                    "change_24h": float(ticker["percentage"]),
                    "volume": float(ticker["baseVolume"]),
                    "high_24h": float(ticker["high"]),
                    "low_24h": float(ticker["low"]),
                    "source": "coinbase",
                }

            # Try Yahoo Finance for stocks
            else:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="2d")

                if not hist.empty:
                    current_price = hist["Close"].iloc[-1]
                    prev_price = hist["Close"].iloc[-2]
                    change_24h = ((current_price - prev_price) / prev_price) * 100

                    return {
                        "price": float(current_price),
                        "change_24h": float(change_24h),
                        "volume": float(hist["Volume"].iloc[-1]),
                        "high_24h": float(hist["High"].iloc[-1]),
                        "low_24h": float(hist["Low"].iloc[-1]),
                        "source": "yahoo_finance",
                    }

        except Exception as e:
            logger.error(f"Error fetching price for {symbol}: {e}")
            return None

    async def get_market_overview(self) -> Dict[str, Any]:
        """Get market overview with major indices and crypto"""
        try:
            # Major crypto pairs
            crypto_symbols = ["BTC/USDT", "ETH/USDT", "ADA/USDT", "SOL/USDT"]
            crypto_data = await self.get_live_prices(crypto_symbols)

            # Major stock indices
            stock_symbols = [
                "^GSPC",
                "^DJI",
                "^IXIC",
                "^VIX",
            ]  # S&P 500, Dow, NASDAQ, VIX
            stock_data = await self.get_live_prices(stock_symbols)

            return {
                "status": "success",
                "crypto": crypto_data.get("data", {}),
                "stocks": stock_data.get("data", {}),
                "timestamp": datetime.now().isoformat(),
                "source": "live_apis",
            }

        except Exception as e:
            logger.error(f"Error fetching market overview: {e}")
            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    async def get_historical_data(
        self, symbol: str, timeframe: str = "1d", limit: int = 100
    ) -> Dict[str, Any]:
        """Get historical data for charting"""
        try:
            if self.binance and "/" in symbol:
                # Get from Binance
                ohlcv = await asyncio.to_thread(
                    self.binance.fetch_ohlcv, symbol, timeframe, limit=limit
                )

                return {
                    "status": "success",
                    "data": {
                        "timestamps": [candle[0] for candle in ohlcv],
                        "opens": [candle[1] for candle in ohlcv],
                        "highs": [candle[2] for candle in ohlcv],
                        "lows": [candle[3] for candle in ohlcv],
                        "closes": [candle[4] for candle in ohlcv],
                        "volumes": [candle[5] for candle in ohlcv],
                    },
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "source": "binance",
                }

            else:
                # Get from Yahoo Finance
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=f"{limit}d")

                return {
                    "status": "success",
                    "data": {
                        "timestamps": [int(dt.timestamp() * 1000) for dt in hist.index],
                        "opens": hist["Open"].tolist(),
                        "highs": hist["High"].tolist(),
                        "lows": hist["Low"].tolist(),
                        "closes": hist["Close"].tolist(),
                        "volumes": hist["Volume"].tolist(),
                    },
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "source": "yahoo_finance",
                }

        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return {"status": "error", "message": str(e), "symbol": symbol}

    async def get_market_data(self, currency: str = "usd", per_page: int = 100) -> Dict[str, Any]:
        """Get market data for top cryptocurrencies"""
        try:
            # Use CoinGecko API for market data
            url = "https://api.coingecko.com/api/v3/coins/markets"
            params = {
                "vs_currency": currency,
                "order": "market_cap_desc",
                "per_page": per_page,
                "page": 1,
                "sparkline": False,
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "status": "success",
                            "coins": data,
                            "timestamp": datetime.now().isoformat(),
                            "source": "coingecko",
                        }
                    else:
                        return {
                            "status": "error",
                            "message": f"API error: {response.status}",
                            "timestamp": datetime.now().isoformat(),
                        }

        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    async def get_global_data(self) -> Dict[str, Any]:
        """Get global cryptocurrency market data"""
        try:
            url = "https://api.coingecko.com/api/v3/global"

            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "status": "success",
                            "data": data["data"],
                            "timestamp": datetime.now().isoformat(),
                            "source": "coingecko",
                        }
                    else:
                        return {
                            "status": "error",
                            "message": f"API error: {response.status}",
                            "timestamp": datetime.now().isoformat(),
                        }

        except Exception as e:
            logger.error(f"Error fetching global data: {e}")
            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    async def get_trending_coins(self) -> Dict[str, Any]:
        """Get trending coins"""
        try:
            url = "https://api.coingecko.com/api/v3/search/trending"

            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "status": "success",
                            "data": data["coins"],
                            "timestamp": datetime.now().isoformat(),
                            "source": "coingecko",
                        }
                    else:
                        return {
                            "status": "error",
                            "message": f"API error: {response.status}",
                            "timestamp": datetime.now().isoformat(),
                        }

        except Exception as e:
            logger.error(f"Error fetching trending coins: {e}")
            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    async def get_coin_price(self, symbol: str, currency: str = "usd") -> Dict[str, Any]:
        """Get current price for a specific coin"""
        try:
            url = "https://api.coingecko.com/api/v3/simple/price"
            params = {"ids": symbol.lower(), "vs_currencies": currency}

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "status": "success",
                            "data": data,
                            "timestamp": datetime.now().isoformat(),
                            "source": "coingecko",
                        }
                    else:
                        return {
                            "status": "error",
                            "message": f"API error: {response.status}",
                            "timestamp": datetime.now().isoformat(),
                        }

        except Exception as e:
            logger.error(f"Error fetching coin price: {e}")
            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    async def get_coin_details(self, symbol: str) -> Dict[str, Any]:
        """Get detailed information for a specific coin"""
        try:
            url = f"https://api.coingecko.com/api/v3/coins/{symbol.lower()}"

            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "status": "success",
                            "data": data,
                            "timestamp": datetime.now().isoformat(),
                            "source": "coingecko",
                        }
                    else:
                        return {
                            "status": "error",
                            "message": f"API error: {response.status}",
                            "timestamp": datetime.now().isoformat(),
                        }

        except Exception as e:
            logger.error(f"Error fetching coin details: {e}")
            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    async def search_coins(self, query: str) -> Dict[str, Any]:
        """Search for coins by name or symbol"""
        try:
            url = "https://api.coingecko.com/api/v3/search"
            params = {"query": query}

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "status": "success",
                            "data": data["coins"],
                            "timestamp": datetime.now().isoformat(),
                            "source": "coingecko",
                        }
                    else:
                        return {
                            "status": "error",
                            "message": f"API error: {response.status}",
                            "timestamp": datetime.now().isoformat(),
                        }

        except Exception as e:
            logger.error(f"Error searching coins: {e}")
            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    async def get_market_summary(self) -> Dict[str, Any]:
        """Get market summary from persistent cache"""
        try:
            # Get real market summary from persistent cache
            from backend.ai.persistent_cache import get_persistent_cache

            cache = get_persistent_cache()
            symbols = []
            total_volume = 0
            total_change = 0
            symbol_count = 0

            # Process all available market data
            for symbol, price_data in cache.get_binance().items():
                base_symbol = symbol.replace("USDT", "")
                coingecko_data = cache.get_coingecko().get(base_symbol.lower(), {})

                if isinstance(price_data, dict) and "price" in price_data:
                    price = price_data["price"]
                else:
                    price = float(price_data) if price_data else 0

                volume = coingecko_data.get("volume_24h", 1000.0)
                change_24h = coingecko_data.get("price_change_24h", 0)

                symbols.append(
                    {
                        "symbol": base_symbol,
                        "price": float(price),
                        "volume": volume,
                        "change_24h": change_24h,
                    }
                )

                total_volume += volume
                total_change += change_24h
                symbol_count += 1

            # Add Coinbase data for symbols not in Binance
            for symbol, price in cache.get_coinbase().items():
                base_symbol = symbol.replace("-USD", "")
                if base_symbol not in [s["symbol"] for s in symbols]:
                    coingecko_data = cache.get_coingecko().get(base_symbol.lower(), {})

                    volume = coingecko_data.get("volume_24h", 1000.0)
                    change_24h = coingecko_data.get("price_change_24h", 0)

                    symbols.append(
                        {
                            "symbol": base_symbol,
                            "price": float(price),
                            "volume": volume,
                            "change_24h": change_24h,
                        }
                    )

                    total_volume += volume
                    total_change += change_24h
                    symbol_count += 1

            average_change = total_change / symbol_count if symbol_count > 0 else 0

            # Return format expected by api_endpoints.py
            return {
                "symbols": symbols,
                "total_symbols": symbol_count,
                "total_volume": total_volume,
                "average_change_24h": average_change,
                "timestamp": time.time(),
                "live_data": True,
            }
        except Exception as e:
            logger.error(f"Error getting market summary: {str(e)}")
            return {
                "symbols": [],
                "total_symbols": 0,
                "total_volume": 0,
                "average_change_24h": 0,
                "timestamp": time.time(),
                "live_data": False,
            }

    async def get_symbol_data(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive data for a specific symbol"""
        try:
            # Get price data
            price_data = await self.get_coin_price(symbol, "usd")

            # Get detailed coin info
            await self.get_coin_details(symbol)

            # Get historical data
            await self.get_historical_data(symbol, "1d", 30)

            # Extract price info for flat structure expected by api_endpoints.py
            price_info = price_data.get("data", {})
            if isinstance(price_info, dict):
                price = price_info.get("price", 0)
                volume = price_info.get("volume", 0)
                change_24h = price_info.get("change_24h", 0)
                high_24h = price_info.get("high_24h", 0)
                low_24h = price_info.get("low_24h", 0)
            else:
                price = float(price_info) if price_info else 0
                volume = 0
                change_24h = 0
                high_24h = 0
                low_24h = 0

            # Return format expected by api_endpoints.py
            return {
                "price": price,
                "volume": volume,
                "change_24h": change_24h,
                "high_24h": high_24h,
                "low_24h": low_24h,
                "exchange": "binance" if self.binance else "coinbase",
                "timestamp": int(time.time()),
                "live_data": True,
            }
        except Exception as e:
            logger.error(f"Error getting symbol data for {symbol}: {e}")
            return {
                "price": 0,
                "volume": 0,
                "change_24h": 0,
                "high_24h": 0,
                "low_24h": 0,
                "exchange": "unknown",
                "timestamp": int(time.time()),
                "live_data": False,
            }

    async def get_candlestick_data(self, symbol: str, interval: str = "1h") -> Dict[str, Any]:
        """Get candlestick data for a symbol"""
        try:
            # Convert interval to timeframe
            timeframe_map = {
                "1m": "1m",
                "5m": "5m",
                "15m": "15m",
                "30m": "30m",
                "1h": "1h",
                "4h": "4h",
                "1d": "1d",
                "1w": "1w",
            }
            timeframe = timeframe_map.get(interval, "1h")

            # Get historical data as candlestick format
            historical_data = await self.get_historical_data(symbol, timeframe, 100)

            if historical_data.get("status") == "success":
                data = historical_data.get("data", {})
                candles = [
                    {
                        "timestamp": data["timestamps"][i],
                        "open": data["opens"][i],
                        "high": data["highs"][i],
                        "low": data["lows"][i],
                        "close": data["closes"][i],
                        "volume": data["volumes"][i],
                    }
                    for i in range(len(data.get("timestamps", [])))
                ]

                # Return format expected by api_endpoints.py
                return candles
            else:
                return []
        except Exception as e:
            logger.error(f"Error getting candlestick data for {symbol}: {e}")
            return []

    async def get_order_book(self, symbol: str, limit: int = 10) -> Dict[str, Any]:
        """Get order book for a symbol"""
        try:
            if self.binance and "/" in symbol:
                # Get from Binance
                order_book = await asyncio.to_thread(self.binance.fetch_order_book, symbol, limit)

                # Return format expected by api_endpoints.py
                return {
                    "bids": order_book["bids"][:limit],
                    "asks": order_book["asks"][:limit],
                }
            else:
                return {"bids": [], "asks": []}
        except Exception as e:
            logger.error(f"Error getting order book for {symbol}: {e}")
            return {"bids": [], "asks": []}

    async def get_technical_indicators(self, symbol: str) -> Dict[str, Any]:
        """Get technical indicators for a symbol"""
        try:
            # Get historical data for calculations
            historical_data = await self.get_historical_data(symbol, "1d", 50)

            if historical_data.get("status") == "success":
                data = historical_data.get("data", {})
                closes = data.get("closes", [])

                if len(closes) >= 14:
                    # Calculate RSI
                    rsi = self._calculate_rsi(closes)

                    # Calculate moving averages
                    sma_20 = (
                        sum(closes[-20:]) / 20 if len(closes) >= 20 else sum(closes) / len(closes)
                    )
                    sma_50 = (
                        sum(closes[-50:]) / 50 if len(closes) >= 50 else sum(closes) / len(closes)
                    )

                    # Calculate MACD
                    macd = self._calculate_macd(closes)

                    # Return format expected by api_endpoints.py
                    return {
                        "rsi": rsi,
                        "sma_20": sma_20,
                        "sma_50": sma_50,
                        "macd": macd,
                        "current_price": closes[-1] if closes else 0,
                    }
                else:
                    return {}
            else:
                return {}
        except Exception as e:
            logger.error(f"Error getting technical indicators for {symbol}: {e}")
            return {}

    def _calculate_rsi(self, prices: list, period: int = 14) -> float:
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50.0

        deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
        gains = [delta if delta > 0 else 0 for delta in deltas]
        losses = [-delta if delta < 0 else 0 for delta in deltas]

        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(self, prices: list) -> Dict[str, float]:
        """Calculate MACD"""
        if len(prices) < 26:
            return {"macd": 0, "signal": 0, "histogram": 0}

        # Calculate EMAs
        ema_12 = self._calculate_ema(prices, 12)
        ema_26 = self._calculate_ema(prices, 26)

        macd_line = ema_12 - ema_26

        # For signal line, we'd need more data, so using a simple approach
        signal_line = macd_line * 0.8  # Simplified

        histogram = macd_line - signal_line

        return {
            "macd": macd_line,
            "signal": signal_line,
            "histogram": histogram,
        }

    def _calculate_ema(self, prices: list, period: int) -> float:
        """Calculate EMA"""
        if len(prices) < period:
            return prices[-1] if prices else 0

        multiplier = 2 / (period + 1)
        ema = prices[0]

        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))

        return ema


# Global instance
live_market_data_service = LiveMarketDataService()


