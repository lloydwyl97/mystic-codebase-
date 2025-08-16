"""
Price Fetcher Service
Handles fetching live prices from multiple exchanges
"""

import logging
from typing import Dict, Any, Optional
import httpx  # type: ignore

logger = logging.getLogger(__name__)


class PriceFetcher:
    def __init__(self):
        self.supported_exchanges = ["binance", "coinbase", "coingecko"]

    async def fetch_price(
        self, symbol: str, exchange: str = "coingecko"
    ) -> Optional[Dict[str, Any]]:
        """Fetch price for a given symbol from specified exchange"""
        try:
            if exchange == "coingecko":
                return await self._fetch_from_coingecko(symbol)
            elif exchange == "binance":
                return await self._fetch_from_binance(symbol)
            elif exchange == "coinbase":
                return await self._fetch_from_coinbase(symbol)
            else:
                logger.warning(f"Unsupported exchange: {exchange}")
                return None
        except Exception as e:
            logger.error(f"Error fetching price for {symbol} from {exchange}: {e}")
            return None

    async def _fetch_from_coingecko(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch price from CoinGecko"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://api.coingecko.com/api/v3/simple/price",
                    params={
                        "ids": symbol.lower(),
                        "vs_currencies": "usd",
                        "include_24hr_change": True,
                        "include_market_cap": True,
                    },
                    timeout=10.0,
                )
                if response.status_code == 200:
                    data = response.json()
                    if symbol.lower() in data:
                        return {
                            "price": data[symbol.lower()]["usd"],
                            "change_24h": (data[symbol.lower()].get("usd_24h_change", 0)),
                            "market_cap": (data[symbol.lower()].get("usd_market_cap", 0)),
                            "source": "coingecko",
                        }
                return None
        except Exception as e:
            logger.error(f"CoinGecko fetch error: {e}")
            return None

    async def _fetch_from_binance(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch price from Binance"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://api.binance.us/api/v3/ticker/price",
                    params={"symbol": f"{symbol.upper()}USDT"},
                    timeout=10.0,
                )
                if response.status_code == 200:
                    data = response.json()
                    return {"price": float(data["price"]), "source": "binance"}
                return None
        except Exception as e:
            logger.error(f"Binance fetch error: {e}")
            return None

    async def _fetch_from_coinbase(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch price from Coinbase"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"https://api.coinbase.us/v2/prices/{symbol.upper()}-USD/spot",
                    timeout=10.0,
                )
                if response.status_code == 200:
                    data = response.json()
                    return {
                        "price": float(data["data"]["amount"]),
                        "source": "coinbase",
                    }
                return None
        except Exception as e:
            logger.error(f"Coinbase fetch error: {e}")
            return None


# Global instance
price_fetcher = PriceFetcher()


