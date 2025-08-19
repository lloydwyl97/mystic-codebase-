"""
CoinGecko Service
Provides cryptocurrency data from CoinGecko API
"""

import logging
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class CoinGeckoService:
    def __init__(self):
        self.base_url = "https://api.coingecko.com/api/v3"
        self.session = None
        self.cache = {}
        self.last_update = 0
        self.update_interval = 60  # 60 seconds

    async def get_simple_price(self, ids: list[str], vs_currencies: list[str] = ["usd"]) -> dict[str, Any]:
        """Get simple price data for multiple coins"""
        try:
            url = f"{self.base_url}/simple/price"
            params = {
                "ids": ",".join(ids),
                "vs_currencies": ",".join(vs_currencies),
                "include_market_cap": "true",
                "include_24hr_vol": "true",
                "include_24hr_change": "true"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"CoinGecko API error: {response.status_code}")
                    return {}
        except Exception as e:
            logger.error(f"Error fetching CoinGecko data: {e}")
            return {}

    async def get_coin_markets(self, vs_currency: str = "usd", order: str = "market_cap_desc", 
                              per_page: int = 100, page: int = 1, sparkline: bool = False) -> list[dict[str, Any]]:
        """Get market data for coins"""
        try:
            url = f"{self.base_url}/coins/markets"
            params = {
                "vs_currency": vs_currency,
                "order": order,
                "per_page": per_page,
                "page": page,
                "sparkline": sparkline
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"CoinGecko markets API error: {response.status_code}")
                    return []
        except Exception as e:
            logger.error(f"Error fetching CoinGecko markets: {e}")
            return []

    async def get_coin_by_id(self, coin_id: str) -> dict[str, Any] | None:
        """Get detailed data for a specific coin"""
        try:
            url = f"{self.base_url}/coins/{coin_id}"
            params = {
                "localization": "false",
                "tickers": "false",
                "market_data": "true",
                "community_data": "false",
                "developer_data": "false",
                "sparkline": "false"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"CoinGecko coin API error: {response.status_code}")
                    return None
        except Exception as e:
            logger.error(f"Error fetching CoinGecko coin data: {e}")
            return None

    async def get_trending(self) -> list[dict[str, Any]]:
        """Get trending coins"""
        try:
            url = f"{self.base_url}/search/trending"
            
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    return data.get("coins", [])
                else:
                    logger.error(f"CoinGecko trending API error: {response.status_code}")
                    return []
        except Exception as e:
            logger.error(f"Error fetching CoinGecko trending data: {e}")
            return []

    async def get_global_data(self) -> dict[str, Any]:
        """Get global cryptocurrency data"""
        try:
            url = f"{self.base_url}/global"
            
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=10)
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"CoinGecko global API error: {response.status_code}")
                    return {}
        except Exception as e:
            logger.error(f"Error fetching CoinGecko global data: {e}")
            return {}

    def get_cached_data(self, key: str) -> dict[str, Any] | None:
        """Get cached data if available and not expired"""
        if key in self.cache:
            data, timestamp = self.cache[key]
            if time.time() - timestamp < self.update_interval:
                return data
        return None

    def cache_data(self, key: str, data: dict[str, Any]):
        """Cache data with timestamp"""
        self.cache[key] = (data, time.time()) 