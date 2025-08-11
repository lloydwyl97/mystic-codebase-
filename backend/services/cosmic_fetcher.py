"""
Cosmic Fetcher Service
Handles fetching cosmic/market data from various sources
"""

import logging
from typing import Dict, Any, Optional
import httpx

logger = logging.getLogger(__name__)


class CosmicFetcher:
    def __init__(self):
        self.base_url = "https://api.coingecko.com/api/v3"
        self.supported_endpoints = ["global", "trending", "exchanges"]

    async def get_global_data(self) -> Optional[Dict[str, Any]]:
        """Fetch global market data"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/global", timeout=10.0)
                if response.status_code == 200:
                    return response.json()
                return None
        except Exception as e:
            logger.error(f"Error fetching global data: {e}")
            return None

    async def get_trending_coins(self) -> Optional[Dict[str, Any]]:
        """Fetch trending coins"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/search/trending", timeout=10.0)
                if response.status_code == 200:
                    return response.json()
                return None
        except Exception as e:
            logger.error(f"Error fetching trending coins: {e}")
            return None

    async def get_exchanges(self) -> Optional[Dict[str, Any]]:
        """Fetch exchange data"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/exchanges",
                    params={"per_page": 100},
                    timeout=10.0,
                )
                if response.status_code == 200:
                    return response.json()
                return None
        except Exception as e:
            logger.error(f"Error fetching exchanges: {e}")
            return None


# Global instance
cosmic_fetcher = CosmicFetcher()
