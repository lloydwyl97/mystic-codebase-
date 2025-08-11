import logging
import requests
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class PriceFetcher:
    """Fetches price data from various exchanges"""

    def __init__(self, api_url: str):
        self.api_url = api_url

    def fetch_price(self, symbol: str) -> Optional[float]:
        """Fetch the current price for a given symbol"""
        try:
            response = requests.get(f"{self.api_url}/price/{symbol}")
            response.raise_for_status()
            data: Dict[str, float] = response.json()
            price = data.get("price")
            logger.info(f"Fetched price for {symbol}: {price}")
            return price
        except requests.RequestException as e:
            logger.error(f"Error fetching price for {symbol}: {e}")
            return None


# Example usage
if __name__ == "__main__":
    fetcher = PriceFetcher(api_url="https://api.example.com")
    btc_price = fetcher.fetch_price("BTCUSDT")
    if btc_price is not None:
        print(f"BTC Price: {btc_price}")
    else:
        print("Failed to fetch BTC price.")
