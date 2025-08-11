"""
Smart Data Feed Engine

Polls top 4 coins from Binance and Coinbase
Hits CoinGecko every 5 minutes
Feeds all cached data into AI + trade system
"""

import logging

logger = logging.getLogger("ai_poller")


class AIPoller:
    """AI Poller class for data polling"""
    
    def __init__(self):
        self.name = "AI Poller"
        self.cache = DataCache()


class DataCache:
    def __init__(self):
        self.binance = {}
        self.coinbase = {}
        self.coingecko = {}
        self.last_update = {}


cache = DataCache()

BINANCE_SYMBOLS = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "SOLUSDT"]
COINBASE_SYMBOLS = ["BTC-USD", "ETH-USD", "ADA-USD", "SOL-USD"]


def get_cache() -> DataCache:
    """Get the current data cache"""
    return cache
