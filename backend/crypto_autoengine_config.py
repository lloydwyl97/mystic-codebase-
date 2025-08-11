#!/usr/bin/env python3
"""
CRYPTO AUTOENGINE Configuration
Central configuration for 20 coins (10 Coinbase + 10 Binance US)
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict, Union


@dataclass
class CoinConfig:
    """Configuration for individual coins"""

    symbol: str
    exchange: str  # 'coinbase' or 'binance'
    base_currency: str
    quote_currency: str
    min_trade_amount: float
    max_trade_amount: float
    enabled: bool = True


@dataclass
class FetcherConfig:
    """Configuration for data fetchers"""

    price_fetch_interval: int = 10  # seconds
    volume_fetch_interval: int = 180  # 3 minutes
    indicator_calc_interval: int = 120  # 2 minutes
    mystic_fetch_interval: int = 3600  # 1 hour
    cache_ttl: int = 300  # 5 minutes
    price_change_threshold: float = 0.002  # 0.2%


@dataclass
class StrategyConfig:
    """Configuration for trading strategies"""

    min_confidence: float = 0.7
    max_confidence: float = 0.95
    min_signal_strength: float = 0.6
    strategy_count: int = 65  # P-wise strategies
    cooldown_period: int = 300  # 5 minutes between trades


@dataclass
class APIConfig:
    """API configuration"""

    binance_base_url: str = "https://api.binance.us/api/v3"
    coinbase_base_url: str = "https://api.coinbase.com/api/v3/brokerage"
    coinbase_rest_url: str = "https://api.coinbase.com/v2"
    noaa_base_url: str = "https://services.swpc.noaa.gov/json"
    schumann_base_url: str = "https://www2.irf.se/maggraphs/schumann"
    max_retries: int = 3
    timeout: int = 10


class PriceCache(TypedDict):
    """Type definition for price cache"""

    price: float
    timestamp: float


class VolumeCache(TypedDict):
    """Type definition for volume cache"""

    volume: float
    timestamp: float


class RSICache(TypedDict):
    """Type definition for RSI cache"""

    value: float
    timestamp: float


class MACDCache(TypedDict):
    """Type definition for MACD cache"""

    value: Dict[str, float]  # Contains 'macd', 'signal', 'histogram'
    timestamp: float


class LastUpdatedCache(TypedDict):
    """Type definition for last updated timestamps"""

    timestamp: float


class StrategySignalCache(TypedDict):
    """Type definition for strategy signals cache"""

    signal: float
    confidence: float
    timestamp: float


class CosmicDataCache(TypedDict):
    """Type definition for cosmic data cache"""

    data: Dict[str, Any]
    timestamp: float


class TradeCooldownCache(TypedDict):
    """Type definition for trade cooldowns"""

    until: float  # Timestamp until cooldown expires


class CryptoAutoEngineConfig:
    """Complete CRYPTO AUTOENGINE configuration"""

    coinbase_coins: List[CoinConfig]
    binance_coins: List[CoinConfig]
    all_coins: List[CoinConfig]
    fetcher_config: FetcherConfig
    strategy_config: StrategyConfig
    api_config: APIConfig
    cache_structure: Dict[
        str,
        Dict[
            str,
            Union[
                PriceCache,
                VolumeCache,
                RSICache,
                MACDCache,
                LastUpdatedCache,
                StrategySignalCache,
                CosmicDataCache,
                TradeCooldownCache,
            ],
        ],
    ]
    throttling_rules: Dict[str, Dict[str, Union[int, float]]]

    def __init__(self):
        # 1. COIN CONFIGURATION - 20 COINS PER EXCHANGE (USER'S SPECIFIED COINS)

        # Top 10 Coinbase coins (U.S.-Accessible, High Potential)
        self.coinbase_coins = [
            CoinConfig("BTC-USD", "coinbase", "BTC", "USD", 10.0, 5000.0),
            CoinConfig("ETH-USD", "coinbase", "ETH", "USD", 10.0, 5000.0),
            CoinConfig("ADA-USD", "coinbase", "ADA", "USD", 10.0, 5000.0),
            CoinConfig("SOL-USD", "coinbase", "SOL", "USD", 10.0, 5000.0),
            CoinConfig("DOT-USD", "coinbase", "DOT", "USD", 10.0, 5000.0),
            CoinConfig("LINK-USD", "coinbase", "LINK", "USD", 10.0, 5000.0),
            CoinConfig("MATIC-USD", "coinbase", "MATIC", "USD", 10.0, 5000.0),
            CoinConfig("AVAX-USD", "coinbase", "AVAX", "USD", 10.0, 5000.0),
            CoinConfig("UNI-USD", "coinbase", "UNI", "USD", 10.0, 5000.0),
            CoinConfig("ATOM-USD", "coinbase", "ATOM", "USD", 10.0, 5000.0),
        ]

        # Top 10 Binance coins (U.S.-Accessible, High Potential)
        self.binance_coins = [
            CoinConfig("BTCUSDT", "binance", "BTC", "USDT", 10.0, 5000.0),
            CoinConfig("ETHUSDT", "binance", "ETH", "USDT", 10.0, 5000.0),
            CoinConfig("ADAUSDT", "binance", "ADA", "USDT", 10.0, 5000.0),
            CoinConfig("SOLUSDT", "binance", "SOL", "USDT", 10.0, 5000.0),
            CoinConfig("DOTUSDT", "binance", "DOT", "USDT", 10.0, 5000.0),
            CoinConfig("LINKUSDT", "binance", "LINK", "USDT", 10.0, 5000.0),
            CoinConfig("MATICUSDT", "binance", "MATIC", "USDT", 10.0, 5000.0),
            CoinConfig("AVAXUSDT", "binance", "AVAX", "USDT", 10.0, 5000.0),
            CoinConfig("UNIUSDT", "binance", "UNI", "USDT", 10.0, 5000.0),
            CoinConfig("ATOMUSDT", "binance", "ATOM", "USDT", 10.0, 5000.0),
        ]

        # All 20 coins combined (10 Coinbase + 10 Binance)
        self.all_coins = self.coinbase_coins + self.binance_coins

        # 2. FETCHER CONFIGURATION
        self.fetcher_config = FetcherConfig()

        # 3. STRATEGY CONFIGURATION
        self.strategy_config = StrategyConfig()

        # 4. API CONFIGURATION
        self.api_config = APIConfig()

        # 5. CACHE STRUCTURE
        self.cache_structure: Dict[
            str,
            Dict[
                str,
                Union[
                    PriceCache,
                    VolumeCache,
                    RSICache,
                    MACDCache,
                    LastUpdatedCache,
                    StrategySignalCache,
                    CosmicDataCache,
                    TradeCooldownCache,
                ],
            ],
        ] = {
            "price": {},  # Dict[str, PriceCache]
            "volume_24h": {},  # Dict[str, VolumeCache]
            "rsi": {},  # Dict[str, RSICache]
            "macd": {},  # Dict[str, MACDCache]
            "last_updated": {},  # Dict[str, LastUpdatedCache]
            "strategy_signals": {},  # Dict[str, StrategySignalCache]
            "cosmic_data": {},  # Dict[str, CosmicDataCache]
            "trade_cooldowns": {},  # Dict[str, TradeCooldownCache]
        }

        # 6. THROTTLING RULES
        self.throttling_rules: Dict[str, Dict[str, Union[int, float]]] = {
            "price": {
                "min_interval": 10,  # seconds
                "max_calls_per_minute": 60,
                "bundle_size": 10,  # coins per call
            },
            "volume": {
                "min_interval": 180,  # seconds
                "max_calls_per_minute": 20,
                "price_change_threshold": 0.002,
            },
            "indicators": {
                "min_interval": 120,  # seconds
                "max_calls_per_minute": 30,
                "price_change_threshold": 0.001,
            },
            "mystic": {
                "min_interval": 3600,
                "max_calls_per_minute": 1,
            },  # seconds
        }

    def get_coin_by_symbol(self, symbol: str) -> Optional[CoinConfig]:
        """Get coin configuration by symbol"""
        for coin in self.all_coins:
            if coin.symbol == symbol:
                return coin
        return None

    def get_coins_by_exchange(self, exchange: str) -> List[CoinConfig]:
        """Get all coins for a specific exchange"""
        if exchange == "coinbase":
            return self.coinbase_coins
        elif exchange == "binance":
            return self.binance_coins
        return []

    def get_original_coins_by_exchange(self, exchange: str) -> List[CoinConfig]:
        """Get original 10 coins for a specific exchange"""
        if exchange == "coinbase":
            return self.coinbase_coins
        elif exchange == "binance":
            return self.binance_coins
        return []

    def get_additional_coins_by_exchange(self, exchange: str) -> List[CoinConfig]:
        """Get additional 10 coins for a specific exchange"""
        if exchange == "coinbase":
            return self.coinbase_coins
        elif exchange == "binance":
            return self.binance_coins
        return []

    def get_all_symbols(self) -> List[str]:
        """Get all coin symbols"""
        return [coin.symbol for coin in self.all_coins]

    def get_enabled_coins(self) -> List[CoinConfig]:
        """Get all enabled coins"""
        return [coin for coin in self.all_coins if coin.enabled]

    def get_enabled_symbols(self) -> List[str]:
        """Get all enabled coin symbols"""
        return [coin.symbol for coin in self.all_coins if coin.enabled]

    def get_coinbase_symbols(self) -> List[str]:
        """Get all Coinbase symbols"""
        return [coin.symbol for coin in self.coinbase_coins]

    def get_binance_symbols(self) -> List[str]:
        """Get all Binance symbols"""
        return [coin.symbol for coin in self.binance_coins]

    def get_current_timestamp(self) -> float:
        """Get current timestamp using datetime module"""
        return datetime.now().timestamp()


# Global configuration instance
config = CryptoAutoEngineConfig()


def get_config() -> CryptoAutoEngineConfig:
    """Get the global configuration instance"""
    return config


def get_coin_config(symbol: str) -> Optional[CoinConfig]:
    """Get coin configuration by symbol"""
    return config.get_coin_by_symbol(symbol)


def get_all_symbols() -> List[str]:
    """Get all coin symbols"""
    return config.get_all_symbols()


def get_enabled_symbols() -> List[str]:
    """Get all enabled coin symbols"""
    return config.get_enabled_symbols()


def get_coinbase_symbols() -> List[str]:
    """Get all Coinbase symbols"""
    return config.get_coinbase_symbols()


def get_binance_symbols() -> List[str]:
    """Get all Binance symbols"""
    return config.get_binance_symbols()


def get_current_time() -> datetime:
    """Get current time as datetime object"""
    return datetime.now()
