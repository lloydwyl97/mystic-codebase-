#!/usr/bin/env python3
"""
Configuration for Tiered Signal System
Defines all settings and parameters for the three-tier signal architecture
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class Tier1Config:
    """Tier 1: Real-Time Signals Configuration"""

    price_fetch_interval: int = 5  # 5-10 seconds
    momentum_fetch_interval: int = 10  # 10-15 seconds
    orderbook_fetch_interval: int = 15  # 15 seconds
    cache_ttl: int = 30  # seconds
    max_retries: int = 3
    retry_delay: int = 1

    # Supported coins per exchange
    binance_coins: list[str] | None = None
    coinbase_coins: list[str] | None = None

    def __post_init__(self):
        if self.binance_coins is None:
            self.binance_coins = [
                "BTCUSDT",
                "ETHUSDT",
                "ADAUSDT",
                "SOLUSDT",
                "DOTUSDT",
                "LINKUSDT",
                "MATICUSDT",
                "AVAXUSDT",
                "UNIUSDT",
                "ATOMUSDT",
            ]
        if self.coinbase_coins is None:
            self.coinbase_coins = [
                "BTC-USD",
                "ETH-USD",
                "ADA-USD",
                "SOL-USD",
                "DOT-USD",
                "LINK-USD",
                "MATIC-USD",
                "AVAX-USD",
                "UNI-USD",
                "ATOM-USD",
            ]


@dataclass
class Tier2Config:
    """Tier 2: Tactical Strategy Configuration"""

    rsi_fetch_interval: int = 180  # 3 minutes
    volume_fetch_interval: int = 120  # 2 minutes
    volatility_fetch_interval: int = 300  # 5 minutes
    cache_ttl: int = 600  # 10 minutes

    # Technical indicator parameters
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # Thresholds for signal generation
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    macd_bullish: float = 0.001
    macd_bearish: float = -0.001


@dataclass
class Tier3Config:
    """Tier 3: Mystic/Cosmic Configuration"""

    schumann_fetch_interval: int = 3600  # 1 hour
    solar_fetch_interval: int = 3600  # 1 hour
    pineal_fetch_interval: int = 7200  # 2 hours
    cache_ttl: int = 7200  # 2 hours
    max_retries: int = 3
    retry_delay: int = 60

    # Cosmic alignment thresholds
    cosmic_alignment_min: float = 60.0
    earth_frequency_ideal: float = 7.83  # Hz


@dataclass
class TradeEngineConfig:
    """Trade Decision Engine Configuration"""

    decision_interval: int = 5  # 3-10 seconds
    cache_ttl: int = 60  # 1 minute
    min_confidence: float = 0.6
    max_confidence: float = 0.95

    # Signal thresholds
    price_deviation_threshold: float = 0.02  # 2%
    volume_spike_threshold: float = 0.2  # 20%
    momentum_flip_threshold: float = 0.05  # 5%

    # Trading thresholds
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    macd_bullish: float = 0.001
    macd_bearish: float = -0.001
    cosmic_alignment_min: float = 60.0
    volatility_max: float = 80.0


@dataclass
class UnifiedManagerConfig:
    """Unified Signal Manager Configuration"""

    sync_interval: int = 10  # Sync all tiers every 10 seconds
    health_check_interval: int = 60  # Health check every minute
    cache_ttl: int = 300  # 5 minutes
    auto_restart: bool = True
    max_restart_attempts: int = 3


@dataclass
class RedisConfig:
    """Redis Configuration"""

    url: str = "redis://localhost:6379"
    host: str = "redis"
    port: int = 6379
    db: int = 0
    password: str | None = None
    decode_responses: bool = True
    socket_timeout: int = 5
    socket_connect_timeout: int = 5


@dataclass
class APIConfig:
    """API Configuration"""

    binance_base_url: str = "https://api.binance.us/api/v3"
    coinbase_base_url: str = "https://api.pro.coinbase.us"
    noaa_base_url: str = "https://services.swpc.noaa.gov/json"
    schumann_base_url: str = "https://www2.irf.se/maggraphs/schumann"

    # Rate limiting
    requests_per_minute: int = 60
    max_concurrent_requests: int = 10


@dataclass
class LoggingConfig:
    """Logging Configuration"""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_handler: str = "tiered_system.log"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5


class TieredSystemConfig:
    """Complete configuration for the tiered signal system"""

    def __init__(self):
        self.tier1 = Tier1Config()
        self.tier2 = Tier2Config()
        self.tier3 = Tier3Config()
        self.trade_engine = TradeEngineConfig()
        self.unified_manager = UnifiedManagerConfig()
        self.redis = RedisConfig()
        self.api = APIConfig()
        self.logging = LoggingConfig()

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "tier1": self.tier1.__dict__,
            "tier2": self.tier2.__dict__,
            "tier3": self.tier3.__dict__,
            "trade_engine": self.trade_engine.__dict__,
            "unified_manager": self.unified_manager.__dict__,
            "redis": self.redis.__dict__,
            "api": self.api.__dict__,
            "logging": self.logging.__dict__,
        }

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "TieredSystemConfig":
        """Create configuration from dictionary"""
        config = cls()

        if "tier1" in config_dict:
            config.tier1 = Tier1Config(**config_dict["tier1"])
        if "tier2" in config_dict:
            config.tier2 = Tier2Config(**config_dict["tier2"])
        if "tier3" in config_dict:
            config.tier3 = Tier3Config(**config_dict["tier3"])
        if "trade_engine" in config_dict:
            config.trade_engine = TradeEngineConfig(**config_dict["trade_engine"])
        if "unified_manager" in config_dict:
            config.unified_manager = UnifiedManagerConfig(**config_dict["unified_manager"])
        if "redis" in config_dict:
            config.redis = RedisConfig(**config_dict["redis"])
        if "api" in config_dict:
            config.api = APIConfig(**config_dict["api"])
        if "logging" in config_dict:
            config.logging = LoggingConfig(**config_dict["logging"])

        return config

    def get_optimized_config(self) -> "TieredSystemConfig":
        """Get optimized configuration for high-frequency trading"""
        optimized = TieredSystemConfig()

        # Optimize Tier 1 for maximum speed
        optimized.tier1.price_fetch_interval = 3  # 3 seconds
        optimized.tier1.momentum_fetch_interval = 5  # 5 seconds
        optimized.tier1.orderbook_fetch_interval = 10  # 10 seconds
        optimized.tier1.cache_ttl = 15  # 15 seconds

        # Optimize Tier 2 for faster analysis
        optimized.tier2.rsi_fetch_interval = 60  # 1 minute
        optimized.tier2.volume_fetch_interval = 60  # 1 minute
        optimized.tier2.volatility_fetch_interval = 120  # 2 minutes

        # Optimize Tier 3 for more frequent cosmic checks
        optimized.tier3.schumann_fetch_interval = 1800  # 30 minutes
        optimized.tier3.solar_fetch_interval = 1800  # 30 minutes
        optimized.tier3.pineal_fetch_interval = 3600  # 1 hour

        # Optimize trade engine for faster decisions
        optimized.trade_engine.decision_interval = 3  # 3 seconds

        # Optimize unified manager
        optimized.unified_manager.sync_interval = 5  # 5 seconds
        optimized.unified_manager.health_check_interval = 30  # 30 seconds

        return optimized

    def get_conservative_config(self) -> "TieredSystemConfig":
        """Get conservative configuration for lower resource usage"""
        conservative = TieredSystemConfig()

        # Conservative Tier 1 settings
        conservative.tier1.price_fetch_interval = 10  # 10 seconds
        conservative.tier1.momentum_fetch_interval = 15  # 15 seconds
        conservative.tier1.orderbook_fetch_interval = 30  # 30 seconds
        conservative.tier1.cache_ttl = 60  # 1 minute

        # Conservative Tier 2 settings
        conservative.tier2.rsi_fetch_interval = 300  # 5 minutes
        conservative.tier2.volume_fetch_interval = 300  # 5 minutes
        conservative.tier2.volatility_fetch_interval = 600  # 10 minutes

        # Conservative Tier 3 settings
        conservative.tier3.schumann_fetch_interval = 7200  # 2 hours
        conservative.tier3.solar_fetch_interval = 7200  # 2 hours
        conservative.tier3.pineal_fetch_interval = 14400  # 4 hours

        # Conservative trade engine
        conservative.trade_engine.decision_interval = 10  # 10 seconds

        # Conservative unified manager
        conservative.unified_manager.sync_interval = 30  # 30 seconds
        conservative.unified_manager.health_check_interval = 120  # 2 minutes

        return conservative


# Default configuration instance
default_config = TieredSystemConfig()

# Predefined configurations
optimized_config = default_config.get_optimized_config()
conservative_config = default_config.get_conservative_config()


def get_config(config_type: str = "default") -> TieredSystemConfig:
    """Get configuration by type"""
    configs = {
        "default": default_config,
        "optimized": optimized_config,
        "conservative": conservative_config,
    }
    return configs.get(config_type, default_config)


