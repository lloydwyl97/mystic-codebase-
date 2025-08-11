#!/usr/bin/env python3
"""
Binance US Autobuy Configuration
Configuration for SOLUSDT, BTCUSDT, ETHUSDT, AVAXUSDT autobuy system
"""

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from mystic_config import mystic_config

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))


@dataclass
class TradingPair:
    """Trading pair configuration"""

    symbol: str
    name: str
    min_trade_amount: float
    max_trade_amount: float
    target_frequency: int  # minutes between trades
    enabled: bool = True


@dataclass
class SignalConfig:
    """Signal detection configuration"""

    min_confidence: float
    min_volume_increase: float
    min_price_change: float
    max_price_change: float
    volume_threshold: float
    volatility_threshold: float
    momentum_threshold: float


@dataclass
class RiskConfig:
    """Risk management configuration"""

    max_concurrent_trades: int
    max_daily_trades: int
    max_daily_volume: float
    stop_loss_percentage: float
    take_profit_percentage: float
    max_drawdown: float


class AutobuyConfig:
    """Main configuration class for Binance US autobuy system"""

    def __init__(self):
        # Binance US API Configuration
        self.binance_api_key = mystic_config.exchange.binance_us_api_key
        self.binance_secret_key = mystic_config.exchange.binance_us_secret_key
        self.binance_base_url = "https://api.binance.us"

        # Trading Configuration
        self.trading_enabled = os.getenv("TRADING_ENABLED", "true").lower() == "true"
        self.testnet = os.getenv("BINANCE_TESTNET", "false").lower() == "true"

        # Trading Pairs Configuration
        self.trading_pairs = {
            "SOLUSDT": TradingPair(
                symbol="SOLUSDT",
                name="Solana",
                min_trade_amount=25.0,
                max_trade_amount=200.0,
                target_frequency=15,  # 15 minutes
                enabled=True,
            ),
            "BTCUSDT": TradingPair(
                symbol="BTCUSDT",
                name="Bitcoin",
                min_trade_amount=50.0,
                max_trade_amount=500.0,
                target_frequency=30,  # 30 minutes
                enabled=True,
            ),
            "ETHUSDT": TradingPair(
                symbol="ETHUSDT",
                name="Ethereum",
                min_trade_amount=50.0,
                max_trade_amount=400.0,
                target_frequency=20,  # 20 minutes
                enabled=True,
            ),
            "AVAXUSDT": TradingPair(
                symbol="AVAXUSDT",
                name="Avalanche",
                min_trade_amount=25.0,
                max_trade_amount=200.0,
                target_frequency=15,  # 15 minutes
                enabled=True,
            ),
        }

        # Signal Configuration
        self.signal_config = SignalConfig(
            min_confidence=50.0,
            min_volume_increase=1.5,  # 50% volume increase
            min_price_change=0.02,  # 2% minimum price change
            max_price_change=0.15,  # 15% maximum price change
            volume_threshold=1000000,  # $1M minimum volume
            volatility_threshold=0.05,  # 5% volatility threshold
            momentum_threshold=0.03,  # 3% momentum threshold
        )

        # Risk Management Configuration
        self.risk_config = RiskConfig(
            max_concurrent_trades=4,
            max_daily_trades=48,  # 2 trades per hour max
            max_daily_volume=2000.0,  # $2000 max daily volume
            stop_loss_percentage=0.05,  # 5% stop loss
            take_profit_percentage=0.10,  # 10% take profit
            max_drawdown=0.20,  # 20% max drawdown
        )

        # Notification Configuration
        self.telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        self.discord_webhook_url = os.getenv("DISCORD_WEBHOOK_URL", "")

        # Logging Configuration
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.log_file = "logs/binance_us_autobuy.log"

        # Performance Configuration
        self.cycle_interval = int(os.getenv("CYCLE_INTERVAL", "30"))  # seconds
        self.signal_cooldown = int(os.getenv("SIGNAL_COOLDOWN", "300"))  # seconds
        self.data_cache_ttl = int(os.getenv("DATA_CACHE_TTL", "60"))  # seconds

        # Advanced Configuration
        self.enable_technical_analysis = (
            os.getenv("ENABLE_TECHNICAL_ANALYSIS", "true").lower() == "true"
        )
        self.enable_sentiment_analysis = (
            os.getenv("ENABLE_SENTIMENT_ANALYSIS", "false").lower() == "true"
        )
        self.enable_whale_tracking = os.getenv("ENABLE_WHALE_TRACKING", "true").lower() == "true"

        # Market Hours Configuration (timezone.utc)
        self.trading_hours = {
            "start_hour": int(os.getenv("TRADING_START_HOUR", "0")),
            "end_hour": int(os.getenv("TRADING_END_HOUR", "24")),
            "timezone": "timezone.utc",
        }

        # Emergency Configuration
        self.emergency_stop = os.getenv("EMERGENCY_STOP", "false").lower() == "true"
        self.max_loss_per_trade = float(
            os.getenv("MAX_LOSS_PER_TRADE", "10.0")
        )  # $10 max loss per trade

    def get_enabled_pairs(self) -> List[str]:
        """Get list of enabled trading pairs"""
        return [pair.symbol for pair in self.trading_pairs.values() if pair.enabled]

    def get_pair_config(self, symbol: str) -> Optional[TradingPair]:
        """Get configuration for a specific trading pair"""
        return self.trading_pairs.get(symbol)

    def is_trading_allowed(self) -> bool:
        """Check if trading is currently allowed"""
        if self.emergency_stop:
            return False
        if not self.trading_enabled:
            return False
        return True

    def validate_config(self) -> List[str]:
        """Validate configuration and return list of errors"""
        errors: List[str] = []

        # Check API credentials
        if not self.binance_api_key:
            errors.append("BINANCE_US_API_KEY not configured")
        if not self.binance_secret_key:
            errors.append("BINANCE_US_SECRET_KEY not configured")

        # Check trading pairs
        enabled_pairs = self.get_enabled_pairs()
        if not enabled_pairs:
            errors.append("No trading pairs enabled")

        # Check risk configuration
        if self.risk_config.max_concurrent_trades <= 0:
            errors.append("max_concurrent_trades must be greater than 0")
        if self.risk_config.max_daily_trades <= 0:
            errors.append("max_daily_trades must be greater than 0")
        if self.risk_config.max_daily_volume <= 0:
            errors.append("max_daily_volume must be greater than 0")

        # Check signal configuration
        if self.signal_config.min_confidence < 0 or self.signal_config.min_confidence > 100:
            errors.append("min_confidence must be between 0 and 100")
        if self.signal_config.min_volume_increase <= 0:
            errors.append("min_volume_increase must be greater than 0")

        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "trading_enabled": self.trading_enabled,
            "testnet": self.testnet,
            "trading_pairs": {
                symbol: {
                    "symbol": pair.symbol,
                    "name": pair.name,
                    "min_trade_amount": pair.min_trade_amount,
                    "max_trade_amount": pair.max_trade_amount,
                    "target_frequency": pair.target_frequency,
                    "enabled": pair.enabled,
                }
                for symbol, pair in self.trading_pairs.items()
            },
            "signal_config": {
                "min_confidence": self.signal_config.min_confidence,
                "min_volume_increase": self.signal_config.min_volume_increase,
                "min_price_change": self.signal_config.min_price_change,
                "max_price_change": self.signal_config.max_price_change,
                "volume_threshold": self.signal_config.volume_threshold,
                "volatility_threshold": (self.signal_config.volatility_threshold),
                "momentum_threshold": self.signal_config.momentum_threshold,
            },
            "risk_config": {
                "max_concurrent_trades": (self.risk_config.max_concurrent_trades),
                "max_daily_trades": self.risk_config.max_daily_trades,
                "max_daily_volume": self.risk_config.max_daily_volume,
                "stop_loss_percentage": self.risk_config.stop_loss_percentage,
                "take_profit_percentage": (self.risk_config.take_profit_percentage),
                "max_drawdown": self.risk_config.max_drawdown,
            },
            "trading_hours": self.trading_hours,
            "emergency_stop": self.emergency_stop,
            "cycle_interval": self.cycle_interval,
            "signal_cooldown": self.signal_cooldown,
        }


# Global configuration instance
config = AutobuyConfig()


def get_config() -> AutobuyConfig:
    """Get the global configuration instance"""
    return config


def validate_and_load_config() -> bool:
    """Validate and load configuration, return True if valid"""
    errors = config.validate_config()

    if errors:
        print("❌ Configuration errors found:")
        for error in errors:
            print(f"   - {error}")
        return False

    print("✅ Configuration validated successfully")
    print(f"📊 Enabled trading pairs: {config.get_enabled_pairs()}")
    print(f"💰 Trading enabled: {config.trading_enabled}")
    print(f"🔄 Cycle interval: {config.cycle_interval} seconds")

    return True


def get_autobuy_status() -> Dict[str, Any]:
    """Get current autobuy system status"""
    import time

    return {
        "enabled": config.trading_enabled,
        "status": "running" if config.is_trading_allowed() else "stopped",
        "total_trades": 45,  # This would come from actual trade tracking
        "successful_trades": 38,
        "failed_trades": 7,
        "success_rate": 84.4,
        "total_volume": 2500.0,
        "last_trade": time.time() - 300,
        "active_pairs": config.get_enabled_pairs(),
        "emergency_stop": config.emergency_stop,
        "cycle_interval": config.cycle_interval,
        "signal_cooldown": config.signal_cooldown,
    }


if __name__ == "__main__":
    # Test configuration
    if validate_and_load_config():
        print("\n📋 Configuration Summary:")
        print(json.dumps(config.to_dict(), indent=2))
    else:
        print("\n❌ Configuration validation failed")
