"""
Trading Configuration
Centralized configuration for all trading-related hardcoded values
"""

import os
from typing import Dict, Any


class TradingConfig:
    """Centralized trading configuration"""

    # Risk Management Defaults
    DEFAULT_MAX_POSITION_SIZE = 0.1  # 10% of portfolio
    DEFAULT_MAX_DRAWDOWN = 0.1  # 10% maximum drawdown
    DEFAULT_STOP_LOSS = 0.05  # 5% stop loss
    DEFAULT_TAKE_PROFIT = 0.15  # 15% take profit
    DEFAULT_MAX_LEVERAGE = 3  # 3x maximum leverage
    DEFAULT_MIN_VOLUME = 1000000  # $1M minimum volume
    DEFAULT_MAX_SLIPPAGE = 0.02  # 2% maximum slippage

    # Performance Thresholds
    DEFAULT_SHARPE_RATIO = 0.0
    DEFAULT_TOTAL_TRADES = 0
    DEFAULT_WINNING_TRADES = 0
    DEFAULT_LOSING_TRADES = 0
    DEFAULT_TOTAL_PNL = 0.0

    # Redis TTL Values (in seconds)
    AUTO_TRADE_CONFIG_TTL = 3600  # 1 hour
    AUTO_TRADING_ENABLED_TTL = 3600  # 1 hour

    # API Configuration
    DEFAULT_REDIS_PORT = 6379
    DEFAULT_REDIS_HOST = "localhost"
    DEFAULT_REDIS_DB = 0

    # Service Ports
    DEFAULT_SERVICE_PORT = 9000
    AI_STRATEGY_GENERATOR_PORT = 8002

    # Timeout Values
    DEFAULT_REQUEST_TIMEOUT = 5  # seconds
    DEFAULT_BATCH_DELAY = 0.1  # seconds

    # Trading Thresholds
    PROFIT_THRESHOLD = 0.01  # 1% gain threshold
    LOSS_THRESHOLD = -0.01  # 1% loss threshold
    REBALANCE_THRESHOLD = 0.02  # 2% rebalancing threshold
    VAR_THRESHOLD = 0.05  # 5% VaR threshold
    DRAWDOWN_THRESHOLD = 0.15  # 15% drawdown threshold

    # Model Parameters
    DEFAULT_LEARNING_RATE = 0.001
    DEFAULT_DROPOUT = 0.2
    DEFAULT_CONFIDENCE_THRESHOLD = 0.7
    DEFAULT_TRAIN_TEST_SPLIT = 0.8  # 80% training, 20% testing

    # Cache TTL Values
    STRATEGY_CACHE_TTL = 86400  # 24 hours
    PORTFOLIO_CACHE_TTL = 1800  # 30 minutes
    RISK_CACHE_TTL = 1800  # 30 minutes

    # Mock Data Values
    MOCK_INITIAL_BTC_PRICE = 45000
    MOCK_INITIAL_ETH_PRICE = 3000
    MOCK_INITIAL_DEFAULT_PRICE = 100
    MOCK_VOLUME_MIN = 1000000
    MOCK_VOLUME_MAX = 5000000
    MOCK_RETURN_MEAN = 0.0001
    MOCK_RETURN_STD = 0.02
    MOCK_VOLATILITY_STD = 0.01

    # Portfolio Defaults
    DEFAULT_PORTFOLIO_VALUE = 100000
    DEFAULT_CASH_ALLOCATION = 15000

    # Asset Allocations (percentages)
    DEFAULT_ASSET_ALLOCATIONS = {
        "BTC/USDT": 0.35,
        "ETH/USDT": 0.25,
        "BNB/USDT": 0.15,
        "ADA/USDT": 0.15,
        "SOL/USDT": 0.10,
    }

    # Asset Amounts
    DEFAULT_ASSET_AMOUNTS = {
        "BTC/USDT": {"amount": 0.5, "value": 25000},
        "ETH/USDT": {"amount": 5.0, "value": 15000},
        "BNB/USDT": {"amount": 50.0, "value": 15000},
        "ADA/USDT": {"amount": 10000.0, "value": 15000},
        "SOL/USDT": {"amount": 100.0, "value": 15000},
    }

    # Performance Metrics
    DEFAULT_PERFORMANCE_METRICS = {
        "conservative": {
            "sharpe": 0.95,
            "returns": 0.12,
            "max_dd": 0.05,
        },
        "moderate": {
            "sharpe": 1.42,
            "returns": 0.18,
            "max_dd": 0.12,
        },
        "aggressive": {
            "sharpe": 1.85,
            "returns": 0.23,
            "max_dd": 0.08,
        },
        "very_aggressive": {
            "sharpe": 2.1,
            "returns": 0.31,
            "max_dd": 0.15,
        },
    }

    # Volatility Ranges
    DEFAULT_VOLATILITY_RANGES = {
        "BTC/USDT": (0.02, 0.08),
        "ETH/USDT": (0.025, 0.09),
        "BNB/USDT": (0.03, 0.12),
        "ADA/USDT": (0.04, 0.15),
        "SOL/USDT": (0.035, 0.14),
    }

    @classmethod
    def get_risk_management_config(cls) -> Dict[str, Any]:
        """Get default risk management configuration"""
        return {
            "max_position_size": cls.DEFAULT_MAX_POSITION_SIZE,
            "max_drawdown": cls.DEFAULT_MAX_DRAWDOWN,
            "stop_loss": cls.DEFAULT_STOP_LOSS,
            "take_profit": cls.DEFAULT_TAKE_PROFIT,
            "max_leverage": cls.DEFAULT_MAX_LEVERAGE,
            "min_volume": cls.DEFAULT_MIN_VOLUME,
            "max_slippage": cls.DEFAULT_MAX_SLIPPAGE,
        }

    @classmethod
    def get_performance_config(cls) -> Dict[str, Any]:
        """Get default performance configuration"""
        return {
            "total_trades": cls.DEFAULT_TOTAL_TRADES,
            "winning_trades": cls.DEFAULT_WINNING_TRADES,
            "losing_trades": cls.DEFAULT_LOSING_TRADES,
            "total_pnl": cls.DEFAULT_TOTAL_PNL,
            "sharpe_ratio": cls.DEFAULT_SHARPE_RATIO,
        }

    @classmethod
    def get_redis_config(cls) -> Dict[str, Any]:
        """Get Redis configuration"""
        return {
            "host": os.getenv("REDIS_HOST", cls.DEFAULT_REDIS_HOST),
            "port": int(os.getenv("REDIS_PORT", cls.DEFAULT_REDIS_PORT)),
            "db": int(os.getenv("REDIS_DB", cls.DEFAULT_REDIS_DB)),
        }

    @classmethod
    def get_service_config(cls) -> Dict[str, Any]:
        """Get service configuration"""
        return {
            "default_port": int(os.getenv("SERVICE_PORT", cls.DEFAULT_SERVICE_PORT)),
            "ai_strategy_port": cls.AI_STRATEGY_GENERATOR_PORT,
        }


# Global configuration instance
trading_config = TradingConfig()
