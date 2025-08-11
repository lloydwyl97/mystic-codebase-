"""
Modern Configuration Management for Mystic Trading Platform
Using Pydantic Settings for type safety and environment variable management
"""

import os
from typing import Any, Dict, List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class DatabaseSettings(BaseSettings):
    """Database configuration"""

    url: str = Field(default="sqlite:///./mystic_trading.db", alias="DATABASE_URL")
    echo: bool = Field(default=False, alias="DATABASE_ECHO")
    pool_size: int = Field(default=10, alias="DATABASE_POOL_SIZE")
    max_overflow: int = Field(default=20, alias="DATABASE_MAX_OVERFLOW")


class RedisSettings(BaseSettings):
    """Redis configuration"""

    url: str = Field(default="redis://localhost:6379", alias="REDIS_URL")
    password: Optional[str] = Field(default=None, alias="REDIS_PASSWORD")
    db: int = Field(default=0, alias="REDIS_DB")
    max_connections: int = Field(default=20, alias="REDIS_MAX_CONNECTIONS")


class APISettings(BaseSettings):
    """API configuration"""

    host: str = Field(default="0.0.0.0", alias="API_HOST")
    port: int = Field(default=8000, alias="API_PORT")
    debug: bool = Field(default=False, alias="API_DEBUG")
    reload: bool = Field(default=True, alias="API_RELOAD")
    workers: int = Field(default=1, alias="API_WORKERS")


class SecuritySettings(BaseSettings):
    """Security configuration"""

    secret_key: str = Field(default="your-secret-key-change-this", alias="SECRET_KEY")
    algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, alias="ACCESS_TOKEN_EXPIRE_MINUTES")
    cors_origins: List[str] = Field(default=["*"], alias="CORS_ORIGINS")


class TradingSettings(BaseSettings):
    """Trading configuration"""

    default_budget: float = Field(default=5000.0, alias="DEFAULT_BUDGET")
    max_position_size: float = Field(default=500.0, alias="MAX_POSITION_SIZE")
    min_signal_score: int = Field(default=70, alias="MIN_SIGNAL_SCORE")
    confirmation_threshold: int = Field(default=3, alias="CONFIRMATION_THRESHOLD")
    max_daily_loss: float = Field(default=2.0, alias="MAX_DAILY_LOSS")
    max_concurrent_positions: int = Field(default=3, alias="MAX_CONCURRENT_POSITIONS")


class ExchangeSettings(BaseSettings):
    """Exchange API configuration"""

    binance_us_api_key: Optional[str] = Field(default=None, alias="BINANCE_US_API_KEY")
    binance_us_secret_key: Optional[str] = Field(default=None, alias="BINANCE_US_SECRET_KEY")
    coinbase_api_key: Optional[str] = Field(default=None, alias="COINBASE_API_KEY")
    coinbase_secret_key: Optional[str] = Field(default=None, alias="COINBASE_SECRET_KEY")
    testnet: bool = Field(default=True, alias="EXCHANGE_TESTNET")


class MonitoringSettings(BaseSettings):
    """Monitoring and logging configuration"""

    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    enable_metrics: bool = Field(default=True, alias="ENABLE_METRICS")
    enable_health_checks: bool = Field(default=True, alias="ENABLE_HEALTH_CHECKS")
    prometheus_port: int = Field(default=9090, alias="PROMETHEUS_PORT")


class AISettings(BaseSettings):
    """AI and ML configuration"""

    ai_model_path: str = Field(default="./models", alias="AI_MODEL_PATH")
    confidence_threshold: float = Field(default=0.75, alias="AI_CONFIDENCE_THRESHOLD")
    enable_auto_training: bool = Field(default=True, alias="AI_AUTO_TRAINING")
    training_interval_hours: int = Field(default=24, alias="AI_TRAINING_INTERVAL")


class Settings(BaseSettings):
    """Main application settings"""

    app_name: str = Field(default="Mystic Trading Platform", alias="APP_NAME")
    version: str = Field(default="1.0.0", alias="APP_VERSION")
    environment: str = Field(default="development", alias="ENVIRONMENT")
    exchange_name: str = Field(default="binance.us", alias="EXCHANGE_NAME")

    # Sub-settings
    database: DatabaseSettings = DatabaseSettings()
    redis: RedisSettings = RedisSettings()
    api: APISettings = APISettings()
    security: SecuritySettings = SecuritySettings()
    trading: TradingSettings = TradingSettings()
    exchange: ExchangeSettings = ExchangeSettings()
    monitoring: MonitoringSettings = MonitoringSettings()
    ai: AISettings = AISettings()

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        allowed = ["development", "staging", "production"]
        if v not in allowed:
            raise ValueError(f"Environment must be one of {allowed}")
        return v

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "env_ignore_empty": True,
        "extra": "allow",
        # Prevent top-level Settings from consuming plain EXCHANGE (string) env
        # which would otherwise try to populate the nested `exchange` model
        # and cause parsing errors. Nested models still read their own env vars.
        "env_prefix": "APP_",
    }


# Global settings instance - with error handling
try:
    settings = Settings()
except Exception as e:
    print(f"Warning: Failed to load settings from .env file: {e}")
    print("Using default settings...")
    # Create settings with defaults only, ignoring env file
    settings = Settings(_env_file=None)

# Legacy compatibility (for existing code)
CORS_ORIGINS = settings.security.cors_origins
IS_PRODUCTION = settings.environment == "production"
LOG_LEVEL = settings.monitoring.log_level

# Additional legacy compatibility variables
DEFAULT_HOST = settings.api.host
DEFAULT_PORT = settings.api.port
SECRET_KEY = settings.security.secret_key

# Auto-buy configuration
AUTO_BUY_CONFIG: Dict[str, Any] = {
    "enabled": False,
    "max_investment": 1000,
    "stop_loss": 5,
    "take_profit": 10,
    "selected_coins": ["bitcoin", "ethereum", "solana"],
    "strategy": "momentum",
    "bot_status": "stopped",
}

# Bot monitoring configuration
BOT_MONITORING: Dict[str, Any] = {
    "is_running": False,
    "total_trades": 0,
    "successful_trades": 0,
    "failed_trades": 0,
    "current_balance": 10000,
    "profit_loss": 0,
    "last_trade": None,
    "active_strategies": [],
    "logs": [],
}

# Additional configuration
API_KEY_HEADER = os.getenv("API_KEY_HEADER", "X-API-Key")
LOG_FILE = os.getenv("LOG_FILE", "logs/mystic_trading.log")
WS_HEARTBEAT_INTERVAL = int(os.getenv("WS_HEARTBEAT_INTERVAL", "30"))
WS_MAX_CONNECTIONS = int(os.getenv("WS_MAX_CONNECTIONS", "1000"))
TRADING_ENABLED = os.getenv("TRADING_ENABLED", "false").lower() == "true"
MAX_ORDER_SIZE = float(os.getenv("MAX_ORDER_SIZE", "10000"))
RISK_LIMIT_PERCENT = float(os.getenv("RISK_LIMIT_PERCENT", "5.0"))
