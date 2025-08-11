"""
Services Package for Mystic Trading

This package contains all service implementations for the Mystic Trading platform.
"""

# Use relative imports for local modules
try:
    from .market_data import MarketDataService
    from .market_data_manager import market_data_manager
    from .notification import get_notification_service
    from .notification_manager import notification_manager
    from .trading import get_trading_service
    from .trading_manager import trading_manager
except ImportError:
    # Fallback for when running as standalone
    pass

__all__ = [
    "MarketDataService",
    "market_data_manager",
    "get_trading_service",
    "trading_manager",
    "get_notification_service",
    "notification_manager",
]
