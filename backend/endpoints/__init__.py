"""
Endpoints Package for Mystic Trading

This package contains all endpoint implementations for the Mystic Trading platform.
"""

from endpoints.analytics_advanced_endpoints import (
    router as analytics_advanced_router,
)
from endpoints.analytics_endpoints import router as analytics_router
from endpoints.live_trading_endpoints import router as live_trading_router
from endpoints.misc_endpoints import router as misc_router
from endpoints.notification_endpoints import router as notification_router
from endpoints.phase5_endpoints import router as phase5_router
from endpoints.signal_advanced_endpoints import router as signal_router
from endpoints.trading_endpoints import router as trading_router
from endpoints.websocket_endpoints import router as websocket_router

__all__ = [
    "analytics_advanced_router",
    "analytics_router",
    "live_trading_router",
    "misc_router",
    "notification_router",
    "phase5_router",
    "signal_router",
    "trading_router",
    "websocket_router",
]

__version__ = "1.0.0"
__author__ = "Mystic Trading Team"
