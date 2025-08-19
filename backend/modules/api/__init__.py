"""
API Module for Mystic Trading Platform

Contains all API-related functionality including endpoints, routers, and API utilities.
"""

from .endpoints import (
    create_comprehensive_health_endpoint,
    create_data_mode_endpoint,
    create_data_status_endpoint,
    create_error_handler,
    create_health_endpoint,
    create_version_endpoint,
    rate_limiter,
)
from .routers import (
    create_analytics_router,
    create_api_router,
    create_portfolio_router,
    create_social_router,
    create_trading_router,
    register_api_endpoints,
)

__all__ = [
    "rate_limiter",
    "create_health_endpoint",
    "create_version_endpoint",
    "create_comprehensive_health_endpoint",
    "create_error_handler",
    "create_data_mode_endpoint",
    "create_data_status_endpoint",
    "create_api_router",
    "register_api_endpoints",
    "create_portfolio_router",
    "create_trading_router",
    "create_analytics_router",
    "create_social_router",
]


