"""
API Routers for Mystic Trading Platform

Centralized router management to reduce code duplication and improve modularity.
"""

import logging

from fastapi import APIRouter

from .endpoints import (
    create_comprehensive_health_endpoint,
    create_health_endpoint,
    create_data_mode_endpoint,
    create_data_status_endpoint,
    create_version_endpoint,
)

logger = logging.getLogger(__name__)


def create_api_router(prefix: str = "/api") -> APIRouter:
    """Create main API router with all endpoints"""
    router = APIRouter()

    # Register health endpoints
    router.add_api_route(
        f"{prefix}/health",
        create_health_endpoint(prefix),
        methods=["GET"],
        tags=["Health"],
    )

    router.add_api_route(
        f"{prefix}/version",
        create_version_endpoint(prefix),
        methods=["GET"],
        tags=["System"],
    )

    router.add_api_route(
        f"{prefix}/comprehensive-health",
        create_comprehensive_health_endpoint(prefix),
        methods=["GET"],
        tags=["Health"],
    )

    # Register data mode endpoints
    router.add_api_route(
        f"{prefix}/data-mode/toggle",
        create_data_mode_endpoint(prefix),
        methods=["POST"],
        tags=["System"],
    )

    router.add_api_route(
        f"{prefix}/data-status",
        create_data_status_endpoint(prefix),
        methods=["GET"],
        tags=["System"],
    )

    return router


def register_api_endpoints(router: APIRouter, prefix: str = "/api") -> None:
    """Register all API endpoints on the given router"""
    try:
        # Create and register main API router
        api_router = create_api_router(prefix)
        router.include_router(api_router)

        # Register shared endpoints
        from shared_endpoints import register_shared_endpoints

        register_shared_endpoints(router, prefix)

        logger.info(f"✅ API endpoints registered with prefix: {prefix}")
    except Exception as e:
        logger.error(f"❌ Failed to register API endpoints: {str(e)}")
        raise


def create_portfolio_router(prefix: str = "/api") -> APIRouter:
    """Create portfolio-specific router"""
    router = APIRouter(prefix=f"{prefix}/portfolio", tags=["Portfolio"])

    # Portfolio endpoints will be added here
    # This reduces duplication from shared_endpoints.py

    return router


def create_trading_router(prefix: str = "/api") -> APIRouter:
    """Create trading-specific router"""
    router = APIRouter(prefix=f"{prefix}/trading", tags=["Trading"])

    # Trading endpoints will be added here
    # This reduces duplication from shared_endpoints.py

    return router


def create_analytics_router(prefix: str = "/api") -> APIRouter:
    """Create analytics-specific router"""
    router = APIRouter(prefix=f"{prefix}/analytics", tags=["Analytics"])

    # Analytics endpoints will be added here
    # This reduces duplication from shared_endpoints.py

    return router


def create_social_router(prefix: str = "/api") -> APIRouter:
    """Create social trading router"""
    router = APIRouter(prefix=f"{prefix}/social", tags=["Social Trading"])

    # Social trading endpoints will be added here
    # This reduces duplication from shared_endpoints.py

    return router
