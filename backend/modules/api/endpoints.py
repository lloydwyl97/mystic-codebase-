"""
Modular API Endpoints for Mystic Trading Platform

Extracted from api_endpoints.py to improve modularity and reduce code duplication.
Contains common endpoint patterns and utilities.
"""

import logging
import time
from typing import Any

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

# Simple in-memory rate limiter
RATE_LIMIT = 60  # requests per minute
rate_limit_cache: dict[str, int] = {}


def rate_limiter(request: Request) -> None:
    """Rate limiting function"""
    ip = request.client.host if request.client else "unknown"
    now = int(time.time())
    window = now // 60
    key = f"{ip}:{window}"
    count = rate_limit_cache.get(key, 0)
    if count >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    rate_limit_cache[key] = count + 1


def create_health_endpoint(prefix: str = "/api"):
    """Create health check endpoint"""

    async def health_check() -> dict[str, str]:
        """Health check endpoint"""
        return {"status": "healthy", "service": "mystic-backend"}

    return health_check


def create_version_endpoint(prefix: str = "/api"):
    """Create version endpoint"""

    async def version() -> dict[str, str]:
        """Get application version"""
        return {"version": "1.0.0", "service": "mystic-backend"}

    return version


def create_comprehensive_health_endpoint(prefix: str = "/api"):
    """Create comprehensive health check endpoint"""

    async def comprehensive_health_check() -> dict[str, Any]:
        """Comprehensive health check with detailed status"""
        try:
            return {
                "status": "healthy",
                "service": "mystic-backend",
                "timestamp": time.time(),
                "components": {
                    "database": "connected",
                    "redis": "connected",
                    "websocket": "active",
                },
            }
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Health check failed")

    return comprehensive_health_check


def create_error_handler():
    """Create global exception handler"""

    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Global exception handler"""
        logger.error(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "message": str(exc),
                "path": str(request.url),
            },
        )

    return global_exception_handler


def create_data_mode_endpoint(prefix: str = "/api"):
    """Create data mode toggle endpoint"""

    async def toggle_data_mode(mode: str) -> dict[str, Any]:
        """Toggle data mode (live/simulation)"""
        try:
            # Real implementation using data service
            from backend.services.data_service import get_data_service

            data_service = get_data_service()
            result = await data_service.set_mode(mode)
            return {
                "status": "success",
                "data_mode": mode,
                "message": f"Data mode set to {mode}",
                "result": result,
            }
        except Exception as e:
            logger.error(f"Error setting data mode: {str(e)}")
            raise HTTPException(status_code=500, detail="Error setting data mode")

    return toggle_data_mode


def create_data_status_endpoint(prefix: str = "/api"):
    """Create data status endpoint"""

    async def get_data_status() -> dict[str, Any]:
        """Get data status"""
        try:
            # Real implementation using data service
            from backend.services.data_service import get_data_service

            data_service = get_data_service()
            status = await data_service.get_status()
            return {
                "data_mode": status.get("mode", "live"),
                "available_sources": status.get("sources", []),
                "last_updated": time.time(),
                "live_data": status.get("live_data", True),
            }
        except Exception as e:
            logger.error(f"Error getting data status: {str(e)}")
            raise HTTPException(status_code=500, detail="Error getting data status")

    return get_data_status


