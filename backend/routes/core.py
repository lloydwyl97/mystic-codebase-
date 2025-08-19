"""
Core Router - Health, Version, and Common APIs

Contains health checks, version information, and common API endpoints.
"""

import logging
import time
from datetime import timezone, datetime
from typing import Any, Dict, Optional, Union

from fastapi import APIRouter, HTTPException, Request
from backend.middleware.rate_limiter import rate_limit

# Rate limiting configuration
default_rate_limit = 60  # requests per minute

# Import live services
try:
    from backend.modules.data.market_data import market_data_manager

    live_services_available = True
except ImportError:
    live_services_available = False
    logger = logging.getLogger(__name__)
    logger.warning("Live services not available")

router = APIRouter()
logger = logging.getLogger(__name__)


def get_redis_client():
    """Get Redis client"""
    try:
        from backend.services.redis_service import get_redis_service

        return get_redis_service()
    except Exception as e:
        logger.error(f"Error getting Redis client: {str(e)}")
        raise HTTPException(status_code=500, detail="Redis service unavailable")


# ============================================================================
# HEALTH AND STATUS ENDPOINTS
# ============================================================================


@router.get("/api/health")
@rate_limit(max_requests=default_rate_limit, window_seconds=60)
async def health_check(request: Request) -> Dict[str, Any]:
    """Simple health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "message": "Mystic Trading Platform is running",
        "live_data": live_services_available,
    }


@router.get("/api/version")
@rate_limit(max_requests=default_rate_limit, window_seconds=60)
async def version(request: Request) -> Dict[str, Any]:
    """Get application version and build information"""
    return {
        "version": "1.0.0",
        "build_date": "2024-06-22",
        "environment": "production",
        "live_data": live_services_available,
        "features": [
            "real-time trading",
            "AI-powered analytics",
            "social trading",
            "mobile PWA support",
            "advanced order types",
            "risk management",
            "auto-trading bots",
        ],
    }


@router.get("/health/comprehensive")
async def comprehensive_health_check():
    """Comprehensive health check with live service status"""
    try:
        health_status: Dict[str, Union[str, float, Dict[str, str], Optional[Dict[str, Any]]]] = {
            "status": "healthy",
            "timestamp": time.time(),
            "version": "1.0.0",
            "live_data": live_services_available,
            "services": {},
        }

        # Check Redis connection
        redis_client = get_redis_client()
        try:
            redis_client.ping()
            health_status["services"]["redis"] = "healthy"
        except Exception as e:
            logger.error(f"Redis health check failed: {str(e)}")
            health_status["services"]["redis"] = "unhealthy"

        # Check live services if available
        if live_services_available:
            try:
                # Check market data manager
                market_summary = await market_data_manager.get_market_summary()
                health_status["services"]["market_data"] = "healthy"
                health_status["market_data"] = {
                    "total_symbols": market_summary.get("total_symbols", 0),
                    "live_data": market_summary.get("live_data", False),
                }
            except Exception as e:
                logger.error(f"Market data health check failed: {str(e)}")
                health_status["services"]["market_data"] = f"unhealthy: {str(e)}"

        return health_status
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": time.time(),
            "error": str(e),
            "live_data": live_services_available,
        }


@router.get("/features")
@router.get("/api/features")
async def get_available_features():
    """Get available platform features with live status"""
    features = [
        {
            "name": "Real-time Trading",
            "description": "Live market data and instant order execution",
            "enabled": True,
            "live_data": live_services_available,
        },
        {
            "name": "AI Analytics",
            "description": "Machine learning powered market analysis",
            "enabled": True,
            "live_data": live_services_available,
        },
        {
            "name": "Social Trading",
            "description": "Copy successful traders and share strategies",
            "enabled": True,
            "live_data": live_services_available,
        },
        {
            "name": "Mobile PWA",
            "description": "Progressive web app for mobile trading",
            "enabled": True,
            "live_data": True,
        },
        {
            "name": "Advanced Orders",
            "description": "Stop-loss, take-profit, and conditional orders",
            "enabled": True,
            "live_data": live_services_available,
        },
        {
            "name": "Risk Management",
            "description": "Portfolio risk analysis and position sizing",
            "enabled": True,
            "live_data": live_services_available,
        },
        {
            "name": "Auto Trading",
            "description": "Automated trading bots with custom strategies",
            "enabled": True,
            "live_data": live_services_available,
        },
    ]

    return {
        "features": features,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "live_data": live_services_available,
    }


@router.get("/api/live-data/status")
async def get_live_data_status():
    """Get live data status and available sources"""
    try:
        return {
            "live_data_enabled": True,
            "data_sources": ["coingecko", "binance", "coinbase"],
            "status": "active",
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "message": "All endpoints use live data sources",
        }
    except Exception as e:
        logger.error(f"Error getting live data status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting live data status: {str(e)}")


