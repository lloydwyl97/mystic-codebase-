"""
Analytics Endpoints

Handles all analytics-related API endpoints including performance and risk analytics.
"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

logger = logging.getLogger("analytics_endpoints")

# Global service references (will be set by main.py)
analytics_service = None


def set_services(aservice):
    """Set service references from main.py"""
    global analytics_service
    analytics_service = aservice


router = APIRouter()


@router.get("/performance")
async def get_performance_analytics() -> Any:
    """Get performance analytics"""
    try:
        if analytics_service and hasattr(analytics_service, "get_analytics"):
            analytics = await analytics_service.get_analytics()
            return {"analytics": analytics}
        else:
            raise HTTPException(status_code=503, detail="Analytics service not available")
    except Exception as e:
        logger.error(f"Error fetching performance analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch performance analytics")


@router.get("/risk")
async def get_risk_analytics() -> Any:
    """Get risk analytics"""
    try:
        if analytics_service and hasattr(analytics_service, "get_analytics"):
            analytics = await analytics_service.get_analytics()
            return {"analytics": analytics}
        else:
            raise HTTPException(status_code=503, detail="Analytics service not available")
    except Exception as e:
        logger.error(f"Error fetching risk analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch risk analytics")



