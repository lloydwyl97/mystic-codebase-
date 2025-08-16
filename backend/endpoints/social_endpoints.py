"""
Social Trading Endpoints

Handles all social trading-related API endpoints including leaders and feed.
"""

import logging
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

logger = logging.getLogger("social_endpoints")

# Global service references (will be set by main.py)
social_trading_manager = None


def set_services(stm):
    """Set service references from main.py"""
    global social_trading_manager
    social_trading_manager = stm


router = APIRouter()


@router.get("/leaders")
async def get_social_leaders() -> Dict[str, Any]:
    """Get social trading leaders"""
    try:
        if social_trading_manager and hasattr(social_trading_manager, "get_leaders"):
            leaders = await social_trading_manager.get_leaders()
            count = len(leaders) if leaders and isinstance(leaders, (list, tuple)) else 0
            return {"leaders": leaders, "count": count}
        else:
            raise HTTPException(status_code=503, detail="Social trading manager not available")
    except Exception as e:
        logger.error(f"Error fetching social leaders: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch social leaders")


@router.get("/feed")
async def get_social_feed() -> Dict[str, Any]:
    """Get social trading feed"""
    try:
        if social_trading_manager and hasattr(social_trading_manager, "get_social_feed"):
            feed = await social_trading_manager.get_social_feed()
            return {"social_feed": feed}
        else:
            raise HTTPException(status_code=503, detail="Social trading manager not available")
    except Exception as e:
        logger.error(f"Error fetching social feed: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch social feed")



