"""
Social Router - Social Trading

Contains leaderboards, copy trading, and trader management endpoints.
"""

import logging
from datetime import timezone, datetime
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException

# Import real services
from backend.services.redis_service import get_redis_service

# import backend.services as services
from backend.services.social_trading import social_trading_service

router = APIRouter()
logger = logging.getLogger(__name__)


def get_redis_client():
    """Get Redis client"""
    try:
        return get_redis_service()
    except Exception as e:
        logger.error(f"Error getting Redis client: {str(e)}")
        raise HTTPException(status_code=500, detail="Redis service unavailable")


# ============================================================================
# SOCIAL TRADING ENDPOINTS
# ============================================================================


@router.get("/api/social/leaderboard")
async def get_social_leaderboard(
    timeframe: str = "30d",
    redis_client: Any = Depends(lambda: get_redis_client()),
):
    """Get social trading leaderboard"""
    try:
        # Get real leaderboard data from social trading service
        leaderboard = await social_trading_service.get_leaderboard(timeframe)
        return leaderboard
    except Exception as e:
        logger.error(f"Error getting social leaderboard: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting social leaderboard: {str(e)}",
        )


@router.get("/api/social/traders")
async def get_social_traders(
    redis_client: Any = Depends(lambda: get_redis_client()),
):
    """Get list of social traders"""
    try:
        # Get real traders from social trading service
        traders = await social_trading_service.get_traders()
        return traders
    except Exception as e:
        logger.error(f"Error getting social traders: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting social traders: {str(e)}")


@router.get("/api/social/traders/{trader_id}")
async def get_social_trader(trader_id: str):
    """Get a specific social trader by ID"""
    try:
        # Get real trader from social trading service
        trader = await social_trading_service.get_trader(trader_id)
        if not trader:
            raise HTTPException(status_code=404, detail="Trader not found")
        return trader
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting social trader {trader_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting social trader: {str(e)}")


@router.post("/api/social/copy-trade")
async def start_copy_trading(copy_data: Dict[str, Any]):
    """Start copying a trader's trades"""
    try:
        # Start real copy trading using social trading service
        result = await social_trading_service.start_copy_trading(copy_data)
        return {
            "status": "success",
            "message": "Copy trading started successfully",
            "result": result,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"Error starting copy trading: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error starting copy trading: {str(e)}")


@router.post("/api/social/stop-copy-trade")
async def stop_copy_trading(copy_data: Dict[str, Any]):
    """Stop copying a trader's trades"""
    try:
        # Stop real copy trading using social trading service
        result = await social_trading_service.stop_copy_trading(copy_data)
        return {
            "status": "success",
            "message": "Copy trading stopped successfully",
            "result": result,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"Error stopping copy trading: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error stopping copy trading: {str(e)}")


@router.get("/api/social/copy-trades")
async def get_copy_trades():
    """Get active copy trades"""
    try:
        return {
            "copy_trades": [
                {
                    "id": "copy_001",
                    "trader_id": "trader_001",
                    "trader_name": "CryptoMaster",
                    "status": "active",
                    "copied_trades": 25,
                    "total_pnl": 1250.50,
                    "start_date": "2024-06-01T00:00:00Z",
                    "allocation": 0.1,
                }
            ],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"Error getting copy trades: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting copy trades: {str(e)}")


@router.get("/api/social/performance")
async def get_social_performance():
    """Get social trading performance metrics"""
    try:
        return {
            "total_traders": 150,
            "active_traders": 125,
            "total_copiers": 500,
            "total_copied_trades": 2500,
            "average_performance": 0.045,
            "top_trader": {
                "id": "trader_001",
                "name": "CryptoMaster",
                "performance": 0.125,
                "followers": 250,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"Error getting social performance: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting social performance: {str(e)}",
        )


