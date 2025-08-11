"""
Auto-Trading Endpoints

Handles all auto-trading related API endpoints including start, stop, and status.
"""

import logging
import time
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Request

from enhanced_logging import log_event, log_operation_performance

logger = logging.getLogger(__name__)

# Simple in-memory rate limiter
RATE_LIMIT = 60  # requests per minute
rate_limit_cache: Dict[str, int] = {}


def rate_limiter(request: Request) -> None:
    ip = request.client.host if request.client else "unknown"
    now = int(time.time())
    window = now // 60
    key = f"{ip}:{window}"
    count = rate_limit_cache.get(key, 0)
    if count >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    rate_limit_cache[key] = count + 1


def get_auto_trading_manager():
    """Get auto trading manager instance"""
    try:
        from auto_trading_manager import AutoTradingManager

        return AutoTradingManager()
    except ImportError as e:
        logger.error(f"AutoTradingManager not available: {str(e)}")
        raise HTTPException(status_code=500, detail="Auto trading service unavailable")


router = APIRouter()


@router.post("/start")
@log_operation_performance("auto_trading_start")
async def start_auto_trading(
    auto_trading_manager: Any = Depends(lambda: get_auto_trading_manager()),
):
    """Start automated trading bot"""
    try:
        result = await auto_trading_manager.start_auto_trading()

        # Log event
        log_event(
            "auto_trading_started",
            f"Auto-trading started with config: {result.get('config', {})}",
        )

        return result
    except Exception as e:
        logger.error(f"Error starting auto-trading: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error starting auto-trading: {str(e)}")


@router.post("/stop")
@log_operation_performance("auto_trading_stop")
async def stop_auto_trading(
    auto_trading_manager: Any = Depends(lambda: get_auto_trading_manager()),
):
    """Stop automated trading bot"""
    try:
        result = await auto_trading_manager.stop_auto_trading()

        # Log event
        log_event(
            "auto_trading_stopped",
            f"Auto-trading stopped at {result.get('timestamp')}",
        )

        return result
    except Exception as e:
        logger.error(f"Error stopping auto-trading: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error stopping auto-trading: {str(e)}")


@router.get("/status")
async def get_auto_trade_status(
    auto_trading_manager: Any = Depends(lambda: get_auto_trading_manager()),
):
    """Get current auto-trading status"""
    try:
        result = await auto_trading_manager.get_auto_trade_status()
        return result
    except Exception as e:
        logger.error(f"Error getting auto-trading status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting auto-trading status: {str(e)}",
        )


@router.get("/auto-bot/status")
async def get_auto_bot_status(
    auto_trading_manager: Any = Depends(lambda: get_auto_trading_manager()),
):
    """Get auto bot status"""
    try:
        status = await auto_trading_manager.get_auto_bot_status()
        return status
    except Exception as e:
        logger.error(f"Error getting auto bot status: {str(e)}")
        raise HTTPException(status_code=500, detail="Error getting auto bot status")


@router.post("/auto-bot/config")
async def update_auto_bot_config(
    config: Dict[str, Any], auto_trading_manager: Any = Depends(lambda: get_auto_trading_manager())
):
    """Update auto bot configuration"""
    try:
        result = await auto_trading_manager.update_auto_bot_config(config)
        return result
    except Exception as e:
        logger.error(f"Error updating auto bot config: {str(e)}")
        raise HTTPException(status_code=500, detail="Error updating auto bot config")
