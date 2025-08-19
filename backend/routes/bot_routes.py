"""
Bot Routes

API endpoints for bot control and management.
"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from backend.services.bot_manager import BotManager

logger = logging.getLogger(__name__)

router = APIRouter()
bot_manager = BotManager()


@router.get("/bot/status")
async def get_bot_status() -> dict[str, Any]:
    """Get current bot status"""
    try:
        return bot_manager.get_bot_status()
    except Exception as e:
        logger.error(f"Error getting bot status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/bot/start")
async def start_bot() -> dict[str, Any]:
    """Start the trading bot"""
    try:
        return bot_manager.start_bot()
    except Exception as e:
        logger.error(f"Error starting bot: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/bot/stop")
async def stop_bot() -> dict[str, Any]:
    """Stop the trading bot"""
    try:
        return bot_manager.stop_bot()
    except Exception as e:
        logger.error(f"Error stopping bot: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/auto-buy/configure")
async def configure_auto_buy(config: dict[str, Any]) -> dict[str, Any]:
    """Configure auto-buy settings"""
    try:
        return bot_manager.configure_auto_buy(config)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error configuring auto-buy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/auto-buy/config")
async def get_auto_buy_config() -> dict[str, Any]:
    """Get current auto-buy configuration"""
    return bot_manager.get_auto_buy_config()


@router.get("/bot/logs")
async def get_bot_logs(limit: int = 100) -> dict[str, Any]:
    """Get bot logs"""
    try:
        return bot_manager.get_bot_logs(limit)
    except Exception as e:
        logger.error(f"Error getting bot logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/trade/execute")
async def execute_trade(
    symbol: str, action: str, amount: float = 1000, strategy: str = "default"
) -> dict[str, Any]:
    """Execute a trade"""
    try:
        return bot_manager.execute_trade(symbol, action, amount, strategy)
    except Exception as e:
        logger.error(f"Error executing trade: {e}")
        raise HTTPException(status_code=500, detail=str(e))


