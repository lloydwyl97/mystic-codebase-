"""
Bot Management Endpoints

Handles all bot-related API endpoints including creation, management, and control.
"""

import logging
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger("bot_endpoints")


# Pydantic models
class BotConfig(BaseModel):
    model_config = {"protected_namespaces": ("settings_",)}
    bot_id: str
    strategy: str
    symbols: List[str]
    risk_level: str
    auto_trade: bool


# Global service references (will be set by main.py)
auto_trading_manager = None


def set_services(atm: Any):
    """Set service references from main.py"""
    global auto_trading_manager
    auto_trading_manager = atm


router = APIRouter()


@router.post("/create")
async def create_bot(config: BotConfig) -> Dict[str, Any]:
    """Create a new trading bot"""
    try:
        if auto_trading_manager and hasattr(auto_trading_manager, "create_bot"):
            result = await auto_trading_manager.create_bot(config.model_dump())
            return {"status": "success", "bot": result}
        else:
            raise HTTPException(status_code=503, detail="Auto trading manager not available")
    except Exception as e:
        logger.error(f"Error creating bot: {e}")
        raise HTTPException(status_code=500, detail="Failed to create bot")


@router.get("/")
async def get_bots() -> Dict[str, Any]:
    """Get all trading bots"""
    try:
        if auto_trading_manager and hasattr(auto_trading_manager, "get_bots"):
            bots: list[dict[str, Any]] = await auto_trading_manager.get_bots()
            count = len(bots)
            return {"bots": bots, "count": count}
        else:
            raise HTTPException(status_code=503, detail="Auto trading manager not available")
    except Exception as e:
        logger.error(f"Error fetching bots: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch bots")


@router.post("/{bot_id}/start")
async def start_bot(bot_id: str) -> Dict[str, Any]:
    """Start a trading bot"""
    try:
        if auto_trading_manager and hasattr(auto_trading_manager, "start_bot"):
            result = await auto_trading_manager.start_bot(bot_id)
            return {"status": "success", "bot_id": bot_id, "result": result}
        else:
            raise HTTPException(status_code=503, detail="Auto trading manager not available")
    except Exception as e:
        logger.error(f"Error starting bot {bot_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to start bot")


@router.post("/{bot_id}/stop")
async def stop_bot(bot_id: str) -> Dict[str, Any]:
    """Stop a trading bot"""
    try:
        if auto_trading_manager and hasattr(auto_trading_manager, "stop_bot"):
            result = await auto_trading_manager.stop_bot(bot_id)
            return {"status": "success", "bot_id": bot_id, "result": result}
        else:
            raise HTTPException(status_code=503, detail="Auto trading manager not available")
    except Exception as e:
        logger.error(f"Error stopping bot {bot_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to stop bot")
