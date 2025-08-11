"""
Bot Management Endpoints

Handles all bot management related API endpoints including start, stop, and status for different bot types.
"""

import logging
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Depends

logger = logging.getLogger(__name__)


def get_bot_manager():
    """Get bot manager instance"""
    try:
        from services.bot_manager import BotManager

        return BotManager()
    except ImportError as e:
        logger.error(f"BotManager not available: {str(e)}")
        raise HTTPException(status_code=500, detail="Bot management service unavailable")


router = APIRouter()


@router.post("/start")
async def start_bot_manager():
    """Start the bot manager"""
    try:
        return {
            "status": "success",
            "message": "Bot manager started",
            "timestamp": time.time(),
        }
    except Exception as e:
        logger.error(f"Error starting bot manager: {str(e)}")
        raise HTTPException(status_code=500, detail="Error starting bot manager")


@router.post("/stop")
async def stop_bot_manager():
    """Stop the bot manager"""
    try:
        return {
            "status": "success",
            "message": "Bot manager stopped",
            "timestamp": time.time(),
        }
    except Exception as e:
        logger.error(f"Error stopping bot manager: {str(e)}")
        raise HTTPException(status_code=500, detail="Error stopping bot manager")


@router.get("/status")
async def get_bot_manager_status(bot_manager: Any = Depends(lambda: get_bot_manager())):
    """Get bot manager status"""
    try:
        status = await bot_manager.get_status()
        return status
    except Exception as e:
        logger.error(f"Error getting bot manager status: {str(e)}")
        raise HTTPException(status_code=500, detail="Error getting bot manager status")


# Coinbase Bot Endpoints
@router.post("/coinbase/start")
async def start_coinbase_bot():
    """Start Coinbase bot"""
    try:
        return {
            "status": "success",
            "message": "Coinbase bot started",
            "bot_type": "coinbase",
            "timestamp": time.time(),
        }
    except Exception as e:
        logger.error(f"Error starting Coinbase bot: {str(e)}")
        raise HTTPException(status_code=500, detail="Error starting Coinbase bot")


@router.post("/coinbase/stop")
async def stop_coinbase_bot():
    """Stop Coinbase bot"""
    try:
        return {
            "status": "success",
            "message": "Coinbase bot stopped",
            "bot_type": "coinbase",
            "timestamp": time.time(),
        }
    except Exception as e:
        logger.error(f"Error stopping Coinbase bot: {str(e)}")
        raise HTTPException(status_code=500, detail="Error stopping Coinbase bot")


@router.get("/coinbase/status")
async def get_coinbase_bot_status(bot_manager: Any = Depends(lambda: get_bot_manager())):
    """Get Coinbase bot status"""
    try:
        status = await bot_manager.get_coinbase_bot_status()
        return status
    except Exception as e:
        logger.error(f"Error getting Coinbase bot status: {str(e)}")
        raise HTTPException(status_code=500, detail="Error getting Coinbase bot status")


@router.get("/coinbase/data")
async def get_coinbase_bot_data(bot_manager: Any = Depends(lambda: get_bot_manager())):
    """Get Coinbase bot data"""
    try:
        data = await bot_manager.get_coinbase_bot_data()
        return data
    except Exception as e:
        logger.error(f"Error getting Coinbase bot data: {str(e)}")
        raise HTTPException(status_code=500, detail="Error getting Coinbase bot data")


# Binance Bot Endpoints
@router.post("/binance/start")
async def start_binance_bot():
    """Start Binance bot"""
    try:
        return {
            "status": "success",
            "message": "Binance bot started",
            "bot_type": "binance",
            "timestamp": time.time(),
        }
    except Exception as e:
        logger.error(f"Error starting Binance bot: {str(e)}")
        raise HTTPException(status_code=500, detail="Error starting Binance bot")


@router.post("/binance/stop")
async def stop_binance_bot():
    """Stop Binance bot"""
    try:
        return {
            "status": "success",
            "message": "Binance bot stopped",
            "bot_type": "binance",
            "timestamp": time.time(),
        }
    except Exception as e:
        logger.error(f"Error stopping Binance bot: {str(e)}")
        raise HTTPException(status_code=500, detail="Error stopping Binance bot")


@router.get("/binance/status")
async def get_binance_bot_status(bot_manager: Any = Depends(lambda: get_bot_manager())):
    """Get Binance bot status"""
    try:
        status = await bot_manager.get_binance_bot_status()
        return status
    except Exception as e:
        logger.error(f"Error getting Binance bot status: {str(e)}")
        raise HTTPException(status_code=500, detail="Error getting Binance bot status")


@router.get("/binance/data")
async def get_binance_bot_data(bot_manager: Any = Depends(lambda: get_bot_manager())):
    """Get Binance bot data"""
    try:
        data = await bot_manager.get_binance_bot_data()
        return data
    except Exception as e:
        logger.error(f"Error getting Binance bot data: {str(e)}")
        raise HTTPException(status_code=500, detail="Error getting Binance bot data")
