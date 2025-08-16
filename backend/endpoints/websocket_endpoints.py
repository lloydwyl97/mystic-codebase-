"""
WebSocket Endpoints

Handles all WebSocket-related endpoints for real-time data streaming.
"""

import logging
from typing import Any

from fastapi import APIRouter, WebSocket

logger = logging.getLogger("websocket_endpoints")

# Global service references (will be set by main.py)
websocket_manager: Any = None


def set_services(wm: Any):
    """Set service references from main.py"""
    global websocket_manager
    websocket_manager = wm


router = APIRouter()


@router.websocket("/market-data")
async def websocket_market_data(websocket: WebSocket):
    """WebSocket endpoint for real-time market data"""
    try:
        if websocket_manager and hasattr(websocket_manager, "handle_market_data_connection"):
            await websocket_manager.handle_market_data_connection(websocket)
        else:
            await websocket.close(code=1000, reason="WebSocket manager not available")
    except Exception as e:
        logger.error(f"WebSocket market data error: {e}")
        await websocket.close(code=1011, reason="Internal server error")


@router.websocket("/trading-signals")
async def websocket_trading_signals(websocket: WebSocket):
    """WebSocket endpoint for real-time trading signals"""
    try:
        if websocket_manager and hasattr(websocket_manager, "handle_trading_signals_connection"):
            await websocket_manager.handle_trading_signals_connection(websocket)
        else:
            await websocket.close(code=1000, reason="WebSocket manager not available")
    except Exception as e:
        logger.error(f"WebSocket trading signals error: {e}")
        await websocket.close(code=1011, reason="Internal server error")



