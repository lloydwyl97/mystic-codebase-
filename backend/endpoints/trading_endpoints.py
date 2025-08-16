"""
Trading Endpoints

Handles all trading-related API endpoints including signals, orders, and portfolio management.
"""

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger("trading_endpoints")


# Pydantic models
class TradingSignal(BaseModel):
    symbol: str
    signal_type: str
    strength: float
    timestamp: float
    strategy: str
    confidence: float
    model_config = {"protected_namespaces": ("settings_",)}


class OrderRequest(BaseModel):
    symbol: str
    side: str
    quantity: float
    order_type: str
    price: Optional[float] = None
    model_config = {"protected_namespaces": ("settings_",)}


# Global service references (will be set by main.py)
signal_manager = None
trade_engine = None
portfolio_service = None


def set_services(sm: Any, te: Any, ps: Any):
    """Set service references from main.py"""
    global signal_manager, trade_engine, portfolio_service
    signal_manager = sm
    trade_engine = te
    portfolio_service = ps


router = APIRouter()


@router.post("/signal")
async def create_trading_signal(signal: TradingSignal) -> Dict[str, Any]:
    """Create a new trading signal"""
    try:
        if signal_manager and hasattr(signal_manager, "create_signal"):
            result: dict[str, Any] = await signal_manager.create_signal(signal.model_dump())
            return {"status": "success", "signal": result}
        else:
            raise HTTPException(status_code=503, detail="Signal manager not available")
    except Exception as e:
        logger.error(f"Error creating trading signal: {e}")
        raise HTTPException(status_code=500, detail="Failed to create trading signal")


@router.get("/signals")
async def get_trading_signals() -> Dict[str, Any]:
    """Get all trading signals"""
    try:
        if signal_manager and hasattr(signal_manager, "get_signals"):
            signals: list[dict[str, Any]] = await signal_manager.get_signals()
            count = len(signals)
            return {"signals": signals, "count": count}
        else:
            raise HTTPException(status_code=503, detail="Signal manager not available")
    except Exception as e:
        logger.error(f"Error fetching trading signals: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch trading signals")


@router.post("/order")
async def place_order(order: OrderRequest) -> Dict[str, Any]:
    """Place a new trading order"""
    try:
        if trade_engine and hasattr(trade_engine, "place_order"):
            result = await trade_engine.place_order(order.model_dump())
            return {"status": "success", "order": result}
        else:
            raise HTTPException(status_code=503, detail="Trade engine not available")
    except Exception as e:
        logger.error(f"Error placing order: {e}")
        raise HTTPException(status_code=500, detail="Failed to place order")


@router.get("/orders")
async def get_orders() -> Dict[str, Any]:
    """Get all trading orders"""
    try:
        if trade_engine and hasattr(trade_engine, "get_orders"):
            orders: list[dict[str, Any]] = await trade_engine.get_orders()
            count = len(orders)
            return {"orders": orders, "count": count}
        else:
            raise HTTPException(status_code=503, detail="Trade engine not available")
    except Exception as e:
        logger.error(f"Error fetching orders: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch orders")


@router.get("/portfolio/overview")
async def get_portfolio_overview() -> Any:
    """Get portfolio overview"""
    try:
        if portfolio_service and hasattr(portfolio_service, "get_overview"):
            overview: Any = await portfolio_service.get_overview()
            return overview
        else:
            raise HTTPException(status_code=503, detail="Portfolio service not available")
    except Exception as e:
        logger.error(f"Error fetching portfolio overview: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch portfolio overview")


@router.get("/portfolio/positions")
async def get_portfolio_positions() -> Dict[str, Any]:
    """Get portfolio positions"""
    try:
        if portfolio_service and hasattr(portfolio_service, "get_positions"):
            positions: list[Any] = await portfolio_service.get_positions()
            count = len(positions)
            return {"positions": positions, "count": count}
        else:
            raise HTTPException(status_code=503, detail="Portfolio service not available")
    except Exception as e:
        logger.error(f"Error fetching portfolio positions: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch portfolio positions")



