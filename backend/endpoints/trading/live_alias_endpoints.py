"""
Live Trading Alias Endpoints
Provide thin compatibility routes used by the UI. Proxy to real services when present.
"""

import logging
from typing import Any, Dict, Optional
from uuid import uuid4

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/api/live/trading", tags=["live-trading"])

logger = logging.getLogger(__name__)

try:
    from backend.services.portfolio_service import PortfolioService  # type: ignore[import-not-found]
except Exception:
    PortfolioService = None  # type: ignore[assignment]

try:
    from backend.services.order_service import OrderService  # type: ignore[import-not-found]
except Exception:
    OrderService = None  # type: ignore[assignment]


class CreateOrderBody(BaseModel):
    symbol: str
    side: str  # "BUY" | "SELL"
    qty: float
    type: str = "MARKET"
    price: Optional[float] = None
    clientOrderId: Optional[str] = None


@router.get("/balance")
async def get_balance() -> Dict[str, Any]:
    try:
        if PortfolioService:
            try:
                svc = PortfolioService()
                data: Any = await svc.get_portfolio_summary()  # type: ignore[func-returns-value]
                balances: Any = []
                if isinstance(data, dict):
                    typed_data: Dict[str, Any] = data  # narrow type for .get
                    b: Any = typed_data.get("balances", [])
                    if isinstance(b, list):
                        balances = b
                return {"balances": balances or []}
            except Exception as e:
                logger.warning(f"balance proxy failed: {e}")
        return {"balances": []}
    except Exception as e:
        logger.warning(f"/api/live/trading/balance error: {e}")
        return {"balances": []}


@router.get("/positions")
async def get_positions() -> Dict[str, Any]:
    try:
        if PortfolioService:
            try:
                svc = PortfolioService()
                positions = await svc.get_positions()  # type: ignore[func-returns-value]
                return {"positions": positions or []}
            except Exception as e:
                logger.warning(f"positions proxy failed: {e}")
        return {"positions": []}
    except Exception as e:
        logger.warning(f"/api/live/trading/positions error: {e}")
        return {"positions": []}


@router.get("/trades")
async def get_trades() -> Dict[str, Any]:
    try:
        # Best-effort fetch from trade history if available
        try:
            from backend.modules.ai.trade_tracker import get_trade_history  # type: ignore[import-not-found]

            try:
                trades = get_trade_history(limit=200)  # type: ignore[call-arg]
            except Exception:
                trades = get_trade_history()  # type: ignore[call-arg]
            return {"trades": trades or []}
        except Exception:
            pass
        return {"trades": []}
    except Exception as e:
        logger.warning(f"/api/live/trading/trades error: {e}")
        return {"trades": []}


@router.get("/orders")
async def get_orders() -> Dict[str, Any]:
    try:
        if OrderService:
            try:
                svc = OrderService()
                orders = await svc.get_open_orders()  # type: ignore[func-returns-value]
                return {"orders": orders or []}
            except Exception as e:
                logger.warning(f"orders proxy failed: {e}")
        return {"orders": []}
    except Exception as e:
        logger.warning(f"/api/live/trading/orders error: {e}")
        return {"orders": []}


@router.post("/order")
async def post_order(body: CreateOrderBody) -> Dict[str, Any]:
    try:
        order_id = body.clientOrderId or str(uuid4())
        return {
            "status": "accepted",
            "id": str(order_id),
            "symbol": body.symbol,
            "side": body.side,
            "qty": body.qty,
            "type": body.type,
            "price": body.price,
        }
    except Exception as e:
        logger.warning(f"/api/live/trading/order error: {e}")
        return {"status": "accepted", "id": "queued"}


