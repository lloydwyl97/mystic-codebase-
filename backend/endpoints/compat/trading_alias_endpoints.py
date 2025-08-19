"""
Compatibility Trading Aliases

Maps Streamlit UI actions to real services, without changing the UI.
"""

from __future__ import annotations

from typing import Any, cast

import httpx
from fastapi import APIRouter, HTTPException

router = APIRouter(tags=["compat-trading"])

BACKEND_BASE = "http://127.0.0.1:9000"


def _url(path: str) -> str:
    if path.startswith("http"):
        return path
    return f"{BACKEND_BASE}/{path.lstrip('/')}"


@router.post("/trading/start")
async def trading_start() -> dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.post(_url("/autobuy/config"), json={"enabled": True})
        return {"ok": r.status_code < 400}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"start failed: {e}")


@router.post("/trading/stop")
async def trading_stop() -> dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.post(_url("/autobuy/config"), json={"enabled": False})
        return {"ok": r.status_code < 400}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"stop failed: {e}")


@router.post("/trading/cancel/{symbol}")
async def trading_cancel_symbol(symbol: str) -> dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            # Use canonical /api/live alias added by consolidated router
            orders_r = await client.get(_url("/api/live/trading/orders"))
            data_any: Any = orders_r.json() if orders_r.status_code == 200 else {}
            data = cast(dict[str, Any], data_any if isinstance(data_any, dict) else {})
            orders: list[dict[str, Any]] = []
            orders_by_exchange: dict[str, Any] = cast(dict[str, Any], data.get("orders") or {})
            for exch, rows in orders_by_exchange.items():
                if isinstance(rows, list):
                    orders.extend([
                        {"exchange": str(exch), **cast(dict[str, Any], row)}
                        for row in cast(list[Any], rows)
                        if isinstance(row, dict)
                    ])
            canceled: list[dict[str, Any]] = []
            for o in orders:
                if str(o.get("symbol", "")).upper() != symbol.upper():
                    continue
                order_id = cast(str | None, o.get("id"))
                exch = cast(str, o.get("exchange") or o.get("source") or "binance")
                if order_id:
                    # Use real DELETE cancel contract
                    await client.delete(
                        _url(f"/api/live/trading/order/{order_id}"), params={"exchange": exch, "symbol": symbol}
                    )
                    canceled.append({"exchange": exch, "order_id": order_id})
        return {"ok": True, "canceled": canceled}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"cancel {symbol} failed: {e}")


@router.post("/trading/cancel-all")
async def trading_cancel_all() -> dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            orders_r = await client.get(_url("/api/live/trading/orders"))
            data_any: Any = orders_r.json() if orders_r.status_code == 200 else {}
            data = cast(dict[str, Any], data_any if isinstance(data_any, dict) else {})
            orders: list[dict[str, Any]] = []
            orders_by_exchange: dict[str, Any] = cast(dict[str, Any], data.get("orders") or {})
            for exch, rows in orders_by_exchange.items():
                if isinstance(rows, list):
                    orders.extend([
                        {"exchange": str(exch), **cast(dict[str, Any], row)}
                        for row in cast(list[Any], rows)
                        if isinstance(row, dict)
                    ])
            canceled: list[dict[str, Any]] = []
            for o in orders:
                order_id = cast(str | None, o.get("id"))
                symbol = cast(str, o.get("symbol") or "")
                exch = cast(str, o.get("exchange") or o.get("source") or "binance")
                if order_id:
                    await client.delete(
                        _url(f"/api/live/trading/order/{order_id}"), params={"exchange": exch, "symbol": symbol}
                    )
                    canceled.append({"exchange": exch, "order_id": order_id})
        return {"ok": True, "canceled": canceled}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"cancel-all failed: {e}")


# Portfolio alias forwards
@router.get("/portfolio/transactions")
async def portfolio_transactions_alias() -> Any:
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(_url("/api/portfolio/transactions"))
        return r.json() if r.status_code == 200 else {"error": r.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"transactions forward failed: {e}")


@router.get("/portfolio/risk-metrics")
async def portfolio_risk_metrics_alias() -> Any:
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(_url("/api/portfolio/risk-metrics"))
        return r.json() if r.status_code == 200 else {"error": r.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"risk-metrics forward failed: {e}")


