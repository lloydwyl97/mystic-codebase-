"""
Portfolio Transactions Endpoint
Provides a minimal transactions feed for the UI. Proxies real data if available.
"""

import logging
from typing import Any, cast

from fastapi import APIRouter

router = APIRouter(prefix="/api/portfolio", tags=["portfolio"])

logger = logging.getLogger(__name__)

try:
    # Prefer a central trade/ledger source if available
    from backend.modules.ai.trade_tracker import get_trade_history  # type: ignore[import-not-found]
except Exception:
    get_trade_history = None  # type: ignore[assignment]


@router.get("/transactions")
async def get_transactions() -> dict[str, Any]:
    try:
        items: list[dict[str, Any]] = []
        if callable(get_trade_history):
            try:
                # Some implementations may not accept "limit"; call safely
                try:
                    history = get_trade_history(limit=200)  # type: ignore[call-arg]
                except Exception:
                    history = get_trade_history()  # type: ignore[call-arg]
                for t in cast(list[dict[str, Any]], history or []):
                    items.append(
                        {
                            "timestamp": t.get("timestamp"),
                            "symbol": t.get("symbol"),
                            "side": t.get("type") or t.get("side"),
                            "qty": t.get("amount") or t.get("qty"),
                            "price": t.get("price"),
                            "fee": t.get("fee", 0.0),
                            "source": t.get("source", "ai"),
                        }
                    )
            except Exception as e:
                logger.warning(f"transactions history fetch failed: {e}")
        return {"items": items, "count": len(items)}
    except Exception as e:
        logger.warning(f"/api/portfolio/transactions error: {e}")
        return {"items": [], "count": 0}


