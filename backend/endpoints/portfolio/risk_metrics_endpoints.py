"""
Portfolio Risk Metrics Endpoint

Computes basic portfolio risk metrics from existing portfolio endpoints/services.
"""

from __future__ import annotations

import math
from typing import Any

from fastapi import APIRouter, HTTPException

try:
    # Prefer calling existing portfolio endpoints/services to avoid duplication
    from backend.services.portfolio_service import PortfolioService  # type: ignore[import-not-found]
except Exception:
    PortfolioService = None  # type: ignore[assignment]


router = APIRouter(prefix="/api/portfolio", tags=["portfolio"]) 


@router.get("/risk-metrics")
async def get_risk_metrics() -> dict[str, Any]:
    try:
        total_value: float = 0.0
        positions: list[dict[str, Any]] = []
        pnl_24h: float | None = None

        # Fetch from service if available
        if PortfolioService:
            svc = PortfolioService()
            overview = svc.get_portfolio_overview()
            total_value = float(overview.get("total_value", 0.0) or 0.0)
            # positions list is derived from overview holdings
            positions = await svc.get_positions()
            # 24h pnl not directly available; leave None unless source is added later
        
        # Derive exposures and weights
        exposure_pct: dict[str, float] = {}
        weights: list[float] = []
        largest_position_pct: float = 0.0
        if total_value > 0 and positions:
            for p in positions:
                symbol = str(p.get("symbol", "")).upper()
                current_value = float(p.get("current_value", 0.0) or 0.0)
                w = (current_value / total_value) * 100.0 if total_value > 0 else 0.0
                exposure_pct[symbol] = w
                weights.append(w)
                if w > largest_position_pct:
                    largest_position_pct = w

        weights_stddev_pct: float | None = None
        if weights:
            mean = sum(weights) / len(weights)
            var = sum((w - mean) ** 2 for w in weights) / len(weights)
            weights_stddev_pct = math.sqrt(var)

        return {
            "risk_metrics": {
                "total_value": total_value,
                "num_positions": len(positions),
                "largest_position_pct": largest_position_pct,
                "exposure_pct": exposure_pct,
                "pnl_24h": pnl_24h,
                "weights_stddev_pct": weights_stddev_pct,
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Risk metrics failed: {e}")


