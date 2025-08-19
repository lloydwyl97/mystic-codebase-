"""
AI Performance Analytics Endpoint
Returns summarized performance data by strategy and totals for the UI.
"""

import logging
from typing import Any

from fastapi import APIRouter

router = APIRouter(prefix="/api/ai", tags=["ai"])

logger = logging.getLogger(__name__)

try:
    from backend.services.metrics.strategy_metrics_service import (
        StrategyMetricsService,  # type: ignore[import-not-found]
    )
except Exception:
    StrategyMetricsService = None  # type: ignore[assignment]


@router.get("/performance/analytics")
async def get_performance_analytics() -> dict[str, Any]:
    try:
        by_strategy: list[dict[str, Any]] = []
        totals: dict[str, Any] = {"profit_pct": 0.0, "trades": 0, "win_rate": 0.0}

        if StrategyMetricsService:
            try:
                svc: Any = StrategyMetricsService()  # type: ignore[no-redef]
                data: Any = await svc.get_performance_summary()  # type: ignore[func-returns-value]
                if isinstance(data, dict):
                    dd: dict[str, Any] = data  # narrow type for .get
                    bs: Any = dd.get("by_strategy", [])
                    if isinstance(bs, list):
                        by_strategy = bs  # type: ignore[assignment]
                    tt: Any = dd.get("totals", totals)
                    if isinstance(tt, dict):
                        totals = tt  # type: ignore[assignment]
            except Exception as e:
                logger.warning(f"analytics service failed: {e}")

        return {"by_strategy": by_strategy, "totals": totals}
    except Exception as e:
        logger.warning(f"/api/ai/performance/analytics error: {e}")
        return {"by_strategy": [], "totals": {"profit_pct": 0.0, "trades": 0, "win_rate": 0.0}}


