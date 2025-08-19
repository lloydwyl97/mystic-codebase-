"""
AI Leaderboard Endpoints
Provide basic leaderboard and analytics-compatible responses for the UI.
"""

import logging
from typing import Any, Dict, List

from fastapi import APIRouter

router = APIRouter(prefix="/api/ai", tags=["ai"])

logger = logging.getLogger(__name__)

try:
    from backend.services.metrics.strategy_metrics_service import StrategyMetricsService  # type: ignore[import-not-found]
except Exception:
    StrategyMetricsService = None  # type: ignore[assignment]


@router.get("/strategies/leaderboard")
async def get_leaderboard() -> List[Dict[str, Any]]:
    try:
        if StrategyMetricsService:
            try:
                svc: Any = StrategyMetricsService()  # type: ignore[no-redef]
                leaderboard: Any = await svc.get_leaderboard()  # type: ignore[func-returns-value]
                if isinstance(leaderboard, list):
                    return leaderboard  # type: ignore[return-value]
            except Exception as e:
                logger.warning(f"leaderboard service failed: {e}")
        return []
    except Exception as e:
        logger.warning(f"/api/ai/strategies/leaderboard error: {e}")
        return []


