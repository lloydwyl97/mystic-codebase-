"""
AI Strategy Service (compat shim)
Exposes an AIStrategyService API expected by endpoints, delegating to AIStrategiesService.
"""

from __future__ import annotations

from typing import Any, Dict, List

try:
    from .ai_strategies import AIStrategiesService
except Exception:  # pragma: no cover
    AIStrategiesService = None  # type: ignore[assignment]


class AIStrategyService:
    def __init__(self) -> None:
        if AIStrategiesService is None:
            raise ImportError("AIStrategiesService is not available")
        self._svc = AIStrategiesService()

    async def get_all_strategies(self) -> List[Dict[str, Any]]:
        return await self._svc.get_leaderboard()

    async def get_strategy_performance(self) -> Dict[str, Any]:
        return await self._svc.get_performance_analytics()

    async def get_system_status(self) -> Dict[str, Any]:
        return await self._svc.get_ai_status()

    async def get_performance_metrics(self) -> Dict[str, Any]:
        return await self._svc.get_performance_analytics()

    async def get_model_status(self) -> Dict[str, Any]:
        # No explicit model registry; reuse AI status
        return await self._svc.get_ai_status()

    async def get_models(self) -> Dict[str, Any]:
        status = await self._svc.get_ai_status()
        return {"active_models": status.get("active_models", 0)}

    async def get_model_configurations(self) -> Dict[str, Any]:
        return {}

    async def get_training_status(self) -> Dict[str, Any]:
        return {"status": "idle"}

    async def get_training_metrics(self) -> Dict[str, Any]:
        return {}

    async def start_retraining(self) -> Dict[str, Any]:
        return {"started": False, "reason": "not_implemented"}

    async def update_strategy(self, strategy_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        # No direct update path; acknowledge receipt
        return {"updated": False, "strategy_id": strategy_id, "reason": "not_implemented"}




