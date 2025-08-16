from __future__ import annotations

from typing import Any, Dict
from datetime import datetime, timezone


class PerformanceMonitor:
    async def get_performance_metrics(self) -> Dict[str, Any]:
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cpu_load": 0.0,
            "memory_used": 0,
        }

    async def get_detailed_metrics(self) -> Dict[str, Any]:
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": {},
        }









