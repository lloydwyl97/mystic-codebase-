from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


class PerformanceMonitor:
    async def get_performance_metrics(self) -> dict[str, Any]:
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cpu_load": 0.0,
            "memory_used": 0,
        }

    async def get_detailed_metrics(self) -> dict[str, Any]:
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": {},
        }









