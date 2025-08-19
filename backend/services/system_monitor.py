from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


class SystemMonitor:
    async def get_services_status(self) -> dict[str, Any]:
        return {"services": [], "timestamp": datetime.now(timezone.utc).isoformat()}

    async def get_recent_events(self, limit: int = 50) -> list[dict[str, Any]]:
        return [
            {
                "type": "info",
                "message": "System monitor initialized",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        ][:limit]









