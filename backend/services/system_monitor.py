from __future__ import annotations

from typing import Any, Dict, List
from datetime import datetime, timezone


class SystemMonitor:
    async def get_services_status(self) -> Dict[str, Any]:
        return {"services": [], "timestamp": datetime.now(timezone.utc).isoformat()}

    async def get_recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        return [
            {
                "type": "info",
                "message": "System monitor initialized",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        ][:limit]







