from datetime import datetime, timezone
from typing import Any


class MiningService:
    async def get_status(self) -> dict[str, Any]:
        return {"timestamp": datetime.now(timezone.utc).isoformat()}

    async def get_performance(self) -> dict[str, Any]:
        return {"timestamp": datetime.now(timezone.utc).isoformat()}

    async def restart(self) -> dict[str, Any]:
        return {"timestamp": datetime.now(timezone.utc).isoformat(), "restarted": True}





