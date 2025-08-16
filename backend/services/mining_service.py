from typing import Any, Dict
from datetime import datetime, timezone


class MiningService:
    async def get_status(self) -> Dict[str, Any]:
        return {"timestamp": datetime.now(timezone.utc).isoformat()}

    async def get_performance(self) -> Dict[str, Any]:
        return {"timestamp": datetime.now(timezone.utc).isoformat()}

    async def restart(self) -> Dict[str, Any]:
        return {"timestamp": datetime.now(timezone.utc).isoformat(), "restarted": True}





