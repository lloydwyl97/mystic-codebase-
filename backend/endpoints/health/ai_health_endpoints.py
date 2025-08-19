"""
AI Health Endpoint
Mirrors system health structure for AI subsystems.
"""

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter

router = APIRouter(prefix="/api/ai/system", tags=["health"])


@router.get("/health")
async def get_ai_system_health() -> dict[str, Any]:
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "services": [
            {"name": "ai-strategy-engine", "status": "ok"},
        ],
    }


