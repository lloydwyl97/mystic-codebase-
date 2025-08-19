"""
AI Signal Service (compat shim)
Provides a minimal AISignalService expected by endpoints.
"""

from __future__ import annotations

from typing import Any


class AISignalService:
    async def get_signals(self) -> dict[str, Any]:
        # Minimal live-compatible response
        return {"signals": []}




