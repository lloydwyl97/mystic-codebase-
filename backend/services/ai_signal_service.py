"""
AI Signal Service (compat shim)
Provides a minimal AISignalService expected by endpoints.
"""

from __future__ import annotations

from typing import Any, Dict


class AISignalService:
    async def get_signals(self) -> Dict[str, Any]:
        # Minimal live-compatible response
        return {"signals": []}




