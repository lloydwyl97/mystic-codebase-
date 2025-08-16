from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

router = APIRouter()


@router.get("/ai/heartbeat")
async def get_ai_heartbeat() -> Dict[str, Any]:
    try:
        try:
            from backend.services.autobuy_service import autobuy_service  # type: ignore[import-not-found]
            hb: Dict[str, Any] = await autobuy_service.heartbeat()  # type: ignore[attr-defined]
            running = str(hb.get("status", "")) == "ready"
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "running": running,
                "strategies_active": len(hb.get("adapters", [])) if isinstance(hb.get("adapters"), list) else 0,
                "last_decision_ts": None,
                "queue_depth": hb.get("active_orders", 0),
            }
        except Exception:
            # Back-compat: try autobuy status
            from backend.endpoints.trading.autobuy_endpoints import get_autobuy_status  # type: ignore

            st: Dict[str, Any] = await get_autobuy_status()  # type: ignore[assignment]
            svc: Dict[str, Any] = st.get("service_status", {}) if isinstance(st, dict) else {}
            running = str(svc.get("status", "")) == "active"
            return {
                "timestamp": st.get("timestamp"),
                "running": running,
                "strategies_active": 0,
                "last_decision_ts": None,
                "queue_depth": svc.get("active_orders", 0),
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI heartbeat failed: {str(e)}")




