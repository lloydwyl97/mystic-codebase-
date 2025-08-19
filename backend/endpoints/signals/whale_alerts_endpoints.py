"""
Whale Alerts Endpoints

Provides live whale alerts with ingest endpoints, backed by in-memory TTL service.
"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from backend.services.whale_alert_service import whale_alert_service

router = APIRouter(prefix="/api/whale", tags=["signals"])

logger = logging.getLogger(__name__)


@router.get("/alerts")
async def get_whale_alerts(limit: int = Query(200, ge=1, le=1000)) -> dict[str, Any]:
    try:
        alerts = whale_alert_service.get_alerts(limit=limit)
        return {"alerts": alerts, "count": len(alerts)}
    except Exception as e:
        logger.warning(f"/api/whale/alerts error: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch whale alerts")


@router.post("/alerts/ingest")
async def ingest_whale_alert(alert: dict[str, Any]) -> dict[str, Any]:
    try:
        whale_alert_service.ingest(alert)
        return {"ok": True}
    except Exception as e:
        logger.error(f"whale alert ingest failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to ingest alert")


@router.post("/alerts/ingest_bulk")
async def ingest_bulk_whale_alerts(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        alerts_any: Any | None = payload.get("alerts")
        alerts: list[dict[str, Any]] = alerts_any if isinstance(alerts_any, list) else []
        count = whale_alert_service.bulk_ingest(alerts)
        return {"ok": True, "ingested": count}
    except Exception as e:
        logger.error(f"whale alert bulk ingest failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to bulk ingest alerts")


