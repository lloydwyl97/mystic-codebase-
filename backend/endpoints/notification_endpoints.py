"""
Notification Endpoints

Handles all notification-related API endpoints including fetching and marking as read.
"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

logger = logging.getLogger("notification_endpoints")

# Global service references (will be set by main.py)
notification_service: Any | None = None


def set_services(ns: Any) -> None:
    """Set service references from main.py"""
    global notification_service
    notification_service = ns


router = APIRouter()


@router.get("/")
async def get_notifications() -> dict[str, Any]:
    """Get all notifications"""
    try:
        if notification_service and hasattr(notification_service, "get_notifications"):
            notifications: list[dict[str, Any]] = await notification_service.get_notifications()
            count = len(notifications)
            return {"notifications": notifications, "count": count}
        else:
            raise HTTPException(status_code=503, detail="Notification service not available")
    except Exception as e:
        logger.error(f"Error fetching notifications: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch notifications")


@router.post("/mark-read")
async def mark_notifications_read(
    notification_ids: list[str],
) -> dict[str, Any]:
    """Mark notifications as read"""
    try:
        if notification_service and hasattr(notification_service, "mark_read"):
            result = await notification_service.mark_read(notification_ids)
            return {"status": "success", "marked_read": result}
        else:
            raise HTTPException(status_code=503, detail="Notification service not available")
    except Exception as e:
        logger.error(f"Error marking notifications as read: {e}")
        raise HTTPException(status_code=500, detail="Failed to mark notifications as read")



