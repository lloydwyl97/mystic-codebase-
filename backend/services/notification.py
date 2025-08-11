"""
Notification Service for Mystic Trading

Provides centralized notification management for the platform.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

from services.notification_manager import notification_manager

logger = logging.getLogger(__name__)


class NotificationService:
    """Centralized notification service for managing and sending notifications"""

    def __init__(self, redis_client: Any):
        self.redis_client = redis_client

    async def send_notification(
        self,
        title: str,
        message: str,
        level: str = "info",
        channels: List[str] = ["in_app"],
    ) -> Dict[str, Any]:
        """Send a notification through specified channels"""
        try:
            # Validate notification data
            is_valid, error_message = notification_manager.validate_notification_data(
                title, message, level, channels
            )
            if not is_valid:
                return {
                    "status": "error",
                    "message": error_message,
                    "timestamp": (datetime.now(timezone.timezone.utc).isoformat()),
                }

            # Create notification object
            notification = notification_manager.create_notification_object(
                title, message, level, channels
            )

            # Store in Redis
            if "in_app" in notification["channels"]:
                # Add to recent notifications list
                notification_data = notification_manager.format_notification_for_storage(
                    notification
                )
                self.redis_client.lpush("recent_notifications", notification_data)
                self.redis_client.ltrim("recent_notifications", 0, 99)  # Keep last 100

            # Send through other channels (placeholder for actual implementations)
            if "email" in notification["channels"]:
                logger.info(f"Would send email: {title}")
                # Email sending logic would go here

            if "sms" in notification["channels"]:
                logger.info(f"Would send SMS: {title}")
                # SMS sending logic would go here

            if "webhook" in notification["channels"]:
                logger.info(f"Would send webhook: {title}")
                # Webhook sending logic would go here

            return {
                "status": "success",
                "notification_id": notification["id"],
                "channels": notification["channels"],
                "timestamp": notification["timestamp"],
            }
        except Exception as e:
            logger.error(f"Error sending notification: {str(e)}")
            return {
                "status": "error",
                "message": f"Error sending notification: {str(e)}",
                "timestamp": datetime.now(timezone.timezone.utc).isoformat(),
            }

    async def get_notifications(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent notifications"""
        try:
            # Get from Redis
            notifications_data = self.redis_client.lrange("recent_notifications", 0, limit - 1)

            # Parse JSON
            notifications: List[Dict[str, Any]] = []
            for notification_data in notifications_data:
                notification = notification_manager.parse_notification_from_storage(
                    notification_data
                )
                if notification:
                    notifications.append(notification)

            return notifications
        except Exception as e:
            logger.error(f"Error getting notifications: {str(e)}")
            return []

    async def mark_notification_read(self, notification_id: str) -> bool:
        """Mark a notification as read"""
        try:
            # Get from Redis
            notifications_data = self.redis_client.lrange("recent_notifications", 0, -1)

            # Parse and update notifications
            notifications: List[Dict[str, Any]] = []
            for notification_data in notifications_data:
                notification = notification_manager.parse_notification_from_storage(
                    notification_data
                )
                if notification:
                    notifications.append(notification)

            # Mark as read
            found, updated_notifications = notification_manager.mark_notification_read(
                notifications, notification_id
            )

            if found:
                # Update Redis with modified notifications
                self.redis_client.delete("recent_notifications")
                for notification in updated_notifications:
                    notification_data = notification_manager.format_notification_for_storage(
                        notification
                    )
                    self.redis_client.rpush("recent_notifications", notification_data)
                return True

            return False
        except Exception as e:
            logger.error(f"Error marking notification as read: {str(e)}")
            return False

    async def clear_all_notifications(self) -> int:
        """Clear all notifications"""
        try:
            # Get count
            count = self.redis_client.llen("recent_notifications")

            # Delete key
            self.redis_client.delete("recent_notifications")

            return count
        except Exception as e:
            logger.error(f"Error clearing notifications: {str(e)}")
            return 0

    async def get_notification_summary(self) -> Dict[str, Any]:
        """Get notification summary statistics"""
        try:
            notifications = await self.get_notifications(limit=1000)  # Get all notifications
            return notification_manager.get_notification_summary(notifications)
        except Exception as e:
            logger.error(f"Error getting notification summary: {str(e)}")
            return {
                "total_count": 0,
                "unread_count": 0,
                "read_count": 0,
                "level_counts": {},
                "timestamp": datetime.now(timezone.timezone.utc).isoformat(),
            }

    async def get_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get alerts (high-priority notifications)"""
        try:
            # Get all notifications
            notifications = await self.get_notifications(limit=1000)

            # Filter for alerts (high priority notifications)
            alerts = []
            for notification in notifications:
                if notification.get("level") in [
                    "error",
                    "warning",
                    "critical",
                ]:
                    alerts.append(
                        {
                            "id": notification.get("id"),
                            "title": notification.get("title"),
                            "message": notification.get("message"),
                            "level": notification.get("level"),
                            "timestamp": notification.get("timestamp"),
                            "read": notification.get("read", False),
                            "type": "alert",
                        }
                    )

            # Sort by timestamp (newest first) and limit
            alerts.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            return alerts[:limit]

        except Exception as e:
            logger.error(f"Error getting alerts: {str(e)}")
            return []

    async def create_alert(
        self, title: str, message: str, level: str = "warning"
    ) -> Dict[str, Any]:
        """Create a new alert"""
        try:
            # Create alert notification
            result = await self.send_notification(
                title=title, message=message, level=level, channels=["in_app"]
            )

            if result.get("status") == "success":
                return {
                    "status": "success",
                    "alert_id": result.get("notification_id"),
                    "message": "Alert created successfully",
                    "timestamp": (datetime.now(timezone.timezone.utc).isoformat()),
                }
            else:
                return {
                    "status": "error",
                    "message": result.get("message", "Failed to create alert"),
                    "timestamp": (datetime.now(timezone.timezone.utc).isoformat()),
                }

        except Exception as e:
            logger.error(f"Error creating alert: {str(e)}")
            return {
                "status": "error",
                "message": f"Error creating alert: {str(e)}",
                "timestamp": datetime.now(timezone.timezone.utc).isoformat(),
            }


# Global notification service instance
notification_service = None


def get_notification_service(redis_client: Any) -> NotificationService:
    """Get or create notification service instance"""
    global notification_service
    if notification_service is None:
        notification_service = NotificationService(redis_client)
    return notification_service
