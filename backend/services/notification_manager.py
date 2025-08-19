"""
Notification Manager

Handles business logic for notification operations including validation,
formatting, and channel management.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


class NotificationManager:
    """Manages notification business logic and operations."""

    def __init__(self):
        self.notification_channels = ["in_app", "email", "sms", "webhook"]
        self.notification_levels = ["info", "warning", "error", "critical"]

    def validate_notification_data(
        self, title: str, message: str, level: str, channels: list[str]
    ) -> tuple[bool, str]:
        """Validate notification data before processing."""
        # Validate title
        if not title or not title.strip():
            return False, "Title cannot be empty"

        # Validate message
        if not message or not message.strip():
            return False, "Message cannot be empty"

        # Validate level
        if level not in self.notification_levels:
            return (
                False,
                f"Invalid level. Must be one of: {', '.join(self.notification_levels)}",
            )

        # Validate channels
        if not channels:
            return False, "At least one channel must be specified"

        valid_channels = [c for c in channels if c in self.notification_channels]
        if not valid_channels:
            return (
                False,
                f"Invalid channels. Must be one or more of: {', '.join(self.notification_channels)}",
            )

        return True, "Notification data is valid"

    def generate_notification_id(self) -> str:
        """Generate a unique notification ID."""
        return f"notification_{datetime.now(timezone.timezone.utc).timestamp()}"

    def create_notification_object(
        self, title: str, message: str, level: str, channels: list[str]
    ) -> dict[str, Any]:
        """Create a notification object with proper formatting."""
        valid_channels = [c for c in channels if c in self.notification_channels]

        return {
            "id": self.generate_notification_id(),
            "title": title.strip(),
            "message": message.strip(),
            "level": level,
            "channels": valid_channels,
            "timestamp": datetime.now(timezone.timezone.utc).isoformat(),
            "read": False,
        }

    def format_notification_for_storage(self, notification: dict[str, Any]) -> str:
        """Format notification for Redis storage."""
        return json.dumps(notification)

    def parse_notification_from_storage(self, notification_data: str) -> dict[str, Any] | None:
        """Parse notification from Redis storage."""
        try:
            return json.loads(notification_data)
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing notification data: {str(e)}")
            return None

    def get_notification_summary(self, notifications: list[dict[str, Any]]) -> dict[str, Any]:
        """Get summary statistics for notifications."""
        total_count = len(notifications)
        unread_count = sum(1 for n in notifications if not n.get("read", False))

        level_counts = {}
        for level in self.notification_levels:
            level_counts[level] = sum(1 for n in notifications if n.get("level") == level)

        return {
            "total_count": total_count,
            "unread_count": unread_count,
            "read_count": total_count - unread_count,
            "level_counts": level_counts,
            "timestamp": datetime.now(timezone.timezone.utc).isoformat(),
        }

    def filter_notifications(
        self,
        notifications: list[dict[str, Any]],
        level: str | None = None,
        read_status: bool | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Filter notifications based on criteria."""
        filtered = notifications

        # Filter by level
        if level and level in self.notification_levels:
            filtered = [n for n in filtered if n.get("level") == level]

        # Filter by read status
        if read_status is not None:
            filtered = [n for n in filtered if n.get("read", False) == read_status]

        # Apply limit
        if limit and limit > 0:
            filtered = filtered[:limit]

        return filtered

    def mark_notification_read(
        self, notifications: list[dict[str, Any]], notification_id: str
    ) -> tuple[bool, list[dict[str, Any]]]:
        """Mark a notification as read and return updated list."""
        updated_notifications: list[dict[str, Any]] = []
        found = False

        for notification in notifications:
            if notification.get("id") == notification_id:
                notification["read"] = True
                notification["read_at"] = datetime.now(timezone.timezone.utc).isoformat()
                found = True
            updated_notifications.append(notification)

        return found, updated_notifications

    def cleanup_old_notifications(
        self, notifications: list[dict[str, Any]], max_age_hours: int = 24
    ) -> list[dict[str, Any]]:
        """Remove notifications older than specified age."""
        cutoff_time = datetime.now(timezone.timezone.utc).timestamp() - (max_age_hours * 3600)

        return [
            n
            for n in notifications
            if datetime.fromisoformat(n.get("timestamp", "")).timestamp() > cutoff_time
        ]

    def validate_redis_operations(self, redis_client: Any) -> bool:
        """Validate that Redis operations are available."""
        try:
            # Test basic Redis operations
            test_key = "notification_test"
            test_value = "test"
            redis_client.set(test_key, test_value)
            result = redis_client.get(test_key)
            redis_client.delete(test_key)
            return result == test_value
        except Exception as e:
            logger.error(f"Redis validation failed: {e}")
            return False


# Global instance
notification_manager = NotificationManager()


