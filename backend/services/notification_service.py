"""
Notification Service
Handles notifications and alerts
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, timezone
import uuid

logger = logging.getLogger(__name__)


class NotificationService:
    def __init__(self):
        self.notifications = {}
        self.alert_settings = {
            "email_enabled": True,
            "push_enabled": True,
            "sms_enabled": False,
            "trade_alerts": True,
            "price_alerts": True,
            "system_alerts": True,
        }
        logger.info("✅ NotificationService initialized")

    async def create_notification(
        self,
        title: str,
        message: str,
        notification_type: str = "info",
        user_id: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a new notification"""
        try:
            notification_id = str(uuid.uuid4())
            notification = {
                "id": notification_id,
                "title": title,
                "message": message,
                "type": notification_type,
                "user_id": user_id,
                "data": data or {},
                "read": False,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            self.notifications[notification_id] = notification
            logger.info(f"✅ Created notification: {notification_id}")

            return {"success": True, "notification": notification}
        except Exception as e:
            logger.error(f"❌ Error creating notification: {e}")
            return {"success": False, "error": str(e)}

    async def get_notifications(
        self,
        user_id: Optional[str] = None,
        limit: int = 50,
        unread_only: bool = False,
    ) -> Dict[str, Any]:
        """Get notifications for a user"""
        try:
            notifications = list(self.notifications.values())

            # Filter by user if specified
            if user_id:
                notifications = [n for n in notifications if n.get("user_id") == user_id]

            # Filter unread only if requested
            if unread_only:
                notifications = [n for n in notifications if not n.get("read", False)]

            # Sort by creation time (newest first)
            notifications.sort(key=lambda x: x["created_at"], reverse=True)

            # Apply limit
            notifications = notifications[:limit]

            return {
                "notifications": notifications,
                "total": len(notifications),
                "unread_count": len([n for n in notifications if not n.get("read", False)]),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            logger.error(f"❌ Error getting notifications: {e}")
            return {"error": str(e)}

    async def mark_notification_read(self, notification_id: str) -> Dict[str, Any]:
        """Mark a notification as read"""
        try:
            if notification_id in self.notifications:
                self.notifications[notification_id]["read"] = True
                self.notifications[notification_id]["read_at"] = datetime.now(
                    timezone.utc
                ).isoformat()
                logger.info(f"✅ Marked notification as read: {notification_id}")
                return {"success": True, "notification_id": notification_id}
            else:
                return {"success": False, "error": "Notification not found"}
        except Exception as e:
            logger.error(f"❌ Error marking notification as read: {e}")
            return {"success": False, "error": str(e)}

    async def mark_all_notifications_read(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Mark all notifications as read for a user"""
        try:
            count = 0
            for notification in self.notifications.values():
                if user_id is None or notification.get("user_id") == user_id:
                    if not notification.get("read", False):
                        notification["read"] = True
                        notification["read_at"] = datetime.now(timezone.utc).isoformat()
                        count += 1

            logger.info(f"✅ Marked {count} notifications as read")
            return {"success": True, "count": count}
        except Exception as e:
            logger.error(f"❌ Error marking all notifications as read: {e}")
            return {"success": False, "error": str(e)}

    async def delete_notification(self, notification_id: str) -> Dict[str, Any]:
        """Delete a notification"""
        try:
            if notification_id in self.notifications:
                del self.notifications[notification_id]
                logger.info(f"✅ Deleted notification: {notification_id}")
                return {"success": True, "notification_id": notification_id}
            else:
                return {"success": False, "error": "Notification not found"}
        except Exception as e:
            logger.error(f"❌ Error deleting notification: {e}")
            return {"success": False, "error": str(e)}

    async def clear_all_notifications(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Clear all notifications for a user"""
        try:
            count = 0
            to_delete = []

            for notification_id, notification in self.notifications.items():
                if user_id is None or notification.get("user_id") == user_id:
                    to_delete.append(notification_id)

            for notification_id in to_delete:
                del self.notifications[notification_id]
                count += 1

            logger.info(f"✅ Cleared {count} notifications")
            return {"success": True, "count": count}
        except Exception as e:
            logger.error(f"❌ Error clearing notifications: {e}")
            return {"success": False, "error": str(e)}

    async def create_trade_alert(
        self,
        symbol: str,
        action: str,
        price: float,
        amount: float,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a trade alert notification"""
        title = f"Trade Executed: {action.upper()} {symbol}"
        message = f"Executed {action} order for {amount} {symbol} at ${price:,.2f}"

        return await self.create_notification(
            title=title,
            message=message,
            notification_type="trade",
            user_id=user_id,
            data={
                "symbol": symbol,
                "action": action,
                "price": price,
                "amount": amount,
            },
        )

    async def create_price_alert(
        self,
        symbol: str,
        current_price: float,
        target_price: float,
        condition: str,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a price alert notification"""
        title = f"Price Alert: {symbol}"
        message = (
            f"{symbol} is now ${current_price:,.2f} ({condition} target of ${target_price:,.2f})"
        )

        return await self.create_notification(
            title=title,
            message=message,
            notification_type="price_alert",
            user_id=user_id,
            data={
                "symbol": symbol,
                "current_price": current_price,
                "target_price": target_price,
                "condition": condition,
            },
        )

    async def create_system_alert(
        self,
        title: str,
        message: str,
        severity: str = "info",
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a system alert notification"""
        return await self.create_notification(
            title=title,
            message=message,
            notification_type=f"system_{severity}",
            user_id=user_id,
            data={"severity": severity},
        )

    async def get_notification_settings(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get notification settings"""
        return {
            "settings": self.alert_settings,
            "user_id": user_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    async def update_notification_settings(
        self, settings: Dict[str, Any], user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Update notification settings"""
        try:
            self.alert_settings.update(settings)
            logger.info(f"✅ Updated notification settings: {settings}")
            return {
                "success": True,
                "settings": self.alert_settings,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            logger.error(f"❌ Error updating notification settings: {e}")
            return {"success": False, "error": str(e)}

    async def get_notification_stats(self) -> Dict[str, Any]:
        """Get notification statistics"""
        try:
            total_notifications = len(self.notifications)
            unread_notifications = len(
                [n for n in self.notifications.values() if not n.get("read", False)]
            )

            # Count by type
            type_counts = {}
            for notification in self.notifications.values():
                notification_type = notification.get("type", "unknown")
                type_counts[notification_type] = type_counts.get(notification_type, 0) + 1

            return {
                "total_notifications": total_notifications,
                "unread_notifications": unread_notifications,
                "type_counts": type_counts,
                "settings": self.alert_settings,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            logger.error(f"❌ Error getting notification stats: {e}")
            return {"error": str(e)}


# Global instance
notification_service = NotificationService()


def get_notification_service() -> NotificationService:
    """Get the notification service instance"""
    return notification_service
