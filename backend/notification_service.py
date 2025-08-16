"""
Notification Service for Mystic Trading

Handles notifications for signal failures, recoveries, and system events.
Supports multiple notification channels: email, Slack, webhook, and in-app.
"""

import json
import logging
import smtplib
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional, Union, cast

import aiohttp
import redis

logger = logging.getLogger(__name__)


class NotificationService:
    def __init__(self, redis_client: Union[redis.Redis, Any]):
        self.redis_client = redis_client
        self.session: Optional[aiohttp.ClientSession] = None
        self.notifications: List[Dict[str, Any]] = []
        self.notification_id_counter = 1

        # Notification configuration
        self.config: Dict[str, Dict[str, Any]] = {
            "email": {
                "enabled": False,
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "username": "",
                "password": "",
                "from_email": "",
                "to_emails": [],
            },
            "slack": {
                "enabled": False,
                "webhook_url": "",
                "channel": "#trading-alerts",
            },
            "webhook": {
                "enabled": False,
                "url": "",
                "headers": {"Content-Type": "application/json"},
            },
            "in_app": {
                "enabled": True,  # Always enabled for in-app notifications
                "max_notifications": 100,
            },
        }

        # Load configuration from Redis
        self._load_config()

    def _load_config(self):
        """Load notification configuration from Redis"""
        try:
            config_data = self.redis_client.get("notification_config")
            if config_data:
                # Handle Redis response which could be bytes or string
                if isinstance(config_data, bytes):
                    config_str = config_data.decode("utf-8")
                else:
                    config_str = str(config_data)

                stored_config: Dict[str, Any] = json.loads(config_str)
                # Merge with defaults, keeping defaults for missing keys
                for channel, config in stored_config.items():
                    if channel in self.config and isinstance(config, dict):
                        if isinstance(self.config[channel], dict):
                            self.config[channel].update(config)
        except Exception as e:
            logger.warning(f"Could not load notification config: {str(e)}")

    def _save_config(self):
        """Save notification configuration to Redis"""
        try:
            # The config is already a dictionary, so we can serialize it directly
            self.redis_client.setex("notification_config", 3600, json.dumps(self.config))
        except Exception as e:
            logger.error(f"Could not save notification config: {str(e)}")

    async def send_notification(
        self,
        title: str,
        message: str,
        level: str = "info",
        channels: Optional[List[str]] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Send notification through specified channels

        Args:
            title: Notification title
            message: Notification message
            level: Notification level (info, warning, error, critical)
            channels: List of channels to use (email, slack, webhook, in_app)
            data: Additional data to include
        """
        if channels is None:
            channels = ["in_app"]  # Default to in-app only

        results: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.timezone.utc).isoformat(),
            "title": title,
            "message": message,
            "level": level,
            "channels": {},
            "success": True,
        }

        # Send to each channel
        for channel in channels:
            if channel in self.config and self.config[channel].get("enabled", False):
                try:
                    if channel == "email":
                        result = await self._send_email(title, message, level, data)
                    elif channel == "slack":
                        result = await self._send_slack(title, message, level, data)
                    elif channel == "webhook":
                        result = await self._send_webhook(title, message, level, data)
                    elif channel == "in_app":
                        result = await self._send_in_app(title, message, level, data)
                    else:
                        result = {
                            "success": False,
                            "error": f"Unknown channel: {channel}",
                        }

                    results["channels"][channel] = result
                    if not result.get("success", False):
                        results["success"] = False

                except Exception as e:
                    logger.error(f"Error sending {channel} notification: {str(e)}")
                    results["channels"][channel] = {
                        "success": False,
                        "error": str(e),
                    }
                    results["success"] = False

        # Log the notification
        logger.info(f"Notification sent: {title} - {message} (Level: {level})")

        return results

    async def _send_email(
        self,
        title: str,
        message: str,
        level: str,
        data: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Send email notification"""
        try:
            config = self.config["email"]

            # Create email message
            msg = MIMEMultipart()
            msg["From"] = config["from_email"]
            msg["To"] = ", ".join(config["to_emails"])
            msg["Subject"] = f"[Mystic Trading] {title}"

            # Create HTML body
            html_body = f"""
            <html>
            <body>
                <h2>{title}</h2>
                <p><strong>Level:</strong> {level.upper()}</p>
                <p><strong>Time:</strong> {datetime.now(timezone.timezone.utc).strftime('%Y-%m-%d %H:%M:%S timezone.utc')}</p>
                <p>{message}</p>
            """

            if data:
                html_body += (
                    "<h3>Additional Data:</h3><pre>" + json.dumps(data, indent=2) + "</pre>"
                )

            html_body += "</body></html>"

            msg.attach(MIMEText(html_body, "html"))

            # Send email
            with smtplib.SMTP(config["smtp_server"], config["smtp_port"]) as server:
                server.starttls()
                server.login(config["username"], config["password"])
                server.send_message(msg)

            return {"success": True, "recipients": len(config["to_emails"])}

        except Exception as e:
            logger.error(f"Email notification failed: {str(e)}")
            return {"success": False, "error": str(e)}

    async def _send_slack(
        self,
        title: str,
        message: str,
        level: str,
        data: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Send Slack notification"""
        try:
            config = self.config["slack"]

            # Create Slack message
            color_map = {
                "info": "#36a64f",  # Green
                "warning": "#ff9500",  # Orange
                "error": "#ff0000",  # Red
                "critical": "#8b0000",  # Dark red
            }

            slack_message: Dict[str, Any] = {
                "channel": config["channel"],
                "attachments": [
                    {
                        "color": color_map.get(level, "#36a64f"),
                        "title": title,
                        "text": message,
                        "fields": [
                            {
                                "title": "Level",
                                "value": level.upper(),
                                "short": True,
                            },
                            {
                                "title": "Time",
                                "value": (
                                    datetime.now(timezone.timezone.utc).strftime(
                                        "%Y-%m-%d %H:%M:%S timezone.utc"
                                    )
                                ),
                                "short": True,
                            },
                        ],
                        "footer": "Mystic Trading Bot",
                    }
                ],
            }

            if data:
                slack_message["attachments"][0]["fields"].append(
                    {
                        "title": "Additional Data",
                        "value": f"```{json.dumps(data, indent=2)}```",
                        "short": False,
                    }
                )

            # Send to Slack
            if not self.session:
                self.session = aiohttp.ClientSession()

            async with self.session.post(config["webhook_url"], json=slack_message) as response:
                if response.status == 200:
                    return {"success": True}
                else:
                    return {
                        "success": False,
                        "error": f"HTTP {response.status}",
                    }

        except Exception as e:
            logger.error(f"Slack notification failed: {str(e)}")
            return {"success": False, "error": str(e)}

    async def _send_webhook(
        self,
        title: str,
        message: str,
        level: str,
        data: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Send webhook notification"""
        try:
            config = self.config["webhook"]

            webhook_data: Dict[str, Any] = {
                "title": title,
                "message": message,
                "level": level,
                "timestamp": datetime.now(timezone.timezone.utc).isoformat(),
                "source": "mystic_trading_bot",
            }

            if data:
                webhook_data["data"] = data

            # Send webhook
            if not self.session:
                self.session = aiohttp.ClientSession()

            async with self.session.post(
                config["url"], json=webhook_data, headers=config["headers"]
            ) as response:
                if response.status in [200, 201, 202]:
                    return {"success": True}
                else:
                    return {
                        "success": False,
                        "error": f"HTTP {response.status}",
                    }

        except Exception as e:
            logger.error(f"Webhook notification failed: {str(e)}")
            return {"success": False, "error": str(e)}

    async def _send_in_app(
        self,
        title: str,
        message: str,
        level: str,
        data: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Store in-app notification"""
        try:
            notification = {
                "id": f"notif_{self.notification_id_counter}",
                "title": title,
                "message": message,
                "level": level,
                "timestamp": datetime.now(timezone.timezone.utc).isoformat(),
                "read": False,
                "data": data or {},
            }

            # Store in Redis
            self.redis_client.lpush("in_app_notifications", json.dumps(notification))

            # Keep only the latest notifications
            config = self.config["in_app"]
            self.redis_client.ltrim("in_app_notifications", 0, config["max_notifications"] - 1)

            self.notifications.append(notification)
            self.notification_id_counter += 1

            return {"success": True, "notification_id": notification["id"]}

        except Exception as e:
            logger.error(f"In-app notification failed: {str(e)}")
            return {"success": False, "error": str(e)}

    async def get_notifications(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get notifications."""
        try:
            # Redis lrange returns a list of bytes or strings
            lrange_result = self.redis_client.lrange("in_app_notifications", 0, limit - 1)
            # Handle both sync and async Redis clients
            if hasattr(lrange_result, "__await__"):
                notifications_data = await lrange_result
            else:
                notifications_data = cast(List[Any], lrange_result)

            notifications: List[Dict[str, Any]] = []

            # Process each notification item
            for data in notifications_data:
                try:
                    if isinstance(data, bytes):
                        data_str = data.decode("utf-8")
                    elif isinstance(data, str):
                        data_str = data
                    else:
                        data_str = str(data)
                    notification: Dict[str, Any] = json.loads(data_str)
                    notifications.append(notification)
                except (json.JSONDecodeError, ValueError, KeyError) as e:
                    logger.warning(f"Could not parse notification data: {e}")
                    continue

            return notifications

        except Exception as e:
            logger.error(f"Error getting notifications: {str(e)}")
            return []

    async def mark_read(self, notification_id: str) -> Dict[str, Any]:
        """Mark notification as read."""
        try:
            # Redis lrange returns a list of bytes or strings
            lrange_result = self.redis_client.lrange("in_app_notifications", 0, -1)
            # Handle both sync and async Redis clients
            if hasattr(lrange_result, "__await__"):
                notifications_data = await lrange_result
            else:
                notifications_data = cast(List[Any], lrange_result)

            # Process each notification item
            try:
                # Try to iterate directly (sync response)
                for i, data in enumerate(notifications_data):
                    try:
                        if isinstance(data, bytes):
                            data_str = data.decode("utf-8")
                        elif isinstance(data, str):
                            data_str = data
                        else:
                            data_str = str(data)
                        notification: Dict[str, Any] = json.loads(data_str)
                        if notification.get("id") == notification_id:
                            notification["read"] = True
                            lset_result = self.redis_client.lset(
                                "in_app_notifications",
                                i,
                                json.dumps(notification),
                            )
                            if hasattr(lset_result, "__await__"):
                                await lset_result
                            return {
                                "status": "success",
                                "message": (f"Notification {notification_id} marked as read"),
                                "notification_id": notification_id,
                                "timestamp": notification["timestamp"],
                            }
                    except (json.JSONDecodeError, ValueError, KeyError) as e:
                        logger.warning(
                            f"Could not parse notification data for marking as read: {e}"
                        )
                        continue
            except TypeError:
                # Handle async response - skip iteration for now
                pass

            return {
                "status": "error",
                "message": f"Notification {notification_id} not found",
            }

        except Exception as e:
            logger.error(f"Error marking notification read: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def clear_all(self) -> Dict[str, Any]:
        """Clear all notifications."""
        try:
            # Redis lrange returns a list of bytes or strings
            lrange_result = self.redis_client.lrange("in_app_notifications", 0, -1)
            # Handle both sync and async Redis clients
            if hasattr(lrange_result, "__await__"):
                notifications_data = await lrange_result
            else:
                notifications_data = cast(List[Any], lrange_result)

            cleared_count = 0
            # Process each notification item
            for data in notifications_data:
                try:
                    if isinstance(data, bytes):
                        data_str = data.decode("utf-8")
                    elif isinstance(data, str):
                        data_str = data
                    else:
                        data_str = str(data)
                    notification: Dict[str, Any] = json.loads(data_str)
                    notification_time = datetime.fromisoformat(
                        notification["timestamp"].replace("Z", "+00:00")
                    ).timestamp()

                    if notification_time < datetime.now(timezone.timezone.utc).timestamp():
                        # Convert data to string for lrem
                        if isinstance(data, bytes):
                            data_str_for_rem = data.decode("utf-8")
                        elif isinstance(data, str):
                            data_str_for_rem = data
                        else:
                            data_str_for_rem = str(data)
                        lrem_result = self.redis_client.lrem(
                            "in_app_notifications", 1, data_str_for_rem
                        )
                        if hasattr(lrem_result, "__await__"):
                            await lrem_result
                        cleared_count += 1
                except (json.JSONDecodeError, ValueError, KeyError) as e:
                    logger.warning(f"Could not parse notification data for clearing: {e}")
                    continue

            logger.info(f"Cleared {cleared_count} old notifications")
            return {
                "status": "success",
                "message": "All notifications cleared",
                "cleared_count": cleared_count,
                "timestamp": datetime.now(timezone.timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error(f"Error clearing notifications: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def create_notification(self, notification_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new notification."""
        try:
            notification = {
                "id": f"notif_{self.notification_id_counter}",
                "type": notification_data.get("type", "info"),
                "title": notification_data.get("title", "Notification"),
                "message": notification_data.get("message", ""),
                "read": False,
                "timestamp": datetime.now(timezone.timezone.utc).isoformat(),
                "priority": notification_data.get("priority", "medium"),
            }

            self.notifications.append(notification)
            self.notification_id_counter += 1

            return notification
        except Exception as e:
            logger.error(f"Error creating notification: {str(e)}")
            return {}

    async def update_config(self, channel: str, config: Dict[str, Any]) -> bool:
        """Update notification configuration"""
        try:
            if channel in self.config:
                self.config[channel].update(config)
                self._save_config()
                logger.info(f"Updated {channel} notification configuration")
                return True
            else:
                logger.error(f"Unknown notification channel: {channel}")
                return False
        except Exception as e:
            logger.error(f"Error updating notification config: {str(e)}")
            return False


# Global notification service instance
notification_service = None


def get_notification_service(
    redis_client: Union[redis.Redis, Any],
) -> NotificationService:
    """Get or create notification service instance"""
    global notification_service
    if notification_service is None:
        notification_service = NotificationService(redis_client)
    return notification_service


