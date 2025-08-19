"""
Message Handler for Mystic Trading Platform

Contains message handling logic for notifications and alerts.
Handles message formatting, routing, and delivery.
"""

import json
import logging
import smtplib
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

import requests

from backend.utils.exceptions import NotificationException

logger = logging.getLogger(__name__)

# Simple usage of imports to avoid unused import errors
_ = json.dumps({"status": "loaded"})


class MessageHandler:
    """Message handler for notifications and alerts delivery"""

    def __init__(self):
        self.channels: dict[str, dict[str, Any]] = {}
        self.message_templates: dict[str, str] = {}
        self.delivery_history: list[dict[str, Any]] = []
        self.messages: dict[str, dict[str, Any]] = {}
        self.message_queue: list[str] = []
        self.message_types: dict[str, dict[str, Any]] = {
            "notification": {"fields": ["title", "content", "recipient"]},
            "alert": {"fields": ["title", "content", "recipient"]},
            "trade": {"fields": ["title", "content", "recipient"]},
            "system": {"fields": ["title", "content", "recipient"]},
            "trade_notification": {"fields": ["title", "content", "recipient"]},
            "system_notification": {"fields": ["title", "content", "recipient"]},
        }
        self.is_active: bool = True

    def add_channel(self, channel_name: str, config: dict[str, Any]) -> dict[str, Any]:
        """Add a new message delivery channel"""
        channel = {
            "name": channel_name,
            "config": config,
            "is_active": True,
            "created_at": datetime.now().isoformat(),
        }

        self.channels[channel_name] = channel
        logger.info(f"Message channel added: {channel_name}")

        return {"success": True, "channel": channel}

    def remove_channel(self, channel_name: str) -> dict[str, Any]:
        """Remove a message delivery channel"""
        if channel_name in self.channels:
            del self.channels[channel_name]
            logger.info(f"Message channel removed: {channel_name}")
            return {"success": True}

        return {"success": False, "error": "Channel not found"}

    def add_template(self, template_name: str, template_content: str) -> dict[str, Any]:
        """Add a message template"""
        self.message_templates[template_name] = template_content
        logger.info(f"Message template added: {template_name}")

        return {"success": True, "template": template_name}

    def remove_template(self, template_name: str) -> dict[str, Any]:
        """Remove a message template"""
        if template_name in self.message_templates:
            del self.message_templates[template_name]
            logger.info(f"Message template removed: {template_name}")
            return {"success": True}

        return {"success": False, "error": "Template not found"}

    def send_message(
        self,
        message_type: str,
        content: dict[str, Any],
        channels: list[str] | None = None,
    ) -> dict[str, Any]:
        """Send a message through specified channels"""
        if not self.is_active:
            return {"error": "Message handler is not active"}

        if not channels:
            channels = list(self.channels.keys())

        results = {}
        successful_deliveries = 0

        for channel_name in channels:
            if channel_name not in self.channels:
                results[channel_name] = {"error": "Channel not found"}
                continue

            channel = self.channels[channel_name]
            if not channel.get("is_active", True):
                results[channel_name] = {"error": "Channel is not active"}
                continue

            try:
                result = self._send_to_channel(channel_name, message_type, content)
                results[channel_name] = result

                if result.get("success"):
                    successful_deliveries += 1

            except Exception as e:
                logger.error(f"Failed to send message to {channel_name}: {e}")
                results[channel_name] = {"error": str(e)}

        # Record delivery history
        delivery_record = {
            "timestamp": datetime.now().isoformat(),
            "message_type": message_type,
            "content": content,
            "channels": channels,
            "results": results,
            "successful_deliveries": successful_deliveries,
            "total_channels": len(channels),
        }

        self.delivery_history.append(delivery_record)

        # Limit history size
        if len(self.delivery_history) > 1000:
            self.delivery_history = self.delivery_history[-500:]

        return {
            "success": successful_deliveries > 0,
            "successful_deliveries": successful_deliveries,
            "total_channels": len(channels),
            "results": results,
        }

    def send_notification(
        self,
        title: str,
        content: str,
        recipient: str,
        priority: str | None = None,
    ) -> dict[str, Any]:
        return self.create_message(
            message_type="notification",
            title=title,
            content=content,
            recipient=recipient,
            priority=priority,
        )

    def send_alert(
        self,
        title: str,
        content: str,
        recipient: str,
        priority: str | None = None,
    ) -> dict[str, Any]:
        return self.create_message(
            message_type="alert",
            title=title,
            content=content,
            recipient=recipient,
            priority=priority,
        )

    def send_trade_notification(
        self, recipient: str, trade_data: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        if trade_data:
            if "title" in trade_data:
                title = trade_data["title"]
            elif "symbol" in trade_data:
                title = f"Trade Alert: {trade_data['symbol']}"
            else:
                title = ""
            if "content" in trade_data:
                content = trade_data["content"]
            elif "side" in trade_data:
                content = f"Trade action: {trade_data['side']}"
            elif "action" in trade_data:
                content = f"Trade action: {trade_data['action']}"
            else:
                content = ""
        else:
            title = ""
            content = ""
        msg = self.create_message(
            message_type="trade_notification",
            title=title,
            content=content,
            recipient=recipient,
        )
        msg["type"] = "trade_notification"
        if trade_data:
            msg["trade_data"] = trade_data
        return msg

    def send_system_notification(
        self,
        title: str,
        content: str,
        recipient: str,
        priority: str | None = None,
    ) -> dict[str, Any]:
        msg = self.create_message(
            message_type="system_notification",
            title=title,
            content=content,
            recipient=recipient,
            priority=priority,
        )
        msg["type"] = "system_notification"
        return msg

    def get_delivery_status(self, message_id: str | None = None) -> dict[str, Any]:
        """Get delivery status for messages"""
        if message_id:
            # Find specific message in history
            for record in self.delivery_history:
                if record.get("message_id") == message_id:
                    return record
            return {"error": "Message not found"}

        # Return summary of recent deliveries
        recent_deliveries = self.delivery_history[-10:] if self.delivery_history else []

        total_messages = len(self.delivery_history)
        successful_messages = sum(
            1 for record in self.delivery_history if record.get("successful_deliveries", 0) > 0
        )

        return {
            "total_messages": total_messages,
            "successful_messages": successful_messages,
            "success_rate": (
                (successful_messages / total_messages * 100) if total_messages > 0 else 0
            ),
            "recent_deliveries": recent_deliveries,
        }

    def get_channel_status(self) -> dict[str, Any]:
        """Get status of all channels"""
        status = {}

        for channel_name, channel in self.channels.items():
            status[channel_name] = {
                "is_active": channel.get("is_active", True),
                "created_at": channel.get("created_at"),
                "config": channel.get("config", {}),
            }

        return status

    def test_channel(self, channel_name: str) -> dict[str, Any]:
        """Test a specific channel"""
        if channel_name not in self.channels:
            return {"error": "Channel not found"}

        self.channels[channel_name]

        try:
            # Send test message
            test_content = {
                "subject": "Test Message",
                "body": "This is a test message from Mystic Trading Platform",
                "timestamp": datetime.now().isoformat(),
            }

            result = self._send_to_channel(channel_name, "test", test_content)

            return {
                "success": result.get("success", False),
                "message": "Channel test completed",
                "result": result,
            }

        except Exception as e:
            logger.error(f"Channel test failed for {channel_name}: {e}")
            return {"error": str(e)}

    def _send_to_channel(
        self, channel_name: str, message_type: str, content: dict[str, Any]
    ) -> dict[str, Any]:
        """Send message to a specific channel"""
        channel = self.channels[channel_name]
        config = channel.get("config", {})

        if channel_name == "email":
            return self._send_email(config, content)
        elif channel_name == "webhook":
            return self._send_webhook(config, content)
        elif channel_name == "slack":
            return self._send_slack(config, content)
        elif channel_name == "telegram":
            return self._send_telegram(config, content)
        else:
            return {"error": f"Unsupported channel: {channel_name}"}

    def _send_email(self, config: dict[str, Any], content: dict[str, Any]) -> dict[str, Any]:
        """Send email message"""
        try:
            # Extract email configuration
            smtp_server = config.get("smtp_server", "localhost")
            smtp_port = config.get("smtp_port", 587)
            username = config.get("username", "")
            password = config.get("password", "")
            from_email = config.get("from_email", "noreply@mystic-trading.com")

            # Create message
            msg = MIMEMultipart()
            msg["From"] = from_email
            msg["To"] = content.get("recipients", [""])[0] if content.get("recipients") else ""
            msg["Subject"] = content.get("subject", "Mystic Trading Notification")

            body = content.get("body", "")
            msg.attach(MIMEText(body, "plain"))

            # Send email
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()

            if username and password:
                server.login(username, password)

            server.send_message(msg)
            server.quit()

            return {"success": True, "message": "Email sent successfully"}

        except Exception as e:
            logger.error(f"Email sending failed: {e}")
            return {"error": str(e)}

    def _send_webhook(self, config: dict[str, Any], content: dict[str, Any]) -> dict[str, Any]:
        """Send webhook message"""
        try:
            webhook_url = config.get("url", "")
            if not webhook_url:
                return {"error": "Webhook URL not configured"}

            # Prepare payload
            payload = {
                "timestamp": datetime.now().isoformat(),
                "content": content,
                "source": "mystic_trading_platform",
            }

            # Send webhook
            response = requests.post(webhook_url, json=payload, timeout=10)

            if response.status_code == 200:
                return {
                    "success": True,
                    "message": "Webhook sent successfully",
                }
            else:
                return {"error": (f"Webhook failed with status {response.status_code}")}

        except Exception as e:
            logger.error(f"Webhook sending failed: {e}")
            return {"error": str(e)}

    def _send_slack(self, config: dict[str, Any], content: dict[str, Any]) -> dict[str, Any]:
        """Send Slack message"""
        try:
            webhook_url = config.get("webhook_url", "")
            if not webhook_url:
                return {"error": "Slack webhook URL not configured"}

            # Format Slack message
            slack_message = {
                "text": content.get("body", ""),
                "attachments": [
                    {
                        "title": content.get("subject", "Mystic Trading Alert"),
                        "color": self._get_severity_color(content.get("severity", "info")),
                        "fields": [
                            {
                                "title": "Type",
                                "value": content.get("type", "notification"),
                                "short": True,
                            },
                            {
                                "title": "Timestamp",
                                "value": content.get("timestamp", ""),
                                "short": True,
                            },
                        ],
                    }
                ],
            }

            # Send to Slack
            response = requests.post(webhook_url, json=slack_message, timeout=10)

            if response.status_code == 200:
                return {
                    "success": True,
                    "message": "Slack message sent successfully",
                }
            else:
                return {"error": f"Slack failed with status {response.status_code}"}

        except Exception as e:
            logger.error(f"Slack sending failed: {e}")
            return {"error": str(e)}

    def _send_telegram(self, config: dict[str, Any], content: dict[str, Any]) -> dict[str, Any]:
        """Send Telegram message"""
        try:
            bot_token = config.get("bot_token", "")
            chat_id = config.get("chat_id", "")

            if not bot_token or not chat_id:
                return {"error": "Telegram bot token or chat ID not configured"}

            # Format message
            message = f"*{content.get('subject', 'Mystic Trading Alert')}*\n\n"
            message += content.get("body", "")
            message += f"\n\n_Time: {content.get('timestamp', '')}_"

            # Send to Telegram
            telegram_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            payload = {
                "chat_id": chat_id,
                "text": message,
                "parse_mode": "Markdown",
            }

            response = requests.post(telegram_url, json=payload, timeout=10)

            if response.status_code == 200:
                return {
                    "success": True,
                    "message": "Telegram message sent successfully",
                }
            else:
                return {"error": (f"Telegram failed with status {response.status_code}")}

        except Exception as e:
            logger.error(f"Telegram sending failed: {e}")
            return {"error": str(e)}

    def _format_notification(self, notification_type: str, data: dict[str, Any]) -> dict[str, Any]:
        """Format notification content"""
        template = self.message_templates.get(notification_type, "")

        if template:
            # Simple template substitution
            body = template.format(**data)
        else:
            # Default formatting
            body = f"Notification: {notification_type}\n"
            body += f"Data: {json.dumps(data, indent=2)}"

        return {
            "subject": f"Mystic Trading - {notification_type.title()}",
            "body": body,
            "type": "notification",
            "timestamp": datetime.now().isoformat(),
        }

    def _format_alert(
        self, alert_type: str, alert_data: dict[str, Any], severity: str
    ) -> dict[str, Any]:
        """Format alert content"""
        template = self.message_templates.get(f"alert_{alert_type}", "")

        if template:
            body = template.format(**alert_data)
        else:
            # Default alert formatting
            body = f"Alert: {alert_type}\n"
            body += f"Severity: {severity}\n"
            body += f"Data: {json.dumps(alert_data, indent=2)}"

        return {
            "subject": f"Mystic Trading Alert - {alert_type.title()}",
            "body": body,
            "type": "alert",
            "severity": severity,
            "timestamp": datetime.now().isoformat(),
        }

    def _get_severity_color(self, severity: str) -> str:
        """Get color for severity level"""
        colors = {
            "info": "#36a64f",  # Green
            "warning": "#ffa500",  # Orange
            "error": "#ff0000",  # Red
            "critical": "#8b0000",  # Dark Red
        }
        return colors.get(severity, "#36a64f")

    def create_message(
        self,
        message_type: str,
        title: str | None = None,
        content: str | None = None,
        recipient: str | None = None,
        priority: str | None = None,
    ) -> dict[str, Any]:
        if message_type not in self.message_types:
            raise NotificationException(f"Invalid message type: {message_type}")
        required = self.message_types[message_type]["fields"]
        for field in required:
            if field == "title" and title is None:
                raise NotificationException("Missing required parameters")
            if field == "content" and content is None:
                raise NotificationException("Missing required parameters")
            if field == "recipient" and recipient is None:
                raise NotificationException("Missing required parameters")
        message_id = f"msg_{len(self.messages) + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        message = {
            "id": message_id,
            "type": message_type,
            "title": title,
            "content": content,
            "recipient": recipient,
            "priority": priority or "medium",
            "created_at": datetime.now(),
            "read": False,
        }
        self.messages[message_id] = message
        self.message_queue.append(message_id)
        logger.info(f"Message created: {message_type} - {title}")
        return message

    def get_message(self, message_id: str) -> dict[str, Any] | None:
        return self.messages.get(message_id)

    def mark_as_read(self, message_id: str) -> bool:
        if message_id in self.messages:
            self.messages[message_id]["read"] = True
            return True
        return False

    def delete_message(self, message_id: str) -> bool:
        if message_id in self.messages:
            del self.messages[message_id]
            if message_id in self.message_queue:
                self.message_queue.remove(message_id)
            return True
        return False

    def list_messages(
        self,
        message_type: str | None = None,
        recipient: str | None = None,
        unread_only: bool = False,
    ) -> list[dict[str, Any]]:
        message_ids = list(self.messages.keys())
        if message_type:
            message_ids = [mid for mid in message_ids if self.messages[mid]["type"] == message_type]
        if recipient:
            message_ids = [
                mid for mid in message_ids if self.messages[mid]["recipient"] == recipient
            ]
        if unread_only:
            message_ids = [mid for mid in message_ids if not self.messages[mid]["read"]]
        return [self.messages[mid] for mid in message_ids]

    def list_unread_messages(self, recipient: str) -> list[dict[str, Any]]:
        return [m for m in self.messages.values() if m["recipient"] == recipient and not m["read"]]

    def get_message_statistics(self, recipient: str | None = None) -> dict[str, Any]:
        if recipient:
            filtered = [m for m in self.messages.values() if m["recipient"] == recipient]
        else:
            filtered = list(self.messages.values())
        total_messages = len(filtered)
        unread_messages = len([m for m in filtered if not m["read"]])
        read_messages = len([m for m in filtered if m["read"]])
        by_type = {}
        for m in filtered:
            t = m["type"]
            by_type[t] = by_type.get(t, 0) + 1
        return {
            "total_messages": total_messages,
            "unread_messages": unread_messages,
            "read_messages": read_messages,
            "messages_by_type": by_type,
        }

    def cleanup_old_messages(self, days: int = 30) -> int:
        now = datetime.now()
        cutoff = now - timedelta(days=days)
        to_delete = [mid for mid, m in self.messages.items() if m["created_at"] < cutoff]
        for mid in to_delete:
            self.delete_message(mid)
        return len(to_delete)

    def bulk_mark_as_read(self, message_ids: list[str]) -> int:
        count = 0
        for mid in message_ids:
            if self.mark_as_read(mid):
                count += 1
        return count

    def bulk_delete_messages(self, message_ids: list[str]) -> int:
        count = 0
        for mid in message_ids:
            if self.delete_message(mid):
                count += 1
        return count

    def search_messages(self, query: str, recipient: str | None = None) -> list[dict[str, Any]]:
        result = []
        for m in self.messages.values():
            if recipient and m.get("recipient") != recipient:
                continue
            if query.lower() in (
                str(m.get("title", "")).lower() + str(m.get("content", "")).lower()
            ):
                result.append(m)
        return result


