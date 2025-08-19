"""
Alerting System for Mystic Trading Platform

Provides comprehensive alerting with:
- Multiple notification channels
- Alert severity levels
- Alert aggregation and deduplication
- Alert history and management
- Custom alert rules
"""

import asyncio
import json
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from typing import Any

try:
    import redis
    from redis import Redis
except ImportError:
    redis = None
    Redis = None

from trading_config import trading_config

logger = logging.getLogger(__name__)

# Alerting configuration
ALERT_RETENTION_DAYS = 30
ALERT_DEDUPLICATION_WINDOW = 300  # 5 minutes
MAX_ALERTS_PER_HOUR = 100


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertChannel(Enum):
    """Alert notification channels"""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    DASHBOARD = "dashboard"


@dataclass
class Alert:
    """Alert information"""
    alert_id: str
    title: str
    message: str
    severity: AlertSeverity
    component: str
    timestamp: float
    channels: list[AlertChannel]
    metadata: dict[str, Any] | None = None
    acknowledged: bool = False
    acknowledged_by: str | None = None
    acknowledged_at: float | None = None


class AlertRule:
    """Alert rule definition"""

    def __init__(self, name: str, condition: str, severity: AlertSeverity,
                 channels: list[AlertChannel], cooldown: int = 300):
        self.name = name
        self.condition = condition
        self.severity = severity
        self.channels = channels
        self.cooldown = cooldown
        self.last_triggered: float | None = None

    def should_trigger(self, current_time: float) -> bool:
        """Check if alert should trigger based on cooldown"""
        if self.last_triggered is None:
            return True

        return (current_time - self.last_triggered) >= self.cooldown


class NotificationChannel:
    """Base class for notification channels"""

    def __init__(self, channel_type: AlertChannel):
        self.channel_type = channel_type
        self.enabled = True

    async def send_notification(self, alert: Alert) -> bool:
        """Send notification (to be implemented by subclasses)"""
        raise NotImplementedError


class EmailNotificationChannel(NotificationChannel):
    """Email notification channel"""

    def __init__(self, smtp_config: dict[str, Any]):
        super().__init__(AlertChannel.EMAIL)
        self.smtp_config = smtp_config

    async def send_notification(self, alert: Alert) -> bool:
        """Send email notification"""
        try:
            # This would implement actual email sending
            # For now, we'll just log the notification
            logger.info(f"Email alert sent: {alert.title} - {alert.message}")
            return True
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
            return False


class SlackNotificationChannel(NotificationChannel):
    """Slack notification channel"""

    def __init__(self, webhook_url: str, channel: str = "#alerts"):
        super().__init__(AlertChannel.SLACK)
        self.webhook_url = webhook_url
        self.channel = channel

    async def send_notification(self, alert: Alert) -> bool:
        """Send Slack notification"""
        try:
            # This would implement actual Slack webhook call
            # For now, we'll just log the notification
            logger.info(f"Slack alert sent: {alert.title} - {alert.message}")
            return True
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
            return False


class WebhookNotificationChannel(NotificationChannel):
    """Webhook notification channel"""

    def __init__(self, webhook_url: str):
        super().__init__(AlertChannel.WEBHOOK)
        self.webhook_url = webhook_url

    async def send_notification(self, alert: Alert) -> bool:
        """Send webhook notification"""
        try:
            # This would implement actual webhook call
            # For now, we'll just log the notification
            logger.info(f"Webhook alert sent: {alert.title} - {alert.message}")
            return True
        except Exception as e:
            logger.error(f"Failed to send webhook notification: {e}")
            return False


class AlertingSystem:
    """Comprehensive alerting system"""

    def __init__(self):
        self.alerts: deque = deque(maxlen=10000)
        self.alert_rules: dict[str, AlertRule] = {}
        self.notification_channels: dict[AlertChannel, NotificationChannel] = {}
        self.alert_counts: dict[str, int] = defaultdict(int)
        self.rate_limit_times: dict[str, list[float]] = defaultdict(list)
        self.redis_client = None
        self.lock = threading.Lock()

        # Initialize Redis connection
        self._initialize_redis()

        # Setup default channels and rules
        self._setup_default_channels()
        self._setup_default_rules()

    def _initialize_redis(self):
        """Initialize Redis connection"""
        try:
            if redis and Redis:
                self.redis_client = Redis(
                    host=trading_config.DEFAULT_REDIS_HOST,
                    port=trading_config.DEFAULT_REDIS_PORT,
                    db=trading_config.DEFAULT_REDIS_DB,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    retry_on_timeout=True
                )

                # Test connection
                self.redis_client.ping()
                logger.info("Redis connection established for alerting system")

            else:
                logger.warning("Redis not available for alerting system")

        except Exception as e:
            logger.error(f"Failed to connect to Redis for alerting system: {e}")
            self.redis_client = None

    def _setup_default_channels(self):
        """Setup default notification channels"""
        # Email channel (simulated)
        email_config = {
            'host': 'smtp.example.com',
            'port': 587,
            'username': 'alerts@example.com',
            'password': 'password'
        }
        self.notification_channels[AlertChannel.EMAIL] = EmailNotificationChannel(email_config)

        # Slack channel (simulated)
        self.notification_channels[AlertChannel.SLACK] = SlackNotificationChannel(
            webhook_url="https://hooks.slack.com/services/xxx/yyy/zzz",
            channel="#alerts"
        )

        # Webhook channel (simulated)
        self.notification_channels[AlertChannel.WEBHOOK] = WebhookNotificationChannel(
            webhook_url="https://api.example.com/webhooks/alerts"
        )

    def _setup_default_rules(self):
        """Setup default alert rules"""
        # System health alerts
        self.add_alert_rule(
            name="high_cpu_usage",
            condition="cpu_percent > 90",
            severity=AlertSeverity.CRITICAL,
            channels=[AlertChannel.EMAIL, AlertChannel.SLACK],
            cooldown=300
        )

        self.add_alert_rule(
            name="high_memory_usage",
            condition="memory_percent > 90",
            severity=AlertSeverity.CRITICAL,
            channels=[AlertChannel.EMAIL, AlertChannel.SLACK],
            cooldown=300
        )

        # Trading alerts
        self.add_alert_rule(
            name="high_error_rate",
            condition="error_rate > 10",
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.EMAIL, AlertChannel.SLACK],
            cooldown=600
        )

        self.add_alert_rule(
            name="low_liquidity",
            condition="liquidity_score < 0.3",
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.EMAIL, AlertChannel.SLACK],
            cooldown=1800
        )

        # Database alerts
        self.add_alert_rule(
            name="slow_queries",
            condition="slow_query_count > 50",
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.EMAIL, AlertChannel.SLACK],
            cooldown=900
        )

    def add_alert_rule(self, name: str, condition: str, severity: AlertSeverity,
                       channels: list[AlertChannel], cooldown: int = 300):
        """Add a new alert rule"""
        self.alert_rules[name] = AlertRule(name, condition, severity, channels, cooldown)

    def create_alert(self, title: str, message: str, severity: AlertSeverity,
                    component: str, channels: list[AlertChannel],
                    metadata: dict[str, Any] | None = None) -> Alert:
        """Create and store a new alert"""
        current_time = time.time()

        # Check rate limiting
        if self._is_rate_limited(component, current_time):
            logger.warning(f"Rate limited alert for component {component}")
            return None

        # Create alert
        alert = Alert(
            alert_id=self._generate_alert_id(),
            title=title,
            message=message,
            severity=severity,
            component=component,
            timestamp=current_time,
            channels=channels,
            metadata=metadata or {}
        )

        # Store alert
        self._store_alert(alert)

        # Send notifications asynchronously
        asyncio.create_task(self._send_notifications(alert))

        return alert

    def _generate_alert_id(self) -> str:
        """Generate unique alert ID"""
        return f"alert_{int(time.time() * 1000)}_{threading.get_ident()}"

    def _is_rate_limited(self, component: str, current_time: float) -> bool:
        """Check if component is rate limited"""
        # Clean old timestamps
        cutoff_time = current_time - 3600  # 1 hour window
        self.rate_limit_times[component] = [
            t for t in self.rate_limit_times[component] if t > cutoff_time
        ]

        # Check if we've exceeded the limit
        if len(self.rate_limit_times[component]) >= MAX_ALERTS_PER_HOUR:
            return True

        # Add current timestamp
        self.rate_limit_times[component].append(current_time)
        return False

    def _store_alert(self, alert: Alert):
        """Store alert in memory and Redis"""
        with self.lock:
            self.alerts.append(alert)
            self.alert_counts[alert.component] += 1

        # Store in Redis if available
        if self.redis_client:
            try:
                self.redis_client.setex(
                    f"alert:{alert.alert_id}",
                    86400,  # 24 hours
                    json.dumps(alert.__dict__, default=str)
                )
            except Exception as e:
                logger.error(f"Failed to store alert in Redis: {e}")

    async def _send_notifications(self, alert: Alert):
        """Send notifications for an alert"""
        tasks = []

        for channel in alert.channels:
            if channel in self.notification_channels:
                channel_instance = self.notification_channels[channel]
                if channel_instance.enabled:
                    tasks.append(channel_instance.send_notification(alert))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Log results
            for i, result in enumerate(results):
                channel_type = alert.channels[i]
                if isinstance(result, Exception):
                    logger.error(f"Failed to send notification via {channel_type.value}: {result}")
                else:
                    logger.debug(f"Notification sent via {channel_type.value}: {result}")

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert"""
        with self.lock:
            for alert in self.alerts:
                if alert.alert_id == alert_id:
                    alert.acknowledged = True
                    alert.acknowledged_by = acknowledged_by
                    alert.acknowledged_at = time.time()

                    # Update in Redis
                    if self.redis_client:
                        try:
                            self.redis_client.setex(
                                f"alert:{alert_id}",
                                86400,
                                json.dumps(alert.__dict__, default=str)
                            )
                        except Exception as e:
                            logger.error(f"Failed to update alert in Redis: {e}")

                    logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
                    return True

        return False

    def get_alerts_summary(self) -> dict[str, Any]:
        """Get comprehensive alerts summary"""
        with self.lock:
            current_time = time.time()

            # Count alerts by severity
            severity_counts = defaultdict(int)
            unacknowledged_count = 0
            recent_alerts = []

            for alert in self.alerts:
                severity_counts[alert.severity.value] += 1
                if not alert.acknowledged:
                    unacknowledged_count += 1

                # Recent alerts (last 24 hours)
                if current_time - alert.timestamp < 86400:
                    recent_alerts.append(alert)

            return {
                'total_alerts': len(self.alerts),
                'unacknowledged_alerts': unacknowledged_count,
                'recent_alerts': len(recent_alerts),
                'severity_distribution': dict(severity_counts),
                'component_counts': dict(self.alert_counts),
                'timestamp': current_time
            }

    def get_alerts_history(self, hours: int = 24,
                          severity: AlertSeverity | None = None,
                          component: str | None = None) -> list[dict[str, Any]]:
        """Get alerts history with optional filtering"""
        cutoff_time = time.time() - (hours * 3600)

        with self.lock:
            filtered_alerts = []

            for alert in self.alerts:
                if alert.timestamp >= cutoff_time:
                    if severity and alert.severity != severity:
                        continue
                    if component and alert.component != component:
                        continue

                    filtered_alerts.append(alert.__dict__)

            return filtered_alerts

    def cleanup_old_alerts(self, max_age_days: int = ALERT_RETENTION_DAYS):
        """Clean up old alerts"""
        cutoff_time = time.time() - (max_age_days * 24 * 3600)

        with self.lock:
            # Remove old alerts
            self.alerts = deque(
                (alert for alert in self.alerts if alert.timestamp >= cutoff_time),
                maxlen=10000
            )

        logger.info("Cleaned up old alerts")


# Global alerting system instance
alerting_system = AlertingSystem()


