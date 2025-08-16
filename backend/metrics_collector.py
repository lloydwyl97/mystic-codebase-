"""
Metrics Collector for Mystic Trading

Collects and exposes comprehensive metrics for Prometheus monitoring.
Includes signal health, auto-trading status, test results, and system performance.
"""

import json
import logging
import time
from datetime import datetime
from typing import Any, Dict

import redis
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
)

from .services.websocket_manager import websocket_manager

logger = logging.getLogger(__name__)


class MetricsCollector:
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client

        # Signal and Auto-trading Metrics
        self.signal_health_gauge = Gauge(
            "mystic_signal_health_status",
            "Overall signal health status (0=critical, 1=degraded, 2=healthy)",
            ["component"],
        )

        self.active_signals_gauge = Gauge(
            "mystic_active_signals_total",
            "Number of active signals",
            ["signal_type"],
        )

        self.active_strategies_gauge = Gauge(
            "mystic_active_strategies_total",
            "Number of active trading strategies",
            ["strategy_type"],
        )

        self.auto_trading_status_gauge = Gauge(
            "mystic_auto_trading_status",
            "Auto-trading status (0=disabled, 1=enabled)",
        )

        # Test and CI Metrics
        self.test_runs_total = Counter(
            "mystic_test_runs_total", "Total number of test runs", ["status"]
        )

        self.test_success_rate_gauge = Gauge(
            "mystic_test_success_rate", "Test success rate percentage"
        )

        self.test_duration_histogram = Histogram(
            "mystic_test_duration_seconds",
            "Test run duration in seconds",
            buckets=[10, 30, 60, 120, 300, 600],
        )

        self.self_healing_triggered_total = Counter(
            "mystic_self_healing_triggered_total",
            "Number of times self-healing was triggered",
            ["reason"],
        )

        # Notification Metrics
        self.notifications_sent_total = Counter(
            "mystic_notifications_sent_total",
            "Number of notifications sent",
            ["channel", "level"],
        )

        self.notification_delivery_duration = Histogram(
            "mystic_notification_delivery_seconds",
            "Notification delivery duration in seconds",
            ["channel"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        )

        # API Performance Metrics
        self.api_requests_total = Counter(
            "mystic_api_requests_total",
            "Total number of API requests",
            ["method", "endpoint", "status"],
        )

        self.api_request_duration = Histogram(
            "mystic_api_request_duration_seconds",
            "API request duration in seconds",
            ["method", "endpoint"],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
        )

        # WebSocket Metrics
        self.websocket_connections_gauge = Gauge(
            "mystic_websocket_connections_active",
            "Number of active WebSocket connections",
        )

        self.websocket_messages_sent_total = Counter(
            "mystic_websocket_messages_sent_total",
            "Total number of WebSocket messages sent",
        )

        # System Health Metrics
        self.system_uptime_gauge = Gauge("mystic_system_uptime_seconds", "System uptime in seconds")

        self.service_health_gauge = Gauge(
            "mystic_service_health_status",
            "Service health status (0=unhealthy, 1=healthy)",
            ["service"],
        )

        # Trading Performance Metrics
        self.live_signals_generated_total = Counter(
            "mystic_live_signals_generated_total",
            "Total number of live signals generated",
            ["symbol", "signal_type"],
        )

        self.signal_confidence_histogram = Histogram(
            "mystic_signal_confidence",
            "Signal confidence levels",
            ["signal_type"],
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        )

        # Memory and Performance Metrics
        self.redis_operations_total = Counter(
            "mystic_redis_operations_total",
            "Total number of Redis operations",
            ["operation", "status"],
        )

        self.redis_operation_duration = Histogram(
            "mystic_redis_operation_duration_seconds",
            "Redis operation duration in seconds",
            ["operation"],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
        )

        # Info metrics
        self.app_info = Info("mystic_app", "Application information")
        self.app_info.info(
            {
                "version": "1.0.0",
                "name": "Mystic Trading Bot",
                "description": ("Advanced trading bot with automated signals and self-healing"),
            }
        )

        # Start time for uptime calculation
        self.start_time = time.time()

    def update_signal_health_metrics(self, health_data: Dict[str, Any]):
        """Update signal health metrics"""
        try:
            overall_health = health_data.get("overall_health", "unknown")
            health_value = {"healthy": 2, "degraded": 1, "critical": 0}.get(overall_health, 0)

            self.signal_health_gauge.labels(component="overall").set(health_value)

            # Update signal counts
            signals = health_data.get("signals", {})
            self.active_signals_gauge.labels(signal_type="total").set(signals.get("total", 0))
            self.active_signals_gauge.labels(signal_type="healthy").set(signals.get("healthy", 0))
            self.active_signals_gauge.labels(signal_type="unhealthy").set(
                signals.get("unhealthy", 0)
            )

            # Update strategy counts
            strategies = health_data.get("strategies", {})
            self.active_strategies_gauge.labels(strategy_type="total").set(
                strategies.get("total", 0)
            )
            self.active_strategies_gauge.labels(strategy_type="healthy").set(
                strategies.get("healthy", 0)
            )
            self.active_strategies_gauge.labels(strategy_type="unhealthy").set(
                strategies.get("unhealthy", 0)
            )

            # Update auto-trading status
            auto_trading = health_data.get("auto_trading", {})
            auto_trading_status = 1 if auto_trading.get("healthy", False) else 0
            self.auto_trading_status_gauge.set(auto_trading_status)

        except Exception as e:
            logger.error(f"Error updating signal health metrics: {str(e)}")

    def update_test_metrics(self, test_run: Dict[str, Any]):
        """Update test metrics"""
        try:
            # Update test run counter
            status = "success" if test_run.get("failed_tests", 0) == 0 else "failure"
            self.test_runs_total.labels(status=status).inc()

            # Update success rate
            success_rate = test_run.get("success_rate", 0.0)
            self.test_success_rate_gauge.set(success_rate)

            # Update test duration
            start_time = datetime.fromisoformat(
                test_run.get("start_time", "").replace("Z", "+00:00")
            )
            end_time = datetime.fromisoformat(test_run.get("end_time", "").replace("Z", "+00:00"))
            duration = (end_time - start_time).total_seconds()
            self.test_duration_histogram.observe(duration)

            # Update self-healing metrics if triggered
            if test_run.get("triggered_healing", False):
                self.self_healing_triggered_total.labels(reason="test_failure").inc()

        except Exception as e:
            logger.error(f"Error updating test metrics: {str(e)}")

    def update_notification_metrics(self, notification_result: Dict[str, Any]):
        """Update notification metrics"""
        try:
            channels = notification_result.get("channels", {})

            for channel, result in channels.items():
                if result.get("success", False):
                    level = notification_result.get("level", "unknown")
                    self.notifications_sent_total.labels(channel=channel, level=level).inc()

                    # Record delivery duration if available
                    if "duration" in result:
                        self.notification_delivery_duration.labels(channel=channel).observe(
                            result["duration"]
                        )

        except Exception as e:
            logger.error(f"Error updating notification metrics: {str(e)}")

    def update_api_metrics(self, method: str, endpoint: str, status: int, duration: float):
        """Update API metrics"""
        try:
            status_category = f"{status // 100}xx"
            self.api_requests_total.labels(
                method=method, endpoint=endpoint, status=status_category
            ).inc()
            self.api_request_duration.labels(method=method, endpoint=endpoint).observe(duration)

        except Exception as e:
            logger.error(f"Error updating API metrics: {str(e)}")

    def update_websocket_metrics(self, connections: int, messages_sent: int = 0):
        """Update WebSocket metrics"""
        try:
            self.websocket_connections_gauge.set(connections)
            if messages_sent > 0:
                self.websocket_messages_sent_total.inc(messages_sent)

        except Exception as e:
            logger.error(f"Error updating WebSocket metrics: {str(e)}")

    def update_system_metrics(self):
        """Update system metrics"""
        try:
            # Update uptime
            uptime = time.time() - self.start_time
            self.system_uptime_gauge.set(uptime)

            # Update service health from Redis
            health_data = self.redis_client.get("service_health")
            if health_data:
                try:
                    # Handle Redis response which could be bytes or string
                    if isinstance(health_data, bytes):
                        health_str = health_data.decode("utf-8")
                    else:
                        health_str = str(health_data)

                    services = json.loads(health_str)
                    for service, status in services.items():
                        health_value = 1 if status.get("status") == "healthy" else 0
                        self.service_health_gauge.labels(service=service).set(health_value)
                except Exception:
                    pass

        except Exception as e:
            logger.error(f"Error updating system metrics: {str(e)}")

    def update_trading_metrics(self, signal_data: Dict[str, Any]):
        """Update trading performance metrics"""
        try:
            symbol = signal_data.get("symbol", "unknown")
            signal_type = signal_data.get("signal_type", "unknown")
            confidence = signal_data.get("confidence", 0.0)

            self.live_signals_generated_total.labels(symbol=symbol, signal_type=signal_type).inc()
            self.signal_confidence_histogram.labels(signal_type=signal_type).observe(confidence)

            # Broadcast metrics update
            import asyncio

            asyncio.create_task(
                websocket_manager.broadcast_json(
                    {
                        "type": "metrics_update",
                        "data": {
                            "symbol": symbol,
                            "signal_type": signal_type,
                            "confidence": confidence,
                            "timestamp": time.time(),
                        },
                    }
                )
            )

        except Exception as e:
            logger.error(f"Error updating trading metrics: {str(e)}")

    def update_redis_metrics(self, operation: str, status: str, duration: float):
        """Update Redis operation metrics"""
        try:
            self.redis_operations_total.labels(operation=operation, status=status).inc()
            self.redis_operation_duration.labels(operation=operation).observe(duration)

        except Exception as e:
            logger.error(f"Error updating Redis metrics: {str(e)}")

    def record_self_healing_event(self, reason: str, actions_taken: int):
        """Record self-healing event"""
        try:
            self.self_healing_triggered_total.labels(reason=reason).inc()

            # You could add more detailed metrics here if needed
            logger.info(f"Self-healing triggered: {reason}, {actions_taken} actions taken")

        except Exception as e:
            logger.error(f"Error recording self-healing event: {str(e)}")

    def get_metrics(self) -> str:
        """Get all metrics in Prometheus format"""
        try:
            return generate_latest().decode("utf-8")  # Decode bytes to string
        except Exception as e:
            logger.error(f"Error generating metrics: {str(e)}")
            return ""

    def get_metrics_content_type(self) -> str:
        """Get the content type for metrics"""
        return CONTENT_TYPE_LATEST


# Global metrics collector instance
metrics_collector = None


def get_metrics_collector(redis_client: redis.Redis) -> MetricsCollector:
    """Get or create metrics collector instance"""
    global metrics_collector
    if metrics_collector is None:
        metrics_collector = MetricsCollector(redis_client)
    return metrics_collector


