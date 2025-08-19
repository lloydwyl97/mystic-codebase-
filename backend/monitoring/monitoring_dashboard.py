"""
Monitoring Dashboard for Mystic Trading Platform

Integrates all monitoring components:
- Enhanced logging
- Performance metrics
- Health checks
- Alerting system
- Real-time monitoring
"""

import asyncio
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from backend.monitoring.alerting_system import AlertChannel, AlertSeverity, alerting_system
from backend.monitoring.enhanced_logger import enhanced_logger
from backend.monitoring.health_checks import health_checker
from backend.monitoring.performance_metrics import metrics_collector

logger = logging.getLogger(__name__)

# Dashboard configuration
DASHBOARD_UPDATE_INTERVAL = 30  # seconds
METRICS_COLLECTION_INTERVAL = 60  # seconds
HEALTH_CHECK_INTERVAL = 60  # seconds


@dataclass
class DashboardMetrics:
    """Comprehensive dashboard metrics"""
    timestamp: float
    system_metrics: dict[str, Any]
    application_metrics: dict[str, Any]
    database_metrics: dict[str, Any]
    health_status: dict[str, Any]
    alert_summary: dict[str, Any]
    logging_stats: dict[str, Any]
    custom_metrics: dict[str, Any]


class MonitoringDashboard:
    """Comprehensive monitoring dashboard"""

    def __init__(self):
        self.enhanced_logger = enhanced_logger
        self.metrics_collector = metrics_collector
        self.health_checker = health_checker
        self.alerting_system = alerting_system

        self.dashboard_metrics: deque = deque(maxlen=1000)
        self.monitoring_active = False
        self.monitoring_task = None
        self.lock = threading.Lock()

        # Start monitoring
        self.start_monitoring()

    def start_monitoring(self):
        """Start the monitoring system"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            logger.info("Monitoring dashboard started")

    def stop_monitoring(self):
        """Stop the monitoring system"""
        if self.monitoring_active:
            self.monitoring_active = False
            if self.monitoring_task:
                self.monitoring_task.cancel()
            logger.info("Monitoring dashboard stopped")

    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect comprehensive metrics
                dashboard_metrics = await self._collect_dashboard_metrics()

                # Store metrics
                with self.lock:
                    self.dashboard_metrics.append(dashboard_metrics)

                # Check for alerts
                await self._check_alerts(dashboard_metrics)

                # Log monitoring status
                self.enhanced_logger.log_info(
                    "Dashboard metrics collected",
                    metrics_count=len(self.dashboard_metrics),
                    timestamp=dashboard_metrics.timestamp
                )

                # Wait for next update
                await asyncio.sleep(DASHBOARD_UPDATE_INTERVAL)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(10)  # Wait before retrying

    async def _collect_dashboard_metrics(self) -> DashboardMetrics:
        """Collect comprehensive dashboard metrics"""
        current_time = time.time()

        # Collect system metrics
        system_metrics = self.metrics_collector.collect_system_metrics()

        # Collect application metrics (simulated)
        application_metrics = self.metrics_collector.collect_application_metrics(
            request_count=100,
            response_times=[0.1, 0.2, 0.15, 0.3, 0.25],
            error_count=2,
            active_connections=50
        )

        # Collect database metrics (simulated)
        database_metrics = self.metrics_collector.collect_database_metrics(
            query_count=500,
            query_times=[0.05, 0.1, 0.08, 0.12, 0.15],
            slow_query_count=5,
            connection_count=20,
            cache_hit_rate=0.85
        )

        # Run health checks
        await self.health_checker.run_all_health_checks()
        health_summary = self.health_checker.get_health_summary()

        # Get alert summary
        alert_summary = self.alerting_system.get_alerts_summary()

        # Get logging stats
        logging_stats = self.enhanced_logger.get_logging_stats()

        # Collect custom metrics
        self.metrics_collector.add_custom_metric("trading_volume", 1000000.0)
        self.metrics_collector.add_custom_metric("active_users", 150)
        self.metrics_collector.add_custom_metric("portfolio_value", 500000.0)

        custom_metrics = self.metrics_collector.get_custom_metrics()

        return DashboardMetrics(
            timestamp=current_time,
            system_metrics=system_metrics,
            application_metrics=application_metrics,
            database_metrics=database_metrics,
            health_status=health_summary,
            alert_summary=alert_summary,
            logging_stats=logging_stats,
            custom_metrics=custom_metrics
        )

    async def _check_alerts(self, dashboard_metrics: DashboardMetrics):
        """Check for alert conditions"""
        # Check system metrics
        cpu_percent = dashboard_metrics.system_metrics.get('cpu_percent', 0)
        memory_percent = dashboard_metrics.system_metrics.get('memory_percent', 0)

        if cpu_percent > 90:
            await self._create_system_alert(
                "High CPU Usage",
                f"CPU usage is {cpu_percent}%",
                AlertSeverity.CRITICAL
            )

        if memory_percent > 90:
            await self._create_system_alert(
                "High Memory Usage",
                f"Memory usage is {memory_percent}%",
                AlertSeverity.CRITICAL
            )

        # Check application metrics
        error_rate = dashboard_metrics.application_metrics.get('error_rate', 0)
        response_time_avg = dashboard_metrics.application_metrics.get('response_time_avg', 0)

        if error_rate > 10:
            await self._create_application_alert(
                "High Error Rate",
                f"Error rate is {error_rate}%",
                AlertSeverity.WARNING
            )

        if response_time_avg > 1.0:
            await self._create_application_alert(
                "Slow Response Time",
                f"Average response time is {response_time_avg}s",
                AlertSeverity.WARNING
            )

        # Check health status
        overall_status = dashboard_metrics.health_status.get('overall_status', 'unknown')
        if overall_status == 'critical':
            await self._create_health_alert(
                "System Health Critical",
                "System health check failed",
                AlertSeverity.CRITICAL
            )

        # Check database metrics
        cache_hit_rate = dashboard_metrics.database_metrics.get('cache_hit_rate', 0)
        if cache_hit_rate < 0.5:
            await self._create_database_alert(
                "Low Cache Hit Rate",
                f"Cache hit rate is {cache_hit_rate}",
                AlertSeverity.WARNING
            )

    async def _create_system_alert(self, title: str, message: str, severity: AlertSeverity):
        """Create system alert"""
        await self.alerting_system.create_alert(
            title=title,
            message=message,
            severity=severity,
            channel=AlertChannel.SYSTEM
        )

    async def _create_application_alert(self, title: str, message: str, severity: AlertSeverity):
        """Create application alert"""
        await self.alerting_system.create_alert(
            title=title,
            message=message,
            severity=severity,
            channel=AlertChannel.APPLICATION
        )

    async def _create_health_alert(self, title: str, message: str, severity: AlertSeverity):
        """Create health alert"""
        await self.alerting_system.create_alert(
            title=title,
            message=message,
            severity=severity,
            channel=AlertChannel.HEALTH
        )

    async def _create_database_alert(self, title: str, message: str, severity: AlertSeverity):
        """Create database alert"""
        await self.alerting_system.create_alert(
            title=title,
            message=message,
            severity=severity,
            channel=AlertChannel.DATABASE
        )

    def get_dashboard_summary(self) -> dict[str, Any]:
        """Get comprehensive dashboard summary"""
        with self.lock:
            if not self.dashboard_metrics:
                return {}

            latest_metrics = self.dashboard_metrics[-1]

            return {
                'timestamp': latest_metrics.timestamp,
                'system': {
                    'cpu_percent': latest_metrics.system_metrics.get('cpu_percent', 0),
                    'memory_percent': latest_metrics.system_metrics.get('memory_percent', 0),
                    'disk_usage_percent': latest_metrics.system_metrics.get('disk_usage_percent', 0)
                },
                'application': {
                    'request_count': latest_metrics.application_metrics.get('request_count', 0),
                    'response_time_avg': latest_metrics.application_metrics.get('response_time_avg', 0),
                    'error_rate': latest_metrics.application_metrics.get('error_rate', 0)
                },
                'database': {
                    'query_count': latest_metrics.database_metrics.get('query_count', 0),
                    'query_time_avg': latest_metrics.database_metrics.get('query_time_avg', 0),
                    'cache_hit_rate': latest_metrics.database_metrics.get('cache_hit_rate', 0)
                },
                'health': {
                    'overall_status': latest_metrics.health_status.get('overall_status', 'unknown'),
                    'critical_components': latest_metrics.health_status.get('critical_components', []),
                    'warning_components': latest_metrics.health_status.get('warning_components', [])
                },
                'alerts': {
                    'total_alerts': latest_metrics.alert_summary.get('total_alerts', 0),
                    'unacknowledged_alerts': latest_metrics.alert_summary.get('unacknowledged_alerts', 0),
                    'recent_alerts': latest_metrics.alert_summary.get('recent_alerts', 0)
                },
                'custom_metrics': latest_metrics.custom_metrics,
                'monitoring_active': self.monitoring_active
            }

    def get_metrics_history(self, hours: int = 24) -> list[dict[str, Any]]:
        """Get metrics history"""
        cutoff_time = time.time() - (hours * 3600)

        with self.lock:
            recent_metrics = [m for m in self.dashboard_metrics if m.timestamp >= cutoff_time]
            return [m.__dict__ for m in recent_metrics]

    def get_component_status(self, component: str) -> dict[str, Any]:
        """Get status of specific component"""
        with self.lock:
            if not self.dashboard_metrics:
                return {}

            latest_metrics = self.dashboard_metrics[-1]

            if component == "system":
                return {
                    'status': 'healthy' if latest_metrics.system_metrics.get('cpu_percent', 0) < 80 else 'warning',
                    'metrics': latest_metrics.system_metrics,
                    'timestamp': latest_metrics.timestamp
                }
            elif component == "application":
                return {
                    'status': 'healthy' if latest_metrics.application_metrics.get('error_rate', 0) < 5 else 'warning',
                    'metrics': latest_metrics.application_metrics,
                    'timestamp': latest_metrics.timestamp
                }
            elif component == "database":
                return {
                    'status': 'healthy' if latest_metrics.database_metrics.get('cache_hit_rate', 0) > 0.8 else 'warning',
                    'metrics': latest_metrics.database_metrics,
                    'timestamp': latest_metrics.timestamp
                }
            elif component == "health":
                return {
                    'status': latest_metrics.health_status.get('overall_status', 'unknown'),
                    'metrics': latest_metrics.health_status,
                    'timestamp': latest_metrics.timestamp
                }
            else:
                return {}

    def get_performance_trends(self, hours: int = 24) -> dict[str, list[float]]:
        """Get performance trends over time"""
        cutoff_time = time.time() - (hours * 3600)

        with self.lock:
            recent_metrics = [m for m in self.dashboard_metrics if m.timestamp >= cutoff_time]

            trends = {
                'cpu_percent': [m.system_metrics.get('cpu_percent', 0) for m in recent_metrics],
                'memory_percent': [m.system_metrics.get('memory_percent', 0) for m in recent_metrics],
                'response_time_avg': [m.application_metrics.get('response_time_avg', 0) for m in recent_metrics],
                'error_rate': [m.application_metrics.get('error_rate', 0) for m in recent_metrics],
                'query_time_avg': [m.database_metrics.get('query_time_avg', 0) for m in recent_metrics]
            }

            return trends


# Global monitoring dashboard instance
monitoring_dashboard = MonitoringDashboard()


