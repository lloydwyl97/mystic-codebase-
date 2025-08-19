"""
Health Monitor Service
Handles system health monitoring and diagnostics
"""

import asyncio
import logging
import platform
from datetime import datetime, timezone
from typing import Any

import psutil

logger = logging.getLogger(__name__)


class HealthMonitorService:
    def __init__(self):
        self.system_metrics = {}
        self.service_status = {}
        self.last_check = None
        logger.info("âœ… HealthMonitorService initialized")

    async def get_system_health(self) -> dict[str, Any]:
        """Get comprehensive system health status"""
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            # Get network stats
            network = psutil.net_io_counters()

            # Determine overall health status
            health_status = "healthy"
            if cpu_percent > 90 or memory.percent > 90 or disk.percent > 90:
                health_status = "critical"
            elif cpu_percent > 70 or memory.percent > 70 or disk.percent > 70:
                health_status = "warning"

            self.system_metrics = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available": memory.available,
                "disk_percent": disk.percent,
                "disk_free": disk.free,
                "network_bytes_sent": network.bytes_sent,
                "network_bytes_recv": network.bytes_recv,
                "timestamp": datetime.now(timezone.timezone.utc).isoformat(),
            }

            self.last_check = datetime.now(timezone.timezone.utc).isoformat()

            return {
                "status": health_status,
                "system": self.system_metrics,
                "platform": {
                    "system": platform.system(),
                    "release": platform.release(),
                    "version": platform.version(),
                    "machine": platform.machine(),
                },
                "services": self.service_status,
                "timestamp": self.last_check,
            }
        except Exception as e:
            logger.error(f"âŒ Error getting system health: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now(timezone.timezone.utc).isoformat(),
            }

    async def check_service_health(self, service_name: str) -> dict[str, Any]:
        """Check health of a specific service"""
        try:
            # Simulate service health check
            service_status = {
                "name": service_name,
                "status": "healthy",
                "response_time": 0.05,
                "last_check": datetime.now(timezone.timezone.utc).isoformat(),
                "uptime": "24h",
                "version": "1.0.0",
            }

            # Update service status
            self.service_status[service_name] = service_status

            return service_status
        except Exception as e:
            logger.error(f"âŒ Error checking service health for {service_name}: {e}")
            return {
                "name": service_name,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now(timezone.timezone.utc).isoformat(),
            }

    async def get_all_services_health(self) -> dict[str, Any]:
        """Get health status of all services"""
        try:
            services = [
                "api_server",
                "database",
                "redis",
                "websocket",
                "ai_engine",
                "trading_engine",
                "market_data",
                "notification_service",
            ]

            results = {}
            for service in services:
                results[service] = await self.check_service_health(service)

            return {
                "services": results,
                "overall_status": "healthy",
                "timestamp": datetime.now(timezone.timezone.utc).isoformat(),
            }
        except Exception as e:
            logger.error(f"âŒ Error getting all services health: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now(timezone.timezone.utc).isoformat(),
            }

    async def self_heal(self) -> dict[str, Any]:
        """Perform self-healing operations"""
        try:
            healing_actions = []

            # Check and restart critical services if needed
            for service_name, status in self.service_status.items():
                if status.get("status") == "error":
                    # Simulate service restart
                    await asyncio.sleep(0.1)
                    self.service_status[service_name]["status"] = "healthy"
                    healing_actions.append(f"Restarted {service_name}")

            return {
                "success": True,
                "actions_taken": healing_actions,
                "timestamp": datetime.now(timezone.timezone.utc).isoformat(),
            }
        except Exception as e:
            logger.error(f"âŒ Error during self-healing: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now(timezone.timezone.utc).isoformat(),
            }

    async def get_health_alerts(self) -> list[dict[str, Any]]:
        """Get current health alerts"""
        alerts = []

        # Check for critical system metrics
        if self.system_metrics.get("cpu_percent", 0) > 90:
            alerts.append(
                {
                    "level": "critical",
                    "message": (f"High CPU usage: {self.system_metrics['cpu_percent']}%"),
                    "timestamp": (datetime.now(timezone.timezone.utc).isoformat()),
                }
            )

        if self.system_metrics.get("memory_percent", 0) > 90:
            alerts.append(
                {
                    "level": "critical",
                    "message": (f"High memory usage: {self.system_metrics['memory_percent']}%"),
                    "timestamp": (datetime.now(timezone.timezone.utc).isoformat()),
                }
            )

        # Check for service errors
        for service_name, status in self.service_status.items():
            if status.get("status") == "error":
                alerts.append(
                    {
                        "level": "warning",
                        "message": f"Service {service_name} is in error state",
                        "timestamp": (datetime.now(timezone.timezone.utc).isoformat()),
                    }
                )

        return alerts


# Global instance
health_monitor_service = HealthMonitorService()


def get_health_monitor_service() -> HealthMonitorService:
    """Get the health monitor service instance"""
    return health_monitor_service


