"""
Health Monitoring Service for Mystic Trading

Monitors the health of all system components and performs self-healing when needed.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from .services.websocket_manager import websocket_manager

logger = logging.getLogger(__name__)


class HealthMonitor:
    """Monitors the health of all system components and performs self-healing when needed."""

    def __init__(
        self,
        signal_manager: Any,
        auto_trading_manager: Any,
        notification_service: Any,
        metrics_collector: Any | None = None,
    ):
        self.signal_manager = signal_manager
        self.auto_trading_manager = auto_trading_manager
        self.notification_service = notification_service
        self.metrics_collector = metrics_collector
        self.is_running = False
        self.monitor_task = None
        self.service_health: dict[str, dict[str, Any]] = {}
        self.system_metrics: dict[str, dict[str, Any]] = {}

    async def start_monitoring(self) -> None:
        """Start the health monitoring background task."""
        if self.is_running:
            logger.warning("Health monitoring is already running")
            return

        self.is_running = True
        self.monitor_task = asyncio.create_task(self._monitor_system_health())
        logger.info("System health monitoring task started")

    async def stop_monitoring(self) -> None:
        """Stop the health monitoring background task."""
        if not self.is_running:
            logger.warning("Health monitoring is not running")
            return

        self.is_running = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("System health monitoring task stopped")

    async def _monitor_system_health(self) -> None:
        """Background task to monitor system health and auto-heal if needed."""
        logger.info("Starting system health monitoring background task...")
        while self.is_running:
            try:
                # Check signal health
                try:
                    signal_health = await self.signal_manager.check_signal_health()
                    auto_trading_health = await self.auto_trading_manager.check_health()

                    # Determine overall health
                    overall_health = signal_health.get("overall_health", "unknown")
                    if (
                        not auto_trading_health.get("healthy", False)
                        and overall_health == "healthy"
                    ):
                        overall_health = "degraded"

                    # If health is degraded or critical, trigger self-healing
                    if overall_health in ["degraded", "critical"]:
                        logger.warning(
                            f"System health is {overall_health}, triggering self-healing..."
                        )

                        # Heal signals
                        signal_healing_result = await self.signal_manager.self_heal_signals()

                        # Heal auto-trading
                        auto_trading_healing_result = await self.auto_trading_manager.self_heal()

                        # Log healing results
                        if (
                            signal_healing_result.get("healing_performed")
                            or auto_trading_healing_result.get("status") == "healed"
                        ):
                            logger.info(
                                f"Self-healing completed: Signals: {signal_healing_result.get('actions_taken', [])}, Auto-trading: {auto_trading_healing_result.get('message', 'No action')}"
                            )

                            # Send notification about healing
                            try:
                                await self.notification_service.send_notification(
                                    title="System Self-Healing Performed",
                                    message=(
                                        f"Automatic healing actions were taken: "
                                        f"Signals: {signal_healing_result.get('actions_taken', [])}, "
                                        f"Auto-trading: {auto_trading_healing_result.get('message', 'No action')}"
                                    ),
                                    level="info",
                                    channels=["in_app"],
                                )
                            except Exception as notify_error:
                                logger.error(
                                    f"Failed to send healing notification: {str(notify_error)}"
                                )
                        else:
                            logger.info("No healing actions needed")

                    # Log health status periodically
                    if overall_health != "healthy":
                        logger.info(
                            f"Current system health: {overall_health} (Signals: {signal_health.get('overall_health')}, Auto-trading: {auto_trading_health.get('healthy')})"
                        )

                except Exception as health_check_error:
                    logger.error(f"Error checking system health: {str(health_check_error)}")

            except Exception as e:
                logger.error(f"Error in system health monitoring: {str(e)}")
                # Send error notification with error handling
                try:
                    await self.notification_service.send_notification(
                        title="Health Monitoring Error",
                        message=f"Background health monitoring failed: {str(e)}",
                        level="error",
                        channels=[
                            "in_app"
                        ],  # Fallback to in-app only to reduce external dependencies
                    )
                except Exception as notify_error:
                    logger.error(f"Failed to send notification: {str(notify_error)}")

            # Check every 60 seconds
            await asyncio.sleep(60)

    async def check_health(self) -> dict[str, Any]:
        """Check the health of all system components."""
        try:
            # Get signal health
            signal_health = await self.signal_manager.check_signal_health()

            # Get auto-trading health
            auto_trading_health = await self.auto_trading_manager.check_health()

            # Combine results
            combined_health = {
                "status": "success",
                "overall_health": signal_health.get("overall_health", "unknown"),
                "signals": signal_health.get("signals", {}),
                "strategies": signal_health.get("strategies", {}),
                "auto_trading": auto_trading_health,
                "timestamp": datetime.now(timezone.timezone.utc).isoformat(),
            }

            # Update overall health if auto-trading is unhealthy
            if (
                not auto_trading_health.get("healthy", False)
                and combined_health["overall_health"] == "healthy"
            ):
                combined_health["overall_health"] = "degraded"

            return combined_health
        except Exception as e:
            logger.error(f"Error getting system health: {str(e)}")
            return {
                "status": "error",
                "message": f"Error getting system health: {str(e)}",
                "timestamp": datetime.now(timezone.timezone.utc).isoformat(),
            }

    async def perform_self_healing(self) -> dict[str, Any]:
        """Manually trigger self-healing of all system components."""
        try:
            # Perform signal self-healing
            signal_result = await self.signal_manager.self_heal_signals()

            # Perform auto-trading self-healing
            auto_trading_result = await self.auto_trading_manager.self_heal()

            # Combine results
            combined_result = {
                "status": "success",
                "signals": signal_result,
                "auto_trading": auto_trading_result,
                "timestamp": datetime.now(timezone.timezone.utc).isoformat(),
            }

            # Update metrics
            if signal_result.get("healing_performed", False) and self.metrics_collector:
                actions_taken = len(signal_result.get("actions_taken", []))
                self.metrics_collector.record_self_healing_event("manual_trigger", actions_taken)

            return combined_result
        except Exception as e:
            logger.error(f"Error performing self-healing: {str(e)}")
            return {
                "status": "error",
                "message": f"Error performing self-healing: {str(e)}",
                "timestamp": datetime.now(timezone.timezone.utc).isoformat(),
            }

    def update_service_health(
        self,
        service_name: str,
        status: str,
        details: dict[str, Any] | None = None,
    ):
        """Update health status for a specific service"""
        try:
            self.service_health[service_name] = {
                "status": status,
                "last_check": datetime.now(timezone.timezone.utc).isoformat(),
                "details": details or {},
            }

            # Broadcast health update
            asyncio.create_task(
                websocket_manager.broadcast_json(
                    {
                        "type": "health_update",
                        "data": {
                            "service": service_name,
                            "status": status,
                            "last_check": self.service_health[service_name]["last_check"],
                            "details": details or {},
                        },
                    }
                )
            )

            logger.info(f"Health updated for {service_name}: {status}")

        except Exception as e:
            logger.error(f"Error updating health for {service_name}: {e}")

    def update_system_metrics(self, metrics: dict[str, Any]):
        """Update system-wide metrics"""
        try:
            # Ensure all metrics are stored as Dict[str, Any]
            for k, v in metrics.items():
                if not isinstance(v, dict):
                    self.system_metrics[k] = {"value": v}
                else:
                    self.system_metrics[k] = v
            self.system_metrics["last_update"] = {
                "value": datetime.now(timezone.timezone.utc).isoformat()
            }

            # Broadcast system metrics update
            asyncio.create_task(
                websocket_manager.broadcast_json(
                    {"type": "system_metrics", "data": self.system_metrics}
                )
            )

        except Exception as e:
            logger.error(f"Error updating system metrics: {e}")


# Global health monitor instance
health_monitor = None


def get_health_monitor(
    signal_manager: Any,
    auto_trading_manager: Any,
    notification_service: Any,
    metrics_collector: Any | None = None,
) -> HealthMonitor:
    """Get or create health monitor instance."""
    global health_monitor
    if health_monitor is None:
        health_monitor = HealthMonitor(
            signal_manager,
            auto_trading_manager,
            notification_service,
            metrics_collector,
        )
    return health_monitor


