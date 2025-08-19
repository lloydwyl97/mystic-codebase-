# watchdog.py
"""
Health Watchdog & Auto-Recovery System
Monitors all trading services and automatically restarts failed components.
"""

import subprocess
import time
import psutil
from typing import Dict, List, Any
from datetime import datetime
import requests
from datetime import datetime, timezone


class TradingWatchdog:
    """
    Advanced watchdog system for trading services.
    """

    def __init__(self, check_interval: int = 60, max_restart_attempts: int = 3):
        """
        Initialize watchdog.

        Args:
            check_interval: Seconds between health checks
            max_restart_attempts: Maximum restart attempts per service
        """
        self.check_interval = check_interval
        self.max_restart_attempts = max_restart_attempts
        self.service_status = {}
        self.restart_history = []
        self.health_log = []

        # Define critical services
        self.critical_services = {
            "mystic-ai": {
                "script": "dashboard_api.py",
                "port": 8000,
                "health_endpoint": "/health",
                "restart_command": ["python", "dashboard_api.py"],
                "max_memory_mb": 1024,
                "max_cpu_percent": 80,
            },
            "trade-logger": {
                "script": "db_logger.py",
                "port": None,
                "health_endpoint": None,
                "restart_command": ["python", "db_logger.py"],
                "max_memory_mb": 512,
                "max_cpu_percent": 50,
            },
            "strategy-mutator": {
                "script": "mutator.py",
                "port": None,
                "health_endpoint": None,
                "restart_command": ["python", "mutator.py"],
                "max_memory_mb": 512,
                "max_cpu_percent": 60,
            },
            "hyper-optimizer": {
                "script": "hyper_tuner.py",
                "port": None,
                "health_endpoint": None,
                "restart_command": ["python", "hyper_tuner.py"],
                "max_memory_mb": 1024,
                "max_cpu_percent": 90,
            },
        }

    def check_service_health(self, service_name: str) -> Dict[str, Any]:
        """
        Check health of a specific service.

        Args:
            service_name: Name of the service to check

        Returns:
            Health status
        """
        if service_name not in self.critical_services:
            return {"status": "unknown", "error": "Service not configured"}

        service_config = self.critical_services[service_name]

        # Check if process is running
        process_running = self._is_process_running(service_config["script"])

        # Check port if configured
        port_healthy = True
        if service_config["port"]:
            port_healthy = self._check_port_health(service_config["port"])

        # Check HTTP health endpoint if configured
        http_healthy = True
        if service_config["health_endpoint"]:
            http_healthy = self._check_http_health(
                service_config["port"], service_config["health_endpoint"]
            )

        # Check resource usage
        resource_usage = self._check_resource_usage(service_config["script"])

        # Determine overall health
        overall_healthy = process_running and port_healthy and http_healthy

        # Check resource limits
        memory_ok = resource_usage["memory_mb"] <= service_config["max_memory_mb"]
        cpu_ok = resource_usage["cpu_percent"] <= service_config["max_cpu_percent"]

        health_status = {
            "service": service_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "overall_healthy": overall_healthy,
            "process_running": process_running,
            "port_healthy": port_healthy,
            "http_healthy": http_healthy,
            "memory_ok": memory_ok,
            "cpu_ok": cpu_ok,
            "resource_usage": resource_usage,
            "last_check": datetime.now(timezone.utc).isoformat(),
        }

        # Update service status
        self.service_status[service_name] = health_status

        return health_status

    def _is_process_running(self, script_name: str) -> bool:
        """Check if a Python script is running."""
        try:
            for proc in psutil.process_iter(["pid", "name", "cmdline"]):
                if proc.info["cmdline"] and script_name in " ".join(proc.info["cmdline"]):
                    return True
            return False
        except Exception as e:
            print(f"Error checking process: {e}")
            return False

    def _check_port_health(self, port: int) -> bool:
        """Check if port is listening."""
        try:
            import socket

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(("localhost", port))
            sock.close()
            return result == 0
        except Exception as e:
            print(f"Error checking port {port}: {e}")
            return False

    def _check_http_health(self, port: int, endpoint: str) -> bool:
        """Check HTTP health endpoint."""
        try:
            url = f"http://localhost:{port}{endpoint}"
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"Error checking HTTP health: {e}")
            return False

    def _check_resource_usage(self, script_name: str) -> Dict[str, float]:
        """Check resource usage of a process."""
        try:
            for proc in psutil.process_iter(["pid", "name", "cmdline"]):
                if proc.info["cmdline"] and script_name in " ".join(proc.info["cmdline"]):
                    process = psutil.Process(proc.info["pid"])
                    return {
                        "memory_mb": process.memory_info().rss / 1024 / 1024,
                        "cpu_percent": process.cpu_percent(),
                        "pid": proc.info["pid"],
                    }
            return {"memory_mb": 0, "cpu_percent": 0, "pid": None}
        except Exception as e:
            print(f"Error checking resource usage: {e}")
            return {"memory_mb": 0, "cpu_percent": 0, "pid": None}

    def restart_service(self, service_name: str) -> Dict[str, Any]:
        """
        Restart a failed service.

        Args:
            service_name: Name of the service to restart

        Returns:
            Restart result
        """
        if service_name not in self.critical_services:
            return {"success": False, "error": "Service not configured"}

        service_config = self.critical_services[service_name]

        # Check restart attempts
        restart_count = self._get_restart_count(service_name)
        if restart_count >= self.max_restart_attempts:
            return {
                "success": False,
                "error": (f"Max restart attempts ({self.max_restart_attempts}) exceeded"),
            }

        try:
            # Kill existing process if running
            self._kill_process(service_config["script"])

            # Wait a moment
            time.sleep(2)

            # Start new process
            process = subprocess.Popen(
                service_config["restart_command"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # Wait a moment for startup
            time.sleep(5)

            # Check if restart was successful
            health_check = self.check_service_health(service_name)

            restart_result = {
                "service": service_name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "restart_attempt": restart_count + 1,
                "process_id": process.pid,
                "success": health_check["overall_healthy"],
                "health_status": health_check,
            }

            # Log restart
            self._log_restart(restart_result)

            if restart_result["success"]:
                print(f"âœ… Successfully restarted {service_name}")
            else:
                print(f"âŒ Failed to restart {service_name}")

            return restart_result

        except Exception as e:
            error_result = {
                "service": service_name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "success": False,
                "error": str(e),
            }
            self._log_restart(error_result)
            return error_result

    def _kill_process(self, script_name: str):
        """Kill a process by script name."""
        try:
            for proc in psutil.process_iter(["pid", "name", "cmdline"]):
                if proc.info["cmdline"] and script_name in " ".join(proc.info["cmdline"]):
                    process = psutil.Process(proc.info["pid"])
                    process.terminate()
                    process.wait(timeout=10)
                    print(f"ðŸ”„ Killed process for {script_name}")
        except Exception as e:
            print(f"Error killing process: {e}")

    def _get_restart_count(self, service_name: str) -> int:
        """Get number of restart attempts for a service."""
        count = 0
        for restart in self.restart_history:
            if restart["service"] == service_name:
                count += 1
        return count

    def _log_restart(self, restart_result: Dict[str, Any]):
        """Log restart attempt."""
        self.restart_history.append(restart_result)

    def monitor_all_services(self) -> Dict[str, Any]:
        """
        Monitor health of all critical services.

        Returns:
            Monitoring results
        """
        print(f"ðŸ” Checking health of {len(self.critical_services)} services...")

        monitoring_results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "services_checked": len(self.critical_services),
            "healthy_services": 0,
            "unhealthy_services": 0,
            "restarts_performed": 0,
            "service_status": {},
        }

        for service_name in self.critical_services:
            health_status = self.check_service_health(service_name)
            monitoring_results["service_status"][service_name] = health_status

            if health_status["overall_healthy"]:
                monitoring_results["healthy_services"] += 1
                print(f"âœ… {service_name}: Healthy")
            else:
                monitoring_results["unhealthy_services"] += 1
                print(f"âŒ {service_name}: Unhealthy")

                # Attempt restart
                restart_result = self.restart_service(service_name)
                if restart_result["success"]:
                    monitoring_results["restarts_performed"] += 1
                    print(f"ðŸ”„ {service_name}: Restarted successfully")
                else:
                    error_msg = restart_result.get("error", "Unknown error")
                print(f"ðŸ’¥ {service_name}: Restart failed - {error_msg}")

        # Log monitoring results
        self.health_log.append(monitoring_results)

        return monitoring_results

    def get_system_summary(self) -> Dict[str, Any]:
        """Get overall system health summary."""
        if not self.service_status:
            return {"status": "unknown", "services": 0}

        healthy_count = sum(
            1 for status in self.service_status.values() if status["overall_healthy"]
        )
        total_count = len(self.service_status)

        # Calculate uptime for each service
        service_uptimes = {}
        for service_name, status in self.service_status.items():
            if status["overall_healthy"]:
                service_uptimes[service_name] = "running"
            else:
                service_uptimes[service_name] = "down"

        return {
            "overall_health": ("healthy" if healthy_count == total_count else "degraded"),
            "healthy_services": healthy_count,
            "total_services": total_count,
            "health_percentage": round((healthy_count / total_count) * 100, 1),
            "service_uptimes": service_uptimes,
            "last_check": datetime.now(timezone.utc).isoformat(),
        }

    def get_health_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get health check history."""
        return self.health_log[-limit:]

    def get_restart_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get restart history."""
        return self.restart_history[-limit:]

    def run_continuous_monitoring(self):
        """Run continuous monitoring loop."""
        print("ðŸš€ Starting continuous monitoring...")
        print(f"â° Check interval: {self.check_interval} seconds")
        print(f"ðŸ”„ Max restart attempts: {self.max_restart_attempts}")

        while True:
            try:
                monitoring_results = self.monitor_all_services()

                # Print summary
                summary = self.get_system_summary()
                print(f"\nðŸ“Š System Summary: {summary['overall_health'].upper()}")
                health_pct = summary["health_percentage"]
                healthy_count = summary["healthy_services"]
                total_count = summary["total_services"]
                print(f"   Healthy: {healthy_count}/{total_count} ({health_pct}%)")
                print(f"   Restarts: {monitoring_results['restarts_performed']}")

                # Wait for next check
                time.sleep(self.check_interval)

            except KeyboardInterrupt:
                print("\nðŸ›‘ Monitoring stopped by user")
                break
            except Exception as e:
                print(f"âŒ Monitoring error: {e}")
                time.sleep(self.check_interval)


# Convenience functions
def check_service_health(service_name: str) -> Dict[str, Any]:
    """Check health of a specific service."""
    watchdog = TradingWatchdog()
    return watchdog.check_service_health(service_name)


def restart_service(service_name: str) -> Dict[str, Any]:
    """Restart a specific service."""
    watchdog = TradingWatchdog()
    return watchdog.restart_service(service_name)


def monitor_services() -> Dict[str, Any]:
    """Monitor all services."""
    watchdog = TradingWatchdog()
    return watchdog.monitor_all_services()


def get_system_health() -> Dict[str, Any]:
    """Get system health summary."""
    watchdog = TradingWatchdog()
    return watchdog.get_system_summary()


# Example usage
if __name__ == "__main__":
    print("ðŸ›¡ï¸ Trading System Watchdog")
    print("=" * 40)

    # Test health monitoring
    watchdog = TradingWatchdog()

    print("\nðŸ” Testing service health checks...")
    for service_name in watchdog.critical_services:
        health = watchdog.check_service_health(service_name)
        status = "âœ… Healthy" if health["overall_healthy"] else "âŒ Unhealthy"
        print(f"   {service_name}: {status}")

    print("\nðŸ“Š System summary:")
    summary = watchdog.get_system_summary()
    print(f"   Overall: {summary['overall_health']}")
    print(f"   Health: {summary['health_percentage']}%")

    # Start continuous monitoring
    print("\nðŸš€ Starting continuous monitoring (Ctrl+C to stop)...")
    watchdog.run_continuous_monitoring()

