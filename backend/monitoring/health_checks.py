"""
Health Check System for Mystic Trading Platform

Provides comprehensive health monitoring with:
- System health checks
- Database connectivity checks
- Service availability monitoring
- Performance health checks
- Custom health checks
"""

import asyncio
import logging
import threading
import time
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

# Health check configuration
HEALTH_CHECK_INTERVAL = 30  # seconds
HEALTH_CHECK_TIMEOUT = 10   # seconds
CRITICAL_THRESHOLD = 3      # consecutive failures


class HealthStatus(Enum):
    """Health status enumeration"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Health check result"""
    component: str
    status: HealthStatus
    message: str
    timestamp: float
    response_time: float
    details: dict[str, Any] | None = None


class HealthChecker:
    """Comprehensive health checker"""

    def __init__(self):
        self.health_results: dict[str, list[HealthCheckResult]] = {}
        self.failure_counts: dict[str, int] = {}
        self.redis_client = None
        self.lock = threading.Lock()

        # Initialize Redis connection
        self._initialize_redis()

    def _initialize_redis(self):
        """Initialize Redis connection"""
        try:
            if redis is None:
                logger.warning("Redis not available for health checks")
                return

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
            logger.info("Redis connection established for health checks")

        except Exception as e:
            logger.error(f"Failed to connect to Redis for health checks: {e}")
            self.redis_client = None

    async def check_system_health(self) -> HealthCheckResult:
        """Check system resource health"""
        start_time = time.time()

        try:
            import psutil

            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_status = HealthStatus.HEALTHY if cpu_percent < 80 else HealthStatus.WARNING if cpu_percent < 95 else HealthStatus.CRITICAL

            # Check memory usage
            memory = psutil.virtual_memory()
            memory_status = HealthStatus.HEALTHY if memory.percent < 80 else HealthStatus.WARNING if memory.percent < 95 else HealthStatus.CRITICAL

            # Check disk usage
            disk = psutil.disk_usage('/')
            disk_status = HealthStatus.HEALTHY if disk.percent < 80 else HealthStatus.WARNING if disk.percent < 95 else HealthStatus.CRITICAL

            # Overall status
            overall_status = max(cpu_status, memory_status, disk_status)

            response_time = time.time() - start_time

            result = HealthCheckResult(
                component="system",
                status=overall_status,
                message="System health check completed",
                timestamp=time.time(),
                response_time=response_time,
                details={
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'disk_percent': disk.percent,
                    'cpu_status': cpu_status.value,
                    'memory_status': memory_status.value,
                    'disk_status': disk_status.value
                }
            )

            self._store_health_result(result)
            return result

        except Exception as e:
            logger.error(f"System health check failed: {e}")
            return HealthCheckResult(
                component="system",
                status=HealthStatus.CRITICAL,
                message=f"System health check failed: {str(e)}",
                timestamp=time.time(),
                response_time=time.time() - start_time
            )

    async def check_database_health(self) -> HealthCheckResult:
        """Check database connectivity and performance"""
        start_time = time.time()

        try:
            # Import database module
            from database_optimized import optimized_db_manager

            # Test database connection
            test_query = "SELECT 1"
            result = optimized_db_manager.execute_query(test_query, (), use_cache=False)

            if result is not None:
                status = HealthStatus.HEALTHY
                message = "Database connection healthy"
            else:
                status = HealthStatus.CRITICAL
                message = "Database connection failed"

            response_time = time.time() - start_time

            return HealthCheckResult(
                component="database",
                status=status,
                message=message,
                timestamp=time.time(),
                response_time=response_time,
                details={'query_result': result}
            )

        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return HealthCheckResult(
                component="database",
                status=HealthStatus.CRITICAL,
                message=f"Database health check failed: {str(e)}",
                timestamp=time.time(),
                response_time=time.time() - start_time
            )

    async def check_redis_health(self) -> HealthCheckResult:
        """Check Redis connectivity and performance"""
        start_time = time.time()

        try:
            if self.redis_client is None:
                return HealthCheckResult(
                    component="redis",
                    status=HealthStatus.UNKNOWN,
                    message="Redis not configured",
                    timestamp=time.time(),
                    response_time=time.time() - start_time
                )

            # Test Redis connection
            self.redis_client.ping()
            status = HealthStatus.HEALTHY
            message = "Redis connection healthy"

            response_time = time.time() - start_time

            return HealthCheckResult(
                component="redis",
                status=status,
                message=message,
                timestamp=time.time(),
                response_time=response_time
            )

        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return HealthCheckResult(
                component="redis",
                status=HealthStatus.CRITICAL,
                message=f"Redis health check failed: {str(e)}",
                timestamp=time.time(),
                response_time=time.time() - start_time
            )

    async def check_api_health(self) -> HealthCheckResult:
        """Check API endpoints health"""
        start_time = time.time()

        try:
            # Test basic API functionality
            import requests

            # Test internal API endpoint
            api_url = "http://localhost:8000/health"
            response = requests.get(api_url, timeout=5)

            if response.status_code == 200:
                status = HealthStatus.HEALTHY
                message = "API endpoints healthy"
            else:
                status = HealthStatus.WARNING
                message = f"API returned status {response.status_code}"

            response_time = time.time() - start_time

            return HealthCheckResult(
                component="api",
                status=status,
                message=message,
                timestamp=time.time(),
                response_time=response_time,
                details={'status_code': response.status_code}
            )

        except Exception as e:
            logger.error(f"API health check failed: {e}")
            return HealthCheckResult(
                component="api",
                status=HealthStatus.CRITICAL,
                message=f"API health check failed: {str(e)}",
                timestamp=time.time(),
                response_time=time.time() - start_time
            )

    async def check_external_services(self) -> HealthCheckResult:
        """Check external service dependencies"""
        start_time = time.time()

        try:
            # Check external services (exchanges, data providers, etc.)
            external_services = {
                'binance_api': 'https://api.binance.us/api/v3/ping',
                'coinbase_api': 'https://api.coinbase.com/v2/time'
            }

            healthy_services = 0
            total_services = len(external_services)

            for service_name, url in external_services.items():
                try:
                    import requests
                    response = requests.get(url, timeout=5)
                    if response.status_code == 200:
                        healthy_services += 1
                except Exception:
                    pass

            health_percentage = (healthy_services / total_services) * 100

            if health_percentage >= 80:
                status = HealthStatus.HEALTHY
                message = f"External services healthy ({health_percentage:.1f}%)"
            elif health_percentage >= 50:
                status = HealthStatus.WARNING
                message = f"External services degraded ({health_percentage:.1f}%)"
            else:
                status = HealthStatus.CRITICAL
                message = f"External services critical ({health_percentage:.1f}%)"

            response_time = time.time() - start_time

            return HealthCheckResult(
                component="external_services",
                status=status,
                message=message,
                timestamp=time.time(),
                response_time=response_time,
                details={
                    'healthy_services': healthy_services,
                    'total_services': total_services,
                    'health_percentage': health_percentage
                }
            )

        except Exception as e:
            logger.error(f"External services health check failed: {e}")
            return HealthCheckResult(
                component="external_services",
                status=HealthStatus.CRITICAL,
                message=f"External services health check failed: {str(e)}",
                timestamp=time.time(),
                response_time=time.time() - start_time
            )

    async def run_all_health_checks(self) -> dict[str, HealthCheckResult]:
        """Run all health checks"""
        checks = [
            self.check_system_health(),
            self.check_database_health(),
            self.check_redis_health(),
            self.check_api_health(),
            self.check_external_services()
        ]

        results = await asyncio.gather(*checks, return_exceptions=True)

        health_results = {}
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                component = ['system', 'database', 'redis', 'api', 'external_services'][i]
                health_results[component] = HealthCheckResult(
                    component=component,
                    status=HealthStatus.CRITICAL,
                    message=f"Health check failed: {str(result)}",
                    timestamp=time.time(),
                    response_time=0.0
                )
            else:
                health_results[result.component] = result

        return health_results

    def _store_health_result(self, result: HealthCheckResult):
        """Store health check result"""
        with self.lock:
            if result.component not in self.health_results:
                self.health_results[result.component] = []

            self.health_results[result.component].append(result)

            # Keep only recent results (last 100)
            if len(self.health_results[result.component]) > 100:
                self.health_results[result.component] = self.health_results[result.component][-100:]

            # Track failure counts
            if result.status in [HealthStatus.CRITICAL, HealthStatus.WARNING]:
                self.failure_counts[result.component] = self.failure_counts.get(result.component, 0) + 1
            else:
                self.failure_counts[result.component] = 0

    def get_health_summary(self) -> dict[str, Any]:
        """Get comprehensive health summary"""
        with self.lock:
            summary = {
                'overall_status': HealthStatus.HEALTHY.value,
                'components': {},
                'failure_counts': dict(self.failure_counts),
                'timestamp': time.time()
            }

            # Check each component
            critical_components = []
            warning_components = []

            for component, results in self.health_results.items():
                if results:
                    latest_result = results[-1]
                    summary['components'][component] = {
                        'status': latest_result.status.value,
                        'message': latest_result.message,
                        'response_time': latest_result.response_time,
                        'timestamp': latest_result.timestamp,
                        'details': latest_result.details
                    }

                    if latest_result.status == HealthStatus.CRITICAL:
                        critical_components.append(component)
                    elif latest_result.status == HealthStatus.WARNING:
                        warning_components.append(component)

            # Determine overall status
            if critical_components:
                summary['overall_status'] = HealthStatus.CRITICAL.value
            elif warning_components:
                summary['overall_status'] = HealthStatus.WARNING.value

            summary['critical_components'] = critical_components
            summary['warning_components'] = warning_components

            return summary

    def get_component_history(self, component: str, hours: int = 24) -> list[dict[str, Any]]:
        """Get health check history for a component"""
        cutoff_time = time.time() - (hours * 3600)

        with self.lock:
            if component not in self.health_results:
                return []

            recent_results = [r for r in self.health_results[component] if r.timestamp >= cutoff_time]
            return [r.__dict__ for r in recent_results]


# Global health checker instance
health_checker = HealthChecker()


