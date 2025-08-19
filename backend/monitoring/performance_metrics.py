"""
Performance Metrics System for Mystic Trading Platform

Provides comprehensive performance monitoring with:
- Real-time metrics collection
- System resource monitoring
- Application performance tracking
- Database performance metrics
- Custom business metrics
"""

import json
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

import psutil

try:
    import redis
    from redis import Redis
except ImportError:
    redis = None
    Redis = None

from trading_config import trading_config

logger = logging.getLogger(__name__)

# Metrics configuration
METRICS_RETENTION_HOURS = 24
METRICS_COLLECTION_INTERVAL = 60  # 1 minute
SYSTEM_METRICS_ENABLED = True
CUSTOM_METRICS_ENABLED = True


@dataclass
class SystemMetrics:
    """System resource metrics"""
    cpu_percent: float
    memory_percent: float
    memory_available: float
    disk_usage_percent: float
    disk_io_read: float
    disk_io_write: float
    network_bytes_sent: float
    network_bytes_recv: float
    timestamp: float


@dataclass
class ApplicationMetrics:
    """Application performance metrics"""
    request_count: int
    response_time_avg: float
    response_time_p95: float
    response_time_p99: float
    error_rate: float
    active_connections: int
    timestamp: float


@dataclass
class DatabaseMetrics:
    """Database performance metrics"""
    query_count: int
    query_time_avg: float
    query_time_p95: float
    slow_query_count: int
    connection_count: int
    cache_hit_rate: float
    timestamp: float


class MetricsCollector:
    """Collects and stores performance metrics"""

    def __init__(self):
        self.system_metrics: deque = deque(maxlen=1440)  # 24 hours at 1-minute intervals
        self.application_metrics: deque = deque(maxlen=1440)
        self.database_metrics: deque = deque(maxlen=1440)
        self.custom_metrics: dict[str, deque] = defaultdict(lambda: deque(maxlen=1440))
        self.redis_client = None
        self.lock = threading.Lock()

        # Initialize Redis for metrics storage
        self._initialize_redis()

    def _initialize_redis(self):
        """Initialize Redis connection for metrics storage"""
        try:
            if redis is None:
                logger.warning("Redis not available, using local metrics storage")
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
            logger.info("Redis connection established for metrics storage")

        except Exception as e:
            logger.error(f"Failed to connect to Redis for metrics: {e}")
            self.redis_client = None

    def collect_system_metrics(self) -> SystemMetrics:
        """Collect system resource metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            network = psutil.net_io_counters()

            metrics = SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_available=memory.available / (1024 * 1024 * 1024),  # GB
                disk_usage_percent=disk.percent,
                disk_io_read=disk_io.read_bytes if disk_io else 0,
                disk_io_write=disk_io.write_bytes if disk_io else 0,
                network_bytes_sent=network.bytes_sent if network else 0,
                network_bytes_recv=network.bytes_recv if network else 0,
                timestamp=time.time()
            )

            with self.lock:
                self.system_metrics.append(metrics)

            # Store in Redis if available
            if self.redis_client:
                try:
                    self.redis_client.setex(
                        f"metrics:system:{int(time.time())}",
                        3600,  # 1 hour TTL
                        json.dumps(metrics.__dict__, default=str)
                    )
                except Exception as e:
                    logger.error(f"Failed to store system metrics in Redis: {e}")

            return metrics

        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return SystemMetrics(0, 0, 0, 0, 0, 0, 0, 0, time.time())

    def collect_application_metrics(self, request_count: int = 0,
                                  response_times: list[float] = None,
                                  error_count: int = 0,
                                  active_connections: int = 0) -> ApplicationMetrics:
        """Collect application performance metrics"""
        try:
            response_times = response_times or []

            if response_times:
                response_time_avg = sum(response_times) / len(response_times)
                sorted_times = sorted(response_times)
                response_time_p95 = sorted_times[int(len(sorted_times) * 0.95)]
                response_time_p99 = sorted_times[int(len(sorted_times) * 0.99)]
            else:
                response_time_avg = 0.0
                response_time_p95 = 0.0
                response_time_p99 = 0.0

            error_rate = (error_count / max(request_count, 1)) * 100

            metrics = ApplicationMetrics(
                request_count=request_count,
                response_time_avg=response_time_avg,
                response_time_p95=response_time_p95,
                response_time_p99=response_time_p99,
                error_rate=error_rate,
                active_connections=active_connections,
                timestamp=time.time()
            )

            with self.lock:
                self.application_metrics.append(metrics)

            # Store in Redis if available
            if self.redis_client:
                try:
                    self.redis_client.setex(
                        f"metrics:application:{int(time.time())}",
                        3600,  # 1 hour TTL
                        json.dumps(metrics.__dict__, default=str)
                    )
                except Exception as e:
                    logger.error(f"Failed to store application metrics in Redis: {e}")

            return metrics

        except Exception as e:
            logger.error(f"Error collecting application metrics: {e}")
            return ApplicationMetrics(0, 0.0, 0.0, 0.0, 0.0, 0, time.time())

    def collect_database_metrics(self, query_count: int = 0,
                               query_times: list[float] = None,
                               slow_query_count: int = 0,
                               connection_count: int = 0,
                               cache_hit_rate: float = 0.0) -> DatabaseMetrics:
        """Collect database performance metrics"""
        try:
            query_times = query_times or []

            if query_times:
                query_time_avg = sum(query_times) / len(query_times)
                sorted_times = sorted(query_times)
                query_time_p95 = sorted_times[int(len(sorted_times) * 0.95)]
            else:
                query_time_avg = 0.0
                query_time_p95 = 0.0

            metrics = DatabaseMetrics(
                query_count=query_count,
                query_time_avg=query_time_avg,
                query_time_p95=query_time_p95,
                slow_query_count=slow_query_count,
                connection_count=connection_count,
                cache_hit_rate=cache_hit_rate,
                timestamp=time.time()
            )

            with self.lock:
                self.database_metrics.append(metrics)

            # Store in Redis if available
            if self.redis_client:
                try:
                    self.redis_client.setex(
                        f"metrics:database:{int(time.time())}",
                        3600,  # 1 hour TTL
                        json.dumps(metrics.__dict__, default=str)
                    )
                except Exception as e:
                    logger.error(f"Failed to store database metrics in Redis: {e}")

            return metrics

        except Exception as e:
            logger.error(f"Error collecting database metrics: {e}")
            return DatabaseMetrics(0, 0.0, 0.0, 0, 0, 0.0, time.time())

    def add_custom_metric(self, metric_name: str, value: float,
                         metadata: dict[str, Any] | None = None):
        """Add custom metric"""
        try:
            metric_data = {
                'value': value,
                'metadata': metadata or {},
                'timestamp': time.time()
            }

            with self.lock:
                self.custom_metrics[metric_name].append(metric_data)

            # Store in Redis if available
            if self.redis_client:
                try:
                    self.redis_client.setex(
                        f"metrics:custom:{metric_name}:{int(time.time())}",
                        3600,  # 1 hour TTL
                        json.dumps(metric_data, default=str)
                    )
                except Exception as e:
                    logger.error(f"Failed to store custom metric in Redis: {e}")

        except Exception as e:
            logger.error(f"Error adding custom metric: {e}")

    def get_metrics_summary(self) -> dict[str, Any]:
        """Get comprehensive metrics summary"""
        with self.lock:
            # System metrics summary
            if self.system_metrics:
                latest_system = self.system_metrics[-1]
                system_summary = {
                    'cpu_percent': latest_system.cpu_percent,
                    'memory_percent': latest_system.memory_percent,
                    'memory_available_gb': latest_system.memory_available,
                    'disk_usage_percent': latest_system.disk_usage_percent,
                    'network_bytes_sent': latest_system.network_bytes_sent,
                    'network_bytes_recv': latest_system.network_bytes_recv
                }
            else:
                system_summary = {}

            # Application metrics summary
            if self.application_metrics:
                latest_app = self.application_metrics[-1]
                app_summary = {
                    'request_count': latest_app.request_count,
                    'response_time_avg': latest_app.response_time_avg,
                    'response_time_p95': latest_app.response_time_p95,
                    'error_rate': latest_app.error_rate,
                    'active_connections': latest_app.active_connections
                }
            else:
                app_summary = {}

            # Database metrics summary
            if self.database_metrics:
                latest_db = self.database_metrics[-1]
                db_summary = {
                    'query_count': latest_db.query_count,
                    'query_time_avg': latest_db.query_time_avg,
                    'slow_query_count': latest_db.slow_query_count,
                    'cache_hit_rate': latest_db.cache_hit_rate
                }
            else:
                db_summary = {}

            # Custom metrics summary
            custom_summary = {}
            for metric_name, metrics in self.custom_metrics.items():
                if metrics:
                    latest_value = metrics[-1]['value']
                    custom_summary[metric_name] = latest_value

            return {
                'system': system_summary,
                'application': app_summary,
                'database': db_summary,
                'custom': custom_summary,
                'timestamp': time.time()
            }

    def get_metrics_history(self, metric_type: str, hours: int = 24) -> list[dict[str, Any]]:
        """Get metrics history for specified type and time range"""
        cutoff_time = time.time() - (hours * 3600)

        with self.lock:
            if metric_type == 'system':
                metrics = [m for m in self.system_metrics if m.timestamp >= cutoff_time]
                return [m.__dict__ for m in metrics]
            elif metric_type == 'application':
                metrics = [m for m in self.application_metrics if m.timestamp >= cutoff_time]
                return [m.__dict__ for m in metrics]
            elif metric_type == 'database':
                metrics = [m for m in self.database_metrics if m.timestamp >= cutoff_time]
                return [m.__dict__ for m in metrics]
            elif metric_type == 'custom':
                all_custom_metrics = []
                for metric_name, metrics in self.custom_metrics.items():
                    recent_metrics = [m for m in metrics if m['timestamp'] >= cutoff_time]
                    all_custom_metrics.extend(recent_metrics)
                return all_custom_metrics
            else:
                return []

    def cleanup_old_metrics(self, max_age_hours: int = METRICS_RETENTION_HOURS):
        """Clean up old metrics data"""
        cutoff_time = time.time() - (max_age_hours * 3600)

        with self.lock:
            # Clean up system metrics
            self.system_metrics = deque(
                (m for m in self.system_metrics if m.timestamp >= cutoff_time),
                maxlen=1440
            )

            # Clean up application metrics
            self.application_metrics = deque(
                (m for m in self.application_metrics if m.timestamp >= cutoff_time),
                maxlen=1440
            )

            # Clean up database metrics
            self.database_metrics = deque(
                (m for m in self.database_metrics if m.timestamp >= cutoff_time),
                maxlen=1440
            )

            # Clean up custom metrics
            for metric_name in list(self.custom_metrics.keys()):
                self.custom_metrics[metric_name] = deque(
                    (m for m in self.custom_metrics[metric_name] if m['timestamp'] >= cutoff_time),
                    maxlen=1440
                )

        logger.info("Cleaned up old metrics data")


# Global metrics collector instance
metrics_collector = MetricsCollector()


