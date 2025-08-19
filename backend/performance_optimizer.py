"""
Performance Optimizer for Mystic Trading Platform
Advanced performance optimizations including caching, connection pooling, and resource management
"""

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import wraps
from typing import Any

import aiohttp
import aioredis
import psutil
import structlog

logger = structlog.get_logger()


@dataclass
class PerformanceMetrics:
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_io: dict[str, float]
    network_io: dict[str, float]
    response_times: dict[str, float]
    cache_hit_rate: float
    active_connections: int
    queue_size: int


@dataclass
class CacheConfig:
    ttl: int = 300  # 5 minutes default
    max_size: int = 1000
    enable_compression: bool = True
    enable_stats: bool = True


class AdvancedCache:
    """Advanced caching system with multiple backends"""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis_client = None
        self.memory_cache = {}
        self.cache_stats = {"hits": 0, "misses": 0, "sets": 0, "deletes": 0}
        self.config = CacheConfig()

    async def initialize(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = await aioredis.from_url(self.redis_url)
            logger.info("Redis cache initialized successfully")
        except Exception as e:
            logger.warning(f"Redis cache initialization failed: {e}")

    async def get(self, key: str) -> Any | None:
        """Get value from cache"""
        try:
            # Try memory cache first
            if key in self.memory_cache:
                item = self.memory_cache[key]
                if item["expires_at"] > datetime.now(timezone.utc):
                    self.cache_stats["hits"] += 1
                    return item["value"]
                else:
                    del self.memory_cache[key]

            # Try Redis cache
            if self.redis_client:
                value = await self.redis_client.get(key)
                if value:
                    self.cache_stats["hits"] += 1
                    # Store in memory cache for faster access
                    self.memory_cache[key] = {
                        "value": value,
                        "expires_at": (datetime.now(timezone.utc) + timedelta(seconds=60)),
                    }
                    return value

            self.cache_stats["misses"] += 1
            return None

        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None

    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set value in cache"""
        try:
            ttl = ttl or self.config.ttl

            # Set in memory cache
            self.memory_cache[key] = {
                "value": value,
                "expires_at": (datetime.now(timezone.utc) + timedelta(seconds=ttl)),
            }

            # Set in Redis cache
            if self.redis_client:
                await self.redis_client.setex(key, ttl, value)

            self.cache_stats["sets"] += 1

            # Clean up memory cache if too large
            if len(self.memory_cache) > self.config.max_size:
                self._cleanup_memory_cache()

            return True

        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        try:
            # Remove from memory cache
            if key in self.memory_cache:
                del self.memory_cache[key]

            # Remove from Redis cache
            if self.redis_client:
                await self.redis_client.delete(key)

            self.cache_stats["deletes"] += 1
            return True

        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False

    def _cleanup_memory_cache(self):
        """Clean up expired items from memory cache"""
        now = datetime.now(timezone.utc)
        expired_keys = [key for key, item in self.memory_cache.items() if item["expires_at"] <= now]

        for key in expired_keys:
            del self.memory_cache[key]

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = self.cache_stats["hits"] / total_requests if total_requests > 0 else 0

        return {
            "hits": self.cache_stats["hits"],
            "misses": self.cache_stats["misses"],
            "hit_rate": hit_rate,
            "sets": self.cache_stats["sets"],
            "deletes": self.cache_stats["deletes"],
            "memory_cache_size": len(self.memory_cache),
            "redis_connected": self.redis_client is not None,
        }


class ConnectionPool:
    """Connection pooling for external services"""

    def __init__(self):
        self.http_session: aiohttp.ClientSession | None = None
        self.redis_pool: aioredis.ConnectionPool | None = None
        self.db_pool = None
        self.max_connections = 100
        self.connection_timeout = 30

    async def initialize(self):
        """Initialize connection pools"""
        try:
            # HTTP session with connection pooling
            connector = aiohttp.TCPConnector(
                limit=self.max_connections,
                limit_per_host=20,
                ttl_dns_cache=300,
                use_dns_cache=True,
            )

            timeout = aiohttp.ClientTimeout(total=self.connection_timeout)
            self.http_session = aiohttp.ClientSession(connector=connector, timeout=timeout)

            logger.info("HTTP connection pool initialized")

        except Exception as e:
            logger.error(f"Connection pool initialization failed: {e}")

    async def get_http_session(self) -> aiohttp.ClientSession | None:
        """Get HTTP session"""
        return self.http_session

    async def close(self):
        """Close all connections"""
        if self.http_session:
            await self.http_session.close()

        if self.redis_pool:
            await self.redis_pool.disconnect()


class AsyncTaskQueue:
    """Asynchronous task queue for background processing"""

    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers
        self.task_queue = asyncio.Queue()
        self.result_queue = asyncio.Queue()
        self.workers: list[asyncio.Task] = []
        self.is_running = False
        self.stats = {
            "tasks_processed": 0,
            "tasks_failed": 0,
            "average_processing_time": 0.0,
        }

    async def start(self):
        """Start worker tasks"""
        self.is_running = True
        for _ in range(self.max_workers):
            worker = asyncio.create_task(self._worker())
            self.workers.append(worker)

        logger.info(f"Started {self.max_workers} async workers")

    async def stop(self):
        """Stop worker tasks"""
        self.is_running = False

        # Cancel all workers
        for worker in self.workers:
            worker.cancel()

        # Wait for workers to finish
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers.clear()

        logger.info("Stopped all async workers")

    async def add_task(self, task_func: Callable, *args, **kwargs) -> str:
        """Add task to queue"""
        task_id = f"task_{int(time.time() * 1000)}"
        task_data = {
            "id": task_id,
            "func": task_func,
            "args": args,
            "kwargs": kwargs,
            "created_at": datetime.now(timezone.utc),
        }

        await self.task_queue.put(task_data)
        return task_id

    async def get_result(self, task_id: str, timeout: float = 30.0) -> Any:
        """Get task result"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                result = await asyncio.wait_for(self.result_queue.get(), timeout=1.0)
                if result["task_id"] == task_id:
                    return result["result"]
            except asyncio.TimeoutError:
                continue

        raise TimeoutError(f"Task {task_id} result not available within {timeout} seconds")

    async def _worker(self):
        """Worker task"""
        while self.is_running:
            try:
                task_data = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)

                start_time = time.time()

                # Execute task
                if asyncio.iscoroutinefunction(task_data["func"]):
                    result = await task_data["func"](*task_data["args"], **task_data["kwargs"])
                else:
                    # Run sync function in thread pool
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None,
                        task_data["func"],
                        *task_data["args"],
                        **task_data["kwargs"],
                    )

                processing_time = time.time() - start_time

                # Update stats
                self.stats["tasks_processed"] += 1
                self.stats["average_processing_time"] = (
                    self.stats["average_processing_time"] * (self.stats["tasks_processed"] - 1)
                    + processing_time
                ) / self.stats["tasks_processed"]

                # Put result
                await self.result_queue.put(
                    {
                        "task_id": task_data["id"],
                        "result": result,
                        "processing_time": processing_time,
                    }
                )

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Worker error: {e}")
                self.stats["tasks_failed"] += 1

    def get_stats(self) -> dict[str, Any]:
        """Get queue statistics"""
        return {
            **self.stats,
            "queue_size": self.task_queue.qsize(),
            "active_workers": len(self.workers),
            "is_running": self.is_running,
        }


class PerformanceMonitor:
    """Real-time performance monitoring"""

    def __init__(self):
        self.metrics_history: list[PerformanceMetrics] = []
        self.max_history_size = 1000
        self.monitoring_interval = 60  # seconds
        self.is_monitoring = False
        self.monitor_task: asyncio.Task | None = None

    async def start_monitoring(self):
        """Start performance monitoring"""
        self.is_monitoring = True
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Performance monitoring started")

    async def stop_monitoring(self):
        """Stop performance monitoring"""
        self.is_monitoring = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Performance monitoring stopped")

    async def _monitor_loop(self):
        """Performance monitoring loop"""
        while self.is_monitoring:
            try:
                metrics = await self._collect_metrics()
                self.metrics_history.append(metrics)

                # Keep history size manageable
                if len(self.metrics_history) > self.max_history_size:
                    self.metrics_history.pop(0)

                await asyncio.sleep(self.monitoring_interval)

            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(self.monitoring_interval)

    async def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics"""
        # CPU usage
        cpu_usage = psutil.cpu_percent(interval=1)

        # Memory usage
        memory = psutil.virtual_memory()
        memory_usage = memory.percent

        # Disk I/O
        disk_io = psutil.disk_io_counters()
        disk_io_metrics = {
            "read_bytes_per_sec": disk_io.read_bytes if disk_io else 0,
            "write_bytes_per_sec": disk_io.write_bytes if disk_io else 0,
        }

        # Network I/O
        network_io = psutil.net_io_counters()
        network_io_metrics = {
            "bytes_sent_per_sec": network_io.bytes_sent if network_io else 0,
            "bytes_recv_per_sec": network_io.bytes_recv if network_io else 0,
        }

        # Response times (placeholder - would be collected from actual endpoints)
        response_times = {
            "/api/v1/trading": 0.1,
            "/api/v1/portfolio": 0.05,
            "/api/v1/market-data": 0.02,
        }

        # Cache hit rate (placeholder - would be collected from cache)
        cache_hit_rate = 0.85

        # Active connections (placeholder - would be collected from connection pools)
        active_connections = 10

        # Queue size (placeholder - would be collected from task queues)
        queue_size = 5

        return PerformanceMetrics(
            timestamp=datetime.now(timezone.utc),
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            disk_io=disk_io_metrics,
            network_io=network_io_metrics,
            response_times=response_times,
            cache_hit_rate=cache_hit_rate,
            active_connections=active_connections,
            queue_size=queue_size,
        )

    def get_current_metrics(self) -> PerformanceMetrics | None:
        """Get most recent metrics"""
        return self.metrics_history[-1] if self.metrics_history else None

    def get_metrics_history(self, hours: int = 24) -> list[PerformanceMetrics]:
        """Get metrics history for specified hours"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        return [metrics for metrics in self.metrics_history if metrics.timestamp >= cutoff_time]

    def get_performance_summary(self) -> dict[str, Any]:
        """Get performance summary"""
        if not self.metrics_history:
            return {}

        recent_metrics = self.get_metrics_history(1)  # Last hour

        if not recent_metrics:
            return {}

        cpu_values = [m.cpu_usage for m in recent_metrics]
        memory_values = [m.memory_usage for m in recent_metrics]

        return {
            "cpu": {
                "current": cpu_values[-1],
                "average": sum(cpu_values) / len(cpu_values),
                "max": max(cpu_values),
                "min": min(cpu_values),
            },
            "memory": {
                "current": memory_values[-1],
                "average": sum(memory_values) / len(memory_values),
                "max": max(memory_values),
                "min": min(memory_values),
            },
            "performance_score": self._calculate_performance_score(recent_metrics),
            "alerts": self._check_performance_alerts(recent_metrics),
        }

    def _calculate_performance_score(self, metrics: list[PerformanceMetrics]) -> float:
        """Calculate overall performance score (0-100)"""
        if not metrics:
            return 0.0

        # Calculate scores for different metrics
        cpu_score = 100.0 - max(m.cpu_usage for m in metrics)
        memory_score = 100.0 - max(m.memory_usage for m in metrics)
        response_score = 100.0 - min(
            100.0, max(max(m.response_times.values()) for m in metrics) * 1000
        )  # Convert to ms

        # Weighted average
        return cpu_score * 0.4 + memory_score * 0.3 + response_score * 0.3

    def _check_performance_alerts(self, metrics: list[PerformanceMetrics]) -> list[dict[str, Any]]:
        """Check for performance alerts"""
        alerts = []

        if not metrics:
            return alerts

        latest = metrics[-1]

        # CPU alert
        if latest.cpu_usage > 80:
            alerts.append(
                {
                    "type": "high_cpu",
                    "severity": "warning",
                    "message": f"High CPU usage: {latest.cpu_usage:.1f}%",
                    "timestamp": latest.timestamp,
                }
            )

        # Memory alert
        if latest.memory_usage > 85:
            alerts.append(
                {
                    "type": "high_memory",
                    "severity": "warning",
                    "message": (f"High memory usage: {latest.memory_usage:.1f}%"),
                    "timestamp": latest.timestamp,
                }
            )

        # Response time alert
        for endpoint, response_time in latest.response_times.items():
            if response_time > 1.0:  # More than 1 second
                alerts.append(
                    {
                        "type": "slow_response",
                        "severity": "warning",
                        "message": (f"Slow response time for {endpoint}: {response_time:.2f}s"),
                        "timestamp": latest.timestamp,
                    }
                )

        return alerts


class PerformanceOptimizer:
    """Main performance optimization system"""

    def __init__(self):
        self.cache = AdvancedCache()
        self.connection_pool = ConnectionPool()
        self.task_queue = AsyncTaskQueue()
        self.performance_monitor = PerformanceMonitor()
        self.optimization_enabled = True

    async def initialize(self):
        """Initialize all performance components"""
        try:
            await self.cache.initialize()
            await self.connection_pool.initialize()
            await self.task_queue.start()
            await self.performance_monitor.start_monitoring()

            logger.info("Performance optimizer initialized successfully")

        except Exception as e:
            logger.error(f"Performance optimizer initialization failed: {e}")

    async def shutdown(self):
        """Shutdown all performance components"""
        try:
            await self.task_queue.stop()
            await self.connection_pool.close()
            await self.performance_monitor.stop_monitoring()

            logger.info("Performance optimizer shutdown completed")

        except Exception as e:
            logger.error(f"Performance optimizer shutdown error: {e}")

    def cache_decorator(self, ttl: int | None = None):
        """Decorator for caching function results"""

        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                if not self.optimization_enabled:
                    return await func(*args, **kwargs)

                # Generate cache key
                cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"

                # Try to get from cache
                cached_result = await self.cache.get(cache_key)
                if cached_result is not None:
                    return cached_result

                # Execute function and cache result
                result = await func(*args, **kwargs)
                await self.cache.set(cache_key, result, ttl)

                return result

            return wrapper

        return decorator

    def async_task_decorator(self):
        """Decorator for running functions as async tasks"""

        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                if not self.optimization_enabled:
                    return await func(*args, **kwargs)

                # Add to task queue
                task_id = await self.task_queue.add_task(func, *args, **kwargs)

                # Wait for result
                result = await self.task_queue.get_result(task_id)
                return result

            return wrapper

        return decorator

    def get_optimization_status(self) -> dict[str, Any]:
        """Get optimization status and statistics"""
        return {
            "optimization_enabled": self.optimization_enabled,
            "cache_stats": self.cache.get_stats(),
            "task_queue_stats": self.task_queue.get_stats(),
            "performance_summary": (self.performance_monitor.get_performance_summary()),
            "current_metrics": self.performance_monitor.get_current_metrics(),
        }

    def enable_optimization(self):
        """Enable performance optimizations"""
        self.optimization_enabled = True
        logger.info("Performance optimizations enabled")

    def disable_optimization(self):
        """Disable performance optimizations"""
        self.optimization_enabled = False
        logger.info("Performance optimizations disabled")


# Global performance optimizer instance
performance_optimizer = PerformanceOptimizer()

