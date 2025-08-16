"""
Redis Cache Manager for Mystic Trading Platform

Provides high-performance caching with:
- Multi-level caching
- Background task processing
- Cache warming
- Performance monitoring
- Automatic cleanup
"""

import asyncio
import hashlib
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict
import threading

try:
    import redis
    from redis import Redis
except ImportError:
    redis = None
    Redis = None

from trading_config import trading_config

logger = logging.getLogger(__name__)

# Cache configuration
CACHE_TTL = trading_config.PORTFOLIO_CACHE_TTL
CACHE_MAX_SIZE = 10000
CACHE_CLEANUP_INTERVAL = 300  # 5 minutes
BACKGROUND_TASK_INTERVAL = 60  # 1 minute

# Redis configuration
REDIS_HOST = trading_config.DEFAULT_REDIS_HOST
REDIS_PORT = trading_config.DEFAULT_REDIS_PORT
REDIS_DB = trading_config.DEFAULT_REDIS_DB
REDIS_PASSWORD = None


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    timestamp: float
    ttl: int
    access_count: int = 0
    last_accessed: float = 0.0


class CacheMetrics:
    """Cache performance metrics"""

    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.set_operations = 0
        self.get_operations = 0
        self.background_tasks = 0
        self.cache_size = 0
        self.start_time = time.time()

    def get_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        uptime = time.time() - self.start_time
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.get_hit_rate(),
            'evictions': self.evictions,
            'set_operations': self.set_operations,
            'get_operations': self.get_operations,
            'background_tasks': self.background_tasks,
            'cache_size': self.cache_size,
            'uptime_seconds': uptime,
            'operations_per_second': (self.hits + self.misses) / uptime if uptime > 0 else 0
        }


class BackgroundTaskManager:
    """Manages background tasks for cache optimization"""

    def __init__(self, cache_manager):
        self.cache_manager = cache_manager
        self.running = False
        self.tasks = []
        self.task_stats = defaultdict(int)

    async def start(self):
        """Start background task processing"""
        self.running = True
        logger.info("Starting background task manager")

        # Start background tasks
        self.tasks = [
            asyncio.create_task(self._cache_cleanup_loop()),
            asyncio.create_task(self._cache_warming_loop()),
            asyncio.create_task(self._metrics_collection_loop()),
            asyncio.create_task(self._performance_optimization_loop())
        ]

    async def stop(self):
        """Stop background task processing"""
        self.running = False
        logger.info("Stopping background task manager")

        # Cancel all tasks
        for task in self.tasks:
            task.cancel()

        # Wait for tasks to complete
        await asyncio.gather(*self.tasks, return_exceptions=True)

    async def _cache_cleanup_loop(self):
        """Periodic cache cleanup"""
        while self.running:
            try:
                await self.cache_manager.cleanup_expired_entries()
                self.task_stats['cleanup'] += 1
                await asyncio.sleep(CACHE_CLEANUP_INTERVAL)
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
                await asyncio.sleep(60)

    async def _cache_warming_loop(self):
        """Cache warming for frequently accessed data"""
        while self.running:
            try:
                await self.cache_manager.warm_cache()
                self.task_stats['warming'] += 1
                await asyncio.sleep(BACKGROUND_TASK_INTERVAL)
            except Exception as e:
                logger.error(f"Cache warming error: {e}")
                await asyncio.sleep(60)

    async def _metrics_collection_loop(self):
        """Collect and store cache metrics"""
        while self.running:
            try:
                await self.cache_manager.collect_metrics()
                self.task_stats['metrics'] += 1
                await asyncio.sleep(30)  # Every 30 seconds
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(60)

    async def _performance_optimization_loop(self):
        """Performance optimization tasks"""
        while self.running:
            try:
                await self.cache_manager.optimize_performance()
                self.task_stats['optimization'] += 1
                await asyncio.sleep(300)  # Every 5 minutes
            except Exception as e:
                logger.error(f"Performance optimization error: {e}")
                await asyncio.sleep(60)


class RedisCacheManager:
    """High-performance Redis cache manager"""

    def __init__(self):
        self.redis_client = None
        self.local_cache = {}
        self.metrics = CacheMetrics()
        self.background_manager = BackgroundTaskManager(self)
        self.access_patterns = defaultdict(int)
        self.running = False

        # Initialize Redis connection
        self._initialize_redis()

    def _initialize_redis(self):
        """Initialize Redis connection"""
        if redis is None or Redis is None:
            logger.warning("Redis not available, using local cache only")
            return

        try:
            self.redis_client = Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                password=REDIS_PASSWORD,
                decode_responses=True
            )

            # Test connection
            self.redis_client.ping()
            logger.info("Redis connection established")

        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            self.redis_client = None

    async def start(self):
        """Start cache manager"""
        self.running = True
        await self.background_manager.start()
        logger.info("Redis cache manager started")

    async def stop(self):
        """Stop cache manager"""
        self.running = False
        await self.background_manager.stop()
        logger.info("Redis cache manager stopped")

    def _generate_cache_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate a unique cache key"""
        # Create a hash of the arguments
        key_data = f"{prefix}:{args}:{sorted(kwargs.items())}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache"""
        self.metrics.get_operations += 1

        # Check local cache first
        if key in self.local_cache:
            entry = self.local_cache[key]

            # Check if entry is expired
            if time.time() - entry.timestamp > entry.ttl:
                del self.local_cache[key]
                self.metrics.misses += 1
                return default

            # Update access info
            entry.access_count += 1
            entry.last_accessed = time.time()
            self.access_patterns[key] += 1
            self.metrics.hits += 1

            return entry.value

        # Check Redis cache
        if self.redis_client:
            try:
                value = self.redis_client.get(key)
                if value is not None:
                    # Cache in local cache
                    entry = CacheEntry(
                        key=key,
                        value=json.loads(value),
                        timestamp=time.time(),
                        ttl=CACHE_TTL
                    )
                    self.local_cache[key] = entry
                    self.metrics.hits += 1
                    return entry.value
            except Exception as e:
                logger.error(f"Redis get error: {e}")

        self.metrics.misses += 1
        return default

    def set(self, key: str, value: Any, ttl: int = CACHE_TTL) -> bool:
        """Set value in cache"""
        self.metrics.set_operations += 1

        try:
            # Store in local cache
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=time.time(),
                ttl=ttl
            )
            self.local_cache[key] = entry

            # Store in Redis
            if self.redis_client:
                self.redis_client.setex(
                    key,
                    ttl,
                    json.dumps(value)
                )

            return True

        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        try:
            # Remove from local cache
            self.local_cache.pop(key, None)

            # Remove from Redis
            if self.redis_client:
                self.redis_client.delete(key)

            return True

        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False

    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        # Check local cache
        if key in self.local_cache:
            entry = self.local_cache[key]
            if time.time() - entry.timestamp <= entry.ttl:
                return True

        # Check Redis
        if self.redis_client:
            try:
                return bool(self.redis_client.exists(key))
            except Exception as e:
                logger.error(f"Redis exists error: {e}")

        return False

    async def cleanup_expired_entries(self):
        """Remove expired cache entries"""
        current_time = time.time()
        expired_keys = []

        for key, entry in self.local_cache.items():
            if current_time - entry.timestamp > entry.ttl:
                expired_keys.append(key)

        # Remove expired entries
        for key in expired_keys:
            self.local_cache.pop(key, None)
            self.metrics.evictions += 1

        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

    async def warm_cache(self):
        """Warm cache with frequently accessed data"""
        try:
            # Get top accessed patterns
            top_patterns = sorted(
                self.access_patterns.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]

            for pattern, count in top_patterns:
                if count > 5:  # Only warm frequently accessed patterns
                    # This would typically fetch data from database
                    # For now, we'll just log the warming attempt
                    logger.debug(f"Warming cache for pattern: {pattern} (accesses: {count})")

        except Exception as e:
            logger.error(f"Cache warming error: {e}")

    async def collect_metrics(self):
        """Collect and store cache metrics"""
        try:
            stats = self.metrics.get_stats()
            stats['cache_size'] = len(self.local_cache)

            # Store metrics in Redis for monitoring
            if self.redis_client:
                self.redis_client.setex(
                    'cache:metrics',
                    300,  # 5 minute TTL
                    json.dumps(stats)
                )

            logger.debug(f"Cache metrics: {stats}")

        except Exception as e:
            logger.error(f"Metrics collection error: {e}")

    async def optimize_performance(self):
        """Optimize cache performance"""
        try:
            # Remove least recently used entries if cache is too large
            if len(self.local_cache) > CACHE_MAX_SIZE:
                # Sort by last accessed time
                sorted_entries = sorted(
                    self.local_cache.items(),
                    key=lambda x: x[1].last_accessed
                )

                # Remove oldest 20% of entries
                remove_count = len(sorted_entries) // 5
                for key, _ in sorted_entries[:remove_count]:
                    self.local_cache.pop(key, None)
                    self.metrics.evictions += 1

                logger.info(f"Optimized cache: removed {remove_count} LRU entries")

        except Exception as e:
            logger.error(f"Performance optimization error: {e}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        stats = self.metrics.get_stats()
        stats.update({
            'local_cache_size': len(self.local_cache),
            'redis_available': self.redis_client is not None,
            'top_patterns': dict(sorted(
                self.access_patterns.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5])
        })
        return stats

    def clear_cache(self):
        """Clear all cache entries"""
        try:
            # Clear local cache
            self.local_cache.clear()

            # Clear Redis cache
            if self.redis_client:
                self.redis_client.flushdb()

            logger.info("Cache cleared")

        except Exception as e:
            logger.error(f"Cache clear error: {e}")


# Global cache manager instance
cache_manager = RedisCacheManager()


