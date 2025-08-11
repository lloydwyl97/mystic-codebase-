import redis
from typing import Optional, List, Dict, Any
import time
import logging
import functools
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)


def log_event(
    event_type: str,
    event_data: Dict[str, Any],
    level: str = "info",
    redis_client: Optional[redis.Redis] = None,
):
    """
    Log an event with structured data and optional Redis storage.

    Args:
        event_type: Type of event (e.g., 'trade_executed', 'signal_generated')
        event_data: Dictionary containing event data
        level: Log level ('debug', 'info', 'warning', 'error')
        redis_client: Optional Redis client for distributed logging
    """
    timestamp = datetime.now().isoformat()
    log_entry = {
        "timestamp": timestamp,
        "event_type": event_type,
        "level": level,
        "data": event_data,
    }

    # Log to standard logger
    log_message = f"[EVENT] {event_type}: {event_data}"
    if level == "debug":
        logger.debug(log_message)
    elif level == "info":
        logger.info(log_message)
    elif level == "warning":
        logger.warning(log_message)
    elif level == "error":
        logger.error(log_message)
    else:
        logger.info(log_message)

    # Store in Redis if available
    if redis_client:
        try:
            redis_key = f"events:{event_type}:{timestamp}"
            redis_client.setex(redis_key, 86400, str(log_entry))  # TTL 24 hours
        except Exception as e:
            logger.warning(f"Failed to store event in Redis: {e}")


def log_operation_performance(operation_name=None):
    """
    Decorator to log the performance (execution time) and exceptions of a function.
    Works for both sync and async functions.
    """

    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)
            op = operation_name or func.__name__
            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                logger.info(f"[PERF] {op} completed in {elapsed:.4f}s")
                return result
            except Exception as e:
                elapsed = time.perf_counter() - start
                logger.error(f"[PERF] {op} failed after {elapsed:.4f}s: {e}")
                raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)
            op = operation_name or func.__name__
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                logger.info(f"[PERF] {op} completed in {elapsed:.4f}s")
                return result
            except Exception as e:
                elapsed = time.perf_counter() - start
                logger.error(f"[PERF] {op} failed after {elapsed:.4f}s: {e}")
                raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


class EnhancedLogger:
    """Enhanced logger with Redis integration for distributed logging."""

    def __init__(
        self,
        redis_client: Optional[redis.Redis] = None,
        key_prefix: str = "logs",
        max_logs: int = 1000,
    ):
        """
        Initialize enhanced logger.

        Args:
            redis_client: Redis client for distributed logging
            key_prefix: Prefix for Redis keys
            max_logs: Maximum number of logs to keep in Redis
        """
        self.redis_client = redis_client or redis.Redis()
        self.key_prefix = key_prefix
        self.max_logs = max_logs
        self.buffer_size = 100
        self.flush_interval = 60  # seconds
        self.log_buffer: List[Dict[str, Any]] = []
        self.last_flush = time.time()

        # Initialize Redis if available
        if self.redis_client:
            try:
                # Test Redis connection - handle both sync and async clients
                try:
                    self.redis_client.ping()
                except TypeError:
                    # Handle async Redis client
                    pass
                logger.info("Enhanced logger initialized with Redis support")
            except Exception as e:
                logger.warning(f"Redis not available for enhanced logging: {e}")
                self.redis_client = None
        else:
            logger.info("Enhanced logger initialized without Redis (local only)")
