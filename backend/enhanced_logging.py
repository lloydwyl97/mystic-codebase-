"""
Enhanced Logging System for Mystic Trading

Provides advanced logging capabilities with Redis integration for distributed logging.
"""

import asyncio
import json
import logging
import logging.handlers
import os
import time
import traceback
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Union

try:
    import redis
except ImportError:
    redis = None

logger = logging.getLogger(__name__)


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging"""

    def format(self, record: logging.LogRecord) -> str:
        # Create structured log entry
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }

        # Add extra fields if present
        if hasattr(record, "extra_fields"):
            extra_fields = getattr(record, "extra_fields", {})
            if isinstance(extra_fields, dict):
                log_entry.update(extra_fields)

        # Add performance metrics if present
        if hasattr(record, "duration"):
            duration = getattr(record, "duration", None)
            if duration is not None:
                log_entry["duration_ms"] = duration

        return json.dumps(log_entry)


class PerformanceLogger:
    """Performance logging decorator and utilities"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def log_performance(self, operation: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator to log operation performance"""

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    duration = (time.time() - start_time) * 1000  # Convert to milliseconds

                    # Create log record with performance data
                    record = self.logger.makeRecord(
                        self.logger.name,
                        logging.INFO,
                        "",
                        0,
                        f"Operation '{operation}' completed successfully",
                        (),
                        None,
                    )
                    record.duration = duration
                    record.extra_fields = {
                        "operation": operation,
                        "status": "success",
                        "duration_ms": duration,
                    }

                    self.logger.handle(record)
                    return result

                except Exception as e:
                    duration = (time.time() - start_time) * 1000

                    # Create log record with error and performance data
                    record = self.logger.makeRecord(
                        self.logger.name,
                        logging.ERROR,
                        "",
                        0,
                        f"Operation '{operation}' failed: {str(e)}",
                        (),
                        (type(e), e, e.__traceback__),
                    )
                    record.duration = duration
                    record.extra_fields = {
                        "operation": operation,
                        "status": "error",
                        "duration_ms": duration,
                        "error": str(e),
                    }

                    self.logger.handle(record)
                    raise

            @wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    duration = (time.time() - start_time) * 1000

                    # Create log record with performance data
                    record = self.logger.makeRecord(
                        self.logger.name,
                        logging.INFO,
                        "",
                        0,
                        f"Operation '{operation}' completed successfully",
                        (),
                        None,
                    )
                    record.duration = duration
                    record.extra_fields = {
                        "operation": operation,
                        "status": "success",
                        "duration_ms": duration,
                    }

                    self.logger.handle(record)
                    return result

                except Exception as e:
                    duration = (time.time() - start_time) * 1000

                    # Create log record with error and performance data
                    record = self.logger.makeRecord(
                        self.logger.name,
                        logging.ERROR,
                        "",
                        0,
                        f"Operation '{operation}' failed: {str(e)}",
                        (),
                        (type(e), e, e.__traceback__),
                    )
                    record.duration = duration
                    record.extra_fields = {
                        "operation": operation,
                        "status": "error",
                        "duration_ms": duration,
                        "error": str(e),
                    }

                    self.logger.handle(record)
                    raise

            # Return appropriate wrapper based on function type
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper

        return decorator

    def log_event(
        self,
        event_type: str,
        details: Dict[str, Any],
        level: int = logging.INFO,
    ):
        """Log a structured event"""
        record = self.logger.makeRecord(
            self.logger.name, level, "", 0, f"Event: {event_type}", (), None
        )
        record.extra_fields = {
            "event_type": event_type,
            "event_details": details,
        }
        self.logger.handle(record)


class EnhancedLogger:
    """Enhanced logger with Redis integration for distributed logging."""

    def __init__(
        self,
        redis_client: Optional[redis.Redis],
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
        self.redis_client = redis_client
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

    def log(self, level: str, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a message with enhanced features.

        Args:
            level: Log level (debug, info, warning, error, critical)
            message: Log message
            extra: Additional data to include in log
        """
        timestamp = datetime.now(timezone.timezone.utc).isoformat()

        log_entry = {
            "timestamp": timestamp,
            "level": level.upper(),
            "message": message,
            "extra": extra or {},
        }

        # Add to buffer
        self.log_buffer.append(log_entry)

        # Flush buffer if full or enough time has passed
        if (
            len(self.log_buffer) >= self.buffer_size
            or time.time() - self.last_flush >= self.flush_interval
        ):
            self._flush_buffer()

    def _flush_buffer(self) -> None:
        """Flush log buffer to Redis."""
        if not self.log_buffer or not self.redis_client:
            return

        try:
            # Create pipeline for batch operations
            pipe = self.redis_client.pipeline()

            for log_entry in self.log_buffer:
                # Store individual log entry
                log_key = f"{self.key_prefix}:{log_entry['timestamp']}"
                pipe.setex(log_key, 86400, json.dumps(log_entry))  # 24 hour TTL

                # Add to sorted set for time-based queries
                score = time.mktime(
                    datetime.fromisoformat(
                        log_entry["timestamp"].replace("Z", "+00:00")
                    ).timetuple()
                )
                pipe.zadd(f"{self.key_prefix}:sorted", {log_key: score})

                # Add to level-based sets
                level_key = f"{self.key_prefix}:level:{log_entry['level']}"
                pipe.sadd(level_key, log_key)
                pipe.expire(level_key, 86400)  # 24 hour TTL

            # Limit sorted set size
            pipe.zremrangebyrank(f"{self.key_prefix}:sorted", 0, -self.max_logs - 1)

            # Execute pipeline
            pipe.execute()

            logger.debug(f"Flushed {len(self.log_buffer)} log entries to Redis")

        except Exception as e:
            logger.error(f"Failed to flush logs to Redis: {e}")
        finally:
            self.log_buffer.clear()
            self.last_flush = time.time()

    async def get_logs(
        self,
        level: Optional[str] = None,
        limit: int = 100,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve logs from Redis.

        Args:
            level: Filter by log level
            limit: Maximum number of logs to return
            start_time: Start time filter (ISO format)
            end_time: End time filter (ISO format)

        Returns:
            List of log entries
        """
        if not self.redis_client:
            return []

        try:
            if level:
                # Get logs by level
                level_key = f"{self.key_prefix}:level:{level.upper()}"
                smembers_result = self.redis_client.smembers(level_key)
                # Handle both sync and async Redis responses
                if hasattr(smembers_result, "__await__"):
                    log_keys = await smembers_result
                else:
                    log_keys = smembers_result
            else:
                # Get all logs from sorted set
                zrevrange_result = self.redis_client.zrevrange(
                    f"{self.key_prefix}:sorted", 0, limit - 1
                )
                # Handle both sync and async Redis responses
                if hasattr(zrevrange_result, "__await__"):
                    log_keys = await zrevrange_result
                else:
                    log_keys = zrevrange_result

            logs: List[Dict[str, Any]] = []
            # Handle both sync and async Redis responses
            try:
                # Ensure log_keys is iterable
                if hasattr(log_keys, "__await__"):
                    # If it's awaitable, we can't iterate directly
                    logger.warning("Async Redis response not fully supported in this context")
                    return logs

                # Try to iterate directly (sync response)
                for key in log_keys:
                    try:
                        # Cast key to string to resolve type issues
                        key_str = str(key) if key is not None else ""
                        get_result = self.redis_client.get(key_str)
                        # Handle both sync and async Redis responses
                        if hasattr(get_result, "__await__"):
                            log_data = await get_result
                        else:
                            log_data = get_result

                        if log_data:
                            # Handle Redis response which could be bytes or string
                            if isinstance(log_data, bytes):
                                log_str = log_data.decode("utf-8")
                            else:
                                log_str = str(log_data)
                            log_entry = json.loads(log_str)

                            # Apply time filters
                            if start_time and log_entry["timestamp"] < start_time:
                                continue
                            if end_time and log_entry["timestamp"] > end_time:
                                continue

                            logs.append(log_entry)

                            if len(logs) >= limit:
                                break

                    except Exception as e:
                        logger.warning(f"Failed to parse log entry {key}: {e}")
            except TypeError:
                # Handle async response - skip iteration for now
                pass

            return logs

        except Exception as e:
            logger.error(f"Failed to retrieve logs from Redis: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """Get logging statistics."""
        if not self.redis_client:
            return {"redis_available": False}

        try:
            stats: Dict[str, Any] = {
                "redis_available": True,
                "buffer_size": len(self.log_buffer),
                "total_logs": self.redis_client.zcard(f"{self.key_prefix}:sorted"),
                "levels": {},
            }

            # Get counts by level
            for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
                level_key = f"{self.key_prefix}:level:{level}"
                count = self.redis_client.scard(level_key)
                stats["levels"][level] = count

            return stats

        except Exception as e:
            logger.error(f"Failed to get logging stats: {e}")
            return {"redis_available": False, "error": str(e)}

    def clear_logs(self, level: Optional[str] = None) -> bool:
        """
        Clear logs from Redis.

        Args:
            level: Clear only specific level (if None, clear all)

        Returns:
            True if successful, False otherwise
        """
        if not self.redis_client:
            return False

        try:
            if level:
                # Clear specific level
                level_key = f"{self.key_prefix}:level:{level.upper()}"
                log_keys = self.redis_client.smembers(level_key)

                if log_keys:
                    pipe = self.redis_client.pipeline()
                    # Handle both sync and async Redis responses
                    try:
                        # Ensure log_keys is iterable
                        if hasattr(log_keys, "__await__"):
                            # If it's awaitable, we can't iterate directly
                            logger.warning("Async Redis response not fully supported in clear_logs")
                            return False

                        # Try to iterate directly (sync response)
                        for key in log_keys:
                            # Cast key to string to resolve type issues
                            key_str = str(key) if key is not None else ""
                            pipe.delete(key_str)
                            pipe.zrem(f"{self.key_prefix}:sorted", key_str)
                    except TypeError:
                        # Handle async response - skip iteration for now
                        pass
                    pipe.delete(level_key)
                    pipe.execute()
            else:
                # Clear all logs
                pattern = f"{self.key_prefix}:*"
                keys = self.redis_client.keys(pattern)
                if keys:
                    self.redis_client.delete(*keys)

            return True

        except Exception as e:
            logger.error(f"Failed to clear logs: {e}")
            return False


class RedisLogHandler(logging.Handler):
    """Redis-based log handler for centralized logging"""

    def __init__(self, redis_client: redis.Redis, key_prefix: str = "logs"):
        super().__init__()
        self.redis_client = redis_client
        self.key_prefix = key_prefix

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record to Redis"""
        try:
            log_entry = {
                "timestamp": record.created,
                "level": record.levelname,
                "message": record.getMessage(),
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno,
            }

            # Add extra fields if present
            if hasattr(record, "extra_fields"):
                log_entry.update(record.extra_fields)

            # Store in Redis with timestamp as score for sorting
            key = f"{self.key_prefix}:{record.levelname.lower()}"
            self.redis_client.zadd(key, {str(log_entry): record.created})

            # Keep only last 1000 entries per level
            self.redis_client.zremrangebyrank(key, 0, -1001)

        except Exception as e:
            # Fallback to console if Redis fails
            print(f"Redis logging failed: {e}")


# Global enhanced logger instance
_enhanced_logger: Optional[EnhancedLogger] = None


def get_enhanced_logger(
    redis_client: Optional[redis.Redis] = None,
) -> EnhancedLogger:
    """Get the global enhanced logger instance."""
    global _enhanced_logger

    if _enhanced_logger is None:
        _enhanced_logger = EnhancedLogger(redis_client)

    return _enhanced_logger


def log_event(level: str, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
    """Log an event using the global enhanced logger."""
    logger = get_enhanced_logger()
    logger.log(level, message, extra)


# Convenience functions
def log_info(message: str, extra: Optional[Dict[str, Any]] = None) -> None:
    """Log info message."""
    log_event("info", message, extra)


def log_warning(message: str, extra: Optional[Dict[str, Any]] = None) -> None:
    """Log warning message."""
    log_event("warning", message, extra)


def log_error(message: str, extra: Optional[Dict[str, Any]] = None) -> None:
    """Log error message."""
    log_event("error", message, extra)


def log_critical(message: str, extra: Optional[Dict[str, Any]] = None) -> None:
    """Log critical message."""
    log_event("critical", message, extra)


def log_debug(message: str, extra: Optional[Dict[str, Any]] = None) -> None:
    """Log debug message."""
    log_event("debug", message, extra)


def setup_enhanced_logging(
    redis_client: Union[redis.Redis, None],
    log_level: str = "INFO",
    log_file: str = "mystic_trading.log",
    enable_redis_logging: bool = True,
) -> Dict[str, logging.Logger]:
    """Setup enhanced logging configuration"""

    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Create formatters
    structured_formatter = StructuredFormatter()
    console_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        f"logs/{log_file}", maxBytes=10 * 1024 * 1024, backupCount=5  # 10MB
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(structured_formatter)
    root_logger.addHandler(file_handler)

    # Error file handler
    error_handler = logging.handlers.RotatingFileHandler(
        f"logs/errors_{log_file}",
        maxBytes=5 * 1024 * 1024,
        backupCount=3,  # 5MB
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(structured_formatter)
    root_logger.addHandler(error_handler)

    # Redis handler for centralized logging
    if enable_redis_logging and redis_client is not None:
        redis_handler = RedisLogHandler(redis_client, "mystic_logs")
        redis_handler.setLevel(logging.INFO)
        redis_handler.setFormatter(structured_formatter)
        root_logger.addHandler(redis_handler)

    # Create specific loggers
    loggers = {}

    # Main application logger
    app_logger = logging.getLogger("mystic.app")
    loggers["app"] = app_logger

    # Signal manager logger
    signal_logger = logging.getLogger("mystic.signals")
    loggers["signals"] = signal_logger

    # Test scheduler logger
    test_logger = logging.getLogger("mystic.tests")
    loggers["tests"] = test_logger

    # Notification logger
    notification_logger = logging.getLogger("mystic.notifications")
    loggers["notifications"] = notification_logger

    # API logger
    api_logger = logging.getLogger("mystic.api")
    loggers["api"] = api_logger

    # Performance logger
    performance_logger = logging.getLogger("mystic.performance")
    loggers["performance"] = performance_logger

    # WebSocket logger
    websocket_logger = logging.getLogger("mystic.websocket")
    loggers["websocket"] = websocket_logger

    # Trading logger
    trading_logger = logging.getLogger("mystic.trading")
    loggers["trading"] = trading_logger

    # Log startup message
    app_logger.info(
        "Enhanced logging system initialized",
        extra={
            "extra_fields": {
                "log_level": log_level,
                "log_file": log_file,
                "redis_logging": enable_redis_logging,
            }
        },
    )

    return loggers


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name"""
    return logging.getLogger(f"mystic.{name}")


def log_operation_performance(
    operation: str,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to log operation performance"""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        logger = get_logger("performance")
        perf_logger = PerformanceLogger(logger)
        return perf_logger.log_performance(operation)(func)

    return decorator


def setup_basic_logging():
    """Setup basic logging with default configuration"""
    try:
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)

        # Setup basic logging configuration
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("logs/mystic_trading.log"),
                logging.StreamHandler()
            ]
        )

        # Log successful setup
        logger = logging.getLogger("mystic.app")
        logger.info("Basic logging setup completed successfully")

        return {"app": logger}

    except Exception as e:
        # Fallback to basic logging if enhanced setup fails
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logging.warning(f"Basic logging setup failed: {e}")
        return {}


