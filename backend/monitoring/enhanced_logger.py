"""
Enhanced Logging System for Mystic Trading Platform

Provides comprehensive logging with:
- Structured logging with JSON format
- Multiple log levels and handlers
- Performance monitoring
- Error tracking and alerting
- Log aggregation and analysis
"""

import logging
import logging.handlers
import json
import time
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional
from collections import defaultdict, deque
import threading
import traceback


logger = logging.getLogger(__name__)

# Logging configuration
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
JSON_LOG_FORMAT = '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'
LOG_RETENTION_DAYS = 30
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10MB
LOG_BACKUP_COUNT = 5


class StructuredFormatter(logging.Formatter):
    """Structured JSON formatter for logs"""

    def __init__(self, include_traceback: bool = True):
        super().__init__()
        self.include_traceback = include_traceback

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON"""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'process_id': record.process,
            'thread_id': record.thread
        }

        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)

        # Add exception info if present
        if record.exc_info and self.include_traceback:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }

        return json.dumps(log_entry, default=str)


class PerformanceLogger:
    """Performance monitoring and logging"""

    def __init__(self):
        self.performance_metrics: deque = deque(maxlen=10000)
        self.slow_operations: deque = deque(maxlen=1000)
        self.error_counts: defaultdict = defaultdict(int)
        self.lock = threading.Lock()

    def log_performance(self, operation: str, duration: float,
                       metadata: Optional[Dict[str, Any]] = None):
        """Log performance metric"""
        metric = {
            'operation': operation,
            'duration': duration,
            'timestamp': time.time(),
            'metadata': metadata or {}
        }

        with self.lock:
            self.performance_metrics.append(metric)

            # Track slow operations
            if duration > 1.0:  # Operations taking more than 1 second
                self.slow_operations.append(metric)

    def log_error(self, error_type: str, error_message: str,
                  context: Optional[Dict[str, Any]] = None):
        """Log error for monitoring"""
        with self.lock:
            self.error_counts[error_type] += 1

        # Log to main logger
        logger.error(f"Error: {error_type} - {error_message}", extra={
            'error_type': error_type,
            'context': context
        })

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        with self.lock:
            if not self.performance_metrics:
                return {}

            durations = [m['duration'] for m in self.performance_metrics]
            recent_metrics = [m for m in self.performance_metrics
                            if time.time() - m['timestamp'] < 3600]  # Last hour

            return {
                'total_operations': len(self.performance_metrics),
                'recent_operations': len(recent_metrics),
                'average_duration': sum(durations) / len(durations),
                'max_duration': max(durations),
                'min_duration': min(durations),
                'slow_operations': len(self.slow_operations),
                'error_counts': dict(self.error_counts)
            }


class LogAggregator:
    """Aggregates and analyzes log data"""

    def __init__(self):
        self.log_entries: deque = deque(maxlen=100000)
        self.log_stats: Dict[str, int] = defaultdict(int)
        self.lock = threading.Lock()

    def add_log_entry(self, entry: Dict[str, Any]):
        """Add log entry to aggregator"""
        with self.lock:
            self.log_entries.append(entry)
            self.log_stats[entry.get('level', 'UNKNOWN')] += 1

    def get_log_analysis(self) -> Dict[str, Any]:
        """Get log analysis and statistics"""
        with self.lock:
            if not self.log_entries:
                return {}

            recent_entries = [e for e in self.log_entries
                            if time.time() - e.get('timestamp', 0) < 3600]

            return {
                'total_entries': len(self.log_entries),
                'recent_entries': len(recent_entries),
                'level_distribution': dict(self.log_stats),
                'unique_loggers': len(set(e.get('logger', 'unknown') for e in self.log_entries))
            }


class EnhancedLogger:
    """Enhanced logging system with performance monitoring"""

    def __init__(self):
        self.loggers: Dict[str, logging.Logger] = {}
        self.handlers: List[logging.Handler] = []
        self.performance_logger = PerformanceLogger()
        self.log_aggregator = LogAggregator()

        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging configuration"""
        # Create logs directory
        os.makedirs("logs", exist_ok=True)

        # Setup root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(LOG_LEVEL)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(LOG_FORMAT)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        self.handlers.append(console_handler)

        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            "logs/mystic_trading.log",
            maxBytes=MAX_LOG_SIZE,
            backupCount=LOG_BACKUP_COUNT
        )
        file_formatter = logging.Formatter(LOG_FORMAT)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        self.handlers.append(file_handler)

        # JSON file handler for structured logging
        json_handler = logging.handlers.RotatingFileHandler(
            "logs/mystic_trading.json",
            maxBytes=MAX_LOG_SIZE,
            backupCount=LOG_BACKUP_COUNT
        )
        json_formatter = StructuredFormatter()
        json_handler.setFormatter(json_formatter)
        root_logger.addHandler(json_handler)
        self.handlers.append(json_handler)

        # Error file handler
        error_handler = logging.handlers.RotatingFileHandler(
            "logs/errors.log",
            maxBytes=MAX_LOG_SIZE,
            backupCount=LOG_BACKUP_COUNT
        )
        error_formatter = logging.Formatter(LOG_FORMAT)
        error_handler.setFormatter(error_formatter)
        error_handler.setLevel(logging.ERROR)
        root_logger.addHandler(error_handler)
        self.handlers.append(error_handler)

        # Performance file handler
        performance_handler = logging.handlers.RotatingFileHandler(
            "logs/performance.log",
            maxBytes=MAX_LOG_SIZE,
            backupCount=LOG_BACKUP_COUNT
        )
        performance_formatter = logging.Formatter(LOG_FORMAT)
        performance_handler.setFormatter(performance_formatter)
        performance_logger = logging.getLogger("performance")
        performance_logger.addHandler(performance_handler)
        performance_logger.setLevel(logging.INFO)
        self.handlers.append(performance_handler)

        # Security file handler
        security_handler = logging.handlers.RotatingFileHandler(
            "logs/security.log",
            maxBytes=MAX_LOG_SIZE,
            backupCount=LOG_BACKUP_COUNT
        )
        security_formatter = logging.Formatter(LOG_FORMAT)
        security_handler.setFormatter(security_formatter)
        security_logger = logging.getLogger("security")
        security_logger.addHandler(security_handler)
        security_logger.setLevel(logging.WARNING)
        self.handlers.append(security_handler)

    def get_logger(self, name: str) -> logging.Logger:
        """Get or create logger with given name"""
        if name not in self.loggers:
            self.loggers[name] = logging.getLogger(name)
        return self.loggers[name]

    def log_info(self, message: str, logger_name: str = "main", **kwargs):
        """Log info message"""
        logger_instance = self.get_logger(logger_name)
        logger_instance.info(message, extra={'extra_fields': kwargs})

        # Add to aggregator
        self.log_aggregator.add_log_entry({
            'timestamp': time.time(),
            'level': 'INFO',
            'message': message,
            'logger': logger_name,
            **kwargs
        })

    def log_warning(self, message: str, logger_name: str = "main", **kwargs):
        """Log warning message"""
        logger_instance = self.get_logger(logger_name)
        logger_instance.warning(message, extra={'extra_fields': kwargs})

        # Add to aggregator
        self.log_aggregator.add_log_entry({
            'timestamp': time.time(),
            'level': 'WARNING',
            'message': message,
            'logger': logger_name,
            **kwargs
        })

    def log_error(self, message: str, error: Optional[Exception] = None,
                  logger_name: str = "main", **kwargs):
        """Log error message with optional exception"""
        logger_instance = self.get_logger(logger_name)

        if error:
            logger_instance.error(message, exc_info=True, extra={'extra_fields': kwargs})
        else:
            logger_instance.error(message, extra={'extra_fields': kwargs})

        # Add to aggregator
        self.log_aggregator.add_log_entry({
            'timestamp': time.time(),
            'level': 'ERROR',
            'message': message,
            'error_type': type(error).__name__ if error else None,
            'logger': logger_name,
            **kwargs
        })

    def log_performance(self, operation: str, duration: float, **kwargs):
        """Log performance metric"""
        self.performance_logger.log_performance(operation, duration, kwargs)

        # Log to performance logger
        performance_logger = self.get_logger("performance")
        performance_logger.info(
            f"Performance: {operation} took {duration:.3f}s",
            extra={'extra_fields': {'operation': operation, 'duration': duration, **kwargs}}
        )

        # Add to aggregator
        self.log_aggregator.add_log_entry({
            'timestamp': time.time(),
            'level': 'INFO',
            'message': f"Performance: {operation}",
            'logger': 'performance',
            'operation': operation,
            'duration': duration,
            **kwargs
        })

    def log_security(self, message: str, severity: str = "medium", **kwargs):
        """Log security event"""
        security_logger = self.get_logger("security")
        security_logger.warning(
            f"[SECURITY-{severity.upper()}] {message}",
            extra={'extra_fields': {'severity': severity, **kwargs}}
        )

        # Add to aggregator
        self.log_aggregator.add_log_entry({
            'timestamp': time.time(),
            'level': 'WARNING',
            'message': f"[SECURITY] {message}",
            'logger': 'security',
            'severity': severity,
            **kwargs
        })

    def log_business_event(self, event_type: str, event_data: Dict[str, Any],
                          logger_name: str = "business"):
        """Log business event"""
        business_logger = self.get_logger(logger_name)
        business_logger.info(
            f"Business event: {event_type}",
            extra={'extra_fields': {'event_type': event_type, 'event_data': event_data}}
        )

        # Add to aggregator
        self.log_aggregator.add_log_entry({
            'timestamp': time.time(),
            'level': 'INFO',
            'message': f"Business event: {event_type}",
            'logger': logger_name,
            'event_type': event_type,
            'event_data': event_data
        })

    def get_logging_stats(self) -> Dict[str, Any]:
        """Get comprehensive logging statistics"""
        return {
            'performance_stats': self.performance_logger.get_performance_stats(),
            'log_analysis': self.log_aggregator.get_log_analysis(),
            'active_loggers': len(self.loggers),
            'active_handlers': len(self.handlers)
        }

    def cleanup_old_logs(self, max_age_days: int = LOG_RETENTION_DAYS):
        """Clean up old log files"""
        cutoff_time = time.time() - (max_age_days * 24 * 3600)

        # Clean up old log files
        log_dir = "logs"
        if os.path.exists(log_dir):
            for filename in os.listdir(log_dir):
                filepath = os.path.join(log_dir, filename)
                if os.path.isfile(filepath):
                    file_time = os.path.getmtime(filepath)
                    if file_time < cutoff_time:
                        os.remove(filepath)
                        logger.info(f"Removed old log file: {filename}")

        logger.info("Cleaned up old log files")


# Global enhanced logger instance
enhanced_logger = EnhancedLogger()
