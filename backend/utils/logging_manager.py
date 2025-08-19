"""
Logging Manager for Mystic Trading

Provides centralized logging configuration and management.
"""

import logging
import logging.handlers
import os
from datetime import datetime, timezone
from typing import Any

from enhanced_logging import StructuredFormatter, log_operation_performance


class LoggingManager:
    """Manages logging configuration and provides logging utilities"""

    def __init__(self):
        self.loggers: dict[str, logging.Logger] = {}
        self.formatters: dict[str, logging.Formatter] = {}
        self.handlers: dict[str, logging.Handler] = {}
        self.log_levels = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }

    def configure(
        self,
        redis_client: Any | None = None,
        log_level: str = "INFO",
        log_file: str = "mystic_trading.log",
        enable_redis_logging: bool = True,
        enable_file_logging: bool = True,
        enable_console_logging: bool = True,
    ) -> dict[str, logging.Logger]:
        """Configure logging system"""

        # Create logs directory if it doesn't exist
        if enable_file_logging:
            os.makedirs("logs", exist_ok=True)

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_levels.get(log_level.upper(), logging.INFO))

        # Clear existing handlers
        root_logger.handlers.clear()

        # Create formatters
        self.formatters["structured"] = self._create_structured_formatter()
        self.formatters["console"] = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Add console handler
        if enable_console_logging:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(self.formatters["console"])
            root_logger.addHandler(console_handler)
            self.handlers["console"] = console_handler

        # Add file handlers
        if enable_file_logging:
            # Main log file with rotation
            file_handler = logging.handlers.RotatingFileHandler(
                f"logs/{log_file}",
                maxBytes=10 * 1024 * 1024,
                backupCount=5,  # 10MB
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(self.formatters["structured"])
            root_logger.addHandler(file_handler)
            self.handlers["file"] = file_handler

            # Error log file
            error_handler = logging.handlers.RotatingFileHandler(
                f"logs/errors_{log_file}",
                maxBytes=5 * 1024 * 1024,  # 5MB
                backupCount=3,
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(self.formatters["structured"])
            root_logger.addHandler(error_handler)
            self.handlers["error"] = error_handler

        # Add Redis handler
        if enable_redis_logging and redis_client is not None:
            # Redis logging is handled by enhanced_logging module
            # No need for separate RedisLogHandler
            pass

        # Create specific loggers
        self.loggers = self._create_loggers()

        # Log startup message
        self.loggers["app"].info(
            "Logging system initialized",
            extra={
                "extra_fields": {
                    "log_level": log_level,
                    "log_file": log_file,
                    "redis_logging": enable_redis_logging,
                    "file_logging": enable_file_logging,
                    "console_logging": enable_console_logging,
                }
            },
        )

        return self.loggers

    def _create_structured_formatter(self) -> logging.Formatter:
        """Create a structured JSON formatter"""
        return StructuredFormatter()

    def _create_loggers(self) -> dict[str, logging.Logger]:
        """Create specific loggers for different components"""
        loggers: dict[str, logging.Logger] = {}

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

        # Database logger
        database_logger = logging.getLogger("mystic.database")
        loggers["database"] = database_logger

        # Security logger
        security_logger = logging.getLogger("mystic.security")
        loggers["security"] = security_logger

        return loggers

    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger by name"""
        if name in self.loggers:
            return self.loggers[name]
        else:
            # Create a new logger if it doesn't exist
            logger = logging.getLogger(f"mystic.{name}")
            self.loggers[name] = logger
            return logger

    def log_event(self, event_type: str, details: dict[str, Any], level: str = "INFO") -> None:
        """Log a structured event"""
        logger = self.get_logger("app")
        log_level = self.log_levels.get(level.upper(), logging.INFO)

        # Create log record
        record = logger.makeRecord(logger.name, log_level, "", 0, f"Event: {event_type}", (), None)

        # Add extra fields as a custom attribute
        record.extra_fields = details

        logger.handle(record)

    def log_performance(
        self,
        operation: str,
        duration_ms: float,
        status: str = "success",
        details: dict[str, Any] | None = None,
    ) -> None:
        """Log a performance metric"""
        logger = self.get_logger("performance")

        # Create log record
        record = logger.makeRecord(
            logger.name,
            logging.INFO,
            "",
            0,
            f"Operation '{operation}' completed with status '{status}'",
            (),
            None,
        )

        # Add performance data
        record.duration = duration_ms

        # Add extra fields
        extra_data = {
            "operation": operation,
            "status": status,
            "duration_ms": duration_ms,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if details:
            extra_data.update(details)

        record.extra_fields = extra_data

        logger.handle(record)

    def performance_decorator(self, operation: str):
        """Decorator to log operation performance"""

        def decorator(func: Any) -> Any:
            return log_operation_performance(operation)(func)

        return decorator


# Global logging manager instance
logging_manager = LoggingManager()


def get_logging_manager():
    """Get the global logging manager instance"""
    return logging_manager


