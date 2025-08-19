"""
Log Rotation Manager for Mystic Trading Platform

Handles automatic log rotation with different policies:
- Regular logs: Rotate daily, keep 7 days
- AI logs: Rotate weekly, keep 4 weeks (preserve learning data)
- Error logs: Rotate daily, keep 14 days
- Audit logs: Rotate daily, keep 30 days
"""

import logging
import logging.handlers
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class LogRotationManager:
    """Manages log rotation with different policies for different log types"""

    def __init__(self, logs_dir: str = "logs"):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(exist_ok=True)

        # Log rotation policies
        self.policies: dict[str, dict[str, Any]] = {
            # Regular application logs - rotate daily, keep 7 days
            "regular": {
                "rotation": "daily",
                "retention_days": 7,
                "max_size_mb": 10,
                "backup_count": 7,
            },
            # AI learning logs - rotate weekly, keep 4 weeks (preserve learning data)
            "ai": {
                "rotation": "weekly",
                "retention_days": 28,
                "max_size_mb": 50,
                "backup_count": 4,
            },
            # Error logs - rotate daily, keep 14 days
            "error": {
                "rotation": "daily",
                "retention_days": 14,
                "max_size_mb": 5,
                "backup_count": 14,
            },
            # Audit logs - rotate daily, keep 30 days
            "audit": {
                "rotation": "daily",
                "retention_days": 30,
                "max_size_mb": 10,
                "backup_count": 30,
            },
            # Bot logs - rotate daily, keep 7 days
            "bot": {
                "rotation": "daily",
                "retention_days": 7,
                "max_size_mb": 10,
                "backup_count": 7,
            },
        }

        # Log file mappings
        self.log_mappings: dict[str, str] = {
            # Regular logs
            "mystic_trading.log": "regular",
            "trading_bots.log": "regular",
            # AI logs (preserve learning data)
            "ai_mutation.log": "ai",
            "ai_strategies.log": "ai",
            "ai_performance.log": "ai",
            "ai_backtest.log": "ai",
            "ai_learning.log": "ai",
            # Error logs
            "errors_mystic_trading.log": "error",
            "errors_ai.log": "error",
            "errors_bots.log": "error",
            # Audit logs
            "audit.log": "audit",
            "security.log": "audit",
            # Bot logs
            "bot_manager.log": "bot",
            "coinbase_bot.log": "bot",
            "binance_bot.log": "bot",
        }

        logger.info("Log Rotation Manager initialized")

    def get_handler(self, log_file: str, policy_type: str | None = None) -> logging.Handler:
        """Get appropriate log handler based on file and policy"""

        # Determine policy type
        if policy_type is None:
            policy_type = self.log_mappings.get(log_file, "regular")

        policy = self.policies[policy_type]
        log_path = self.logs_dir / log_file

        # Create handler based on rotation policy
        if policy["rotation"] == "daily":
            handler = logging.handlers.TimedRotatingFileHandler(
                filename=log_path,
                when="midnight",
                interval=1,
                backupCount=int(policy["backup_count"]),
                encoding="utf-8",
            )
        elif policy["rotation"] == "weekly":
            handler = logging.handlers.TimedRotatingFileHandler(
                filename=log_path,
                when="W0",  # Monday
                interval=1,
                backupCount=int(policy["backup_count"]),
                encoding="utf-8",
            )
        else:
            # Fallback to size-based rotation
            handler = logging.handlers.RotatingFileHandler(
                filename=log_path,
                maxBytes=int(policy["max_size_mb"] * 1024 * 1024),
                backupCount=int(policy["backup_count"]),
                encoding="utf-8",
            )

        # Set formatter
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)

        return handler

    def cleanup_old_logs(self) -> dict[str, int]:
        """Clean up old log files based on retention policies"""
        cleanup_stats: dict[str, int] = {}

        for log_file, policy_type in self.log_mappings.items():
            policy = self.policies[policy_type]
            retention_days = int(policy["retention_days"])
            cutoff_date = datetime.now() - timedelta(days=retention_days)

            # Find old log files
            deleted_count = 0

            # Check for rotated files (e.g., mystic_trading.log.2024-01-01)
            for old_file in self.logs_dir.glob(f"{log_file}.*"):
                try:
                    # Extract date from filename
                    date_str = old_file.name.split(".")[-1]
                    file_date = datetime.strptime(date_str, "%Y-%m-%d")

                    if file_date < cutoff_date:
                        old_file.unlink()
                        deleted_count += 1
                        logger.debug(f"Deleted old log file: {old_file}")

                except (ValueError, IndexError):
                    # Skip files that don't match date pattern
                    continue

            cleanup_stats[log_file] = deleted_count

        logger.info(f"Log cleanup completed: {cleanup_stats}")
        return cleanup_stats

    def get_log_stats(self) -> dict[str, dict[str, Any]]:
        """Get statistics about log files"""
        stats: dict[str, dict[str, Any]] = {}

        for log_file, policy_type in self.log_mappings.items():
            log_path = self.logs_dir / log_file
            policy = self.policies[policy_type]

            file_stats: dict[str, Any] = {
                "policy_type": policy_type,
                "rotation": policy["rotation"],
                "retention_days": policy["retention_days"],
                "exists": log_path.exists(),
                "size_mb": 0.0,
                "backup_files": 0,
            }

            if log_path.exists():
                file_stats["size_mb"] = log_path.stat().st_size / (1024 * 1024)

            # Count backup files
            backup_pattern = f"{log_file}.*"
            backup_files = list(self.logs_dir.glob(backup_pattern))
            file_stats["backup_files"] = len(backup_files)

            stats[log_file] = file_stats

        return stats

    def create_ai_log_handler(self, log_file: str) -> logging.Handler:
        """Create a specialized handler for AI logs with extended retention"""
        ai_policy = self.policies["ai"]
        log_path = self.logs_dir / log_file

        # AI logs use weekly rotation with extended retention
        handler = logging.handlers.TimedRotatingFileHandler(
            filename=log_path,
            when="W0",  # Monday
            interval=1,
            backupCount=int(ai_policy["backup_count"]),
            encoding="utf-8",
        )

        # Use structured formatter for AI logs
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)

        return handler

    def setup_logger(
        self, logger_name: str, log_file: str, level: int = logging.INFO
    ) -> logging.Logger:
        """Setup a logger with appropriate rotation policy"""
        logger_instance = logging.getLogger(logger_name)
        logger_instance.setLevel(level)

        # Clear existing handlers
        logger_instance.handlers.clear()

        # Add file handler with rotation
        file_handler = self.get_handler(log_file)
        logger_instance.addHandler(file_handler)

        # Add console handler for important logs
        if log_file in [
            "mystic_trading.log",
            "errors_mystic_trading.log",
            "audit.log",
        ]:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            console_handler.setFormatter(formatter)
            logger_instance.addHandler(console_handler)

        return logger_instance


# Global log rotation manager instance
log_rotation_manager = LogRotationManager()


def get_log_rotation_manager() -> LogRotationManager:
    """Get the global log rotation manager instance"""
    return log_rotation_manager


def setup_rotated_logging() -> dict[str, logging.Logger]:
    """Setup all loggers with rotation policies"""
    loggers: dict[str, logging.Logger] = {}

    # Setup main application logger
    loggers["app"] = log_rotation_manager.setup_logger("mystic.app", "mystic_trading.log")

    # Setup AI loggers (with extended retention)
    loggers["ai_mutation"] = log_rotation_manager.setup_logger(
        "mystic.ai.mutation", "ai_mutation.log"
    )
    loggers["ai_strategies"] = log_rotation_manager.setup_logger(
        "mystic.ai.strategies", "ai_strategies.log"
    )
    loggers["ai_performance"] = log_rotation_manager.setup_logger(
        "mystic.ai.performance", "ai_performance.log"
    )

    # Setup error logger
    loggers["errors"] = log_rotation_manager.setup_logger(
        "mystic.errors", "errors_mystic_trading.log", logging.ERROR
    )

    # Setup audit logger
    loggers["audit"] = log_rotation_manager.setup_logger("mystic.audit", "audit.log")

    # Setup bot loggers
    loggers["bot_manager"] = log_rotation_manager.setup_logger(
        "mystic.bot.manager", "bot_manager.log"
    )
    loggers["coinbase_bot"] = log_rotation_manager.setup_logger(
        "mystic.bot.coinbase", "coinbase_bot.log"
    )
    loggers["binance_bot"] = log_rotation_manager.setup_logger(
        "mystic.bot.binance", "binance_bot.log"
    )

    # Setup trading bots logger
    loggers["trading_bots"] = log_rotation_manager.setup_logger(
        "mystic.trading.bots", "trading_bots.log"
    )

    # Log startup message
    loggers["app"].info("Rotated logging system initialized")

    return loggers


