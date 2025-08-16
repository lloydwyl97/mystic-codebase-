"""
Auto Trading Manager for Mystic Trading

Manages automated trading operations, separated from signal management for better modularity.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict
import os

from notification_service import get_notification_service
from trading_config import trading_config

logger = logging.getLogger(__name__)


class AutoTradingManager:
    def __init__(self, redis_client: Any) -> None:
        self.redis_client = redis_client
        self.auto_trading_enabled: bool = False

        # Initialize notification service
        try:
            self.notification_service = get_notification_service(redis_client)
        except Exception as e:
            logger.warning(f"Failed to initialize notification service: {str(e)}")
            self.notification_service = None

        # Load trading strategies
        self.trading_strategies: Dict[str, Dict[str, Any]] = self._load_trading_strategies()

        # Initialize auto-trading status from Redis if available
        self._initialize_status()

    def _load_trading_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Load trading strategies from Redis or config file. Raise if unavailable."""
        try:
            strategy_data = self.redis_client.get("trading_strategies")
            if strategy_data:
                return json.loads(strategy_data)
        except Exception as e:
            logger.warning(f"Error loading trading strategies from Redis: {str(e)}")
        # Try to load from config file
        config_path = os.path.join(os.path.dirname(__file__), "autobuy_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
                if "trading_strategies" in config:
                    return config["trading_strategies"]
        raise RuntimeError("No trading strategies found in Redis or config file.")

    def _initialize_status(self) -> None:
        """Initialize auto-trading status from Redis"""
        try:
            auto_trade_data = self.redis_client.get("auto_trading_enabled")
            if auto_trade_data:
                auto_trading = json.loads(auto_trade_data)
                self.auto_trading_enabled = auto_trading.get("enabled", False)

            config_data = self.redis_client.get("auto_trade_config")
            if config_data:
                # Auto-trading config exists, no need to initialize
                logger.info("Auto-trading configuration loaded from Redis")
            else:
                # Initialize with default disabled config
                self._save_default_config(enabled=False)
                logger.info("Initialized default auto-trading configuration")
        except Exception as e:
            logger.warning(f"Error initializing auto-trading status: {str(e)}")
            self.auto_trading_enabled = False

    def _save_default_config(self, enabled: bool = False) -> None:
        """Save default auto-trading configuration to Redis"""
        try:
            auto_trade_config: Dict[str, Any] = {
                "enabled": enabled,
                "start_time": (
                    datetime.now(timezone.timezone.utc).isoformat() if enabled else None
                ),
                "strategies": (list(self.trading_strategies.keys()) if enabled else []),
                "risk_management": trading_config.get_risk_management_config(),
                "performance": trading_config.get_performance_config(),
            }

            # Store in Redis with error handling
            self.redis_client.setex("auto_trade_config", trading_config.AUTO_TRADE_CONFIG_TTL, json.dumps(auto_trade_config))

            # Update enabled flag in Redis
            self.redis_client.setex(
                "auto_trading_enabled",
                trading_config.AUTO_TRADING_ENABLED_TTL,
                json.dumps(
                    {
                        "enabled": enabled,
                        "updated_at": (datetime.now(timezone.timezone.utc).isoformat()),
                    }
                ),
            )
        except Exception as e:
            logger.error(f"Error saving default auto-trading config: {str(e)}")

    async def start_auto_trading(self) -> Dict[str, Any]:
        """Start automated trading"""
        try:
            logger.info("Starting automated trading...")

            # Define auto_trade_config with proper typing
            auto_trade_config: Dict[str, Any] = {
                "enabled": True,
                "start_time": datetime.now(timezone.timezone.utc).isoformat(),
                "strategies": list(self.trading_strategies.keys()),
                "risk_management": trading_config.get_risk_management_config(),
                "performance": trading_config.get_performance_config(),
            }

            # Store in Redis with error handling
            try:
                self.redis_client.setex("auto_trade_config", trading_config.AUTO_TRADE_CONFIG_TTL, json.dumps(auto_trade_config))

                # Update enabled flag in Redis
                self.redis_client.setex(
                    "auto_trading_enabled",
                    trading_config.AUTO_TRADING_ENABLED_TTL,
                    json.dumps(
                        {
                            "enabled": True,
                            "activated_at": (datetime.now(timezone.timezone.utc).isoformat()),
                        }
                    ),
                )

                self.auto_trading_enabled = True
                logger.info("Automated trading started successfully")

                # Send notification if notification service is available
                if self.notification_service:
                    await self.notification_service.send_notification(
                        "Auto-trading started",
                        "Automated trading has been started successfully.",
                        "info",
                        channels=["in_app"],
                        data={"auto_trading": True},
                    )
            except Exception as e:
                logger.error(f"Redis error when starting auto-trading: {str(e)}")
                # Continue with local state even if Redis fails
                self.auto_trading_enabled = True
                logger.warning("Auto-trading started with local state only (Redis failed)")

            return {
                "status": "success",
                "message": "Automated trading started",
                "config": auto_trade_config,
                "timestamp": datetime.now(timezone.timezone.utc).isoformat(),
            }
        except Exception as e:
            logger.error(f"Error starting auto-trading: {str(e)}")
            raise

    async def stop_auto_trading(self) -> Dict[str, Any]:
        """Stop automated trading"""
        try:
            logger.info("Stopping automated trading...")

            # Get current config to preserve structure
            config_data = self.redis_client.get("auto_trade_config")

            # Define auto_trade_config with proper typing
            auto_trade_config: Dict[str, Any] = {}

            if config_data:
                auto_trade_config = json.loads(config_data)
                auto_trade_config["enabled"] = False
                auto_trade_config["stop_time"] = datetime.now(timezone.timezone.utc).isoformat()
                auto_trade_config["strategies"] = []  # Empty list of strings
            else:
                # Create a new config with the same structure as in start_auto_trading
                auto_trade_config = {
                    "enabled": False,
                    "stop_time": (datetime.now(timezone.timezone.utc).isoformat()),
                    "strategies": [],  # Empty list of strategy names (strings)
                    "risk_management": trading_config.get_risk_management_config(),
                    "performance": trading_config.get_performance_config(),
                }

            # Store in Redis with error handling
            try:
                self.redis_client.setex("auto_trade_config", trading_config.AUTO_TRADE_CONFIG_TTL, json.dumps(auto_trade_config))

                # Update enabled flag in Redis
                self.redis_client.setex(
                    "auto_trading_enabled",
                    trading_config.AUTO_TRADING_ENABLED_TTL,
                    json.dumps(
                        {
                            "enabled": False,
                            "deactivated_at": (datetime.now(timezone.timezone.utc).isoformat()),
                        }
                    ),
                )

                self.auto_trading_enabled = False
                logger.info("Automated trading stopped successfully")

                # Send notification if notification service is available
                if self.notification_service:
                    await self.notification_service.send_notification(
                        "Auto-trading stopped",
                        "Automated trading has been stopped successfully.",
                        "info",
                        channels=["in_app"],
                        data={"auto_trading": False},
                    )
            except Exception as e:
                logger.error(f"Redis error when stopping auto-trading: {str(e)}")
                # Continue with local state even if Redis fails
                self.auto_trading_enabled = False
                logger.warning("Auto-trading stopped with local state only (Redis failed)")

            return {
                "status": "success",
                "message": "Automated trading stopped",
                "timestamp": datetime.now(timezone.timezone.utc).isoformat(),
            }
        except Exception as e:
            logger.error(f"Error stopping auto-trading: {str(e)}")
            raise

    async def get_auto_trade_status(self) -> Dict[str, Any]:
        """Get auto-trading status. Fail if Redis is unavailable or data is missing."""
        try:
            config_data = self.redis_client.get("auto_trade_config")
            if not config_data:
                raise RuntimeError("Auto-trading config not found in Redis.")
            config = json.loads(config_data)
            return {
                "status": "success",
                "auto_trading": config,
                "timestamp": datetime.now(timezone.timezone.utc).isoformat(),
            }
        except Exception as e:
            logger.error(f"Error getting auto-trade status: {str(e)}")
            raise

    async def check_health(self) -> Dict[str, bool]:
        """Check the health of auto-trading. Fail if Redis is unavailable or data is missing."""
        try:
            config_data = self.redis_client.get("auto_trade_config")
            if not config_data:
                raise RuntimeError("Auto-trading config not found in Redis.")
            config = json.loads(config_data)
            return {"healthy": (config.get("enabled", False) == self.auto_trading_enabled)}
        except Exception as e:
            logger.error(f"Error checking auto-trading health: {str(e)}")
            raise

    async def self_heal(self) -> Dict[str, Any]:
        """Self-heal auto-trading if needed"""
        try:
            # Check if auto-trading is supposed to be enabled but isn't
            auto_trade_data = self.redis_client.get("auto_trading_enabled")
            if auto_trade_data:
                auto_trading = json.loads(auto_trade_data)
                should_be_enabled = auto_trading.get("enabled", False)

                if should_be_enabled and not self.auto_trading_enabled:
                    # Auto-trading should be enabled but isn't
                    logger.warning(
                        "Auto-trading is supposed to be enabled but isn't. Self-healing..."
                    )
                    await self.start_auto_trading()
                    return {
                        "status": "healed",
                        "message": "Auto-trading was reactivated",
                        "timestamp": (datetime.now(timezone.timezone.utc).isoformat()),
                    }
                elif not should_be_enabled and self.auto_trading_enabled:
                    # Auto-trading should be disabled but isn't
                    logger.warning(
                        "Auto-trading is supposed to be disabled but isn't. Self-healing..."
                    )
                    await self.stop_auto_trading()
                    return {
                        "status": "healed",
                        "message": "Auto-trading was deactivated",
                        "timestamp": (datetime.now(timezone.timezone.utc).isoformat()),
                    }

            # No healing needed
            return {
                "status": "healthy",
                "message": "Auto-trading is in the correct state",
                "timestamp": datetime.now(timezone.timezone.utc).isoformat(),
            }
        except Exception as e:
            logger.error(f"Error self-healing auto-trading: {str(e)}")
            return {
                "status": "error",
                "message": f"Error self-healing auto-trading: {str(e)}",
                "timestamp": datetime.now(timezone.timezone.utc).isoformat(),
            }


# Global auto trading manager instance
auto_trading_manager = None


def get_auto_trading_manager(redis_client: Any) -> AutoTradingManager:
    """Get or create auto trading manager instance"""
    global auto_trading_manager
    if auto_trading_manager is None:
        auto_trading_manager = AutoTradingManager(redis_client)
    return auto_trading_manager


