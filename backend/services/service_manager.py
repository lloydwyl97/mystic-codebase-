"""
Service Manager

Handles initialization and management of all application services.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ServiceManager:
    """Manages all application services."""

    def __init__(self):
        self.market_data_service = None
        self.connection_manager = None
        self.redis_client = None
        self.unified_signal_manager = None
        self._initialized = False

    async def initialize_services(self):
        """Initialize all services."""
        try:
            # Initialize connection manager
            try:
                from connection_manager import get_connection_manager

                logger.info("Initializing connection manager...")
                self.connection_manager = get_connection_manager()
                await self.connection_manager.initialize_connections()
                self.redis_client = self.connection_manager.redis_client

                redis_health = self.connection_manager.check_redis_health()
                if redis_health["connected"]:
                    logger.info("âœ… Redis connection established")
                else:
                    logger.error("âŒ Redis not available - limited functionality")
            except Exception as e:
                logger.warning(f"âš ï¸ ConnectionManager not available: {e}")

            # Initialize market data service
            try:
                from backend.services.market_data import MarketDataService

                logger.info("Initializing market data service...")
                self.market_data_service = MarketDataService()
                await self.market_data_service.initialize()
                logger.info("âœ… Market data service initialized")
            except Exception as e:
                logger.warning(f"âš ï¸ MarketDataService not available: {e}")

            # Initialize unified signal manager
            if self.redis_client:
                try:
                    from unified_signal_manager import (
                        get_unified_signal_manager,
                    )

                    self.unified_signal_manager = get_unified_signal_manager(self.redis_client)
                    logger.info("âœ… Unified signal manager initialized")
                except Exception as e:
                    logger.warning(f"âš ï¸ Unified signal manager failed: {e}")

            self._initialized = True
            logger.info("âœ… Service initialization completed")

        except Exception as e:
            logger.error(f"âŒ Service initialization error: {e}")
            raise

    async def shutdown_services(self):
        """Shutdown all services."""
        try:
            logger.info("Shutting down services...")
            if self.market_data_service:
                await self.market_data_service.close()
            if self.connection_manager:
                await self.connection_manager.close_connections()
            logger.info("âœ… Services shut down successfully")
        except Exception as e:
            logger.error(f"âŒ Shutdown error: {e}")

    def get_health_status(self) -> dict[str, Any]:
        """Get health status of all services."""
        return {
            "backend": "online",
            "database": "online",
            "api": "online",
            "market_data": "online" if self.market_data_service else "offline",
            "connection_manager": ("online" if self.connection_manager else "offline"),
            "redis": "online" if self.redis_client else "offline",
            "signal_manager": ("online" if self.unified_signal_manager else "offline"),
        }

    @property
    def is_initialized(self) -> bool:
        """Check if services are initialized."""
        return self._initialized


# Global service manager instance
service_manager = ServiceManager()


