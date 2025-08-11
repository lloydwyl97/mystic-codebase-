"""
Application Lifespan Management for Mystic Trading

Handles application startup and shutdown processes.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

# Import services
try:
    from .connection_manager import get_connection_manager
    from .enhanced_logging import setup_enhanced_logging
    from .health_monitor import get_health_monitor
    from .services.market_data import market_data_service
    from .services.service_manager import service_manager
except ImportError:
    from backend.connection_manager import get_connection_manager
    from backend.enhanced_logging import setup_enhanced_logging
    from backend.health_monitor import get_health_monitor
    from backend.services.market_data import market_data_service
    from backend.services.service_manager import service_manager

# Get logger
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifespan - startup and shutdown processes."""
    # Startup
    logger.info("Starting up application...")

    conn_manager = None

    try:
        # Get connection manager
        conn_manager = get_connection_manager()

        # Initialize connections
        await conn_manager.initialize_connections()

        # Initialize services with timeout
        try:
            initialization_task = asyncio.create_task(market_data_service.initialize())
            await asyncio.wait_for(initialization_task, timeout=30.0)
            logger.info("Market data service initialized successfully")
        except asyncio.TimeoutError:
            logger.error("Market data service initialization timed out")
        except Exception as e:
            logger.error(f"Error initializing market data service: {str(e)}")

        # Initialize enhanced logging
        try:
            global websocket_logger, signal_logger, test_logger, notification_logger, api_logger, performance_logger, trading_logger
            loggers = setup_enhanced_logging(
                conn_manager.redis_client, "INFO", "mystic_trading.log", True
            )
            signal_logger = loggers["signals"]
            test_logger = loggers["tests"]
            notification_logger = loggers["notifications"]
            api_logger = loggers["api"]
            performance_logger = loggers["performance"]
            websocket_logger = loggers["websocket"]
            trading_logger = loggers["trading"]
            logger.info("Enhanced logging initialized successfully")
        except Exception as e:
            logger.warning(f"Enhanced logging initialization failed: {str(e)}")
            # Use basic logging as fallback
            signal_logger = logger
            test_logger = logger
            notification_logger = logger
            api_logger = logger
            performance_logger = logger
            websocket_logger = logger
            trading_logger = logger

        # Initialize health monitor
        try:
            health_monitor = get_health_monitor(
                conn_manager.signal_manager,
                conn_manager.auto_trading_manager,
                conn_manager.notification_service,
                conn_manager.metrics_collector,
            )
            await health_monitor.start_monitoring()
            logger.info("Health monitor started successfully")
        except Exception as e:
            logger.warning(f"Health monitor initialization failed: {str(e)}")

        # Initialize service manager
        try:
            if service_manager:
                await service_manager.initialize_services()
            logger.info("✅ Application startup completed")
        except Exception as e:
            logger.error(f"❌ Startup error: {e}")

    except Exception as e:
        logger.error(f"❌ Critical startup error: {e}")
        # Continue with limited functionality

    yield

    # Shutdown
    logger.info("Shutting down application...")

    try:
        # Close market data service
        try:
            shutdown_task = asyncio.create_task(market_data_service.close())
            await asyncio.wait_for(shutdown_task, timeout=10.0)
            logger.info("Market data service shut down successfully")
        except asyncio.TimeoutError:
            logger.error("Market data service shutdown timed out")
        except Exception as e:
            logger.error(f"Error shutting down market data service: {str(e)}")

        # Stop health monitor
        if conn_manager:
            try:
                health_monitor = get_health_monitor(
                    conn_manager.signal_manager,
                    conn_manager.auto_trading_manager,
                    conn_manager.notification_service,
                    conn_manager.metrics_collector,
                )
                await health_monitor.stop_monitoring()
                logger.info("Health monitor stopped successfully")
            except Exception as e:
                logger.warning(f"Error stopping health monitor: {str(e)}")

            # Close connections
            await conn_manager.close_connections()

        # Shutdown service manager
        try:
            if service_manager:
                await service_manager.shutdown_services()
            logger.info("✅ Application shutdown completed")
        except Exception as e:
            logger.error(f"❌ Shutdown error: {e}")

    except Exception as e:
        logger.error(f"❌ Critical shutdown error: {e}")


# Alternative lifespan for compatibility
async def startup():
    """Startup event handler for compatibility"""
    logger.info("Starting up application (compatibility mode)...")
    # Basic startup without complex initialization
    logger.info("✅ Application startup completed (compatibility mode)")


async def shutdown():
    """Shutdown event handler for compatibility"""
    logger.info("Shutting down application (compatibility mode)...")
    logger.info("✅ Application shutdown completed (compatibility mode)")
