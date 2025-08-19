#!/usr/bin/env python3
"""
CRYPTO AUTOENGINE Startup Script
Main entry point for the complete system
"""

import asyncio
import logging
import os
import sys
from collections.abc import Callable
from typing import cast

import redis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from crypto_autoengine_api import initialize_managers, router
    from crypto_autoengine_config import get_config
    from trading_config import trading_config
except ImportError:
    # Fallback if the path modification didn't work
    sys.path.insert(0, current_dir)
    from crypto_autoengine_api import initialize_managers, router
    from crypto_autoengine_config import get_config
    from trading_config import trading_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("crypto_autoengine.log"),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)


class CryptoAutoEngine:
    """Main CRYPTO AUTOENGINE application"""

    def __init__(self):
        self.app = FastAPI(
            title="CRYPTO AUTOENGINE",
            description="Complete cryptocurrency trading automation system",
            version="1.0.0",
        )

        self.config = get_config()
        self.redis_client: redis.Redis | None = None

        # Setup CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Include API routes
        self.app.include_router(router)

        # Add startup and shutdown events
        self.app.add_event_handler("startup", self.startup_event)  # type: ignore
        self.app.add_event_handler("shutdown", self.shutdown_event)  # type: ignore

        logger.info("CRYPTO AUTOENGINE application initialized")

    async def startup_event(self) -> None:
        """Application startup event"""
        logger.info("Starting CRYPTO AUTOENGINE...")

        try:
            # Initialize Redis connection
            await self._initialize_redis()

            # Initialize all managers
            if self.redis_client:
                initialize_managers(self.redis_client)

            logger.info("CRYPTO AUTOENGINE startup completed successfully")

        except Exception as e:
            logger.error(f"Startup failed: {e}")
            raise

    async def shutdown_event(self) -> None:
        """Application shutdown event"""
        logger.info("Shutting down CRYPTO AUTOENGINE...")

        try:
            # Close Redis connection
            if self.redis_client:
                self.redis_client.close()
                logger.info("Redis connection closed")

            logger.info("CRYPTO AUTOENGINE shutdown completed")

        except Exception as e:
            logger.error(f"Shutdown error: {e}")

    async def _initialize_redis(self) -> None:
        """Initialize Redis connection"""
        try:
            # Try to connect to Redis
            self.redis_client = redis.Redis(
                host=trading_config.DEFAULT_REDIS_HOST,
                port=trading_config.DEFAULT_REDIS_PORT,
                db=trading_config.DEFAULT_REDIS_DB,
                decode_responses=True,
                socket_connect_timeout=trading_config.DEFAULT_REQUEST_TIMEOUT,
                socket_timeout=trading_config.DEFAULT_REQUEST_TIMEOUT,
            )

            # Test connection - run ping in a separate thread to avoid blocking
            # the event loop since redis-py is synchronous
            loop = asyncio.get_event_loop()
            if self.redis_client:
                ping_func: Callable[[], str] = cast(Callable[[], str], self.redis_client.ping)
                await loop.run_in_executor(None, ping_func)
            logger.info("Redis connection established")

        except redis.ConnectionError:
            logger.warning("Redis not available, using in-memory cache only")
            self.redis_client = None
        except Exception as e:
            logger.error(f"Redis initialization error: {e}")
            self.redis_client = None

    def get_app(self) -> FastAPI:
        """Get the FastAPI application"""
        return self.app


def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    crypto_engine = CryptoAutoEngine()
    return crypto_engine.get_app()


async def main() -> None:
    """Main entry point"""
    logger.info("=" * 60)
    logger.info("CRYPTO AUTOENGINE STARTING")
    logger.info("=" * 60)

    # Display configuration
    config = get_config()
    logger.info("Configuration loaded:")
    logger.info(f"  - Total coins: {len(config.all_coins)}")
    logger.info(f"  - Coinbase coins: {len(config.coinbase_coins)}")
    logger.info(f"  - Binance coins: {len(config.binance_coins)}")
    logger.info(f"  - Price fetch interval: {config.fetcher_config.price_fetch_interval}s")
    logger.info(f"  - Volume fetch interval: {config.fetcher_config.volume_fetch_interval}s")
    logger.info(f"  - Indicator calc interval: {config.fetcher_config.indicator_calc_interval}s")
    logger.info(f"  - Mystic fetch interval: {config.fetcher_config.mystic_fetch_interval}s")

    # Create application
    app = create_app()

    # Start the server
    import uvicorn

    logger.info("Starting FastAPI server...")
    config = uvicorn.Config(app=app, host="127.0.0.1", port=trading_config.DEFAULT_SERVICE_PORT, log_level="info", reload=False)

    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("CRYPTO AUTOENGINE stopped by user")
    except Exception as e:
        logger.error(f"CRYPTO AUTOENGINE failed: {e}")
        sys.exit(1)


