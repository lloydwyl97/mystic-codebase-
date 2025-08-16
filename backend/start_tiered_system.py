#!/usr/bin/env python3
"""
Startup Script for Tiered Signal System
Demonstrates how to initialize and run the new tiered signal architecture
"""

import asyncio
import logging
import time
from typing import Any, Optional

import redis.asyncio as redis

from backend.services.market_data import MarketDataService
from .services.notification import get_notification_service
from .services.service_manager import service_manager
from .services.trading import get_trading_service

# Import the tiered system components
from unified_signal_manager import UnifiedSignalManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("tiered_system.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("tiered_system")


class TieredSystemManager:
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        self.unified_manager: Optional[UnifiedSignalManager] = None
        self.is_running = False

        # Service instances
        self.service_manager: Any = None
        self.market_service: Optional[MarketDataService] = None
        self.notification_service: Any = None
        self.trading_service: Any = None

        logger.info("Tiered System Manager initialized")

    async def initialize_redis(self) -> bool:
        """Initialize Redis connection"""
        try:
            # Connect to Redis using the async client
            self.redis_client = redis.from_url(self.redis_url)
            # Test connection with await
            if self.redis_client:
                ping_result = await self.redis_client.ping()
                logger.debug(f"Redis ping result: {ping_result}")
            logger.info("Redis connection established")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            logger.info("Running in fallback mode without Redis")
            return False

    async def initialize_components(self) -> bool:
        """Initialize all tiered system components"""
        try:
            if self.redis_client:
                # Initialize unified signal manager
                self.unified_manager = UnifiedSignalManager(self.redis_client)
                logger.info("All tiered system components initialized")
                return True
            else:
                logger.warning("Redis not available - running in limited mode")
                return False
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            return False

    async def start_system(self) -> bool:
        """Start the complete tiered signal system"""
        logger.info("ðŸš€ Starting Tiered Signal System...")

        try:
            # Initialize Redis
            redis_available = await self.initialize_redis()

            # Initialize components only if Redis is available
            if redis_available:
                components_ready = await self.initialize_components()

                # Initialize services
                logger.info("Initializing services...")

                # Initialize service manager
                self.service_manager = service_manager
                logger.info(f"Service manager initialized: {self.service_manager is not None}")

                # Initialize market data service
                self.market_service = MarketDataService()
                logger.info("Market data service initialized")

                # Get notification service with Redis client
                self.notification_service = get_notification_service(self.redis_client)
                logger.info("Notification service initialized")

                # Get trading service with Redis client
                self.trading_service = get_trading_service(self.redis_client)
                logger.info("Trading service initialized")
            else:
                logger.warning("Skipping component initialization - Redis not available")
                components_ready = False

            if not components_ready:
                logger.error("Failed to initialize components")
                return False

            self.is_running = True

            # Start the unified signal manager
            if self.unified_manager:
                logger.info("Starting unified signal manager...")
                await self.unified_manager.run()

            return True

        except Exception as e:
            logger.error(f"Error starting tiered system: {e}")
            return False

    async def stop_system(self) -> None:
        """Stop the tiered signal system"""
        logger.info("ðŸ›‘ Stopping Tiered Signal System...")
        self.is_running = False

        if self.unified_manager:
            await self.unified_manager.stop_all_components()

        if self.redis_client:
            await self.redis_client.close()

        logger.info("Tiered system stopped")

    async def run_demo(self) -> None:
        """Run a demonstration of the tiered system"""
        logger.info("ðŸŽ¯ Running Tiered System Demo...")

        try:
            # Start the system
            if not await self.start_system():
                return

            # Run for a demonstration period
            demo_duration = 300  # 5 minutes
            start_time = time.time()

            while self.is_running and (time.time() - start_time) < demo_duration:
                try:
                    # Get system status
                    if self.unified_manager:
                        status = self.unified_manager.get_status()
                        logger.info(f"System Status: {status['manager_status']}")

                        # Get signal summary
                        summary = await self.unified_manager.get_signal_summary()
                        logger.info(f"Signal Summary: {summary}")

                        # Get active signals
                        active_signals = await self.unified_manager.get_active_signals()
                        if active_signals:
                            logger.info(f"Active Signals: {len(active_signals)}")
                            for signal in active_signals[:3]:  # Show first 3
                                logger.info(
                                    f"  - {signal['symbol']}: {signal['action']} "
                                    f"(confidence: {signal['confidence']:.2f})"
                                )

                    # Wait before next check
                    await asyncio.sleep(30)  # Check every 30 seconds

                except Exception as e:
                    logger.error(f"Error in demo loop: {e}")
                    await asyncio.sleep(10)

            logger.info("Demo completed")

        except Exception as e:
            logger.error(f"Error in demo: {e}")
        finally:
            await self.stop_system()

    async def run_continuous(self) -> None:
        """Run the system continuously"""
        logger.info("ðŸ”„ Running Tiered System Continuously...")

        try:
            # Start the system
            if not await self.start_system():
                return

            # Keep running until interrupted
            while self.is_running:
                try:
                    await asyncio.sleep(60)  # Check every minute
                except KeyboardInterrupt:
                    logger.info("Received interrupt signal")
                    break
                except Exception as e:
                    logger.error(f"Error in continuous run: {e}")
                    await asyncio.sleep(10)

        except Exception as e:
            logger.error(f"Error in continuous run: {e}")
        finally:
            await self.stop_system()


def print_system_info() -> None:
    """Print information about the tiered system"""
    print(
        """
ðŸ”§ TIERED SIGNAL SYSTEM - MYSTIC TRADING PLATFORM
================================================

ðŸ“Š SYSTEM ARCHITECTURE:
â”œâ”€â”€ Tier 1: Real-Time Signals (5-15 sec)
â”‚   â”œâ”€â”€ Price data (5-10 sec)
â”‚   â”œâ”€â”€ Momentum spikes (10-15 sec)
â”‚   â””â”€â”€ Order book depth (15 sec)
â”‚
â”œâ”€â”€ Tier 2: Tactical Strategy (1-5 min)
â”‚   â”œâ”€â”€ RSI / MACD (1-3 min)
â”‚   â”œâ”€â”€ 24h Volume (2-3 min)
â”‚   â”œâ”€â”€ Volatility Index (2-5 min)
â”‚   â””â”€â”€ Time-based changes (1-3 min)
â”‚
â””â”€â”€ Tier 3: Mystic/Cosmic (30 min - 1 hr)
    â”œâ”€â”€ Schumann Resonance (1 hr)
    â”œâ”€â”€ Solar Flare Index (1 hr)
    â””â”€â”€ Pineal Alignment (1+ hr)

ðŸŽ¯ TRADE DECISION ENGINE:
â”œâ”€â”€ Combines all three tiers
â”œâ”€â”€ Makes decisions every 3-10 seconds
â”œâ”€â”€ Provides unified coin state objects
â””â”€â”€ Generates confidence-based signals

ðŸ“ˆ API ENDPOINTS:
â”œâ”€â”€ /api/signals/tier1 - Real-time price signals
â”œâ”€â”€ /api/signals/tier2 - Technical indicators
â”œâ”€â”€ /api/signals/tier3 - Cosmic/mystic signals
â”œâ”€â”€ /api/signals/unified - Combined signals
â”œâ”€â”€ /api/signals/trade-decisions - Trading decisions
â””â”€â”€ /api/signals/summary - System status

ðŸš€ USAGE:
python start_tiered_system.py [demo|continuous]
    """
    )


async def main() -> None:
    """Main function"""
    import sys

    # Print system information
    print_system_info()

    # Parse command line arguments
    mode = "demo"
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()

    # Create system manager
    manager = TieredSystemManager()

    try:
        if mode == "demo":
            logger.info("Running in DEMO mode (5 minutes)")
            await manager.run_demo()
        elif mode == "continuous":
            logger.info("Running in CONTINUOUS mode")
            await manager.run_continuous()
        else:
            logger.error(f"Unknown mode: {mode}")
            logger.info("Available modes: demo, continuous")
            return

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        await manager.stop_system()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Tiered system stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        exit(1)


