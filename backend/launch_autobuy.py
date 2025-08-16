#!/usr/bin/env python3
"""
Binance US Autobuy Launcher
Main launcher for the SOLUSDT, BTCUSDT, ETHUSDT, AVAXUSDT autobuy system
"""

import asyncio
import logging
import os
import signal
import sys
from datetime import datetime, timezone
from typing import Optional

from backend.endpoints.autobuy.autobuy_config import (
    validate_and_load_config,
    get_config,
)
from binance_us_autobuy import autobuy_system
from autobuy_dashboard import app as dashboard_app
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/autobuy_launcher.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class AutobuyLauncher:
    """Main launcher for the Binance US autobuy system"""

    def __init__(self):
        self.config = get_config()
        self.autobuy_task: Optional[asyncio.Task] = None
        self.dashboard_task: Optional[asyncio.Task] = None
        self.is_running = False
        self.start_time = datetime.now(timezone.utc)

    async def start_autobuy_system(self):
        """Start the autobuy system"""
        try:
            logger.info("ðŸš€ Starting Binance US Autobuy System...")
            await autobuy_system.run()
        except Exception as e:
            logger.error(f"âŒ Autobuy system error: {e}")

    async def start_dashboard(self):
        """Start the dashboard server"""
        try:
            logger.info("ðŸŒ Starting Autobuy Dashboard...")
            config = uvicorn.Config(
                dashboard_app,
                host="0.0.0.0",
                port=8080,
                log_level="info",
                access_log=True,
            )
            server = uvicorn.Server(config)
            await server.serve()
        except Exception as e:
            logger.error(f"âŒ Dashboard error: {e}")

    async def run_system(self):
        """Run the complete autobuy system"""
        try:
            # Validate configuration
            if not validate_and_load_config():
                logger.error("âŒ Configuration validation failed")
                return

            logger.info("âœ… Configuration validated successfully")

            # Start both autobuy system and dashboard
            self.autobuy_task = asyncio.create_task(self.start_autobuy_system())
            self.dashboard_task = asyncio.create_task(self.start_dashboard())

            self.is_running = True

            # Wait for both tasks
            await asyncio.gather(self.autobuy_task, self.dashboard_task)

        except KeyboardInterrupt:
            logger.info("â¹ï¸ Shutdown requested...")
        except Exception as e:
            logger.error(f"âŒ System error: {e}")
        finally:
            await self.shutdown()

    async def shutdown(self):
        """Shutdown the system gracefully"""
        logger.info("ðŸ”„ Shutting down autobuy system...")

        self.is_running = False

        # Cancel tasks
        if self.autobuy_task:
            self.autobuy_task.cancel()
        if self.dashboard_task:
            self.dashboard_task.cancel()

        # Wait for tasks to complete
        if self.autobuy_task:
            try:
                await self.autobuy_task
            except asyncio.CancelledError:
                pass

        if self.dashboard_task:
            try:
                await self.dashboard_task
            except asyncio.CancelledError:
                pass

        # Cleanup autobuy system
        await autobuy_system.cleanup()

        logger.info("âœ… System shutdown complete")

    def get_status(self) -> dict:
        """Get system status"""
        uptime = datetime.now(timezone.utc) - self.start_time

        return {
            "is_running": self.is_running,
            "uptime": str(uptime),
            "start_time": self.start_time.isoformat(),
            "autobuy_task_running": (self.autobuy_task and not self.autobuy_task.done()),
            "dashboard_task_running": (self.dashboard_task and not self.dashboard_task.done()),
            "config": self.config.to_dict(),
        }


# Global launcher instance
launcher = AutobuyLauncher()


def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"ðŸ“¡ Received signal {signum}, shutting down...")
    asyncio.create_task(launcher.shutdown())


async def main():
    """Main function"""
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("ðŸš€ Binance US Autobuy Launcher Starting...")
    logger.info("=" * 60)
    logger.info("ðŸŽ¯ Trading Pairs: SOLUSDT, BTCUSDT, ETHUSDT, AVAXUSDT")
    logger.info("ðŸ’° Strategy: Aggressive Autobuy")
    logger.info("ðŸŒ Dashboard: http://localhost:8080")
    logger.info("=" * 60)

    try:
        await launcher.run_system()
    except Exception as e:
        logger.error(f"âŒ Fatal error: {e}")
        sys.exit(1)


def print_banner():
    """Print startup banner"""
    banner = """
    â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—
    â•‘                    BINANCE US AUTOBUY SYSTEM                 â•‘
    â•‘                                                              â•‘
    â•‘  ðŸŽ¯ Trading Pairs: SOLUSDT, BTCUSDT, ETHUSDT, AVAXUSDT      â•‘
    â•‘  ðŸ’° Strategy: Aggressive Autobuy                            â•‘
    â•‘  ðŸŒ Dashboard: http://localhost:8080                        â•‘
    â•‘  ðŸ“Š Real-time monitoring and control                        â•‘
    â•‘                                                              â•‘
    â•‘  âš ï¸  WARNING: This system executes real trades!              â•‘
    â•‘  ðŸ’¡ Ensure proper API configuration before starting        â•‘
    â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    """
    print(banner)


def check_environment():
    """Check environment setup"""
    print("ðŸ” Checking environment...")

    # Check required environment variables
    required_vars = ["BINANCE_US_API_KEY", "BINANCE_US_SECRET_KEY"]

    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        print("âŒ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nðŸ’¡ Please set these variables in your .env file")
        return False

    print("âœ… Environment check passed")
    return True


if __name__ == "__main__":
    print_banner()

    if not check_environment():
        sys.exit(1)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nâ¹ï¸ Shutdown requested by user")
    except Exception as e:
        print(f"\nâŒ Fatal error: {e}")
        sys.exit(1)


