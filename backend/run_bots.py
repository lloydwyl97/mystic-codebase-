#!/usr/bin/env python3
"""
Trading Bots Startup Script
Runs both Coinbase and Binance bots with proper error handling
"""

import asyncio
import signal
import sys
from datetime import timezone, datetime
from typing import Any

# Import the bot manager
from bot_manager import BotManager

# Import rotated logging system
from backend.utils.log_rotation_manager import get_log_rotation_manager

# Configure logging with rotation
log_manager = get_log_rotation_manager()
logger = log_manager.setup_logger("trading_bots", "trading_bots.log")


class TradingBotsRunner:
    def __init__(self):
        self.bot_manager = BotManager()
        self.is_running = False

        # Set up signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        logger.info("Trading Bots Runner initialized")

    def signal_handler(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.is_running = False

    async def run(self):
        """Main runner function"""
        logger.info("Starting Trading Bots Runner...")
        self.is_running = True

        try:
            # Start the bot manager
            await self.bot_manager.run()

        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        except Exception as e:
            logger.error(f"Fatal error in Trading Bots Runner: {e}")
        finally:
            await self.cleanup()

    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up resources...")
        self.is_running = False

        try:
            await self.bot_manager.stop_all_bots()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

        logger.info("Trading Bots Runner stopped")


def main():
    """Main function"""
    print("=" * 60)
    print("MYSTIC TRADING BOTS")
    print("=" * 60)
    print(f"Starting at: {datetime.now(timezone.utc).isoformat()}")
    print("Press Ctrl+C to stop")
    print("=" * 60)

    try:
        runner = TradingBotsRunner()
        asyncio.run(runner.run())
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


