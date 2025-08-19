#!/usr/bin/env python3
"""
Bot Manager - Coordinates Coinbase and Binance trading bots
Ensures bots run independently without bandwidth conflicts
"""

import asyncio
import signal
import sys
from datetime import datetime, timezone
from types import FrameType
from typing import Any

# Import the bots
from coinbase_bot import CoinbaseBot

# Import rotated logging system
from backend.utils.log_rotation_manager import get_log_rotation_manager
from binance_bot import BinanceBot

# Configure logging with rotation
log_manager = get_log_rotation_manager()
logger = log_manager.setup_logger("bot_manager", "bot_manager.log")


class BotManager:
    def __init__(self):
        self.coinbase_bot = CoinbaseBot()
        self.binance_bot = BinanceBot()
        self.is_running = False
        self.tasks = []

        # Manager configuration
        self.config = {
            "auto_restart": True,
            "restart_delay": 30,  # seconds
            "health_check_interval": 60,  # seconds
            "max_restart_attempts": 3,
        }

        # Performance tracking
        self.stats: dict[str, str | int | None] = {
            "start_time": None,
            "restart_count": 0,
            "last_health_check": None,
        }

        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        logger.info("Bot Manager initialized with rotated logging")

    def signal_handler(self, signum: int, frame: FrameType | None) -> None:
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        if frame is not None:
            logger.debug(
                f"Signal received in function: {frame.f_code.co_name if frame.f_code else 'unknown'}"
            )
        self.is_running = False

    async def start_bot(self, bot: CoinbaseBot | BinanceBot, bot_name: str) -> None:
        """Start a bot with error handling and auto-restart"""
        restart_attempts = 0

        while self.is_running and restart_attempts < self.config["max_restart_attempts"]:
            try:
                logger.info(f"Starting {bot_name} bot...")
                await bot.run()

            except Exception as e:
                restart_attempts += 1
                logger.error(f"Error in {bot_name} bot (attempt {restart_attempts}): {e}")

                if self.is_running and self.config["auto_restart"]:
                    logger.info(
                        f"Restarting {bot_name} bot in {self.config['restart_delay']} seconds..."
                    )
                    current_restarts = self.stats.get("restart_count", 0)
                    if isinstance(current_restarts, int):
                        self.stats["restart_count"] = current_restarts + 1
                    await asyncio.sleep(self.config["restart_delay"])
                else:
                    break
            else:
                # Bot exited normally
                logger.info(f"{bot_name} bot stopped normally")
                break

        if restart_attempts >= self.config["max_restart_attempts"]:
            logger.error(f"{bot_name} bot failed to start after {restart_attempts} attempts")

    async def health_check(self):
        """Periodic health check of both bots"""
        while self.is_running:
            try:
                coinbase_status = self.coinbase_bot.get_status()
                binance_status = self.binance_bot.get_status()

                logger.info("=== Bot Health Check ===")
                logger.info(
                    f"Coinbase Bot: {coinbase_status['status']} - {coinbase_status['market_data_count']} coins"
                )
                logger.info(
                    f"Binance Bot: {binance_status['status']} - {binance_status['market_data_count']} coins"
                )

                # Check for issues
                if coinbase_status["status"] == "error":
                    logger.warning("Coinbase bot is in error state")

                if binance_status["status"] == "error":
                    logger.warning("Binance bot is in error state")

                self.stats["last_health_check"] = datetime.now(timezone.timezone.utc).isoformat()

            except Exception as e:
                logger.error(f"Error in health check: {e}")

            await asyncio.sleep(self.config["health_check_interval"])

    async def start_all_bots(self):
        """Start both bots concurrently"""
        logger.info("Starting all trading bots...")
        self.is_running = True
        self.stats["start_time"] = datetime.now(timezone.timezone.utc).isoformat()

        # Create tasks for both bots and health check
        coinbase_task = asyncio.create_task(self.start_bot(self.coinbase_bot, "Coinbase"))
        binance_task = asyncio.create_task(self.start_bot(self.binance_bot, "Binance"))
        health_task = asyncio.create_task(self.health_check())

        self.tasks = [coinbase_task, binance_task, health_task]

        try:
            # Wait for all tasks to complete
            await asyncio.gather(*self.tasks, return_exceptions=True)

        except Exception as e:
            logger.error(f"Error in bot manager: {e}")
        finally:
            await self.stop_all_bots()

    async def stop_all_bots(self):
        """Stop all bots gracefully"""
        logger.info("Stopping all trading bots...")
        self.is_running = False

        # Stop individual bots
        try:
            await self.coinbase_bot.close()
        except Exception as e:
            logger.error(f"Error closing Coinbase bot: {e}")

        try:
            await self.binance_bot.close()
        except Exception as e:
            logger.error(f"Error closing Binance bot: {e}")

        # Cancel all tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()

        logger.info("All bots stopped")

    def get_status(self) -> dict[str, Any]:
        """Get overall status of all bots"""
        return {
            "manager_status": "running" if self.is_running else "stopped",
            "start_time": self.stats["start_time"],
            "last_health_check": self.stats["last_health_check"],
            "restart_count": self.stats["restart_count"],
            "coinbase_bot": self.coinbase_bot.get_status(),
            "binance_bot": self.binance_bot.get_status(),
            "config": self.config,
        }

    async def run(self):
        """Main manager loop"""
        logger.info("Bot Manager starting...")

        try:
            await self.start_all_bots()
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        except Exception as e:
            logger.error(f"Fatal error in Bot Manager: {e}")
        finally:
            await self.stop_all_bots()
            logger.info("Bot Manager stopped")


# Global manager instance
bot_manager = BotManager()


async def main():
    """Main function"""
    print("=" * 60)
    print("MYSTIC BOT MANAGER")
    print("=" * 60)
    print(f"Starting at: {datetime.now(timezone.timezone.utc).isoformat()}")
    print("Press Ctrl+C to stop")
    print("=" * 60)

    try:
        await bot_manager.run()
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())


