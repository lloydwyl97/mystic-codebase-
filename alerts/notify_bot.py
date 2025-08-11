#!/usr/bin/env python3
"""
Notification Bot - Discord & Telegram Alerts
Runs in a dedicated container for trading alerts
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class NotificationBot:
    """Main notification bot for trading alerts"""

    def __init__(self):
        self.running = False
        self.discord_webhook = os.getenv("DISCORD_WEBHOOK_URL", "")
        self.telegram_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID", "")

    async def initialize(self):
        """Initialize the notification bot"""
        logger.info("🚀 Initializing Notification Bot...")

        # Check configuration
        if not self.discord_webhook and not self.telegram_token:
            logger.warning("⚠️ No notification channels configured")
        else:
            if self.discord_webhook:
                logger.info("✅ Discord webhook configured")
            if self.telegram_token:
                logger.info("✅ Telegram bot configured")

        logger.info("✅ Notification Bot initialized")

    async def send_discord_alert(self, message: str):
        """Send alert to Discord"""
        if not self.discord_webhook:
            return

        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                payload = {"content": message}
                async with session.post(self.discord_webhook, json=payload) as response:
                    if response.status == 204:
                        logger.info("✅ Discord alert sent")
                    else:
                        logger.error(f"❌ Discord alert failed: {response.status}")
        except Exception as e:
            logger.error(f"❌ Discord alert error: {e}")

    async def send_telegram_alert(self, message: str):
        """Send alert to Telegram"""
        if not self.telegram_token or not self.telegram_chat_id:
            return

        try:
            import aiohttp

            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            payload = {
                "chat_id": self.telegram_chat_id,
                "text": message,
                "parse_mode": "HTML",
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        logger.info("✅ Telegram alert sent")
                    else:
                        logger.error(f"❌ Telegram alert failed: {response.status}")
        except Exception as e:
            logger.error(f"❌ Telegram alert error: {e}")

    async def send_alert(self, message: str):
        """Send alert to all configured channels"""
        await self.send_discord_alert(message)
        await self.send_telegram_alert(message)

    async def start(self):
        """Start the notification bot"""
        logger.info("🚀 Starting Notification Bot...")
        self.running = True

        await self.initialize()

        # Send startup notification
        await self.send_alert("🚀 Mystic Trading Notification Bot is online!")

        # Main loop - listen for Redis messages
        try:
            import redis.asyncio as redis

            r = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))

            while self.running:
                try:
                    # Listen for notification messages
                    message = await r.blpop("notifications", timeout=1)
                    if message:
                        _, alert_data = message
                        await self.send_alert(alert_data.decode())

                except Exception as e:
                    logger.error(f"❌ Error in notification loop: {e}")
                    await asyncio.sleep(5)

        except Exception as e:
            logger.error(f"💥 Fatal error in notification bot: {e}")
            self.running = False

    async def stop(self):
        """Stop the notification bot"""
        logger.info("🛑 Stopping Notification Bot...")
        self.running = False
        await self.send_alert("🛑 Mystic Trading Notification Bot is shutting down...")


async def main():
    """Main entry point"""
    bot = NotificationBot()

    try:
        await bot.start()
    except KeyboardInterrupt:
        logger.info("🛑 Received shutdown signal")
        await bot.stop()
    except Exception as e:
        logger.error(f"💥 Unexpected error: {e}")
        await bot.stop()


if __name__ == "__main__":
    asyncio.run(main())
