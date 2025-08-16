#!/usr/bin/env python3
"""
AI Trading Platform Startup Script
Run this to start the complete AI trading system
"""

import asyncio
import logging

from ai_trading_integration import AITradingIntegration
from backup_utils import snapshot
from notifier import send_alert


async def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    logger.info("ðŸš€ Starting Mystic AI Trading Platform...")

    # Send startup notification
    send_alert("ðŸ¤– Mystic AI Trading Platform is starting up...")

    # Create backup
    snapshot()

    # Initialize AI trading integration
    ai_trader = AITradingIntegration()

    try:
        # Start trading loop and daily tasks concurrently
        await asyncio.gather(ai_trader.start_trading_loop(), ai_trader.run_daily_tasks())
    except KeyboardInterrupt:
        logger.info("Received shutdown signal...")
        ai_trader.stop_trading()
        send_alert("ðŸ›‘ Mystic AI Trading Platform shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        send_alert(f"ðŸ’¥ Fatal error in AI Trading Platform: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())


