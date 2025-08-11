"""
Alerts - Discord + Telegram Live Push

Sends real-time alerts to Discord and Telegram
"""

import logging
import os
from typing import Optional

import aiohttp

logger = logging.getLogger("alerts")

# Configuration
DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK_URL")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


async def send_discord(message: str) -> bool:
    """Send message to Discord webhook"""
    if not DISCORD_WEBHOOK:
        logger.debug("Discord webhook not configured")
        return False

    try:
        async with aiohttp.ClientSession() as session:
            payload = {
                "content": message,
                "username": "Mystic Trading Bot",
                "avatar_url": ("https://img.icons8.com/color/96/000000/crystal-ball.png"),
            }

            async with session.post(DISCORD_WEBHOOK, json=payload) as resp:
                if resp.status == 204:
                    logger.debug("✅ Discord message sent")
                    return True
                else:
                    logger.error(f"❌ Discord API error: {resp.status}")
                    return False
    except Exception as e:
        logger.error(f"❌ Discord send error: {e}")
        return False


async def send_telegram(message: str) -> bool:
    """Send message to Telegram bot"""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.debug("Telegram bot not configured")
        return False

    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "HTML",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=payload) as resp:
                if resp.status == 200:
                    logger.debug("✅ Telegram message sent")
                    return True
                else:
                    logger.error(f"❌ Telegram API error: {resp.status}")
                    return False
    except Exception as e:
        logger.error(f"❌ Telegram send error: {e}")
        return False


async def broadcast_alert(message: str) -> bool:
    """Send alert to all configured channels"""
    try:
        discord_sent = await send_discord(message)
        telegram_sent = await send_telegram(message)

        if discord_sent or telegram_sent:
            logger.info(f"📢 Alert broadcast: {message[:50]}...")
            return True
        else:
            logger.warning("⚠️ No alert channels configured")
            return False
    except Exception as e:
        logger.error(f"❌ Broadcast error: {e}")
        return False


async def send_trade_alert(
    symbol: str,
    action: str,
    price: float,
    amount: float,
    pnl: Optional[float] = None,
):
    """Send formatted trade alert"""
    try:
        if action == "BUY":
            message = (
                f"🟢 **BUY ORDER**\n"
                f"Symbol: {symbol}\n"
                f"Price: ${price:.4f}\n"
                f"Amount: {amount:.4f}\n"
                f"Total: ${price * amount:.2f}"
            )
        elif action == "SELL":
            pnl_text = f"P&L: {pnl:+.2f}%" if pnl is not None else ""
            message = (
                f"🔴 **SELL ORDER**\n"
                f"Symbol: {symbol}\n"
                f"Price: ${price:.4f}\n"
                f"Amount: {amount:.4f}\n"
                f"Total: ${price * amount:.2f}\n"
                f"{pnl_text}"
            )
        else:
            message = f"⚪ **{action}**\n" f"Symbol: {symbol}\n" f"Price: ${price:.4f}"

        await broadcast_alert(message)
    except Exception as e:
        logger.error(f"❌ Trade alert error: {e}")


async def send_market_alert(alert_type: str, data: dict):
    """Send formatted market alert"""
    try:
        if alert_type == "BREAKOUT":
            message = (
                f"🚀 **BREAKOUT DETECTED**\n"
                f"Symbol: {data.get('symbol', 'Unknown')}\n"
                f"Change: {data.get('change', 0):.2f}%\n"
                f"Price: ${data.get('price', 0):.4f}"
            )
        elif alert_type == "PUMP":
            message = (
                f"💥 **PUMP DETECTED**\n"
                f"Symbol: {data.get('symbol', 'Unknown')}\n"
                f"Volume: ${data.get('volume', 0):,.0f}\n"
                f"Rank: #{data.get('rank', 0)}"
            )
        elif alert_type == "MYSTIC":
            message = (
                f"🔮 **MYSTIC SIGNAL**\n"
                f"Message: {data.get('message', 'Unknown')}\n"
                f"Confidence: {data.get('confidence', 0)}%"
            )
        else:
            message = f"📊 **{alert_type}**\n" f"Data: {str(data)[:100]}..."

        await broadcast_alert(message)
    except Exception as e:
        logger.error(f"❌ Market alert error: {e}")


def get_alert_status() -> dict:
    """Get current alert configuration status"""
    return {
        "discord_configured": bool(DISCORD_WEBHOOK),
        "telegram_configured": bool(TELEGRAM_TOKEN and TELEGRAM_CHAT_ID),
        "total_channels": sum([bool(DISCORD_WEBHOOK), bool(TELEGRAM_TOKEN and TELEGRAM_CHAT_ID)]),
    }
