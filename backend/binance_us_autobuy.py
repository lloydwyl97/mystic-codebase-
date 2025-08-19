#!/usr/bin/env python3
"""
Binance US Autobuy System
Focused on SOLUSDT, BTCUSDT, ETHUSDT with aggressive autobuy logic
"""

import asyncio
import hashlib
import hmac
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlencode

import aiohttp
import requests
from dotenv import load_dotenv
from mystic_config import mystic_config

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/binance_us_autobuy.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Binance US Configuration
BINANCE_API_KEY = mystic_config.exchange.binance_us_api_key
BINANCE_SECRET_KEY = mystic_config.exchange.binance_us_secret_key
BINANCE_BASE_URL = "https://api.binance.us"

# Trading Configuration
TRADING_PAIRS = ["SOLUSDT", "BTCUSDT", "ETHUSDT"]
USD_AMOUNT_PER_TRADE = float(os.getenv("USD_AMOUNT_PER_TRADE", "50"))
MAX_CONCURRENT_TRADES = int(os.getenv("MAX_CONCURRENT_TRADES", "4"))
TRADING_ENABLED = os.getenv("TRADING_ENABLED", "true").lower() == "true"

# Signal Configuration
MIN_VOLUME_INCREASE = float(os.getenv("MIN_VOLUME_INCREASE", "1.5"))  # 50% volume increase
MIN_PRICE_CHANGE = float(os.getenv("MIN_PRICE_CHANGE", "0.02"))  # 2% price change
SIGNAL_COOLDOWN = int(os.getenv("SIGNAL_COOLDOWN", "300"))  # 5 minutes

# Notification Configuration
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")


class BinanceUSAutobuy:
    def __init__(self):
        self.session: aiohttp.ClientSession | None = None
        self.active_trades: dict[str, dict[str, Any]] = {}
        self.trade_history: list[dict[str, Any]] = []
        self.signal_history: dict[str, list[dict[str, Any]]] = {}
        self.last_signal_time: dict[str, float] = {}
        self.is_running = False

        # Statistics
        self.total_trades = 0
        self.successful_trades = 0
        self.failed_trades = 0
        self.total_volume = 0.0

        logger.info(f"Binance US Autobuy initialized for pairs: {TRADING_PAIRS}")

    async def initialize(self):
        """Initialize the autobuy system"""
        if not BINANCE_API_KEY or not BINANCE_SECRET_KEY:
            logger.error("âŒ Binance US API credentials not configured")
            return False

        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10, connect=5))

        # Test connection
        try:
            account_info: dict[str, Any] | None = await self.get_account_info()
            if account_info:
                balance = account_info.get("totalWalletBalance", "N/A")
                logger.info("âœ… Binance US connection successful")
                logger.info(f"Account balance: {balance} USDT")
                return True
            else:
                logger.error("âŒ Failed to connect to Binance US")
                return False
        except Exception as e:
            logger.error(f"âŒ Connection test failed: {e}")
            return False

    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
        logger.info("âœ… Binance US Autobuy cleaned up")

    def send_notification(self, message: str):
        """Send notification via Telegram and Discord"""
        try:
            # Telegram
            if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                requests.post(
                    f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                    data={"chat_id": TELEGRAM_CHAT_ID, "text": message},
                    timeout=5,
                )

            # Discord
            if DISCORD_WEBHOOK_URL:
                requests.post(DISCORD_WEBHOOK_URL, json={"content": message}, timeout=5)

            logger.info(f"ðŸ“¢ Notification sent: {message}")
        except Exception as e:
            logger.error(f"âŒ Failed to send notification: {e}")

    async def get_account_info(self) -> dict[str, Any] | None:
        """Get Binance US account information"""
        if self.session is None:
            raise RuntimeError("Session is not initialized. Call initialize() first.")

        try:
            timestamp = int(time.time() * 1000)
            params = {"timestamp": timestamp}
            query = urlencode(params)

            signature = hmac.new(
                BINANCE_SECRET_KEY.encode(), query.encode(), hashlib.sha256
            ).hexdigest()

            url = f"{BINANCE_BASE_URL}/api/v3/account?{query}&signature={signature}"
            headers = {"X-MBX-APIKEY": BINANCE_API_KEY}

            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"âŒ Account info failed: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"âŒ Error getting account info: {e}")
            return None

    async def get_ticker_24hr(self, symbol: str) -> dict[str, Any] | None:
        """Get 24hr ticker for a symbol"""
        if self.session is None:
            raise RuntimeError("Session is not initialized. Call initialize() first.")

        try:
            url = f"{BINANCE_BASE_URL}/api/v3/ticker/24hr"
            params = {"symbol": symbol}

            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"âŒ Ticker failed for {symbol}: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"âŒ Error getting ticker for {symbol}: {e}")
            return None

    async def get_current_price(self, symbol: str) -> float | None:
        """Get current price for a symbol"""
        if self.session is None:
            raise RuntimeError("Session is not initialized. Call initialize() first.")

        try:
            url = f"{BINANCE_BASE_URL}/api/v3/ticker/price"
            params = {"symbol": symbol}

            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return float(data["price"])
                else:
                    logger.error(f"âŒ Price failed for {symbol}: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"âŒ Error getting price for {symbol}: {e}")
            return None

    async def place_market_buy_order(
        self, symbol: str, quote_amount: float
    ) -> dict[str, Any] | None:
        """Place a market buy order"""
        if not TRADING_ENABLED:
            logger.info(f"ðŸ”„ Trading disabled - simulating buy order for {symbol}")
            return {
                "symbol": symbol,
                "orderId": f"sim_{int(time.time())}",
                "status": "FILLED",
                "executedQty": str(quote_amount),
                "price": "0",
            }

        if self.session is None:
            raise RuntimeError("Session is not initialized. Call initialize() first.")

        try:
            timestamp = int(time.time() * 1000)
            params = {
                "symbol": symbol,
                "side": "BUY",
                "type": "MARKET",
                "quoteOrderQty": str(quote_amount),
                "timestamp": timestamp,
            }

            query = urlencode(params)
            signature = hmac.new(
                BINANCE_SECRET_KEY.encode(), query.encode(), hashlib.sha256
            ).hexdigest()

            url = f"{BINANCE_BASE_URL}/api/v3/order?{query}&signature={signature}"
            headers = {"X-MBX-APIKEY": BINANCE_API_KEY}

            async with self.session.post(url, headers=headers) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"âœ… Buy order executed: {symbol} ${quote_amount}")
                    return result
                else:
                    error_text = await response.text()
                    logger.error(
                        f"âŒ Buy order failed for {symbol}: {response.status} - {error_text}"
                    )
                    return None
        except Exception as e:
            logger.error(f"âŒ Error placing buy order for {symbol}: {e}")
            return None

    def analyze_buy_signal(self, symbol: str, ticker_data: dict[str, Any]) -> dict[str, Any]:
        """Analyze if we should buy based on ticker data"""
        if self.session is None:
            raise RuntimeError("Session is not initialized. Call initialize() first.")

        try:
            # Extract key metrics
            price_change = float(ticker_data.get("priceChangePercent", "0"))
            quote_volume = float(ticker_data.get("quoteVolume", "0"))
            high_24h = float(ticker_data.get("highPrice", "0"))
            low_24h = float(ticker_data.get("lowPrice", "0"))
            current_price = float(ticker_data.get("lastPrice", "0"))

            # Calculate additional metrics
            price_range = high_24h - low_24h
            price_position = (current_price - low_24h) / price_range if price_range > 0 else 0.5

            # Signal conditions
            signals: list[str] = []
            confidence = 0

            # 1. Price momentum (positive change)
            if price_change > MIN_PRICE_CHANGE:
                signals.append(f"Price up {price_change:.2f}%")
                confidence += 25

            # 2. Volume spike
            if quote_volume > 1000000:  # $1M+ volume
                signals.append(f"High volume: ${quote_volume:,.0f}")
                confidence += 20

            # 3. Price near 24h low (bounce opportunity)
            if price_position < 0.3:
                signals.append("Price near 24h low")
                confidence += 15

            # 4. Strong upward momentum
            if price_change > 5:
                signals.append("Strong upward momentum")
                confidence += 20

            # 5. High volatility (opportunity)
            if price_range / current_price > 0.05:
                signals.append("High volatility")
                confidence += 10

            # 6. Recent signal history
            if symbol in self.signal_history:
                recent_signals = [
                    s for s in self.signal_history[symbol] if time.time() - s["timestamp"] < 3600
                ]  # Last hour
                if len(recent_signals) >= 2:
                    signals.append("Multiple recent signals")
                    confidence += 10

            # Determine if we should buy
            should_buy = confidence >= 50 and len(signals) >= 2

            return {
                "should_buy": should_buy,
                "confidence": confidence,
                "signals": signals,
                "price_change": price_change,
                "volume": quote_volume,
                "price_position": price_position,
                "current_price": current_price,
            }

        except Exception as e:
            logger.error(f"âŒ Error analyzing signal for {symbol}: {e}")
            return {
                "should_buy": False,
                "confidence": 0,
                "signals": [],
                "error": str(e),
            }

    async def execute_buy_signal(self, symbol: str, signal_data: dict[str, Any]):
        """Execute a buy signal"""
        if self.session is None:
            raise RuntimeError("Session is not initialized. Call initialize() first.")

        try:
            # Check if we already have an active trade for this symbol
            if symbol in self.active_trades:
                logger.info(f"â³ Already have active trade for {symbol}, skipping")
                return

            # Check if we've exceeded max concurrent trades
            if len(self.active_trades) >= MAX_CONCURRENT_TRADES:
                logger.info(f"â³ Max concurrent trades reached ({MAX_CONCURRENT_TRADES}), skipping")
                return

            # Check cooldown
            current_time = time.time()
            if symbol in self.last_signal_time:
                if current_time - self.last_signal_time[symbol] < SIGNAL_COOLDOWN:
                    logger.info(f"â³ Signal cooldown for {symbol}, skipping")
                    return

            # Execute the buy order
            logger.info(f"ðŸš€ Executing buy signal for {symbol}")
            logger.info(f"   Confidence: {signal_data['confidence']}%")
            logger.info(f"   Signals: {', '.join(signal_data['signals'])}")

            order_result: dict[str, Any] | None = await self.place_market_buy_order(
                symbol, USD_AMOUNT_PER_TRADE
            )

            if order_result:
                # Record the trade
                trade_record = {
                    "symbol": symbol,
                    "order_id": order_result.get("orderId", "unknown"),
                    "amount_usd": USD_AMOUNT_PER_TRADE,
                    "price": float(order_result.get("price", 0)),
                    "quantity": float(order_result.get("executedQty", 0)),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "confidence": signal_data["confidence"],
                    "signals": signal_data["signals"],
                    "status": "executed",
                }

                self.active_trades[symbol] = trade_record
                self.trade_history.append(trade_record)
                self.total_trades += 1
                self.successful_trades += 1
                self.total_volume += USD_AMOUNT_PER_TRADE
                self.last_signal_time[symbol] = current_time

                # Send notification
                message = (
                    f"ðŸš€ BINANCE US AUTOBUY EXECUTED\n"
                    f"Symbol: {symbol}\n"
                    f"Amount: ${USD_AMOUNT_PER_TRADE}\n"
                    f"Confidence: {signal_data['confidence']}%\n"
                    f"Signals: {', '.join(signal_data['signals'])}\n"
                    f"Order ID: {order_result.get('orderId', 'N/A')}"
                )
                self.send_notification(message)

                logger.info(f"âœ… Buy order successful for {symbol}")
            else:
                logger.error(f"âŒ Buy order failed for {symbol}")
                self.failed_trades += 1

        except Exception as e:
            logger.error(f"âŒ Error executing buy signal for {symbol}: {e}")
            self.failed_trades += 1

    async def process_trading_pair(self, symbol: str):
        """Process a single trading pair"""
        if self.session is None:
            raise RuntimeError("Session is not initialized. Call initialize() first.")

        try:
            # Get 24hr ticker data
            ticker_data = await self.get_ticker_24hr(symbol)
            if not ticker_data:
                return

            # Analyze for buy signals
            signal_data = self.analyze_buy_signal(symbol, ticker_data)

            # Record signal history
            if symbol not in self.signal_history:
                self.signal_history[symbol] = []

            self.signal_history[symbol].append(
                {
                    "timestamp": time.time(),
                    "confidence": signal_data["confidence"],
                    "signals": signal_data["signals"],
                    "price_change": signal_data.get("price_change", 0),
                }
            )

            # Keep only last 100 signals
            if len(self.signal_history[symbol]) > 100:
                self.signal_history[symbol] = self.signal_history[symbol][-100:]

            # Execute buy signal if conditions are met
            if signal_data["should_buy"]:
                await self.execute_buy_signal(symbol, signal_data)

        except Exception as e:
            logger.error(f"âŒ Error processing {symbol}: {e}")

    async def run_trading_cycle(self):
        """Run one complete trading cycle"""
        logger.info("ðŸ”„ Starting trading cycle...")

        # Process all trading pairs
        tasks = [self.process_trading_pair(symbol) for symbol in TRADING_PAIRS]
        await asyncio.gather(*tasks, return_exceptions=True)

        # Log statistics
        logger.info(f"ðŸ“Š Cycle complete - Active trades: {len(self.active_trades)}")
        logger.info(
            f"ðŸ“Š Total trades: {self.total_trades}, Success: {self.successful_trades}, Failed: {self.failed_trades}"
        )

    async def run(self):
        """Main trading loop"""
        logger.info("ðŸš€ Starting Binance US Autobuy System")
        logger.info(f"ðŸ“Š Trading pairs: {TRADING_PAIRS}")
        logger.info(f"ðŸ’° Amount per trade: ${USD_AMOUNT_PER_TRADE}")
        logger.info(f"ðŸ”„ Trading enabled: {TRADING_ENABLED}")

        # Initialize
        if not await self.initialize():
            logger.error("âŒ Failed to initialize autobuy system")
            return

        self.is_running = True

        try:
            while self.is_running:
                await self.run_trading_cycle()

                # Wait before next cycle (30 seconds)
                await asyncio.sleep(30)

        except KeyboardInterrupt:
            logger.info("â¹ï¸ Shutting down autobuy system...")
        except Exception as e:
            logger.error(f"âŒ Error in main loop: {e}")
        finally:
            self.is_running = False
            await self.cleanup()

    def get_status(self) -> dict[str, Any]:
        """Get current system status"""
        return {
            "is_running": self.is_running,
            "trading_pairs": TRADING_PAIRS,
            "active_trades": len(self.active_trades),
            "total_trades": self.total_trades,
            "successful_trades": self.successful_trades,
            "failed_trades": self.failed_trades,
            "total_volume": self.total_volume,
            "trading_enabled": TRADING_ENABLED,
            "last_update": datetime.now(timezone.utc).isoformat(),
        }


# Global instance
autobuy_system = BinanceUSAutobuy()


async def main():
    """Main function"""
    await autobuy_system.run()


if __name__ == "__main__":
    asyncio.run(main())


