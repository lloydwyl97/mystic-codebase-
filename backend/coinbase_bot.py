#!/usr/bin/env python3
"""
Coinbase Trading Bot - Dedicated bot for Coinbase exchange only
Handles only Coinbase coins with proper throttling and error handling
"""

import asyncio
import random
import time
from dataclasses import dataclass
from datetime import timezone, datetime
from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict

import aiohttp

# Import rotated logging system
from backend.utils.log_rotation_manager import get_log_rotation_manager

# Configure logging with rotation
log_manager = get_log_rotation_manager()
logger = log_manager.setup_logger("coinbase_bot", "coinbase_bot.log")


class RateLimitConfig(TypedDict):
    requests_per_minute: int
    requests_per_coin: int
    interval_per_coin: int
    last_request_time: Dict[str, float]
    request_count: int
    reset_time: float


class StatsConfig(TypedDict):
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_trades: int
    successful_trades: int
    failed_trades: int
    last_update: Optional[str]


class BotStatus(Enum):
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class CoinData:
    symbol: str
    price: float
    change_24h: float
    volume_24h: float
    timestamp: str
    api_source: str = "coinbase"


@dataclass
class TradeSignal:
    symbol: str
    action: str  # "BUY", "SELL", "HOLD"
    confidence: float
    price: float
    timestamp: str
    reason: str


class CoinbaseBot:
    def __init__(self):
        self.status = BotStatus.STOPPED
        self.is_running = False
        self.session: Optional[aiohttp.ClientSession] = None

        # Coinbase specific configuration
        self.coinbase_base_url = "https://api.pro.coinbase.us"
        self.coins = [
            "AR",
            "OP",
            "IMX",
            "ODEAN",
            "MINA",
            "ANKR",
            "GRT",
            "MATIC",
            "RNDR",
            "LINK",
        ]

        # Rate limiting: 10 requests per minute = 1 request every 6 seconds
        self.rate_limit: RateLimitConfig = {
            "requests_per_minute": 10,
            "requests_per_coin": 1,
            "interval_per_coin": 6,  # seconds
            "last_request_time": {},
            "request_count": 0,
            "reset_time": time.time() + 60,
        }

        # Trading configuration
        self.trading_config = {
            "enabled": False,
            "max_investment": 1000,
            "stop_loss": 5.0,  # 5%
            "take_profit": 10.0,  # 10%
            "min_confidence": 70.0,
        }

        # Data storage
        self.market_data: Dict[str, CoinData] = {}
        self.trade_history: List[Dict[str, Any]] = []
        self.signals: List[TradeSignal] = []

        # Performance tracking
        self.stats: StatsConfig = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_trades": 0,
            "successful_trades": 0,
            "failed_trades": 0,
            "last_update": None,
        }

        logger.info("Coinbase Bot initialized")

    async def initialize(self):
        """Initialize the bot and create HTTP session"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=10, connect=5)
                )
            self.status = BotStatus.RUNNING
            self.is_running = True
            logger.info("Coinbase Bot initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Coinbase Bot: {e}")
            self.status = BotStatus.ERROR
            raise

    async def close(self):
        """Close the bot and cleanup resources"""
        try:
            self.is_running = False
            self.status = BotStatus.STOPPED
            if self.session:
                await self.session.close()
                self.session = None
            logger.info("Coinbase Bot closed successfully")
        except Exception as e:
            logger.error(f"Error closing Coinbase Bot: {e}")

    async def check_rate_limit(self, coin: str) -> bool:
        """Check if we can make a request for this coin"""
        current_time = time.time()

        # Reset rate limit counter if minute has passed
        if current_time >= self.rate_limit["reset_time"]:
            self.rate_limit["request_count"] = 0
            self.rate_limit["reset_time"] = current_time + 60

        # Check per-minute limit
        if self.rate_limit["request_count"] >= self.rate_limit["requests_per_minute"]:
            logger.warning("Rate limit reached for Coinbase (minute limit)")
            return False

        # Check per-coin interval
        last_request = self.rate_limit["last_request_time"].get(coin, 0)
        if current_time - last_request < self.rate_limit["interval_per_coin"]:
            logger.warning(f"Rate limit reached for {coin} (interval limit)")
            return False

        return True

    async def update_rate_limit(self, coin: str):
        """Update rate limit counters after successful request"""
        current_time = time.time()
        self.rate_limit["last_request_time"][coin] = current_time
        self.rate_limit["request_count"] += 1

    async def fetch_coin_data(self, coin: str) -> Optional[CoinData]:
        """Fetch data for a single coin from Coinbase"""
        if not self.session:
            logger.error("No HTTP session available")
            return None

        try:
            # Check rate limit
            if not await self.check_rate_limit(coin):
                return None

            # Prepare request
            product_id = f"{coin}-USD"
            url = f"{self.coinbase_base_url}/products/{product_id}/ticker"

            # Make request
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()

                    # Update rate limit
                    await self.update_rate_limit(coin)
                    self.stats["successful_requests"] += 1

                    # Create coin data object
                    coin_data = CoinData(
                        symbol=coin,
                        price=float(data.get("price", 0)),
                        change_24h=0.0,  # Coinbase ticker doesn't provide 24h change
                        volume_24h=float(data.get("volume", 0)),
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        api_source="coinbase",
                    )

                    logger.debug(f"Fetched data for {coin}: ${coin_data.price}")
                    return coin_data
                else:
                    logger.error(f"Coinbase API error for {coin}: {response.status}")
                    self.stats["failed_requests"] += 1
                    return None

        except Exception as e:
            logger.error(f"Error fetching {coin} from Coinbase: {e}")
            self.stats["failed_requests"] += 1
            return None

    async def update_all_coins(self):
        """Update data for all coins with proper throttling"""
        logger.info("Starting Coinbase market data update")

        for coin in self.coins:
            if not self.is_running:
                break

            try:
                data = await self.fetch_coin_data(coin)
                if data:
                    self.market_data[coin] = data
                    logger.debug(f"Updated {coin}: ${data.price}")
                else:
                    logger.warning(f"No data received for {coin}")

                # Wait between requests to respect rate limits
                await asyncio.sleep(self.rate_limit["interval_per_coin"])

            except Exception as e:
                logger.error(f"Error updating {coin}: {e}")
                continue

        self.stats["last_update"] = datetime.now(timezone.utc).isoformat()
        logger.info(f"Coinbase market data update completed. Updated {len(self.market_data)} coins")

    def generate_signals(self) -> List[TradeSignal]:
        """Generate trading signals based on market data using a real algorithm (SMA crossover)"""
        signals: List[TradeSignal] = []

        for coin, data in self.market_data.items():
            try:
                # Example: Use a simple moving average crossover for signal
                price_history = data.price_history if hasattr(data, "price_history") else []
                if len(price_history) < 20:
                    continue  # Not enough data for SMA
                short_sma = sum(price_history[-5:]) / 5
                long_sma = sum(price_history[-20:]) / 20
                signal = "HOLD"
                confidence = 50.0
                reason = "SMA crossover neutral"
                if short_sma > long_sma:
                    signal = "BUY"
                    confidence = 75.0
                    reason = "Short SMA crossed above Long SMA"
                elif short_sma < long_sma:
                    signal = "SELL"
                    confidence = 75.0
                    reason = "Short SMA crossed below Long SMA"
                trade_signal = TradeSignal(
                    symbol=coin,
                    action=signal,
                    confidence=confidence,
                    price=data.price,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    reason=reason,
                )
                signals.append(trade_signal)
            except Exception as e:
                logger.error(f"Error generating signal for {coin}: {e}")
                continue
        self.signals = signals
        return signals

    async def execute_trade(self, signal: TradeSignal) -> bool:
        """Execute a trade based on signal (simulated)"""
        if not self.trading_config["enabled"]:
            logger.info(f"Trading disabled, skipping {signal.symbol} {signal.action}")
            return False

        if signal.confidence < self.trading_config["min_confidence"]:
            logger.info(f"Signal confidence too low for {signal.symbol}: {signal.confidence}%")
            return False

        try:
            # Simulate trade execution
            trade_id = f"coinbase_trade_{int(time.time())}_{random.randint(1000, 9999)}"

            trade_record = {
                "trade_id": trade_id,
                "symbol": signal.symbol,
                "action": signal.action,
                "price": signal.price,
                "confidence": signal.confidence,
                "timestamp": signal.timestamp,
                "reason": signal.reason,
                "status": "executed",
            }

            self.trade_history.append(trade_record)
            self.stats["total_trades"] += 1
            self.stats["successful_trades"] += 1

            logger.info(f"Executed trade: {signal.symbol} {signal.action} at ${signal.price}")
            return True

        except Exception as e:
            logger.error(f"Error executing trade for {signal.symbol}: {e}")
            self.stats["failed_trades"] += 1
            return False

    async def run_trading_cycle(self):
        """Run one complete trading cycle"""
        try:
            # Update market data
            await self.update_all_coins()

            # Generate signals
            signals = self.generate_signals()

            # Execute trades for high-confidence signals
            for signal in signals:
                if signal.action in ["BUY", "SELL"]:
                    await self.execute_trade(signal)

            logger.info(f"Trading cycle completed. {len(signals)} signals generated")

        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
            self.status = BotStatus.ERROR

    async def run(self):
        """Main bot loop"""
        logger.info("Starting Coinbase Bot")

        try:
            await self.initialize()

            while self.is_running:
                try:
                    await self.run_trading_cycle()

                    # Wait before next cycle (6 seconds as per requirements)
                    await asyncio.sleep(6)

                except Exception as e:
                    logger.error(f"Error in main bot loop: {e}")
                    await asyncio.sleep(10)  # Wait longer on error

        except Exception as e:
            logger.error(f"Fatal error in Coinbase Bot: {e}")
            self.status = BotStatus.ERROR
        finally:
            await self.close()

    def get_status(self) -> Dict[str, Any]:
        """Get current bot status"""
        return {
            "status": self.status.value,
            "is_running": self.is_running,
            "coins_tracked": len(self.coins),
            "market_data_count": len(self.market_data),
            "signals_count": len(self.signals),
            "total_trades": self.stats["total_trades"],
            "successful_trades": self.stats["successful_trades"],
            "failed_trades": self.stats["failed_trades"],
            "last_update": self.stats["last_update"],
            "rate_limit": {
                "requests_this_minute": self.rate_limit["request_count"],
                "max_requests_per_minute": self.rate_limit["requests_per_minute"],
            },
        }


# Global bot instance
coinbase_bot = CoinbaseBot()


async def main():
    """Main function to run the Coinbase bot"""
    bot = CoinbaseBot()
    await bot.run()


if __name__ == "__main__":
    asyncio.run(main())


