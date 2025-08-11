"""
Auto Trading Loop Service
Runs every 60 seconds to process signals and execute trades
Integrates with existing alert and signal systems
"""

import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, List

from ai_strategy_execution import execute_ai_strategy_signal
from signal_manager import SignalManager
from utils.alerts import broadcast_alert
from .services.websocket_manager import websocket_manager

logger = logging.getLogger("auto_trading_loop")

# Configuration
TRADE_AMOUNT_USD = float(os.getenv("TRADE_AMOUNT_USD", "15"))
MIN_SIGNAL_CONFIDENCE = float(os.getenv("MIN_SIGNAL_CONFIDENCE", "0.7"))
AUTO_TRADING_ENABLED = os.getenv("AUTO_TRADING_ENABLED", "false").lower() == "true"

# Exchange routing configuration
EXCHANGE_ROUTING = {
    "bitcoin": "coinbase",
    "ethereum": "binance",
    "solana": "binance",
    "cardano": "coinbase",
    "dogecoin": "binance",
    "litecoin": "coinbase",
    "avalanche-2": "binance",
    "chainlink": "coinbase",
    "polkadot": "binance",
    "polygon": "coinbase",
}


class AutoTradingLoop:
    """Auto trading loop that processes signals every 60 seconds"""

    def __init__(self, redis_client: Any):
        self.redis_client = redis_client
        self.signal_manager = SignalManager(redis_client)
        self.is_running = False
        self.last_execution_time = 0
        self.total_trades = 0
        self.successful_trades = 0
        self.failed_trades = 0

        logger.info(f"Auto Trading Loop initialized - Enabled: {AUTO_TRADING_ENABLED}")
        logger.info(f"Trade amount: ${TRADE_AMOUNT_USD}, Min confidence: {MIN_SIGNAL_CONFIDENCE}")

    async def start(self):
        """Start the auto trading loop"""
        if self.is_running:
            logger.warning("Auto trading loop is already running")
            return

        if not AUTO_TRADING_ENABLED:
            logger.info("Auto trading is disabled - running in simulation mode")

        self.is_running = True
        logger.info("🚀 Starting Auto Trading Loop (60s intervals)")

        # Send startup alert
        await broadcast_alert(
            f"🤖 Auto Trading Loop Started\n"
            f"Trade Amount: ${TRADE_AMOUNT_USD}\n"
            f"Min Confidence: {MIN_SIGNAL_CONFIDENCE}\n"
            f"Mode: {'LIVE' if AUTO_TRADING_ENABLED else 'SIMULATION'}"
        )

        # Start the main loop
        asyncio.create_task(self._run_loop())

    async def stop(self):
        """Stop the auto trading loop"""
        if not self.is_running:
            return

        self.is_running = False
        logger.info("🛑 Auto Trading Loop stopped")

        # Send shutdown alert
        await broadcast_alert(
            f"🛑 Auto Trading Loop Stopped\n"
            f"Total Trades: {self.total_trades}\n"
            f"Success Rate: {(self.successful_trades/self.total_trades*100):.1f}%"
            if self.total_trades > 0
            else "No trades"
        )

    async def _run_loop(self):
        """Main trading loop - runs every 60 seconds"""
        while self.is_running:
            try:
                start_time = time.time()

                # Process signals and execute trades
                await self._process_signals()

                # Calculate execution time
                execution_time = time.time() - start_time
                logger.info(f"Trading cycle completed in {execution_time:.2f}s")

                # Wait for next cycle (60 seconds)
                await asyncio.sleep(60)

            except asyncio.CancelledError:
                logger.info("Auto trading loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in auto trading loop: {e}")
                await asyncio.sleep(60)  # Continue after error

    async def _process_signals(self):
        """Process all available signals and execute trades"""
        try:
            # Get all signals from Redis
            signals = await self._get_all_signals()

            if not signals:
                logger.debug("No signals to process")
                return

            logger.info(f"Processing {len(signals)} signals")

            # Process each signal
            for signal in signals:
                await self._process_single_signal(signal)

        except Exception as e:
            logger.error(f"Error processing signals: {e}")

    async def _get_all_signals(self) -> List[Dict[str, Any]]:
        """Get all available signals from Redis"""
        try:
            # Get signal history for all coins
            signal_keys = await self.redis_client.keys("signal_history:*")
            all_signals = []

            for key in signal_keys:
                # Get the most recent signal for each coin
                signal_data = await self.redis_client.lindex(key, 0)
                if signal_data:
                    try:
                        signal = json.loads(signal_data)
                        all_signals.append(signal)
                    except json.JSONDecodeError:
                        continue

            return all_signals

        except Exception as e:
            logger.error(f"Error getting signals: {e}")
            return []

    async def _process_single_signal(self, signal: Dict[str, Any]):
        """Process a single signal and execute trade if conditions are met"""
        try:
            symbol = signal.get("symbol", "").lower()
            confidence = signal.get("confidence", 0) / 100  # Convert percentage to decimal
            # signal_type = signal.get("signal_type", "")

            # Check if signal meets minimum confidence
            if confidence < MIN_SIGNAL_CONFIDENCE:
                logger.debug(f"Signal confidence too low for {symbol}: {confidence:.2f}")
                return

            # Check if signal indicates a breakout or strong signal
            if not self._is_strong_signal(signal):
                logger.debug(f"Signal not strong enough for {symbol}")
                return

            # Determine exchange to use
            exchange = EXCHANGE_ROUTING.get(symbol, "binance")

            # Execute trade
            success = await self._execute_trade(symbol, exchange, signal)

            if success:
                self.successful_trades += 1
                logger.info(f"✅ Trade executed successfully: {symbol} via {exchange}")
            else:
                self.failed_trades += 1
                logger.warning(f"❌ Trade failed: {symbol} via {exchange}")

            self.total_trades += 1

            # After processing the signal (success/failure), broadcast it:
            await websocket_manager.broadcast_json({"type": "signal", "data": signal})

        except Exception as e:
            logger.error(f"Error processing signal for {signal.get('symbol', 'unknown')}: {e}")

    def _is_strong_signal(self, signal: Dict[str, Any]) -> bool:
        """Check if signal is strong enough to execute trade"""
        try:
            # Check for breakout signals
            if signal.get("breakout", False):
                return True

            # Check for high confidence signals
            confidence = signal.get("confidence", 0) / 100
            if confidence > 0.8:  # 80% confidence
                return True

            # Check for strong technical indicators
            indicators = signal.get("indicators", {})
            if indicators.get("rsi", 0) < 20 or indicators.get("rsi", 0) > 80:
                return True

            # Check for momentum signals
            if signal.get("signal_type") == "momentum" and confidence > 0.7:
                return True

            return False

        except Exception as e:
            logger.error(f"Error checking signal strength: {e}")
            return False

    async def _execute_trade(self, symbol: str, exchange: str, signal: Dict[str, Any]) -> bool:
        """Execute a trade on the specified exchange"""
        try:
            if not AUTO_TRADING_ENABLED:
                # Simulation mode
                logger.info(f"🔄 SIMULATION: Would execute {symbol} trade via {exchange}")
                await broadcast_alert(
                    f"🔄 SIMULATION TRADE\n"
                    f"Symbol: {symbol.upper()}\n"
                    f"Exchange: {exchange.upper()}\n"
                    f"Amount: ${TRADE_AMOUNT_USD}\n"
                    f"Confidence: {signal.get('confidence', 0):.1f}%"
                )
                return True

            # Live trading mode
            logger.info(f"🎯 Executing live trade: {symbol} via {exchange}")

            # Map symbol to exchange format
            if exchange == "binance":
                binance_symbol = f"{symbol.upper()}USDT"
                coinbase_symbol = f"{symbol.upper()}-USD"
            else:  # coinbase
                binance_symbol = f"{symbol.upper()}USDT"
                coinbase_symbol = f"{symbol.upper()}-USD"

            # Execute trade using existing AI strategy execution
            result = execute_ai_strategy_signal(
                binance_symbol, coinbase_symbol, TRADE_AMOUNT_USD, signal
            )

            if result and "error" not in result:
                await broadcast_alert(
                    f"✅ LIVE TRADE EXECUTED\n"
                    f"Symbol: {symbol.upper()}\n"
                    f"Exchange: {exchange.upper()}\n"
                    f"Amount: ${TRADE_AMOUNT_USD}\n"
                    f"Order ID: {result.get('orderId', 'N/A')}"
                )
                return True
            else:
                error_msg = result.get("error", "Unknown error") if result else "No result"
                await broadcast_alert(
                    f"❌ TRADE FAILED\n"
                    f"Symbol: {symbol.upper()}\n"
                    f"Exchange: {exchange.upper()}\n"
                    f"Error: {error_msg}"
                )
                return False

        except Exception as e:
            logger.error(f"Error executing trade for {symbol}: {e}")
            await broadcast_alert(f"❌ Trade execution error for {symbol}: {str(e)}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get trading statistics"""
        return {
            "is_running": self.is_running,
            "auto_trading_enabled": AUTO_TRADING_ENABLED,
            "total_trades": self.total_trades,
            "successful_trades": self.successful_trades,
            "failed_trades": self.failed_trades,
            "success_rate": (
                (self.successful_trades / self.total_trades * 100) if self.total_trades > 0 else 0
            ),
            "trade_amount_usd": TRADE_AMOUNT_USD,
            "min_confidence": MIN_SIGNAL_CONFIDENCE,
            "last_execution_time": self.last_execution_time,
        }


# Global instance
auto_trading_loop = None


def get_auto_trading_loop(redis_client: Any) -> AutoTradingLoop:
    """Get or create auto trading loop instance"""
    global auto_trading_loop
    if auto_trading_loop is None:
        auto_trading_loop = AutoTradingLoop(redis_client)
    return auto_trading_loop
