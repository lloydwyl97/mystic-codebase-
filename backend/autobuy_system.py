#!/usr/bin/env python3
"""
Autobuy System for Mystic Trading Platform

Handles automated buying of cryptocurrencies based on signals and strategies.
Now integrated with AI training, model versioning, and experimental services.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any

from ai_model_versioning import get_ai_model_versioning

# Import new AI and experimental systems
from ai_training_pipeline import get_ai_training_pipeline

# Use absolute imports
from crypto_autoengine_config import get_config
from experimental_integration import get_experimental_integration
from shared_cache import SharedCache
from strategy_system import StrategyManager
from websocket_manager import websocket_manager

from backend.services.mystic_signal_engine import mystic_signal_engine

logger = logging.getLogger(__name__)


class TradeOrder:
    """Trade order representation"""

    def __init__(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        confidence: float,
        mystic_factors: dict[str, Any] | None = None,
    ):
        self.symbol = symbol
        self.side = side  # 'buy' or 'sell'
        self.amount = amount
        self.price = price
        self.confidence = confidence
        self.mystic_factors = mystic_factors or {}
        self.timestamp = datetime.now(timezone.timezone.utc).isoformat()
        self.status = "pending"  # 'pending', 'executed', 'failed', 'cancelled'
        self.order_id: str | None = None
        self.execution_price: float | None = None
        self.execution_time: str | None = None


class AutobuySystem:
    """Autobuy system that executes trades based on strategy signals and mystic factors"""

    def __init__(self, cache: SharedCache, strategy_manager: StrategyManager):
        self.cache = cache
        self.strategy_manager = strategy_manager
        self.config = get_config()

        # Trade tracking
        self.pending_orders: dict[str, TradeOrder] = {}
        self.executed_orders: list[TradeOrder] = []
        self.failed_orders: list[TradeOrder] = []

        # Performance tracking
        self.total_trades = 0
        self.successful_trades = 0
        self.failed_trades = 0
        self.total_profit = 0.0

        # Trading state
        self.is_running = False
        self.last_execution_time = 0

        # Mystic signal tracking
        self.last_mystic_signal = None
        self.mystic_signal_cache: dict[str, tuple[Any, float]] = {}
        self.mystic_signal_ttl = 60  # 1 minute cache

        # Initialize AI and experimental systems
        self.ai_training_pipeline = get_ai_training_pipeline(cache)
        self.ai_model_versioning = get_ai_model_versioning()
        self.experimental_integration = get_experimental_integration()

        # AI learning state
        self.ai_learning_enabled = True
        self.model_performance_tracking = True
        self.experimental_influence_enabled = True

        logger.info(
            "Autobuy system initialized with AI training, model versioning, and experimental integration"
        )

    async def start(self):
        """Start the autobuy system with all integrated systems"""
        self.is_running = True
        logger.info(
            "Autobuy system started with AI training, model versioning, and experimental integration"
        )

        # Start AI training pipeline
        if self.ai_learning_enabled:
            try:
                asyncio.create_task(self.ai_training_pipeline.start())
                logger.info("âœ… AI Training Pipeline started")
            except Exception as e:
                logger.error(f"âŒ Failed to start AI Training Pipeline: {e}")

        # Start experimental integration
        if self.experimental_influence_enabled:
            try:
                asyncio.create_task(self.experimental_integration.start())
                logger.info("âœ… Experimental Integration started")
            except Exception as e:
                logger.error(f"âŒ Failed to start Experimental Integration: {e}")

        # Start auto-optimization task
        if self.model_performance_tracking:
            try:
                asyncio.create_task(self._auto_optimize_models())
                logger.info("âœ… Model Auto-Optimization started")
            except Exception as e:
                logger.error(f"âŒ Failed to start Model Auto-Optimization: {e}")

        while self.is_running:
            try:
                await self.process_strategy_signals()
                await asyncio.sleep(5)  # Check every 5 seconds
            except Exception as e:
                logger.error(f"Error in autobuy system: {e}")
                await asyncio.sleep(10)

    async def stop(self):
        """Stop the autobuy system and all integrated systems"""
        self.is_running = False
        logger.info("Stopping autobuy system and all integrated systems")

        # Stop AI training pipeline
        if self.ai_training_pipeline:
            try:
                await self.ai_training_pipeline.stop()
                logger.info("âœ… AI Training Pipeline stopped")
            except Exception as e:
                logger.error(f"âŒ Error stopping AI Training Pipeline: {e}")

        # Stop experimental integration
        if self.experimental_integration:
            try:
                await self.experimental_integration.stop()
                logger.info("âœ… Experimental Integration stopped")
            except Exception as e:
                logger.error(f"âŒ Error stopping Experimental Integration: {e}")

        logger.info("Autobuy system and all integrated systems stopped")

    async def emergency_stop(self):
        """Emergency stop the autobuy system - cancels all pending orders and stops trading"""
        try:
            logger.warning("ðŸš¨ EMERGENCY STOP ACTIVATED - Cancelling all pending orders")

            # Stop the system
            self.is_running = False

            # Cancel all pending orders
            cancelled_orders = []
            for symbol, order in list(self.pending_orders.items()):
                try:
                    # Cancel the order
                    order.status = "cancelled"
                    order.execution_time = datetime.now(timezone.timezone.utc).isoformat()

                    # Move to failed orders
                    self.failed_orders.append(order)
                    cancelled_orders.append(symbol)

                    logger.warning(
                        f"ðŸš¨ Cancelled pending order for {symbol}: {order.amount} @ {order.price}"
                    )

                except Exception as e:
                    logger.error(f"Error cancelling order for {symbol}: {e}")

            # Clear pending orders
            self.pending_orders.clear()

            # Broadcast emergency stop
            await websocket_manager.broadcast_json(
                {
                    "type": "emergency_stop",
                    "data": {
                        "timestamp": (datetime.now(timezone.timezone.utc).isoformat()),
                        "cancelled_orders": cancelled_orders,
                        "message": ("Emergency stop activated - all trading stopped"),
                    },
                }
            )

            logger.warning(
                f"ðŸš¨ Emergency stop completed - {len(cancelled_orders)} orders cancelled"
            )

        except Exception as e:
            logger.error(f"Error during emergency stop: {e}")

    async def _auto_optimize_models(self):
        """Auto-optimize AI models based on performance"""
        while self.is_running:
            try:
                # Run auto-optimization every hour
                await asyncio.sleep(3600)  # 1 hour

                if self.model_performance_tracking and self.ai_model_versioning:
                    optimization_result = await self.ai_model_versioning.auto_optimize_models()

                    if optimization_result.get("actions_taken"):
                        logger.info(
                            f"ðŸ¤– Auto-optimization completed: {optimization_result['actions_taken']}"
                        )

                        # Broadcast optimization results
                        await websocket_manager.broadcast_json(
                            {
                                "type": "model_optimization",
                                "data": optimization_result,
                            }
                        )

            except Exception as e:
                logger.error(f"âŒ Error in auto-optimization: {e}")
                await asyncio.sleep(300)  # 5 minutes on error

    async def process_strategy_signals(self):
        """Process strategy signals and execute trades with AI and experimental integration"""
        current_time = time.time()

        # Check if enough time has passed since last execution
        if current_time - self.last_execution_time < 10:  # Minimum 10 seconds between trades
            return

        # Get mystic signal
        mystic_signal = await self._get_mystic_signal()

        # Get experimental influence
        experimental_influence = None
        if self.experimental_influence_enabled and self.experimental_integration:
            try:
                experimental_influence = (
                    await self.experimental_integration.get_experimental_influence("BTCUSDT")
                )
            except Exception as e:
                logger.error(f"âŒ Error getting experimental influence: {e}")

        # Get all strategy signals
        strategy_results = self.strategy_manager.run_all_strategies()

        for symbol, result in strategy_results.items():
            try:
                await self._process_coin_signals(
                    symbol, result, mystic_signal, experimental_influence
                )
            except Exception as e:
                logger.error(f"Error processing signals for {symbol}: {e}")

        # Update AI training data after processing
        if self.ai_learning_enabled and self.ai_training_pipeline:
            try:
                await self._update_ai_training_data(
                    strategy_results, mystic_signal, experimental_influence
                )
            except Exception as e:
                logger.error(f"âŒ Error updating AI training data: {e}")

    async def _get_mystic_signal(self) -> Any | None:
        """Get current mystic signal with caching"""
        current_time = time.time()

        # Check cache first
        if "mystic_signal" in self.mystic_signal_cache:
            cached_signal, cache_time = self.mystic_signal_cache["mystic_signal"]
            if current_time - cache_time < self.mystic_signal_ttl:
                return cached_signal

        try:
            # Generate new mystic signal
            mystic_signal = await mystic_signal_engine.generate_comprehensive_signal()
            self.mystic_signal_cache["mystic_signal"] = (
                mystic_signal,
                current_time,
            )
            self.last_mystic_signal = mystic_signal

            logger.info(
                f"Mystic signal generated: {mystic_signal.signal_type.value} "
                f"(confidence: {mystic_signal.confidence:.2f}, strength: {mystic_signal.strength:.2f})"
            )

            return mystic_signal
        except Exception as e:
            logger.error(f"Error generating mystic signal: {e}")
            return None

    async def _process_coin_signals(
        self,
        symbol: str,
        strategy_result: dict[str, Any],
        mystic_signal: Any | None,
        experimental_influence: dict[str, Any] | None,
    ):
        """Process signals for a specific coin with mystic integration"""
        aggregated = strategy_result.get("aggregated", {})
        decision = aggregated.get("decision", "hold")
        confidence = aggregated.get("confidence", 0.0)
        strength = aggregated.get("strength", 0.0)

        # Integrate mystic signal into decision making
        final_decision, final_confidence, mystic_factors = self._integrate_mystic_signals(
            decision, confidence, strength, mystic_signal
        )

        # Integrate experimental influence into decision making
        if experimental_influence:
            final_decision, final_confidence, mystic_factors = (
                self._integrate_experimental_influence(
                    final_decision,
                    final_confidence,
                    mystic_factors,
                    experimental_influence,
                )
            )

        # Check if we should trade
        if not self._should_execute_trade(symbol, final_decision, final_confidence, strength):
            return

        # Get coin data
        coin_data = self.cache.get_coin_cache(symbol)
        if not coin_data:
            return

        # Calculate trade amount (now influenced by mystic factors)
        trade_amount = self._calculate_trade_amount(
            symbol, coin_data.price, final_confidence, mystic_factors
        )
        if trade_amount <= 0:
            return

        # Create trade order with mystic factors
        order = TradeOrder(
            symbol=symbol,
            side=final_decision,
            amount=trade_amount,
            price=coin_data.price,
            confidence=final_confidence,
            mystic_factors=mystic_factors,
        )

        # Execute trade
        success = await self._execute_trade(order)

        if success:
            self.last_execution_time = time.time()
            self.cache.set_trade_cooldown(symbol)
            logger.info(
                f"Executed {final_decision} order for {symbol}: {trade_amount} @ {coin_data.price} "
                f"(Mystic: {mystic_signal.signal_type.value if mystic_signal else 'None'})"
            )
        else:
            logger.warning(f"Failed to execute {final_decision} order for {symbol}")

    def _integrate_mystic_signals(
        self,
        strategy_decision: str,
        strategy_confidence: float,
        strategy_strength: float,
        mystic_signal: Any | None,
    ) -> tuple[str, float, dict[str, Any]]:
        """Integrate mystic signals with strategy signals for final decision"""
        if not mystic_signal:
            return strategy_decision, strategy_confidence, {}

        # Mystic signal influence weights
        mystic_weight = 0.4  # 40% influence from mystic factors
        strategy_weight = 0.6  # 60% influence from strategy factors

        # Convert decisions to numerical values
        decision_map = {"buy": 1.0, "sell": -1.0, "hold": 0.0}
        strategy_value = decision_map.get(strategy_decision, 0.0)
        mystic_value = decision_map.get(
            mystic_signal.signal_type.value.lower().replace("strong_", ""), 0.0
        )

        # Weighted combination
        combined_value = strategy_value * strategy_weight + mystic_value * mystic_weight

        # Determine final decision
        if combined_value > 0.3:
            final_decision = "buy"
        elif combined_value < -0.3:
            final_decision = "sell"
        else:
            final_decision = "hold"

        # Calculate final confidence (weighted average)
        final_confidence = (
            strategy_confidence * strategy_weight + mystic_signal.confidence * mystic_weight
        )

        # Prepare mystic factors for logging
        mystic_factors = {
            "mystic_signal_type": mystic_signal.signal_type.value,
            "mystic_confidence": mystic_signal.confidence,
            "mystic_strength": mystic_signal.strength,
            "mystic_reasoning": mystic_signal.reasoning,
            "combined_value": combined_value,
            "strategy_original": strategy_decision,
            "strategy_confidence": strategy_confidence,
            "strategy_strength": strategy_strength,
        }

        return final_decision, final_confidence, mystic_factors

    def _integrate_experimental_influence(
        self,
        strategy_decision: str,
        strategy_confidence: float,
        mystic_factors: dict[str, Any],
        experimental_influence: dict[str, Any],
    ) -> tuple[str, float, dict[str, Any]]:
        """Integrate experimental influence into decision making"""
        if not experimental_influence:
            return strategy_decision, strategy_confidence, mystic_factors

        # Experimental influence weights
        experimental_weight = 0.3  # 30% influence from experimental factors
        strategy_weight = 0.7  # 70% influence from strategy factors

        # Convert decisions to numerical values
        decision_map = {"buy": 1.0, "sell": -1.0, "hold": 0.0}
        strategy_value = decision_map.get(strategy_decision, 0.0)
        experimental_value = decision_map.get(
            experimental_influence.get("signal_type", "").lower().replace("strong_", ""),
            0.0,
        )

        # Weighted combination
        combined_value = strategy_value * strategy_weight + experimental_value * experimental_weight

        # Determine final decision
        if combined_value > 0.3:
            final_decision = "buy"
        elif combined_value < -0.3:
            final_decision = "sell"
        else:
            final_decision = "hold"

        # Calculate final confidence (weighted average)
        final_confidence = (
            strategy_confidence * strategy_weight
            + experimental_influence.get("confidence", 0.0) * experimental_weight
        )

        # Update mystic factors with experimental influence
        mystic_factors.update(experimental_influence)
        mystic_factors["combined_value"] = combined_value

        return final_decision, final_confidence, mystic_factors

    def _should_execute_trade(
        self, symbol: str, decision: str, confidence: float, strength: float
    ) -> bool:
        """Determine if we should execute a trade with mystic considerations"""
        # Check if coin is in cooldown
        if not self.cache.can_trade(symbol):
            return False

        # Check if decision is valid
        if decision not in ["buy", "sell"]:
            return False

        # Check confidence threshold (now includes mystic confidence)
        if confidence < self.config.strategy_config.min_confidence:
            return False

        # Check signal strength
        if strength < self.config.strategy_config.min_signal_strength:
            return False

        # Check if we already have a pending order for this symbol
        if symbol in self.pending_orders:
            return False

        # Check if we have enough data
        coin_data = self.cache.get_coin_cache(symbol)
        if not coin_data or coin_data.price <= 0:
            return False

        return True

    def _calculate_trade_amount(
        self,
        symbol: str,
        price: float,
        confidence: float,
        mystic_factors: dict[str, Any],
    ) -> float:
        """Calculate trade amount based on confidence, coin config, and mystic factors"""
        coin_config = self.config.get_coin_by_symbol(symbol)
        if not coin_config:
            return 0.0

        # Base amount from coin config
        base_amount = coin_config.min_trade_amount

        # Scale by confidence (higher confidence = larger position)
        confidence_multiplier = 1.0 + (confidence - 0.5) * 2  # 0.5 -> 1.0, 1.0 -> 2.0

        # Mystic factor multiplier
        mystic_multiplier = 1.0
        if mystic_factors:
            mystic_signal_type = mystic_factors.get("mystic_signal_type", "")
            mystic_strength = mystic_factors.get("mystic_strength", 0.5)

            # Strong mystic signals get higher multipliers
            if "STRONG" in mystic_signal_type:
                mystic_multiplier = 1.0 + mystic_strength * 0.5  # Up to 1.5x for strong signals
            elif mystic_strength > 0.7:
                mystic_multiplier = 1.0 + mystic_strength * 0.3  # Up to 1.3x for high strength
            else:
                mystic_multiplier = 1.0 + mystic_strength * 0.1  # Up to 1.1x for normal strength

        # Calculate final amount
        trade_amount = base_amount * confidence_multiplier * mystic_multiplier

        # Ensure within limits
        trade_amount = max(trade_amount, coin_config.min_trade_amount)
        trade_amount = min(trade_amount, coin_config.max_trade_amount)

        return trade_amount

    async def _execute_trade(self, order: TradeOrder) -> bool:
        """Execute a trade order"""
        try:
            # Add to pending orders
            self.pending_orders[order.symbol] = order

            # Simulate trade execution (replace with actual exchange API calls)
            success = await self._simulate_trade_execution(order)

            if success:
                # Update order status
                order.status = "executed"
                order.execution_price = order.price
                order.execution_time = datetime.now(timezone.timezone.utc).isoformat()
                order.order_id = f"order_{int(time.time())}_{order.symbol}"

                # Move to executed orders
                self.executed_orders.append(order)
                del self.pending_orders[order.symbol]

                # Update statistics
                self.total_trades += 1
                self.successful_trades += 1

                # Calculate profit (simplified)
                if order.side == "sell":
                    # Find corresponding buy order for profit calculation
                    buy_order = self._find_corresponding_buy_order(order.symbol)
                    if buy_order is not None and buy_order.execution_price is not None:
                        profit = (order.execution_price - buy_order.execution_price) * order.amount
                        self.total_profit += profit

                # Broadcast trade execution
                await websocket_manager.broadcast_json(
                    {
                        "type": "trade_execution",
                        "data": {
                            "symbol": order.symbol,
                            "side": order.side,
                            "amount": order.amount,
                            "price": order.execution_price,
                            "order_id": order.order_id,
                            "timestamp": order.execution_time,
                            "mystic_factors": order.mystic_factors,
                        },
                    }
                )

                return True
            else:
                # Mark as failed
                order.status = "failed"
                self.failed_orders.append(order)
                del self.pending_orders[order.symbol]

                self.total_trades += 1
                self.failed_trades += 1

                # Broadcast trade failure
                await websocket_manager.broadcast_json(
                    {
                        "type": "trade_failure",
                        "data": {
                            "symbol": order.symbol,
                            "side": order.side,
                            "amount": order.amount,
                            "price": order.price,
                            "timestamp": (datetime.now(timezone.timezone.utc).isoformat()),
                            "reason": "Execution failed",
                        },
                    }
                )

                return False

        except Exception as e:
            logger.error(f"Error executing trade for {order.symbol}: {e}")

            # Mark as failed
            order.status = "failed"
            self.failed_orders.append(order)
            if order.symbol in self.pending_orders:
                del self.pending_orders[order.symbol]

            self.total_trades += 1
            self.failed_trades += 1

            # Broadcast trade error
            await websocket_manager.broadcast_json(
                {
                    "type": "trade_error",
                    "data": {
                        "symbol": order.symbol,
                        "side": order.side,
                        "amount": order.amount,
                        "price": order.price,
                        "timestamp": (datetime.now(timezone.timezone.utc).isoformat()),
                        "error": str(e),
                    },
                }
            )

            return False

    async def _simulate_trade_execution(self, order: TradeOrder) -> bool:
        """Execute trade on Binance US (real trading)"""
        try:
            # Import Binance US API
            from backend.ai.auto_trade import BinanceUSAPI

            async with BinanceUSAPI() as binance:
                # Get current price
                ticker_data = await binance.get_ticker_price(order.symbol)
                if not ticker_data:
                    logger.error(f"Failed to get price for {order.symbol}")
                    return False

                current_price = ticker_data["price"]

                # Calculate quantity based on USD amount
                order.amount / current_price

                # Prepare order parameters
                params = {
                    "symbol": order.symbol,
                    "side": order.side.upper(),
                    "type": "MARKET",
                    "quoteOrderQty": order.amount,  # USD amount
                    "timestamp": int(time.time() * 1000),
                }

                # Generate signature
                import hashlib
                import hmac
                from urllib.parse import urlencode

                query_string = urlencode(params)
                signature = hmac.new(
                    binance.secret_key.encode("utf-8"),
                    query_string.encode("utf-8"),
                    hashlib.sha256,
                ).hexdigest()

                # Execute order
                url = f"{binance.base_url}/api/v3/order"
                headers = {"X-MBX-APIKEY": binance.api_key}

                async with binance.session.post(
                    url,
                    params={**params, "signature": signature},
                    headers=headers,
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(
                            f"âœ… Real trade executed: {order.symbol} {order.side} ${order.amount}"
                        )
                        logger.info(f"Order result: {result}")
                        return True
                    else:
                        error_text = await response.text()
                        logger.error(f"âŒ Trade execution failed: {response.status} - {error_text}")
                        return False

        except Exception as e:
            logger.error(f"Error executing real trade for {order.symbol}: {e}")
            return False

    def _find_corresponding_buy_order(self, symbol: str) -> TradeOrder | None:
        """Find the most recent buy order for a symbol"""
        for order in reversed(self.executed_orders):
            if order.symbol == symbol and order.side == "buy":
                return order
        return None

    def get_trading_stats(self) -> dict[str, Any]:
        """Get comprehensive trading statistics including AI and experimental data"""
        try:
            stats = {
                "trading_performance": {
                    "total_trades": self.total_trades,
                    "successful_trades": self.successful_trades,
                    "failed_trades": self.failed_trades,
                    "total_profit": self.total_profit,
                    "win_rate": ((self.successful_trades / max(self.total_trades, 1)) * 100),
                    "average_profit": (self.total_profit / max(self.total_trades, 1)),
                    "max_drawdown": self._calculate_max_drawdown(),
                },
                "system_status": {
                    "is_running": self.is_running,
                    "ai_learning_enabled": self.ai_learning_enabled,
                    "model_performance_tracking": (self.model_performance_tracking),
                    "experimental_influence_enabled": (self.experimental_influence_enabled),
                },
                "ai_systems": {},
                "experimental_systems": {},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            # Add AI system status
            if self.ai_training_pipeline:
                stats["ai_systems"]["training_pipeline"] = self.ai_training_pipeline.get_status()

            if self.ai_model_versioning:
                stats["ai_systems"]["model_versioning"] = self.ai_model_versioning.get_status()

            # Add experimental system status
            if self.experimental_integration:
                stats["experimental_systems"] = self.experimental_integration.get_status()

            return stats

        except Exception as e:
            logger.error(f"âŒ Error getting trading stats: {e}")
            return {"error": str(e)}

    def get_recent_orders(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent trade orders"""
        recent_orders: list[dict[str, Any]] = []

        # Add pending orders
        for order in self.pending_orders.values():
            recent_orders.append(
                {
                    "symbol": order.symbol,
                    "side": order.side,
                    "amount": order.amount,
                    "price": order.price,
                    "confidence": order.confidence,
                    "status": order.status,
                    "timestamp": order.timestamp,
                }
            )

        # Add recent executed orders
        for order in self.executed_orders[-limit:]:
            recent_orders.append(
                {
                    "symbol": order.symbol,
                    "side": order.side,
                    "amount": order.amount,
                    "price": order.price,
                    "confidence": order.confidence,
                    "status": order.status,
                    "execution_price": order.execution_price,
                    "execution_time": order.execution_time,
                    "order_id": order.order_id,
                    "timestamp": order.timestamp,
                }
            )

        # Sort by timestamp
        recent_orders.sort(key=lambda x: x["timestamp"], reverse=True)

        return recent_orders[:limit]

    def cancel_pending_order(self, symbol: str) -> bool:
        """Cancel a pending order"""
        if symbol in self.pending_orders:
            order = self.pending_orders[symbol]
            order.status = "cancelled"
            self.failed_orders.append(order)
            del self.pending_orders[symbol]
            logger.info(f"Cancelled pending order for {symbol}")
            return True
        return False

    def cancel_all_pending_orders(self):
        """Cancel all pending orders"""
        symbols_to_cancel = list(self.pending_orders.keys())
        for symbol in symbols_to_cancel:
            self.cancel_pending_order(symbol)
        logger.info(f"Cancelled {len(symbols_to_cancel)} pending orders")

    async def _update_ai_training_data(
        self,
        strategy_results: dict[str, Any],
        mystic_signal: Any | None,
        experimental_influence: dict[str, Any] | None,
    ):
        """Update AI training data with current trading results"""
        try:
            # Get current trading performance
            trading_performance = {
                "total_trades": self.total_trades,
                "successful_trades": self.successful_trades,
                "failed_trades": self.failed_trades,
                "total_profit": self.total_profit,
                "win_rate": ((self.successful_trades / max(self.total_trades, 1)) * 100),
                "average_profit": (self.total_profit / max(self.total_trades, 1)),
            }

            # Update model performance if we have an active model
            if self.ai_model_versioning and self.ai_model_versioning.active_model:
                active_model = self.ai_model_versioning.active_model

                # Calculate performance metrics
                performance_data = {
                    "accuracy": trading_performance["win_rate"] / 100,
                    "total_profit": trading_performance["total_profit"],
                    "win_rate": trading_performance["win_rate"] / 100,
                    "total_trades": trading_performance["total_trades"],
                    "average_profit": trading_performance["average_profit"],
                    "max_drawdown": self._calculate_max_drawdown(),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

                # Update model performance
                await self.ai_model_versioning.update_model_performance(
                    active_model, performance_data
                )

                # Evaluate model performance
                evaluation = await self.ai_model_versioning.evaluate_model_performance(active_model)

                if evaluation.get("recommendation") == "deactivate":
                    logger.warning(f"âš ï¸ Model {active_model} underperforming - consider rollback")

            # Store training data for future model training
            {
                "strategy_results": strategy_results,
                "mystic_signal": mystic_signal,
                "experimental_influence": experimental_influence,
                "trading_performance": trading_performance,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            # This would normally be stored in the AI training pipeline
            # For now, we'll log it
            logger.debug(
                f"ðŸ“Š Updated AI training data: {len(strategy_results)} strategies, {trading_performance['total_trades']} trades"
            )

        except Exception as e:
            logger.error(f"âŒ Error updating AI training data: {e}")

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from executed trades"""
        try:
            if not self.executed_orders:
                return 0.0

            # Calculate cumulative profit over time
            cumulative_profits = []
            running_profit = 0.0

            for order in sorted(self.executed_orders, key=lambda x: x.timestamp):
                if order.status == "executed" and order.execution_price:
                    # Simplified profit calculation
                    if order.side == "buy":
                        running_profit -= order.amount * order.execution_price
                    else:  # sell
                        running_profit += order.amount * order.execution_price

                    cumulative_profits.append(running_profit)

            if not cumulative_profits:
                return 0.0

            # Calculate maximum drawdown
            peak = cumulative_profits[0]
            max_drawdown = 0.0

            for profit in cumulative_profits:
                if profit > peak:
                    peak = profit
                drawdown = (peak - profit) / max(peak, 1.0)
                max_drawdown = max(max_drawdown, drawdown)

            return max_drawdown

        except Exception as e:
            logger.error(f"âŒ Error calculating max drawdown: {e}")
            return 0.0


class AutobuyManager:
    """Manages the autobuy system"""

    def __init__(self, cache: SharedCache, strategy_manager: StrategyManager):
        self.cache = cache
        self.strategy_manager = strategy_manager
        self.autobuy_system = AutobuySystem(cache, strategy_manager)
        self.task = None

    async def start(self):
        """Start the autobuy manager"""
        if self.task and not self.task.done():
            logger.warning("Autobuy system already running")
            return

        self.task = asyncio.create_task(self.autobuy_system.start())
        logger.info("Autobuy manager started")

    async def stop(self):
        """Stop the autobuy manager"""
        if self.task:
            await self.autobuy_system.stop()
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
            self.task = None
            logger.info("Autobuy manager stopped")

    def get_status(self) -> dict[str, Any]:
        """Get autobuy system status"""
        return {
            "is_running": self.autobuy_system.is_running,
            "task_running": self.task is not None and not self.task.done(),
            "trading_stats": self.autobuy_system.get_trading_stats(),
        }

    def get_recent_orders(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent trade orders"""
        return self.autobuy_system.get_recent_orders(limit)

    def cancel_pending_order(self, symbol: str) -> bool:
        """Cancel a pending order"""
        return self.autobuy_system.cancel_pending_order(symbol)

    def cancel_all_pending_orders(self):
        """Cancel all pending orders"""
        self.autobuy_system.cancel_all_pending_orders()


