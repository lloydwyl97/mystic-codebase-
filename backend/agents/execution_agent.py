"""
Execution Agent
Handles order execution, trade management, and position tracking
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class ExecutionAgent(BaseAgent):
    """Execution Agent - Handles order execution and trade management"""

    def __init__(self, agent_id: str = "execution_agent_001"):
        super().__init__(agent_id, "execution")

        # Execution-specific state
        self.state.update(
            {
                "active_orders": {},
                "positions": {},
                "trade_history": [],
                "execution_stats": {
                    "total_trades": 0,
                    "successful_trades": 0,
                    "failed_trades": 0,
                    "total_volume": 0,
                    "total_fees": 0,
                },
                "exchange_status": "connected",
                "last_execution": None,
            }
        )

        # Register execution-specific handlers
        self.register_handler("approved_trading_signal", self.handle_approved_signal)
        self.register_handler("cancel_order", self.handle_cancel_order)
        self.register_handler("modify_order", self.handle_modify_order)
        self.register_handler("portfolio_update", self.handle_portfolio_update)
        self.register_handler("market_data", self.handle_market_data)

        print(f"⚡ Execution Agent {agent_id} initialized")

    async def initialize(self):
        """Initialize execution agent resources"""
        try:
            # Load existing positions and orders
            await self.load_existing_data()

            # Initialize exchange connection
            await self.initialize_exchange()

            # Subscribe to market data
            await self.subscribe_to_market_data()

            print(f"✅ Execution Agent {self.agent_id} initialized successfully")

        except Exception as e:
            print(f"❌ Error initializing Execution Agent: {e}")
            self.update_health_status("error")

    async def process_loop(self):
        """Main execution processing loop"""
        while self.running:
            try:
                # Monitor active orders
                await self.monitor_active_orders()

                # Update positions
                await self.update_positions()

                # Check for order timeouts
                await self.check_order_timeouts()

                # Update execution metrics
                await self.update_execution_metrics()

                await asyncio.sleep(10)  # Check every 10 seconds

            except Exception as e:
                print(f"❌ Error in execution processing loop: {e}")
                await asyncio.sleep(30)

    async def load_existing_data(self):
        """Load existing positions and orders from Redis"""
        try:
            # Load positions
            positions_data = self.redis_client.get("positions")
            if positions_data:
                self.state["positions"] = json.loads(positions_data)

            # Load active orders
            orders_data = self.redis_client.get("active_orders")
            if orders_data:
                self.state["active_orders"] = json.loads(orders_data)

            # Load trade history
            history_data = self.redis_client.lrange("trade_history", 0, -1)
            self.state["trade_history"] = [json.loads(item) for item in history_data]

            print(
                f"📊 Loaded {len(self.state['positions'])} positions and {len(self.state['active_orders'])} orders"
            )

        except Exception as e:
            print(f"❌ Error loading existing data: {e}")

    async def initialize_exchange(self):
        """Initialize exchange connection"""
        try:
            # This would typically initialize exchange API connections
            # For now, simulate exchange connection

            exchange_config = {
                "status": "connected",
                "last_heartbeat": datetime.now().isoformat(),
                "supported_pairs": ["BTCUSDT", "ETHUSDT", "ADAUSDT"],
                "min_order_size": 0.001,
                "max_order_size": 1000,
                "fee_rate": 0.001,
            }

            self.redis_client.set("exchange_config", json.dumps(exchange_config), ex=3600)

            print("🔗 Exchange connection initialized")

        except Exception as e:
            print(f"❌ Error initializing exchange: {e}")
            self.state["exchange_status"] = "disconnected"

    async def subscribe_to_market_data(self):
        """Subscribe to market data updates"""
        try:
            # Subscribe to market data channel
            pubsub = self.redis_client.pubsub()
            pubsub.subscribe("market_data")

            # Start market data listener
            asyncio.create_task(self.listen_market_data(pubsub))

            print("📡 Execution Agent subscribed to market data")

        except Exception as e:
            print(f"❌ Error subscribing to market data: {e}")

    async def listen_market_data(self, pubsub):
        """Listen for market data updates"""
        try:
            for message in pubsub.listen():
                if not self.running:
                    break

                if message["type"] == "message":
                    market_data = json.loads(message["data"])
                    await self.process_market_data(market_data)

        except Exception as e:
            print(f"❌ Error in market data listener: {e}")
        finally:
            pubsub.close()

    async def process_market_data(self, market_data: Dict[str, Any]):
        """Process incoming market data for execution"""
        try:
            symbol = market_data.get("symbol")
            price = market_data.get("price")
            market_data.get("timestamp")

            # Update order prices
            await self.update_order_prices(symbol, price)

            # Check for order triggers
            await self.check_order_triggers(symbol, price)

        except Exception as e:
            print(f"❌ Error processing market data: {e}")

    async def handle_approved_signal(self, message: Dict[str, Any]):
        """Handle approved trading signal"""
        try:
            strategy_id = message.get("strategy_id")
            signal = message.get("signal", {})
            market_data = message.get("market_data", {})
            position_size = message.get("position_size", {})
            message.get("risk_validation", {})

            symbol = market_data.get("symbol")
            signal_type = signal.get("type")
            confidence = signal.get("confidence", 0)
            price = market_data.get("price", 0)

            print(f"🎯 Executing approved signal for {symbol} ({signal_type})")

            # Execute the trade
            execution_result = await self.execute_trade(
                symbol=symbol,
                signal_type=signal_type,
                position_size=position_size,
                price=price,
                strategy_id=strategy_id,
                confidence=confidence,
            )

            # Send execution result to strategy agent
            await self.send_message(
                "strategy_agent",
                {
                    "type": "execution_result",
                    "strategy_id": strategy_id,
                    "execution": execution_result,
                    "signal": signal,
                },
            )

            # Broadcast execution
            await self.broadcast_message({"type": "trade_executed", "execution": execution_result})

        except Exception as e:
            print(f"❌ Error handling approved signal: {e}")
            await self.broadcast_error(f"Execution error: {e}")

    async def handle_cancel_order(self, message: Dict[str, Any]):
        """Handle order cancellation request"""
        try:
            order_id = message.get("order_id")

            print(f"❌ Cancelling order {order_id}")

            # Cancel order
            cancellation_result = await self.cancel_order(order_id)

            # Send response
            response = {
                "type": "order_cancelled",
                "order_id": order_id,
                "result": cancellation_result,
                "timestamp": datetime.now().isoformat(),
            }

            sender = message.get("from_agent")
            if sender:
                await self.send_message(sender, response)

        except Exception as e:
            print(f"❌ Error cancelling order: {e}")
            await self.broadcast_error(f"Order cancellation error: {e}")

    async def handle_modify_order(self, message: Dict[str, Any]):
        """Handle order modification request"""
        try:
            order_id = message.get("order_id")
            new_params = message.get("new_params", {})

            print(f"✏️ Modifying order {order_id}")

            # Modify order
            modification_result = await self.modify_order(order_id, new_params)

            # Send response
            response = {
                "type": "order_modified",
                "order_id": order_id,
                "result": modification_result,
                "timestamp": datetime.now().isoformat(),
            }

            sender = message.get("from_agent")
            if sender:
                await self.send_message(sender, response)

        except Exception as e:
            print(f"❌ Error modifying order: {e}")
            await self.broadcast_error(f"Order modification error: {e}")

    async def handle_portfolio_update(self, message: Dict[str, Any]):
        """Handle portfolio update"""
        try:
            portfolio = message.get("portfolio", {})

            # Update positions
            await self.update_positions_from_portfolio(portfolio)

        except Exception as e:
            print(f"❌ Error handling portfolio update: {e}")

    async def execute_trade(
        self,
        symbol: str,
        signal_type: str,
        position_size: Dict[str, Any],
        price: float,
        strategy_id: str,
        confidence: float,
    ) -> Dict[str, Any]:
        """Execute a trade"""
        try:
            # Validate execution parameters
            if not await self.validate_execution_params(symbol, position_size, price):
                from backend.utils.exceptions import TradingException, ErrorCode
            raise TradingException(
                message="Invalid execution parameters",
                error_code=ErrorCode.TRADING_ORDER_ERROR,
                details={"symbol": symbol, "position_size": position_size, "price": price}
            )

            # Calculate order parameters
            order_params = await self.calculate_order_params(
                symbol, signal_type, position_size, price
            )

            # Place order
            order_result = await self.place_order(order_params)

            # Create execution record
            execution = {
                "execution_id": f"exec_{int(datetime.now().timestamp())}",
                "strategy_id": strategy_id,
                "symbol": symbol,
                "signal_type": signal_type,
                "order_id": order_result.get("order_id"),
                "quantity": order_params["quantity"],
                "price": order_params["price"],
                "total_value": (order_params["quantity"] * order_params["price"]),
                "confidence": confidence,
                "status": order_result.get("status", "pending"),
                "timestamp": datetime.now().isoformat(),
                "fees": order_result.get("fees", 0),
            }

            # Update execution stats
            self.state["execution_stats"]["total_trades"] += 1
            self.state["execution_stats"]["total_volume"] += execution["total_value"]
            self.state["execution_stats"]["total_fees"] += execution["fees"]

            # Add to trade history
            self.state["trade_history"].append(execution)

            # Store in Redis
            self.redis_client.lpush("trade_history", json.dumps(execution))
            self.redis_client.ltrim("trade_history", 0, 999)  # Keep last 1000 trades

            # Update last execution
            self.state["last_execution"] = execution

            print(f"✅ Trade executed: {execution['execution_id']}")

            return execution

        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            self.state["execution_stats"]["failed_trades"] += 1
            from backend.utils.exceptions import TradingException, ErrorCode
            raise TradingException(
                message="Failed to execute trade",
                error_code=ErrorCode.TRADING_ORDER_ERROR,
                details={"symbol": symbol, "signal_type": signal_type, "original_error": str(e)},
                original_exception=e
            )

    async def validate_execution_params(
        self, symbol: str, position_size: Dict[str, Any], price: float
    ) -> bool:
        """Validate execution parameters"""
        try:
            # Check if exchange is connected
            if self.state["exchange_status"] != "connected":
                return False

            # Check if symbol is supported
            exchange_config = self.redis_client.get("exchange_config")
            if exchange_config:
                config = json.loads(exchange_config)
                supported_pairs = config.get("supported_pairs", [])
                if symbol not in supported_pairs:
                    return False

            # Check position size
            size = position_size.get("position_size", 0)
            if size <= 0:
                return False

            # Check price
            if price <= 0:
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating execution parameters: {e}")
            return False

    async def calculate_order_params(
        self,
        symbol: str,
        signal_type: str,
        position_size: Dict[str, Any],
        price: float,
    ) -> Dict[str, Any]:
        """Calculate order parameters"""
        try:
            quantity = position_size.get("position_size", 0)

            # Determine order type based on signal
            if signal_type == "buy":
                order_type = "market"  # Use market order for immediate execution
                side = "buy"
            elif signal_type == "sell":
                order_type = "market"
                side = "sell"
            else:
                from backend.utils.exceptions import TradingException, ErrorCode
            raise TradingException(
                message=f"Invalid signal type: {signal_type}",
                error_code=ErrorCode.TRADING_ORDER_ERROR,
                details={"signal_type": signal_type}
            )

            # Calculate fees
            exchange_config = self.redis_client.get("exchange_config")
            fee_rate = 0.001  # Default fee rate
            if exchange_config:
                config = json.loads(exchange_config)
                fee_rate = config.get("fee_rate", 0.001)

            total_value = quantity * price
            fees = total_value * fee_rate

            return {
                "symbol": symbol,
                "side": side,
                "order_type": order_type,
                "quantity": quantity,
                "price": price,
                "total_value": total_value,
                "fees": fees,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            print(f"❌ Error calculating order parameters: {e}")
            raise

    async def place_order(self, order_params: Dict[str, Any]) -> Dict[str, Any]:
        """Place order on exchange"""
        try:
            # Simulate order placement
            order_id = f"order_{int(datetime.now().timestamp())}"

            # Simulate order result
            order_result = {
                "order_id": order_id,
                "status": "filled",  # Assume immediate fill for market orders
                "filled_quantity": order_params["quantity"],
                "filled_price": order_params["price"],
                "fees": order_params["fees"],
                "timestamp": datetime.now().isoformat(),
            }

            # Add to active orders
            self.state["active_orders"][order_id] = {
                **order_params,
                **order_result,
            }

            # Store in Redis
            self.redis_client.set(
                f"order:{order_id}",
                json.dumps(self.state["active_orders"][order_id]),
                ex=3600,
            )
            self.redis_client.set(
                "active_orders",
                json.dumps(self.state["active_orders"]),
                ex=3600,
            )

            print(f"📋 Order placed: {order_id}")

            return order_result

        except Exception as e:
            print(f"❌ Error placing order: {e}")
            raise

    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an order"""
        try:
            if order_id not in self.state["active_orders"]:
                raise ValueError(f"Order {order_id} not found")

            # Simulate order cancellation
            cancellation_result = {
                "order_id": order_id,
                "status": "cancelled",
                "timestamp": datetime.now().isoformat(),
            }

            # Remove from active orders
            del self.state["active_orders"][order_id]

            # Update Redis
            self.redis_client.delete(f"order:{order_id}")
            self.redis_client.set(
                "active_orders",
                json.dumps(self.state["active_orders"]),
                ex=3600,
            )

            print(f"❌ Order cancelled: {order_id}")

            return cancellation_result

        except Exception as e:
            print(f"❌ Error cancelling order: {e}")
            raise

    async def modify_order(self, order_id: str, new_params: Dict[str, Any]) -> Dict[str, Any]:
        """Modify an order"""
        try:
            if order_id not in self.state["active_orders"]:
                raise ValueError(f"Order {order_id} not found")

            # Update order parameters
            self.state["active_orders"][order_id].update(new_params)

            # Simulate order modification
            modification_result = {
                "order_id": order_id,
                "status": "modified",
                "new_params": new_params,
                "timestamp": datetime.now().isoformat(),
            }

            # Update Redis
            self.redis_client.set(
                f"order:{order_id}",
                json.dumps(self.state["active_orders"][order_id]),
                ex=3600,
            )
            self.redis_client.set(
                "active_orders",
                json.dumps(self.state["active_orders"]),
                ex=3600,
            )

            print(f"✏️ Order modified: {order_id}")

            return modification_result

        except Exception as e:
            print(f"❌ Error modifying order: {e}")
            raise

    async def monitor_active_orders(self):
        """Monitor active orders"""
        try:
            for order_id, order in self.state["active_orders"].items():
                # Check if order is still active
                if order.get("status") == "filled":
                    # Remove filled orders
                    del self.state["active_orders"][order_id]
                    self.redis_client.delete(f"order:{order_id}")

                    # Update execution stats
                    self.state["execution_stats"]["successful_trades"] += 1

                    print(f"✅ Order filled: {order_id}")

                # Check for order timeouts
                order_time = datetime.fromisoformat(
                    order.get("timestamp", datetime.now().isoformat())
                )
                if (datetime.now() - order_time) > timedelta(minutes=5):
                    # Cancel timed out orders
                    await self.cancel_order(order_id)

            # Update Redis
            self.redis_client.set(
                "active_orders",
                json.dumps(self.state["active_orders"]),
                ex=3600,
            )

        except Exception as e:
            print(f"❌ Error monitoring active orders: {e}")

    async def update_positions(self):
        """Update position information"""
        try:
            # Get current positions from exchange
            positions = await self.get_exchange_positions()

            # Update state
            self.state["positions"] = positions

            # Store in Redis
            self.redis_client.set("positions", json.dumps(positions), ex=300)

        except Exception as e:
            print(f"❌ Error updating positions: {e}")

    async def update_positions_from_portfolio(self, portfolio: Dict[str, Any]):
        """Update positions from portfolio data"""
        try:
            positions = portfolio.get("positions", [])

            # Convert to position dict
            position_dict = {}
            for position in positions:
                symbol = position.get("symbol")
                if symbol:
                    position_dict[symbol] = position

            self.state["positions"] = position_dict

            # Store in Redis
            self.redis_client.set("positions", json.dumps(position_dict), ex=300)

        except Exception as e:
            print(f"❌ Error updating positions from portfolio: {e}")

    async def get_exchange_positions(self) -> Dict[str, Any]:
        """Get current positions from exchange"""
        try:
            # Simulate getting positions from exchange
            # In real implementation, this would call exchange API

            positions = {}

            # Get positions from Redis as fallback
            positions_data = self.redis_client.get("positions")
            if positions_data:
                positions = json.loads(positions_data)

            return positions

        except Exception as e:
            print(f"❌ Error getting exchange positions: {e}")
            return {}

    async def check_order_timeouts(self):
        """Check for order timeouts"""
        try:
            current_time = datetime.now()

            for order_id, order in list(self.state["active_orders"].items()):
                order_time = datetime.fromisoformat(
                    order.get("timestamp", current_time.isoformat())
                )

                # Cancel orders older than 5 minutes
                if (current_time - order_time) > timedelta(minutes=5):
                    print(f"⏰ Order timeout: {order_id}")
                    await self.cancel_order(order_id)

        except Exception as e:
            print(f"❌ Error checking order timeouts: {e}")

    async def update_order_prices(self, symbol: str, price: float):
        """Update order prices for a symbol"""
        try:
            for order_id, order in self.state["active_orders"].items():
                if order.get("symbol") == symbol:
                    # Update current price
                    order["current_price"] = price

                    # Check for price-based triggers
                    await self.check_price_triggers(order_id, order, price)

        except Exception as e:
            print(f"❌ Error updating order prices: {e}")

    async def check_price_triggers(
        self, order_id: str, order: Dict[str, Any], current_price: float
    ):
        """Check for price-based order triggers"""
        try:
            # Check stop loss
            stop_loss = order.get("stop_loss")
            if stop_loss and current_price <= stop_loss:
                print(f"🛑 Stop loss triggered for order {order_id}")
                await self.cancel_order(order_id)

            # Check take profit
            take_profit = order.get("take_profit")
            if take_profit and current_price >= take_profit:
                print(f"🎯 Take profit triggered for order {order_id}")
                await self.cancel_order(order_id)

        except Exception as e:
            print(f"❌ Error checking price triggers: {e}")

    async def check_order_triggers(self, symbol: str, price: float):
        """Check for order triggers"""
        try:
            # This would check for various order triggers
            # For now, just update prices
            await self.update_order_prices(symbol, price)

        except Exception as e:
            print(f"❌ Error checking order triggers: {e}")

    async def update_execution_metrics(self):
        """Update execution metrics"""
        try:
            metrics = {
                "agent_id": self.agent_id,
                "execution_stats": self.state["execution_stats"],
                "active_orders_count": len(self.state["active_orders"]),
                "positions_count": len(self.state["positions"]),
                "exchange_status": self.state["exchange_status"],
                "last_execution": self.state["last_execution"],
                "timestamp": datetime.now().isoformat(),
            }

            # Store metrics in Redis
            self.redis_client.set(f"agent_metrics:{self.agent_id}", json.dumps(metrics), ex=300)

        except Exception as e:
            print(f"❌ Error updating execution metrics: {e}")

    async def handle_market_data(self, message: Dict[str, Any]):
        """Handle market data message"""
        try:
            market_data = message.get("market_data", {})
            print(f"📊 Execution Agent received market data for {len(market_data)} symbols")
            
            # Process market data
            await self.process_market_data(market_data)
            
            # Check order triggers for each symbol
            for symbol, data in market_data.items():
                price = data.get("price", 0)
                if price > 0:
                    await self.check_order_triggers(symbol, price)
            
            # Update execution metrics
            await self.update_execution_metrics()
            
        except Exception as e:
            print(f"❌ Error handling market data: {e}")
            await self.broadcast_error(f"Market data handling error: {e}")
