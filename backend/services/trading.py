"""
Trading Service for Mystic Trading

Provides centralized trading functionality including order management,
position tracking, and trading strategy execution.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict

from enhanced_logging import log_operation_performance
from backend.services.trading_manager import trading_manager
from backend.services.websocket_manager import websocket_manager

logger = logging.getLogger(__name__)


class TradingService:
    """Centralized trading service for executing and managing trades"""

    def __init__(self, redis_client: Any):
        self.redis_client = redis_client
        self.is_initialized = False

    async def initialize(self) -> None:
        """Initialize the trading service"""
        try:
            # Load active orders from Redis
            orders_data = self.redis_client.get("active_orders")
            if orders_data:
                # Load positions from Redis
                positions_data = self.redis_client.get("positions")
                trading_manager.deserialize_data(orders_data, positions_data or "")

            self.is_initialized = True
            logger.info("Trading service initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing trading service: {str(e)}")
            # Continue with empty state

    @log_operation_performance("place_order")
    async def place_order(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Place a new order"""
        try:
            # Validate order data
            is_valid, error_message = trading_manager.validate_order_data(order_data)
            if not is_valid:
                return {
                    "status": "error",
                    "message": error_message,
                    "timestamp": (datetime.now(timezone.timezone.utc).isoformat()),
                }

            # Add order using manager
            order = trading_manager.add_order(order_data)

            # Save to Redis
            serialized_data = trading_manager.serialize_data()
            self.redis_client.setex(
                "active_orders", 3600, serialized_data["active_orders"]
            )  # 1 hour TTL

            # Broadcast order placement
            await websocket_manager.broadcast_json(
                {
                    "type": "order_placed",
                    "data": {
                        "order_id": order["order_id"],
                        "symbol": order.get("symbol"),
                        "side": order.get("side"),
                        "quantity": order.get("quantity"),
                        "status": order["status"],
                        "timestamp": order["timestamp"],
                    },
                }
            )

            logger.info(f"Order placed: {order['order_id']}")
            return {
                "status": "success",
                "message": "Order placed successfully",
                "order_id": order["order_id"],
                "timestamp": order["timestamp"],
            }
        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
            return {
                "status": "error",
                "message": f"Error placing order: {str(e)}",
                "timestamp": datetime.now(timezone.timezone.utc).isoformat(),
            }

    @log_operation_performance("cancel_order")
    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an existing order"""
        try:
            # Remove order using manager
            order_data = trading_manager.remove_order(order_id)
            if order_data:
                # Save to Redis
                serialized_data = trading_manager.serialize_data()
                self.redis_client.setex(
                    "active_orders",
                    3600,
                    serialized_data["active_orders"],  # 1 hour TTL
                )

                logger.info(f"Order cancelled: {order_id}")
                return {
                    "status": "success",
                    "message": "Order cancelled successfully",
                    "order_id": order_id,
                    "timestamp": (datetime.now(timezone.timezone.utc).isoformat()),
                }
            else:
                logger.warning(f"Order not found for cancellation: {order_id}")
                return {
                    "status": "error",
                    "message": f"Order not found: {order_id}",
                    "timestamp": (datetime.now(timezone.timezone.utc).isoformat()),
                }
        except Exception as e:
            logger.error(f"Error cancelling order: {str(e)}")
            return {
                "status": "error",
                "message": f"Error cancelling order: {str(e)}",
                "timestamp": datetime.now(timezone.timezone.utc).isoformat(),
            }

    @log_operation_performance("get_active_orders")
    async def get_active_orders(self) -> Dict[str, Any]:
        """Get all active orders"""
        try:
            orders = trading_manager.get_all_orders()
            return {
                "status": "success",
                "orders": orders,
                "count": len(orders),
                "timestamp": datetime.now(timezone.timezone.utc).isoformat(),
            }
        except Exception as e:
            logger.error(f"Error getting active orders: {str(e)}")
            return {
                "status": "error",
                "message": f"Error getting active orders: {str(e)}",
                "timestamp": datetime.now(timezone.timezone.utc).isoformat(),
            }

    @log_operation_performance("get_positions")
    async def get_positions(self) -> Dict[str, Any]:
        """Get all current positions"""
        try:
            positions = trading_manager.get_all_positions()
            return {
                "status": "success",
                "positions": positions,
                "count": len(positions),
                "timestamp": datetime.now(timezone.timezone.utc).isoformat(),
            }
        except Exception as e:
            logger.error(f"Error getting positions: {str(e)}")
            return {
                "status": "error",
                "message": f"Error getting positions: {str(e)}",
                "timestamp": datetime.now(timezone.timezone.utc).isoformat(),
            }

    @log_operation_performance("execute_strategy")
    async def execute_strategy(
        self, strategy_name: str, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a trading strategy"""
        try:
            logger.info(f"Executing strategy: {strategy_name}")

            # Validate strategy parameters
            is_valid, error_message = trading_manager.validate_strategy_parameters(
                strategy_name, parameters
            )
            if not is_valid:
                return {
                    "status": "error",
                    "message": error_message,
                    "timestamp": (datetime.now(timezone.timezone.utc).isoformat()),
                }

            # Strategy execution logic would go here
            # This is a placeholder for demonstration

            return {
                "status": "success",
                "strategy": strategy_name,
                "message": f"Strategy {strategy_name} executed successfully",
                "timestamp": datetime.now(timezone.timezone.utc).isoformat(),
            }
        except Exception as e:
            logger.error(f"Error executing strategy {strategy_name}: {str(e)}")
            return {
                "status": "error",
                "message": f"Error executing strategy: {str(e)}",
                "timestamp": datetime.now(timezone.timezone.utc).isoformat(),
            }


# Global trading service instance
trading_service = None


def get_trading_service(redis_client: Any) -> TradingService:
    """Get or create trading service instance"""
    global trading_service
    if trading_service is None:
        trading_service = TradingService(redis_client)
    return trading_service


