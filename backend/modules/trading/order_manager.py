"""
Order Manager Module for Mystic Trading Platform

Extracted from trade_engine.py and auto_trading_manager.py to improve modularity.
Handles order creation, management, and execution.
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types"""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order sides"""

    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order statuses"""

    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    PARTIALLY_FILLED = "partially_filled"


@dataclass
class Order:
    """Order data structure"""

    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    timestamp: float = None
    exchange: str = "binance"

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class OrderManager:
    """Manages order creation, execution, and tracking"""

    def __init__(self):
        self.orders: Dict[str, Order] = {}
        self.order_counter = 0
        logger.info("✅ OrderManager initialized")

    def create_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        exchange: str = "binance",
    ) -> Order:
        """Create a new order"""
        try:
            self.order_counter += 1
            order_id = f"order_{self.order_counter}_{int(time.time())}"

            order = Order(
                id=order_id,
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                exchange=exchange,
            )

            self.orders[order_id] = order
            logger.info(f"✅ Order created: {order_id} - {symbol} {side.value} {quantity}")

            return order
        except Exception as e:
            logger.error(f"❌ Error creating order: {str(e)}")
            raise

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID"""
        return self.orders.get(order_id)

    def get_orders(
        self,
        symbol: Optional[str] = None,
        status: Optional[OrderStatus] = None,
        limit: int = 100,
    ) -> List[Order]:
        """Get orders with optional filtering"""
        try:
            orders = list(self.orders.values())

            if symbol:
                orders = [o for o in orders if o.symbol == symbol]

            if status:
                orders = [o for o in orders if o.status == status]

            return orders[:limit]
        except Exception as e:
            logger.error(f"❌ Error getting orders: {str(e)}")
            return []

    def update_order_status(self, order_id: str, status: OrderStatus) -> bool:
        """Update order status"""
        try:
            if order_id in self.orders:
                self.orders[order_id].status = status
                logger.info(f"✅ Order {order_id} status updated to {status.value}")
                return True
            return False
        except Exception as e:
            logger.error(f"❌ Error updating order status: {str(e)}")
            return False

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        try:
            if order_id in self.orders:
                self.orders[order_id].status = OrderStatus.CANCELLED
                logger.info(f"✅ Order {order_id} cancelled")
                return True
            return False
        except Exception as e:
            logger.error(f"❌ Error cancelling order: {str(e)}")
            return False

    def get_order_history(
        self, symbol: Optional[str] = None, limit: int = 100, offset: int = 0
    ) -> Dict[str, Any]:
        """Get order history with pagination"""
        try:
            orders = self.get_orders(symbol=symbol, limit=limit + offset)
            orders = orders[offset : offset + limit]

            return {
                "orders": [self._order_to_dict(order) for order in orders],
                "total": len(orders),
                "limit": limit,
                "offset": offset,
            }
        except Exception as e:
            logger.error(f"❌ Error getting order history: {str(e)}")
            return {"orders": [], "total": 0, "limit": limit, "offset": offset}

    def _order_to_dict(self, order: Order) -> Dict[str, Any]:
        """Convert order to dictionary"""
        return {
            "id": order.id,
            "symbol": order.symbol,
            "side": order.side.value,
            "order_type": order.order_type.value,
            "quantity": order.quantity,
            "price": order.price,
            "status": order.status.value,
            "timestamp": order.timestamp,
            "exchange": order.exchange,
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get order statistics"""
        try:
            total_orders = len(self.orders)
            pending_orders = len(
                [o for o in self.orders.values() if o.status == OrderStatus.PENDING]
            )
            filled_orders = len([o for o in self.orders.values() if o.status == OrderStatus.FILLED])
            cancelled_orders = len(
                [o for o in self.orders.values() if o.status == OrderStatus.CANCELLED]
            )

            return {
                "total_orders": total_orders,
                "pending_orders": pending_orders,
                "filled_orders": filled_orders,
                "cancelled_orders": cancelled_orders,
                "fill_rate": (filled_orders / total_orders if total_orders > 0 else 0),
            }
        except Exception as e:
            logger.error(f"❌ Error getting order statistics: {str(e)}")
            return {}


# Global order manager instance
order_manager = OrderManager()
