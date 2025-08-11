"""
Trading Manager

Handles business logic for trading operations including order validation,
position management, and strategy execution.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from services.websocket_manager import websocket_manager

logger = logging.getLogger(__name__)


class TradingManager:
    """Manages trading business logic and operations."""

    def __init__(self):
        self.active_orders: Dict[str, Dict[str, Any]] = {}
        self.positions: Dict[str, Dict[str, Any]] = {}

    def validate_order_data(self, order_data: Dict[str, Any]) -> tuple[bool, str]:
        """Validate order data before processing."""
        required_fields = ["symbol", "side", "quantity"]

        for field in required_fields:
            if field not in order_data:
                return False, f"Missing required field: {field}"

        # Validate symbol
        if not order_data.get("symbol"):
            return False, "Symbol cannot be empty"

        # Validate side
        if order_data.get("side") not in ["buy", "sell"]:
            return False, "Side must be 'buy' or 'sell'"

        # Validate quantity
        try:
            quantity = float(order_data.get("quantity", 0))
            if quantity <= 0:
                return False, "Quantity must be greater than 0"
        except (ValueError, TypeError):
            return False, "Quantity must be a valid number"

        return True, "Order data is valid"

    def generate_order_id(self) -> str:
        """Generate a unique order ID."""
        return f"order_{datetime.now(timezone.timezone.utc).timestamp()}"

    def add_order(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add a new order to active orders."""
        order_id = order_data.get("order_id", self.generate_order_id())
        order_data["order_id"] = order_id
        order_data["timestamp"] = datetime.now(timezone.timezone.utc).isoformat()
        order_data["status"] = "pending"

        self.active_orders[order_id] = order_data
        return order_data

    def remove_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Remove an order from active orders."""
        return self.active_orders.pop(order_id, None)

    def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get an order by ID."""
        return self.active_orders.get(order_id)

    def get_all_orders(self) -> Dict[str, Dict[str, Any]]:
        """Get all active orders."""
        return self.active_orders.copy()

    def update_order_status(self, order_id: str, status: str) -> bool:
        """Update order status."""
        if order_id in self.active_orders:
            self.active_orders[order_id]["status"] = status
            self.active_orders[order_id]["updated_at"] = datetime.now(
                timezone.timezone.utc
            ).isoformat()
            return True
        return False

    def add_position(self, position_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add a new position."""
        symbol = position_data.get("symbol")
        if not symbol:
            raise ValueError("Position must have a symbol")

        position_id = f"pos_{symbol}_{datetime.now(timezone.timezone.utc).timestamp()}"
        position_data["position_id"] = position_id
        position_data["timestamp"] = datetime.now(timezone.timezone.utc).isoformat()

        self.positions[position_id] = position_data
        return position_data

    def update_position(self, position_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing position."""
        if position_id in self.positions:
            self.positions[position_id].update(updates)
            self.positions[position_id]["updated_at"] = datetime.now(
                timezone.timezone.utc
            ).isoformat()
            return True
        return False

    def get_position(self, position_id: str) -> Optional[Dict[str, Any]]:
        """Get a position by ID."""
        return self.positions.get(position_id)

    def get_all_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get all positions."""
        return self.positions.copy()

    def calculate_position_pnl(
        self, position: Dict[str, Any], current_price: float
    ) -> Dict[str, Any]:
        """Calculate position profit/loss."""
        entry_price = position.get("entry_price", 0)
        quantity = position.get("quantity", 0)
        side = position.get("side", "buy")

        if entry_price <= 0 or quantity <= 0:
            return {"pnl": 0.0, "pnl_percentage": 0.0, "unrealized_pnl": 0.0}

        if side == "buy":
            pnl = (current_price - entry_price) * quantity
            pnl_percentage = ((current_price - entry_price) / entry_price) * 100
        else:  # sell
            pnl = (entry_price - current_price) * quantity
            pnl_percentage = ((entry_price - current_price) / entry_price) * 100

        result = {
            "pnl": pnl,
            "pnl_percentage": pnl_percentage,
            "unrealized_pnl": pnl,
        }

        # Broadcast position P&L update
        import asyncio

        asyncio.create_task(
            websocket_manager.broadcast_json(
                {
                    "type": "position_pnl",
                    "data": {
                        "symbol": position.get("symbol"),
                        "side": side,
                        "entry_price": entry_price,
                        "current_price": current_price,
                        "quantity": quantity,
                        "pnl": pnl,
                        "pnl_percentage": pnl_percentage,
                        "timestamp": (datetime.now(timezone.timezone.utc).isoformat()),
                    },
                }
            )
        )

        return result

    def validate_strategy_parameters(
        self, strategy_name: str, parameters: Dict[str, Any]
    ) -> tuple[bool, str]:
        """Validate strategy parameters."""
        if not strategy_name:
            return False, "Strategy name is required"

        # Add specific validation for different strategies
        if strategy_name == "momentum":
            required_params = ["lookback_period", "threshold"]
            for param in required_params:
                if param not in parameters:
                    return (
                        False,
                        f"Missing required parameter for momentum strategy: {param}",
                    )

        elif strategy_name == "mean_reversion":
            required_params = ["window_size", "std_dev"]
            for param in required_params:
                if param not in parameters:
                    return (
                        False,
                        f"Missing required parameter for mean reversion strategy: {param}",
                    )

        return True, "Strategy parameters are valid"

    def serialize_data(self) -> Dict[str, str]:
        """Serialize data for storage."""
        return {
            "active_orders": json.dumps(self.active_orders),
            "positions": json.dumps(self.positions),
        }

    def deserialize_data(self, orders_data: str, positions_data: str):
        """Deserialize data from storage."""
        try:
            if orders_data:
                self.active_orders = json.loads(orders_data)
            if positions_data:
                self.positions = json.loads(positions_data)
        except json.JSONDecodeError as e:
            logger.error(f"Error deserializing trading data: {e}")
            # Continue with empty state
            self.active_orders = {}
            self.positions = {}


# Global instance
trading_manager = TradingManager()
