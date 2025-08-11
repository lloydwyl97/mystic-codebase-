"""
Autosell Service for Mystic AI Trading Platform
Monitors cached trades and current prices to execute automated sell orders.
"""

import logging
import uuid
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone
import sys
import os

# Add backend to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from modules.ai.persistent_cache import PersistentCache

logger = logging.getLogger(__name__)


class AutoExecutionService:
    """Mock AutoExecutionService for now - will be replaced with actual implementation"""

    def __init__(self):
        self.cache = PersistentCache()
        self.simulation_mode = True  # Default to simulation mode

    def execute_sell_order(self, exchange: str, symbol: str, quantity: float,
                          price: float, order_type: str = "market") -> Dict[str, Any]:
        """Execute a sell order (simulated or real)"""
        try:
            order_id = str(uuid.uuid4())
            timestamp = datetime.now(timezone.utc)

            # Calculate total value
            total_value = quantity * price

            # Log the trade
            trade_success = self.cache.log_trade(
                trade_id=order_id,
                symbol=symbol,
                side="SELL",
                quantity=quantity,
                price=price,
                exchange=exchange,
                status="completed" if self.simulation_mode else "pending"
            )

            if trade_success:
                logger.info(f"✅ Sell order executed: {symbol} {quantity} @ ${price:.2f} = ${total_value:.2f}")

                return {
                    "success": True,
                    "order_id": order_id,
                    "symbol": symbol,
                    "side": "SELL",
                    "quantity": quantity,
                    "price": price,
                    "total_value": total_value,
                    "exchange": exchange,
                    "order_type": order_type,
                    "simulation": self.simulation_mode,
                    "timestamp": timestamp.isoformat()
                }
            else:
                logger.error(f"❌ Failed to log sell trade for {symbol}")
                return {"success": False, "error": "Failed to log trade"}

        except Exception as e:
            logger.error(f"❌ Failed to execute sell order for {symbol}: {e}")
            return {"success": False, "error": str(e)}

    def set_simulation_mode(self, enabled: bool):
        """Enable or disable simulation mode"""
        self.simulation_mode = enabled
        logger.info(f"Simulation mode: {'ENABLED' if enabled else 'DISABLED'}")


class AutoSellService:
    def __init__(self):
        """Initialize autosell service with dependencies"""
        self.cache = PersistentCache()
        self.execution_service = AutoExecutionService()

        # Configuration
        self.take_profit_percentage = 0.05  # 5% take profit
        self.stop_loss_percentage = 0.03     # 3% stop loss
        self.trailing_stop_percentage = 0.02  # 2% trailing stop
        self.min_profit_percentage = 0.01    # 1% minimum profit to start trailing

        # Track active sell orders to prevent duplicates
        self.active_sells = set()

        # Track trailing stops for each position
        self.trailing_stops = {}  # {symbol: highest_price_seen}

        logger.info("✅ AutoSellService initialized")

    def get_open_buy_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get open buy positions from trade journal"""
        try:
            # Get all buy trades
            all_trades = self.cache.get_trades(symbol=symbol)
            buy_trades = [trade for trade in all_trades if trade['side'] == 'BUY']

            # Group by symbol to calculate average buy price and total quantity
            positions = {}
            for trade in buy_trades:
                symbol = trade['symbol']
                if symbol not in positions:
                    positions[symbol] = {
                        'symbol': symbol,
                        'total_quantity': 0,
                        'total_value': 0,
                        'trades': []
                    }

                positions[symbol]['total_quantity'] += trade['quantity']
                positions[symbol]['total_value'] += trade['total_value']
                positions[symbol]['trades'].append(trade)

            # Calculate average buy price and convert to list
            open_positions = []
            for symbol, position in positions.items():
                avg_buy_price = position['total_value'] / position['total_quantity']

                open_positions.append({
                    'symbol': symbol,
                    'quantity': position['total_quantity'],
                    'avg_buy_price': avg_buy_price,
                    'total_invested': position['total_value'],
                    'trades': position['trades']
                })

            return open_positions

        except Exception as e:
            logger.error(f"❌ Failed to get open positions: {e}")
            return []

    def check_sell_conditions(self, symbol: str, avg_buy_price: float,
                            current_price: float) -> Tuple[bool, Dict[str, Any]]:
        """Check if sell conditions are met for a position"""
        try:
            # Calculate profit/loss percentage
            profit_percentage = (current_price - avg_buy_price) / avg_buy_price

            # Update trailing stop if we have a position
            if symbol in self.trailing_stops:
                highest_price = self.trailing_stops[symbol]
                if current_price > highest_price:
                    self.trailing_stops[symbol] = current_price
                    highest_price = current_price
            else:
                # Initialize trailing stop if we have profit
                if profit_percentage >= self.min_profit_percentage:
                    self.trailing_stops[symbol] = current_price
                    highest_price = current_price
                else:
                    highest_price = avg_buy_price

            # Calculate trailing stop price
            trailing_stop_price = highest_price * (1 - self.trailing_stop_percentage)

            # Check sell conditions
            take_profit_hit = profit_percentage >= self.take_profit_percentage
            stop_loss_hit = profit_percentage <= -self.stop_loss_percentage
            trailing_stop_hit = current_price <= trailing_stop_price and profit_percentage >= self.min_profit_percentage

            should_sell = take_profit_hit or stop_loss_hit or trailing_stop_hit

            conditions = {
                "current_price": current_price,
                "avg_buy_price": avg_buy_price,
                "profit_percentage": profit_percentage,
                "take_profit_hit": take_profit_hit,
                "stop_loss_hit": stop_loss_hit,
                "trailing_stop_hit": trailing_stop_hit,
                "highest_price": highest_price,
                "trailing_stop_price": trailing_stop_price,
                "should_sell": should_sell
            }

            if should_sell:
                logger.info(f"🎯 Sell conditions met for {symbol}: Profit={profit_percentage:.2%}, "
                          f"Take Profit={take_profit_hit}, Stop Loss={stop_loss_hit}, "
                          f"Trailing Stop={trailing_stop_hit}")

            return should_sell, conditions

        except Exception as e:
            logger.error(f"❌ Failed to check sell conditions for {symbol}: {e}")
            return False, {"error": str(e)}

    def execute_autosell(self, exchange: str, symbol: str, quantity: float,
                        avg_buy_price: float) -> Dict[str, Any]:
        """Execute an automated sell order for a position"""
        try:
            # Check if sell order already active
            order_key = f"{exchange}:{symbol}"
            if order_key in self.active_sells:
                return {"success": False, "reason": "Sell order already active"}

            # Get current price
            latest_price = self.cache.get_latest_price('aggregated', symbol)
            if not latest_price:
                return {"success": False, "reason": "No price data available"}

            current_price = float(latest_price['price'])

            # Check sell conditions
            should_sell, conditions = self.check_sell_conditions(symbol, avg_buy_price, current_price)

            if not should_sell:
                return {
                    "success": False,
                    "reason": "Sell conditions not met",
                    "conditions": conditions
                }

            # Mark order as active
            self.active_sells.add(order_key)

            try:
                # Execute sell order
                result = self.execution_service.execute_sell_order(
                    exchange=exchange,
                    symbol=symbol,
                    quantity=quantity,
                    price=current_price
                )

                if result["success"]:
                    logger.info(f"🚀 Autosell executed: {symbol} {quantity} @ ${current_price:.2f}")

                    # Remove from trailing stops if sold
                    if symbol in self.trailing_stops:
                        del self.trailing_stops[symbol]

                return result

            finally:
                # Remove from active orders
                self.active_sells.discard(order_key)

        except Exception as e:
            logger.error(f"❌ Failed to execute autosell for {symbol}: {e}")
            return {"success": False, "error": str(e)}

    def execute_all_autosells(self) -> Dict[str, Any]:
        """Execute autosell for all open positions"""
        try:
            logger.info("🔄 Starting autosell execution for all positions...")

            # Get all open positions
            open_positions = self.get_open_buy_positions()

            if not open_positions:
                return {
                    "success": True,
                    "message": "No open positions to sell",
                    "total_positions": 0,
                    "successful_sells": 0
                }

            results = []
            successful_sells = 0

            for position in open_positions:
                try:
                    # Try multiple exchanges
                    exchanges = ["coinbase_us", "binance_us", "kraken_us"]

                    for exchange in exchanges:
                        result = self.execute_autosell(
                            exchange=exchange,
                            symbol=position['symbol'],
                            quantity=position['quantity'],
                            avg_buy_price=position['avg_buy_price']
                        )
                        results.append({
                            "symbol": position['symbol'],
                            "exchange": exchange,
                            "quantity": position['quantity'],
                            "avg_buy_price": position['avg_buy_price'],
                            "result": result
                        })

                        if result.get("success"):
                            successful_sells += 1
                            break  # Move to next position if successful

                except Exception as e:
                    logger.error(f"❌ Failed to process position {position['symbol']}: {e}")
                    results.append({
                        "symbol": position['symbol'],
                        "exchange": "unknown",
                        "quantity": position['quantity'],
                        "avg_buy_price": position['avg_buy_price'],
                        "result": {"success": False, "error": str(e)}
                    })

            summary = {
                "success": True,
                "total_positions": len(open_positions),
                "successful_sells": successful_sells,
                "results": results,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            logger.info(f"✅ Autosell execution completed: {successful_sells}/{len(open_positions)} successful")
            return summary

        except Exception as e:
            logger.error(f"❌ Failed to execute all autosells: {e}")
            return {"success": False, "error": str(e)}

    def get_open_orders(self) -> List[Dict[str, Any]]:
        """Get all open autosell orders"""
        try:
            return [
                {
                    "order_id": order.get("order_id", ""),
                    "symbol": order.get("symbol", ""),
                    "quantity": order.get("quantity", 0.0),
                    "price": order.get("price", 0.0),
                    "status": order.get("status", "pending"),
                    "exchange": order.get("exchange", ""),
                    "timestamp": order.get("timestamp", ""),
                    "order_type": "autosell"
                }
                for order in self.active_sells
            ]
        except Exception as e:
            logger.error(f"❌ Failed to get open orders: {e}")
            return []

    def get_trailing_stops(self) -> List[Dict[str, Any]]:
        """Get all active trailing stops"""
        try:
            return [
                {
                    "symbol": symbol,
                    "initial_price": stop_data.get("initial_price", 0.0),
                    "current_price": stop_data.get("current_price", 0.0),
                    "trailing_percentage": stop_data.get("trailing_percentage", 0.0),
                    "quantity": stop_data.get("quantity", 0.0),
                    "timestamp": stop_data.get("timestamp", ""),
                    "status": "active"
                }
                for symbol, stop_data in self.trailing_stops.items()
            ]
        except Exception as e:
            logger.error(f"❌ Failed to get trailing stops: {e}")
            return []

    def get_service_status(self) -> Dict[str, Any]:
        """Get current service status and statistics"""
        try:
            # Get cache stats
            cache_stats = self.cache.get_cache_stats()

            # Get open positions
            open_positions = self.get_open_buy_positions()

            # Get recent trades
            recent_trades = self.cache.get_trades(limit=10)

            # Calculate total portfolio value
            total_invested = sum(pos['total_invested'] for pos in open_positions)

            return {
                "service": "AutoSellService",
                "status": "active",
                "simulation_mode": self.execution_service.simulation_mode,
                "active_sells": len(self.active_sells),
                "open_positions": len(open_positions),
                "total_invested": total_invested,
                "trailing_stops": len(self.trailing_stops),
                "cache_stats": cache_stats,
                "recent_trades": recent_trades,
                "configuration": {
                    "take_profit_percentage": self.take_profit_percentage,
                    "stop_loss_percentage": self.stop_loss_percentage,
                    "trailing_stop_percentage": self.trailing_stop_percentage,
                    "min_profit_percentage": self.min_profit_percentage
                }
            }

        except Exception as e:
            logger.error(f"❌ Failed to get service status: {e}")
            return {"success": False, "error": str(e)}

    def reset_trailing_stops(self):
        """Reset all trailing stops (useful for testing)"""
        self.trailing_stops.clear()
        logger.info("🔄 Trailing stops reset")


# Global service instance
autosell_service = AutoSellService()


def get_autosell_service() -> AutoSellService:
    """Get the global autosell service instance"""
    return autosell_service


if __name__ == "__main__":
    # Test the service
    service = AutoSellService()
    print(f"✅ AutoSellService initialized: {service}")

    # Test status
    status = service.get_service_status()
    print(f"Service status: {status['status']}")
    print(f"Simulation mode: {status['simulation_mode']}")
    print(f"Active sells: {status['active_sells']}")
    print(f"Open positions: {status['open_positions']}")
    print(f"Total invested: ${status['total_invested']:.2f}")
