# integration_hook.py
"""
Simple integration hook for existing trading systems.

This file shows how to add trade logging to your existing trading bots
with minimal code changes.
"""

from trade_memory_integration import log_trade_entry, log_trade_exit
from db_logger import get_strategy_stats
import logging

logger = logging.getLogger(__name__)


class TradingHook:
    """
    Simple hook class that can be added to existing trading systems.
    """

    def __init__(self):
        self.active_trades = {}

    def on_trade_entry(
        self,
        coin: str,
        strategy_name: str,
        entry_price: float,
        quantity: float = 1.0,
        entry_reason: str = "",
    ):
        """
        Call this when entering a trade

        Args:
            coin: Trading pair (e.g., 'BTCUSDT')
            strategy_name: Name of strategy used
            entry_price: Entry price
            quantity: Trade quantity
            entry_reason: Reason for entry
        """
        try:
            trade_id = log_trade_entry(
                coin=coin,
                strategy_name=strategy_name,
                entry_price=entry_price,
                quantity=quantity,
                entry_reason=entry_reason,
            )

            if trade_id:
                self.active_trades[trade_id] = {
                    "coin": coin,
                    "strategy_name": strategy_name,
                    "entry_price": entry_price,
                    "quantity": quantity,
                }
                logger.info(f"Trade logged: {coin} | Strategy: {strategy_name} | ID: {trade_id}")
                return trade_id

        except Exception as e:
            logger.error(f"Failed to log trade entry: {e}")
            return None

    def on_trade_exit(self, trade_id: int, exit_price: float, exit_reason: str = ""):
        """
        Call this when exiting a trade

        Args:
            trade_id: Trade ID from entry
            exit_price: Exit price
            exit_reason: Reason for exit
        """
        try:
            success = log_trade_exit(trade_id, exit_price, exit_reason)

            if success and trade_id in self.active_trades:
                trade_info = self.active_trades[trade_id]
                profit = (exit_price - trade_info["entry_price"]) * trade_info["quantity"]
                logger.info(f"Trade exit logged: {trade_info['coin']} | Profit: {profit:.2f}")
                del self.active_trades[trade_id]

            return success

        except Exception as e:
            logger.error(f"Failed to log trade exit: {e}")
            return False

    def get_strategy_performance(self, strategy_name: str):
        """Get performance stats for a strategy"""
        try:
            return get_strategy_stats(strategy_name)
        except Exception as e:
            logger.error(f"Failed to get strategy performance: {e}")
            return None


# Global hook instance
trading_hook = TradingHook()


# Example: How to integrate with existing trading bot
class ExampleTradingBot:
    """
    Example showing how to integrate with existing trading bot.
    """

    def __init__(self):
        self.hook = trading_hook

    def buy(
        self,
        coin: str,
        price: float,
        strategy_name: str,
        quantity: float = 1.0,
    ):
        """
        Example buy method with logging
        """
        # Your existing buy logic here
        print(f"Buying {quantity} {coin} at ${price}")

        # Log the trade entry
        trade_id = self.hook.on_trade_entry(
            coin=coin,
            strategy_name=strategy_name,
            entry_price=price,
            quantity=quantity,
            entry_reason="Buy signal triggered",
        )

        return trade_id

    def sell(self, trade_id: int, price: float):
        """
        Example sell method with logging
        """
        # Your existing sell logic here
        print(f"Selling trade {trade_id} at ${price}")

        # Log the trade exit
        success = self.hook.on_trade_exit(
            trade_id=trade_id,
            exit_price=price,
            exit_reason="Sell signal triggered",
        )

        return success


# Example: Simple function-based integration
def simple_trade_logger(
    coin: str,
    strategy: str,
    entry_price: float,
    exit_price: float = None,
    action: str = "entry",
):
    """
    Simple function for logging trades.

    Args:
        coin: Trading pair
        strategy: Strategy name
        entry_price: Entry price
        exit_price: Exit price (for exit action)
        action: "entry" or "exit"
    """
    if action == "entry":
        return log_trade_entry(coin, strategy, entry_price)
    elif action == "exit" and exit_price:
        # For simplicity, assume trade_id is 1
        return log_trade_exit(1, exit_price)


# Example: Decorator for automatic logging
def log_trades(func):
    """
    Decorator to automatically log trades.
    """

    def wrapper(*args, **kwargs):
        # Extract trade info from function call
        # This is a simplified example
        result = func(*args, **kwargs)

        # Log the trade (you'd need to extract actual trade data)
        if hasattr(result, "trade_info"):
            log_trade_entry(
                coin=result.trade_info.get("coin"),
                strategy_name=result.trade_info.get("strategy"),
                entry_price=result.trade_info.get("price"),
            )

        return result

    return wrapper


# Example usage
if __name__ == "__main__":
    # Example 1: Using the hook class
    bot = ExampleTradingBot()

    # Buy
    trade_id = bot.buy("BTCUSDT", 45000.0, "Breakout_EMA", 0.1)

    # Sell later
    if trade_id:
        bot.sell(trade_id, 46000.0)

    # Example 2: Using simple function
    simple_trade_logger("ETHUSDT", "RSI_Dip", 3000.0, action="entry")
    simple_trade_logger("ETHUSDT", "RSI_Dip", 3000.0, 3100.0, action="exit")

    print("Integration examples completed!")


