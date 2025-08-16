# trade_memory_integration.py
"""
Integration module for trade logging and strategy memory engine.
This module hooks into the existing trading execution system to automatically
log trades and track strategy performance.
"""

import logging
import datetime
from typing import Dict, Any, Optional, List
from db_logger import (
    log_trade,
    register_strategy,
    get_strategy_id,
    update_trade_exit,
)
from reward_engine import evaluate_strategies
from mutator import run_evolution_cycle
from alerts import (
    alert_trade_execution,
    alert_strategy_mutation,
    alert_evolution_cycle,
)

logger = logging.getLogger(__name__)


class TradeMemoryIntegration:
    """
    Integration class that hooks into the trading execution engine
    to automatically log trades and manage strategy evolution.
    """

    def __init__(self):
        self.active_trades = {}  # Track open trades
        self.strategy_cache = {}  # Cache strategy IDs
        self.evaluation_interval = 100  # Evaluate strategies every N trades
        self.evolution_interval = 500  # Run evolution every N trades
        self.trade_counter = 0

        # Initialize default strategies
        self._initialize_default_strategies()

    def _initialize_default_strategies(self):
        """Initialize default strategies if they don't exist"""
        default_strategies = [
            ("Breakout_EMA", "EMA crossover with breakout detection"),
            ("RSI_Dip", "RSI oversold with volume confirmation"),
            ("MACD_Signal", "MACD signal line crossover strategy"),
            ("Bollinger_Bands", "Bollinger Bands mean reversion"),
            ("Volume_Spike", "Volume spike breakout strategy"),
            ("Trend_Following", "Trend following with momentum"),
            ("Mean_Reversion", "Mean reversion with support/resistance"),
            ("Volatility_Breakout", "Volatility breakout strategy"),
        ]

        for name, description in default_strategies:
            strategy_id = register_strategy(name, description)
            if strategy_id:
                self.strategy_cache[name] = strategy_id
                logger.info(f"Initialized strategy: {name} (ID: {strategy_id})")

    def log_trade_entry(
        self,
        coin: str,
        strategy_name: str,
        entry_price: float,
        quantity: float = 1.0,
        entry_reason: str = "",
        trade_type: str = "spot",
        risk_level: str = "medium",
        tags: str = "",
    ) -> Optional[int]:
        """
        Log a trade entry

        Args:
            coin: Trading pair (e.g., 'BTCUSDT')
            strategy_name: Name of the strategy used
            entry_price: Entry price
            quantity: Trade quantity
            entry_reason: Reason for entry
            trade_type: Type of trade
            risk_level: Risk level
            tags: Comma-separated tags

        Returns:
            int: Trade ID if successful, None otherwise
        """
        try:
            # Get or create strategy ID
            if strategy_name not in self.strategy_cache:
                strategy_id = get_strategy_id(strategy_name)
                if not strategy_id:
                    strategy_id = register_strategy(
                        strategy_name,
                        f"Auto-registered strategy: {strategy_name}",
                    )
                if strategy_id:
                    self.strategy_cache[strategy_name] = strategy_id

            strategy_id = self.strategy_cache.get(strategy_name)
            if not strategy_id:
                logger.error(f"Could not get strategy ID for: {strategy_name}")
                return None

            # Log the trade entry
            success = log_trade(
                coin=coin,
                strategy_id=strategy_id,
                entry_price=entry_price,
                exit_price=None,  # Will be updated when trade exits
                quantity=quantity,
                duration_minutes=None,
                entry_reason=entry_reason,
                exit_reason="",
                trade_type=trade_type,
                risk_level=risk_level,
                tags=tags,
            )

            if success:
                # Get the trade ID (this would need to be returned from log_trade)
                # For now, we'll use a simple counter
                trade_id = len(self.active_trades) + 1
                self.active_trades[trade_id] = {
                    "coin": coin,
                    "strategy_name": strategy_name,
                    "entry_price": entry_price,
                    "quantity": quantity,
                    "entry_time": datetime.datetime.timezone.utcnow(),
                }

                self.trade_counter += 1
                logger.info(
                    f"Logged trade entry: {coin} | Strategy: {strategy_name} | Entry: {entry_price}"
                )

                # Check if we should run evaluation/evolution
                self._check_evaluation_triggers()

                return trade_id

            return None

        except Exception as e:
            logger.error(f"Failed to log trade entry: {e}")
            return None

    def log_trade_exit(self, trade_id: int, exit_price: float, exit_reason: str = "") -> bool:
        """
        Log a trade exit

        Args:
            trade_id: Trade ID from entry
            exit_price: Exit price
            exit_reason: Reason for exit

        Returns:
            bool: True if successful
        """
        try:
            if trade_id not in self.active_trades:
                logger.error(f"Trade ID {trade_id} not found in active trades")
                return False

            trade_info = self.active_trades[trade_id]

            # Calculate duration
            (
                datetime.datetime.timezone.utcnow() - trade_info["entry_time"]
            ).total_seconds() / 60

            # Update the trade with exit information
            success = update_trade_exit(trade_id, exit_price, exit_reason)

            if success:
                # Calculate profit for alert
                profit = (exit_price - trade_info["entry_price"]) * trade_info["quantity"]
                success_bool = profit > 0

                # Send trade execution alert
                trade_alert_info = {
                    "coin": trade_info["coin"],
                    "strategy_name": trade_info["strategy_name"],
                    "entry_price": trade_info["entry_price"],
                    "exit_price": exit_price,
                    "profit": profit,
                    "success": success_bool,
                }
                alert_trade_execution(trade_alert_info)

                # Remove from active trades
                del self.active_trades[trade_id]

                logger.info(
                    f"Logged trade exit: {trade_info['coin']} | Profit: {profit:.2f} | Success: {success_bool}"
                )
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to log trade exit: {e}")
            return False

    def _check_evaluation_triggers(self):
        """Check if we should run evaluation or evolution cycles"""
        try:
            # Run strategy evaluation every N trades
            if self.trade_counter % self.evaluation_interval == 0:
                logger.info(f"Running strategy evaluation (trade #{self.trade_counter})")
                evaluation_results = evaluate_strategies(min_trades=3, days=7)
                logger.info(
                    f"Evaluation completed: {evaluation_results.get('updated_strategies', 0)} strategies updated"
                )

            # Run evolution cycle every N trades
            if self.trade_counter % self.evolution_interval == 0:
                logger.info(f"Running evolution cycle (trade #{self.trade_counter})")
                evolution_results = run_evolution_cycle()

                # Send evolution alert
                alert_evolution_cycle(evolution_results)

                # Send alerts for new mutations
                for detail in evolution_results.get("details", []):
                    if detail["type"] == "mutation":
                        alert_strategy_mutation(detail["info"])

                logger.info(
                    f"Evolution completed: {evolution_results.get('total_new_strategies', 0)} new strategies created"
                )

        except Exception as e:
            logger.error(f"Error in evaluation triggers: {e}")

    def get_strategy_performance(self, strategy_name: str, days: int = 30) -> Dict[str, Any]:
        """
        Get performance statistics for a strategy

        Args:
            strategy_name: Name of the strategy
            days: Number of days to look back

        Returns:
            Dict with performance statistics
        """
        try:
            strategy_id = self.strategy_cache.get(strategy_name)
            if not strategy_id:
                strategy_id = get_strategy_id(strategy_name)
                if strategy_id:
                    self.strategy_cache[strategy_name] = strategy_id

            if not strategy_id:
                return {
                    "error": f"Strategy {strategy_name} not found",
                    "total_trades": 0,
                    "win_rate": 0.0,
                    "avg_profit": 0.0,
                    "total_profit": 0.0,
                }

            from db_logger import get_strategy_stats

            stats = get_strategy_stats(strategy_id, days=days)
            stats["strategy_name"] = strategy_name
            return stats

        except Exception as e:
            logger.error(f"Error getting strategy performance: {e}")
            return {"error": str(e)}

    def get_active_trades(self) -> List[Dict[str, Any]]:
        """Get list of currently active trades"""
        return [
            {
                "trade_id": trade_id,
                "coin": info["coin"],
                "strategy_name": info["strategy_name"],
                "entry_price": info["entry_price"],
                "quantity": info["quantity"],
                "entry_time": info["entry_time"].isoformat(),
                "duration_minutes": (
                    (datetime.datetime.timezone.utcnow() - info["entry_time"]).total_seconds() / 60
                ),
            }
            for trade_id, info in self.active_trades.items()
        ]

    def force_evaluation(self) -> Dict[str, Any]:
        """Force run strategy evaluation"""
        try:
            logger.info("Forcing strategy evaluation...")
            results = evaluate_strategies(min_trades=1, days=1)
            return results
        except Exception as e:
            logger.error(f"Error in forced evaluation: {e}")
            return {"error": str(e)}

    def force_evolution(self) -> Dict[str, Any]:
        """Force run evolution cycle"""
        try:
            logger.info("Forcing evolution cycle...")
            results = run_evolution_cycle()
            alert_evolution_cycle(results)
            return results
        except Exception as e:
            logger.error(f"Error in forced evolution: {e}")
            return {"error": str(e)}


# Global instance
trade_memory = TradeMemoryIntegration()


# Convenience functions for easy integration
def log_trade_entry(coin: str, strategy_name: str, entry_price: float, **kwargs) -> Optional[int]:
    """Convenience function to log trade entry"""
    return trade_memory.log_trade_entry(coin, strategy_name, entry_price, **kwargs)


def log_trade_exit(trade_id: int, exit_price: float, exit_reason: str = "") -> bool:
    """Convenience function to log trade exit"""
    return trade_memory.log_trade_exit(trade_id, exit_price, exit_reason)


def get_strategy_performance(strategy_name: str, days: int = 30) -> Dict[str, Any]:
    """Convenience function to get strategy performance"""
    return trade_memory.get_strategy_performance(strategy_name, days)


def get_active_trades() -> List[Dict[str, Any]]:
    """Convenience function to get active trades"""
    return trade_memory.get_active_trades()


def force_evaluation() -> Dict[str, Any]:
    """Convenience function to force evaluation"""
    return trade_memory.force_evaluation()


def force_evolution() -> Dict[str, Any]:
    """Convenience function to force evolution"""
    return trade_memory.force_evolution()


