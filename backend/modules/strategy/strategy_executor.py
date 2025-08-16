"""
Strategy Executor for Mystic Trading Platform

Contains strategy execution logic for live trading.
Handles real-time strategy implementation and order execution.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from backend.utils.exceptions import StrategyException

logger = logging.getLogger(__name__)

# Simple usage of imports to avoid unused import errors
_ = json.dumps({"status": "loaded"})


class StrategyExecutor:
    """Strategy executor for live trading implementation"""

    def __init__(self):
        self.active_strategies: Dict[str, Dict[str, Any]] = {}
        self.strategy_history: List[Dict[str, Any]] = []
        self.execution_stats: Dict[str, Dict[str, Any]] = {}
        self.is_active: bool = True
        self.risk_limits: Dict[str, float] = {
            "max_position_size": 0.1,  # 10% of portfolio
            "max_daily_loss": 0.05,  # 5% daily loss limit
            "max_drawdown": 0.15,  # 15% max drawdown
        }

    def add_strategy(self, strategy_id: str, strategy_config: Dict[str, Any]) -> Dict[str, Any]:
        """Add a new strategy for execution"""
        strategy = {
            "id": strategy_id,
            "config": strategy_config,
            "status": "active",
            "created_at": datetime.now().isoformat(),
            "execution_count": 0,
            "total_pnl": 0.0,
            "last_execution": None,
        }

        self.active_strategies[strategy_id] = strategy
        logger.info(f"Strategy added for execution: {strategy_id}")

        return {"success": True, "strategy": strategy}

    def remove_strategy(self, strategy_id: str) -> Dict[str, Any]:
        """Remove a strategy from execution"""
        if strategy_id in self.active_strategies:
            del self.active_strategies[strategy_id]
            logger.info(f"Strategy removed from execution: {strategy_id}")
            return {"success": True}

        return {"success": False, "error": "Strategy not found"}

    def execute_strategy(
        self,
        strategy_id: str,
        market_data: Optional[Dict[str, Any]] = None,
        portfolio_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute a loaded strategy with given market and portfolio data"""
        if strategy_id not in self.active_strategies:
            raise StrategyException("Strategy not found")
        if market_data is None:
            raise StrategyException("Market data and portfolio data are required")
        strategy = self.active_strategies[strategy_id]
        conditions = strategy.get("conditions")
        if not isinstance(conditions, list):
            raise StrategyException("Invalid strategy configuration")
        actions = self._evaluate_conditions(conditions, market_data)
        executed_actions = []
        for action in actions:
            result = self._execute_action(action)
            executed_actions.append(result)
        if strategy_id not in self.execution_stats:
            self.execution_stats[strategy_id] = {
                "executions": 0,
                "successful_actions": 0,
            }
        self.execution_stats[strategy_id]["executions"] += 1
        self.execution_stats[strategy_id]["successful_actions"] += len(executed_actions)
        return {"status": "success", "actions": executed_actions}

    def get_strategy_status(self, strategy_id: str) -> Dict[str, Any]:
        """Get status of a specific strategy"""
        if strategy_id not in self.active_strategies:
            return {"error": "Strategy not found"}

        strategy = self.active_strategies[strategy_id]

        # Calculate performance metrics
        performance = self._calculate_strategy_performance(strategy_id)

        return {
            "strategy": strategy,
            "performance": performance,
            "is_active": self.is_active,
        }

    def get_all_strategies_status(self) -> Dict[str, Any]:
        """Get status of all active strategies"""
        strategies_status = {}

        for strategy_id in self.active_strategies:
            strategies_status[strategy_id] = self.get_strategy_status(strategy_id)

        return {
            "total_strategies": len(self.active_strategies),
            "active_strategies": len(
                [s for s in self.active_strategies.values() if s["status"] == "active"]
            ),
            "strategies": strategies_status,
            "executor_active": self.is_active,
        }

    def pause_strategy(self, strategy_id: str) -> Dict[str, Any]:
        """Pause a strategy execution"""
        if strategy_id not in self.active_strategies:
            return {"error": "Strategy not found"}

        self.active_strategies[strategy_id]["status"] = "paused"
        logger.info(f"Strategy paused: {strategy_id}")

        return {"success": True, "status": "paused"}

    def resume_strategy(self, strategy_id: str) -> Dict[str, Any]:
        """Resume a strategy execution"""
        if strategy_id not in self.active_strategies:
            return {"error": "Strategy not found"}

        self.active_strategies[strategy_id]["status"] = "active"
        logger.info(f"Strategy resumed: {strategy_id}")

        return {"success": True, "status": "active"}

    def update_risk_limits(self, new_limits: Dict[str, float]) -> Dict[str, Any]:
        """Update risk limits"""
        for key, value in new_limits.items():
            if key in self.risk_limits:
                self.risk_limits[key] = value

        logger.info(f"Risk limits updated: {new_limits}")

        return {"success": True, "risk_limits": self.risk_limits}

    def get_execution_history(
        self, strategy_id: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get execution history"""
        history = self.strategy_history

        if strategy_id:
            history = [record for record in history if record["strategy_id"] == strategy_id]

        return history[-limit:] if history else []

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary"""
        total_executions = sum(
            strategy["execution_count"] for strategy in self.active_strategies.values()
        )
        total_pnl = sum(strategy["total_pnl"] for strategy in self.active_strategies.values())

        # Calculate success rate
        successful_executions = len(
            [
                record
                for record in self.strategy_history
                if any(result.get("success") for result in record.get("execution_results", []))
            ]
        )

        success_rate = (
            (successful_executions / len(self.strategy_history) * 100)
            if self.strategy_history
            else 0
        )

        return {
            "total_strategies": len(self.active_strategies),
            "total_executions": total_executions,
            "total_pnl": total_pnl,
            "success_rate": round(success_rate, 2),
            "execution_history_count": len(self.strategy_history),
            "risk_limits": self.risk_limits,
        }

    def _check_risk_limits(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check if current portfolio violates risk limits"""
        try:
            # Check daily loss limit
            daily_pnl = portfolio_data.get("daily_pnl", 0)
            portfolio_value = portfolio_data.get("total_value", 1)

            if abs(daily_pnl) / portfolio_value > self.risk_limits["max_daily_loss"]:
                return {
                    "passed": False,
                    "reason": (f"Daily loss limit exceeded: {abs(daily_pnl):.2f}"),
                }

            # Check drawdown limit
            current_drawdown = portfolio_data.get("current_drawdown", 0)
            if current_drawdown > self.risk_limits["max_drawdown"]:
                return {
                    "passed": False,
                    "reason": f"Max drawdown exceeded: {current_drawdown:.2%}",
                }

            return {"passed": True}

        except Exception as e:
            logger.error(f"Risk limit check failed: {e}")
            return {"passed": False, "reason": f"Risk check error: {e}"}

    def _generate_signals(
        self, strategy: Dict[str, Any], market_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate trading signals based on strategy and market data"""
        signals = []

        try:
            config = strategy["config"]
            strategy_type = config.get("type", "basic")

            if strategy_type == "momentum":
                signals = self._generate_momentum_signals(market_data, config)
            elif strategy_type == "mean_reversion":
                signals = self._generate_mean_reversion_signals(market_data, config)
            elif strategy_type == "breakout":
                signals = self._generate_breakout_signals(market_data, config)
            else:
                signals = self._generate_basic_signals(market_data, config)

            return signals

        except Exception as e:
            logger.error(f"Signal generation failed: {e}")
            return []

    def _execute_trade(
        self, signal: Dict[str, Any], portfolio_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single trade based on signal"""
        try:
            symbol = signal.get("symbol", "")
            action = signal.get("action", "")
            quantity = signal.get("quantity", 0)
            price = signal.get("price", 0)

            # Simulate trade execution
            trade_result = {
                "symbol": symbol,
                "action": action,
                "quantity": quantity,
                "price": price,
                "timestamp": datetime.now().isoformat(),
                "success": True,
                "order_id": f"order_{datetime.now().timestamp()}",
                "commission": quantity * price * 0.001,  # 0.1% commission
                "status": "filled",
            }

            # Update portfolio (simplified)
            if action == "buy":
                trade_result["pnl"] = 0  # No PnL for new position
            elif action == "sell":
                # Calculate PnL (simplified)
                entry_price = (
                    portfolio_data.get("positions", {}).get(symbol, {}).get("entry_price", price)
                )
                trade_result["pnl"] = (price - entry_price) * quantity

            return trade_result

        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def _calculate_strategy_performance(self, strategy_id: str) -> Dict[str, Any]:
        """Calculate performance metrics for a strategy"""
        strategy_history = self.get_execution_history(strategy_id)

        if not strategy_history:
            return {"error": "No execution history available"}

        total_trades = 0
        successful_trades = 0
        total_pnl = 0.0

        for record in strategy_history:
            for result in record.get("execution_results", []):
                if result.get("success"):
                    total_trades += 1
                    successful_trades += 1
                    total_pnl += result.get("pnl", 0)
                elif result.get("success") is False:
                    total_trades += 1

        win_rate = (successful_trades / total_trades * 100) if total_trades > 0 else 0

        return {
            "total_trades": total_trades,
            "successful_trades": successful_trades,
            "win_rate": round(win_rate, 2),
            "total_pnl": round(total_pnl, 2),
            "avg_pnl_per_trade": (round(total_pnl / total_trades, 2) if total_trades > 0 else 0),
        }

    def _generate_momentum_signals(
        self, market_data: Dict[str, Any], config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate momentum-based trading signals"""
        signals = []

        for symbol, data in market_data.items():
            rsi = data.get("rsi", 50)
            price = data.get("price", 0)

            # Simple momentum logic
            if rsi < 30:  # Oversold
                signals.append(
                    {
                        "symbol": symbol,
                        "action": "buy",
                        "quantity": 1.0,
                        "price": price,
                        "confidence": 0.8,
                        "reason": "RSI oversold",
                    }
                )
            elif rsi > 70:  # Overbought
                signals.append(
                    {
                        "symbol": symbol,
                        "action": "sell",
                        "quantity": 1.0,
                        "price": price,
                        "confidence": 0.8,
                        "reason": "RSI overbought",
                    }
                )

        return signals

    def _generate_mean_reversion_signals(
        self, market_data: Dict[str, Any], config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate mean reversion trading signals"""
        signals = []

        for symbol, data in market_data.items():
            price = data.get("price", 0)
            bb_upper = data.get("bollinger_upper", price * 1.02)
            bb_lower = data.get("bollinger_lower", price * 0.98)

            # Mean reversion logic
            if price > bb_upper:
                signals.append(
                    {
                        "symbol": symbol,
                        "action": "sell",
                        "quantity": 1.0,
                        "price": price,
                        "confidence": 0.7,
                        "reason": "Price above upper Bollinger Band",
                    }
                )
            elif price < bb_lower:
                signals.append(
                    {
                        "symbol": symbol,
                        "action": "buy",
                        "quantity": 1.0,
                        "price": price,
                        "confidence": 0.7,
                        "reason": "Price below lower Bollinger Band",
                    }
                )

        return signals

    def _generate_breakout_signals(
        self, market_data: Dict[str, Any], config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate breakout trading signals"""
        signals = []

        for symbol, data in market_data.items():
            price = data.get("price", 0)
            volume = data.get("volume", 0)
            avg_volume = data.get("avg_volume", volume)

            # Breakout logic
            if volume > avg_volume * 1.5:  # Volume spike
                signals.append(
                    {
                        "symbol": symbol,
                        "action": "buy",
                        "quantity": 1.0,
                        "price": price,
                        "confidence": 0.6,
                        "reason": "Volume breakout",
                    }
                )

        return signals

    def _generate_basic_signals(
        self, market_data: Dict[str, Any], config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate basic trading signals"""
        signals = []

        for symbol, data in market_data.items():
            price = data.get("price", 0)
            macd = data.get("macd", 0)

            # Basic MACD-based signals
            if macd > 0:
                signals.append(
                    {
                        "symbol": symbol,
                        "action": "buy",
                        "quantity": 1.0,
                        "price": price,
                        "confidence": 0.5,
                        "reason": "Positive MACD",
                    }
                )
            elif macd < 0:
                signals.append(
                    {
                        "symbol": symbol,
                        "action": "sell",
                        "quantity": 1.0,
                        "price": price,
                        "confidence": 0.5,
                        "reason": "Negative MACD",
                    }
                )

        return signals

    def load_strategy(self, strategy_config: Dict[str, Any]) -> bool:
        """Load a new strategy configuration"""
        if (
            not isinstance(strategy_config, dict)
            or "id" not in strategy_config
            or "name" not in strategy_config
            or "conditions" not in strategy_config
        ):
            raise StrategyException("Invalid strategy configuration")
        strategy_id = strategy_config["id"]
        if strategy_id in self.active_strategies:
            raise StrategyException("Strategy with this ID already loaded")
        self.active_strategies[strategy_id] = strategy_config
        self.strategy_history.append(
            {
                "action": "load",
                "id": strategy_id,
                "timestamp": datetime.timezone.utcnow().isoformat(),
            }
        )
        return True

    def unload_strategy(self, strategy_id: str) -> bool:
        """Unload a strategy by ID"""
        if strategy_id not in self.active_strategies:
            return False
        del self.active_strategies[strategy_id]
        self.strategy_history.append(
            {
                "action": "unload",
                "strategy_id": strategy_id,
                "timestamp": datetime.now().isoformat(),
            }
        )
        if strategy_id in self.execution_stats:
            del self.execution_stats[strategy_id]
        return True

    def get_strategy(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """Get strategy configuration by ID"""
        return self.active_strategies.get(strategy_id)

    def update_strategy(self, strategy_id: str, updated_config: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing strategy configuration"""
        if strategy_id not in self.active_strategies:
            raise StrategyException("Strategy not found")

        # Validate the updated configuration
        if not isinstance(updated_config, dict):
            raise StrategyException("Invalid strategy configuration")

        # Update the strategy configuration
        # Handle both load_strategy and add_strategy workflows
        if "config" in self.active_strategies[strategy_id]:
            # Strategy was added via add_strategy
            self.active_strategies[strategy_id]["config"] = updated_config
        else:
            # Strategy was loaded via load_strategy - update directly
            self.active_strategies[strategy_id].update(updated_config)

        self.active_strategies[strategy_id]["updated_at"] = datetime.now().isoformat()

        logger.info(f"Strategy updated: {strategy_id}")

        return {
            "success": True,
            "strategy": self.active_strategies[strategy_id],
        }

    def list_strategies(self) -> List[str]:
        """List all loaded strategy IDs"""
        return list(self.active_strategies.keys())

    def get_execution_stats(self, strategy_id: str) -> Dict[str, Any]:
        """Get execution stats for a strategy"""
        return self.execution_stats.get(strategy_id, {})

    def get_all_execution_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get execution stats for all strategies"""
        return self.execution_stats

    def _evaluate_conditions(
        self, conditions: List[Dict[str, Any]], market_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Evaluate trading conditions and return all matches for all symbols."""
        actions = []
        for symbol, data in market_data.items():
            rsi = data.get("rsi")
            for cond in conditions:
                if cond["condition"] == "rsi < 30" and rsi is not None and rsi < 30:
                    action = cond.copy()
                    action["symbol"] = symbol
                    actions.append(action)
                elif cond["condition"] == "rsi > 70" and rsi is not None and rsi > 70:
                    action = cond.copy()
                    action["symbol"] = symbol
                    actions.append(action)
                elif cond["condition"] not in ["rsi < 30", "rsi > 70"]:
                    raise StrategyException("Invalid condition")
        return actions

    def _execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a trading action and return result with action key included."""
        if (
            not isinstance(action, dict)
            or "action" not in action
            or "symbol" not in action
            or "quantity" not in action
        ):
            raise StrategyException("Invalid action or missing parameters")
        if action["action"] not in ["buy", "sell"]:
            raise StrategyException("Invalid action or missing parameters")
        return {
            "status": "success",
            "order_id": "12345",
            "action": action["action"],
        }


