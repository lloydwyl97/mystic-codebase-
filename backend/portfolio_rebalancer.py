"""
Portfolio Rebalancer Module

Handles portfolio rebalancing operations and health monitoring.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, cast

from .services.websocket_manager import websocket_manager

logger = logging.getLogger(__name__)


def rebalance_portfolio(
    current_holdings: Dict[str, Any],
    stablecoin: str = "USDT",
    threshold: float = 0.05,
) -> Dict[str, Any]:
    """
    Rebalance portfolio based on current holdings and target allocation.

    Args:
        current_holdings: Dictionary of current coin holdings
        stablecoin: Stablecoin symbol to use for rebalancing
        threshold: Rebalancing threshold percentage

    Returns:
        Dictionary with rebalancing results and health metrics
    """
    try:
        # Calculate current allocation
        total_value = sum(holding.get("value", 0) for holding in current_holdings.values())
        if total_value == 0:
            return {
                "success": False,
                "error": "No portfolio value found",
                "health_score": 0,
                "risk_level": "unknown",
                "timestamp": datetime.now(timezone.timezone.utc).isoformat(),
            }

        # Calculate health metrics
        health_score = 0
        for data in current_holdings.values():
            change_pct = data.get("change_pct", 0)
            if change_pct < -10:
                health_score -= 2
            elif change_pct < -5:
                health_score -= 1
            elif change_pct > 10:
                health_score += 1

        # Determine risk level
        if health_score < -5:
            risk_level = "high"
        elif health_score < 0:
            risk_level = "medium"
        else:
            risk_level = "low"

        return {
            "success": True,
            "total_value": total_value,
            "holdings_count": len(current_holdings),
            "health_score": health_score,
            "risk_level": risk_level,
            "timestamp": datetime.now(timezone.timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.error(f"Portfolio rebalancing failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "health_score": 0,
            "risk_level": "unknown",
            "timestamp": datetime.now(timezone.timezone.utc).isoformat(),
        }


class PortfolioRebalancer:
    """Advanced portfolio rebalancing with real-time updates."""

    def __init__(self):
        self.current_positions: Dict[str, Dict[str, Any]] = {}
        self.target_allocation: Dict[str, float] = {}
        self.rebalancing_history: List[Dict[str, Any]] = []

    async def get_current_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get current portfolio positions."""
        return self.current_positions.copy()

    def _calculate_current_allocation(
        self, positions: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate current allocation percentages."""
        total_value = sum(pos.get("value", 0) for pos in positions.values())
        if total_value == 0:
            return cast(Dict[str, float], {})

        allocation = {}
        for symbol, position in positions.items():
            value = position.get("value", 0)
            allocation[symbol] = (value / total_value) * 100

        return allocation

    def _calculate_rebalancing_trades(
        self, current: Dict[str, float], target: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Calculate required trades to reach target allocation."""
        trades: List[Dict[str, Any]] = []
        threshold = 0.05  # 5% threshold

        for symbol in set(current.keys()) | set(target.keys()):
            current_pct = current.get(symbol, 0)
            target_pct = target.get(symbol, 0)
            difference = target_pct - current_pct

            if abs(difference) > threshold:
                trades.append(
                    {
                        "symbol": symbol,
                        "action": "buy" if difference > 0 else "sell",
                        "amount": abs(difference),
                        "current_pct": current_pct,
                        "target_pct": target_pct,
                    }
                )
            else:
                trades.append(
                    {
                        "symbol": symbol,
                        "action": "hold",
                        "amount": 0,
                        "current_pct": current_pct,
                        "target_pct": target_pct,
                    }
                )

        return trades

    async def _execute_trade(self, trade: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single trade."""
        try:
            # Simulate trade execution
            trade_id = (
                f"rebalance_{trade['symbol']}_{datetime.now(timezone.timezone.utc).timestamp()}"
            )

            result = {
                "trade_id": trade_id,
                "symbol": trade["symbol"],
                "action": trade["action"],
                "amount": trade["amount"],
                "status": "executed",
                "timestamp": datetime.now(timezone.timezone.utc).isoformat(),
            }

            # Broadcast trade execution
            await websocket_manager.broadcast_json({"type": "rebalance_trade", "data": result})

            return result

        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            return {
                "trade_id": None,
                "symbol": trade["symbol"],
                "action": trade["action"],
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now(timezone.timezone.utc).isoformat(),
            }

    async def rebalance_portfolio(self, target_allocation: Dict[str, float]) -> Dict[str, Any]:
        """Rebalance portfolio to target allocation."""
        try:
            current_positions = await self.get_current_positions()
            current_allocation = self._calculate_current_allocation(current_positions)

            # Calculate required trades
            trades = self._calculate_rebalancing_trades(current_allocation, target_allocation)

            # Execute trades
            executed_trades: List[Dict[str, Any]] = []
            for trade in trades:
                if trade["action"] != "hold":
                    result = await self._execute_trade(trade)
                    if result["status"] == "executed":
                        executed_trades.append(result)

            # Update portfolio
            new_positions = await self.get_current_positions()
            new_allocation = self._calculate_current_allocation(new_positions)

            result: Dict[str, Any] = {
                "success": True,
                "trades_executed": len(executed_trades),
                "current_allocation": new_allocation,
                "target_allocation": target_allocation,
                "timestamp": datetime.now(timezone.timezone.utc).isoformat(),
            }

            # Broadcast portfolio rebalancing
            await websocket_manager.broadcast_json(
                {
                    "type": "portfolio_rebalanced",
                    "data": {
                        "trades_executed": len(executed_trades),
                        "current_allocation": new_allocation,
                        "target_allocation": target_allocation,
                        "timestamp": result["timestamp"],
                    },
                }
            )

            return result

        except Exception as e:
            logger.error(f"Portfolio rebalancing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now(timezone.timezone.utc).isoformat(),
            }


