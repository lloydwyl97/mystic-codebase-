# capital_allocator.py
"""
Smart Capital Allocator
Manages capital distribution across multiple trading strategies based on performance.
"""

from typing import Dict, List, Any
from datetime import datetime
from strategy_leaderboard import get_strategy_leaderboard
from position_sizer import PositionSizer
from datetime import datetime, timezone


class CapitalAllocator:
    """
    Advanced capital allocation engine with multiple allocation strategies.
    """

    def __init__(self, total_capital: float = 10000.0, max_strategies: int = 10):
        """
        Initialize capital allocator.

        Args:
            total_capital: Total capital to allocate
            max_strategies: Maximum number of strategies to fund
        """
        self.total_capital = total_capital
        self.max_strategies = max_strategies
        self.allocation_history = []
        self.current_allocations = {}
        self.sizer = PositionSizer()

    def allocate_by_performance(
        self, hours_back: int = 24, min_win_rate: float = 0.55
    ) -> Dict[str, float]:
        """
        Allocate capital based on strategy performance.

        Args:
            hours_back: Hours to look back for performance data
            min_win_rate: Minimum win rate to consider

        Returns:
            Capital allocation per strategy
        """
        try:
            leaderboard = get_strategy_leaderboard(hours_back)

            # Filter winning strategies
            winners = [
                s for s in leaderboard if s["win_rate"] > min_win_rate and s["total_profit"] > 0
            ]

            if not winners:
                print("âš ï¸ No winning strategies found for allocation")
                return {}

            # Sort by total profit
            winners.sort(key=lambda x: x["total_profit"], reverse=True)

            # Limit to max strategies
            winners = winners[: self.max_strategies]

            # Calculate total performance score
            total_score = sum(s["total_profit"] for s in winners)

            # Allocate capital proportionally
            allocations = {}
            for strategy in winners:
                weight = strategy["total_profit"] / total_score
                allocated_capital = self.total_capital * weight
                allocations[strategy["strategy"]] = round(allocated_capital, 2)

            # Log allocation
            self._log_allocation("performance_based", allocations, winners)

            return allocations

        except Exception as e:
            print(f"âŒ Error in performance-based allocation: {e}")
            return {}

    def allocate_by_risk_parity(self, target_volatility: float = 0.15) -> Dict[str, float]:
        """
        Allocate capital using risk parity approach.

        Args:
            target_volatility: Target portfolio volatility

        Returns:
            Capital allocation per strategy
        """
        try:
            leaderboard = get_strategy_leaderboard(hours_back=24)

            if not leaderboard:
                return {}

            # Estimate volatility for each strategy (simplified)
            allocations = {}
            total_risk_contribution = 0

            for strategy in leaderboard[: self.max_strategies]:
                # Estimate volatility based on win rate and profit consistency
                win_rate = strategy["win_rate"]
                profit_consistency = min(strategy["total_profit"] / max(strategy["trades"], 1), 100)

                # Simplified volatility estimation
                volatility = (1 - win_rate) * 0.1 + (profit_consistency / 1000)
                volatility = max(volatility, 0.01)  # Minimum volatility

                # Risk contribution should be equal
                risk_contribution = target_volatility / volatility
                total_risk_contribution += risk_contribution

                allocations[strategy["strategy"]] = {
                    "volatility": volatility,
                    "risk_contribution": risk_contribution,
                }

            # Calculate actual capital allocation
            final_allocations = {}
            for strategy, data in allocations.items():
                weight = data["risk_contribution"] / total_risk_contribution
                allocated_capital = self.total_capital * weight
                final_allocations[strategy] = round(allocated_capital, 2)

            self._log_allocation(
                "risk_parity",
                final_allocations,
                leaderboard[: self.max_strategies],
            )

            return final_allocations

        except Exception as e:
            print(f"âŒ Error in risk parity allocation: {e}")
            return {}

    def allocate_by_equal_weight(self) -> Dict[str, float]:
        """
        Allocate capital equally across top strategies.

        Returns:
            Capital allocation per strategy
        """
        try:
            leaderboard = get_strategy_leaderboard(hours_back=24)

            if not leaderboard:
                return {}

            # Take top strategies
            top_strategies = leaderboard[: self.max_strategies]

            # Equal allocation
            allocation_per_strategy = self.total_capital / len(top_strategies)

            allocations = {}
            for strategy in top_strategies:
                allocations[strategy["strategy"]] = round(allocation_per_strategy, 2)

            self._log_allocation("equal_weight", allocations, top_strategies)

            return allocations

        except Exception as e:
            print(f"âŒ Error in equal weight allocation: {e}")
            return {}

    def allocate_by_kelly_criterion(self) -> Dict[str, float]:
        """
        Allocate capital using Kelly Criterion for each strategy.

        Returns:
            Capital allocation per strategy
        """
        try:
            leaderboard = get_strategy_leaderboard(hours_back=24)

            if not leaderboard:
                return {}

            allocations = {}
            total_kelly = 0

            for strategy in leaderboard[: self.max_strategies]:
                win_rate = strategy["win_rate"]
                avg_profit = strategy["avg_profit"]

                # Estimate average win/loss
                if avg_profit > 0:
                    avg_win = avg_profit
                    avg_loss = abs(avg_profit * 0.5)  # Assume losses are half of wins
                else:
                    avg_win = 10.0  # Default values
                    avg_loss = 10.0

                # Calculate Kelly percentage
                kelly_pct = self.sizer.calculate_kelly_criterion(win_rate, avg_win, avg_loss)
                total_kelly += kelly_pct

                allocations[strategy["strategy"]] = {
                    "kelly_percentage": kelly_pct,
                    "win_rate": win_rate,
                    "avg_profit": avg_profit,
                }

            # Normalize Kelly allocations
            final_allocations = {}
            for strategy, data in allocations.items():
                if total_kelly > 0:
                    weight = data["kelly_percentage"] / total_kelly
                    allocated_capital = self.total_capital * weight
                else:
                    allocated_capital = 0

                final_allocations[strategy] = round(allocated_capital, 2)

            self._log_allocation(
                "kelly_criterion",
                final_allocations,
                leaderboard[: self.max_strategies],
            )

            return final_allocations

        except Exception as e:
            print(f"âŒ Error in Kelly criterion allocation: {e}")
            return {}

    def allocate_by_momentum(self, momentum_period: int = 7) -> Dict[str, float]:
        """
        Allocate capital based on recent momentum (performance trend).

        Args:
            momentum_period: Days to calculate momentum

        Returns:
            Capital allocation per strategy
        """
        try:
            # Get performance over different periods
            recent_performance = get_strategy_leaderboard(hours_back=24)
            older_performance = get_strategy_leaderboard(hours_back=24 * momentum_period)

            if not recent_performance or not older_performance:
                return {}

            # Calculate momentum scores
            momentum_scores = {}

            for recent in recent_performance[: self.max_strategies]:
                strategy_name = recent["strategy"]

                # Find older performance for this strategy
                older = next(
                    (s for s in older_performance if s["strategy"] == strategy_name),
                    None,
                )

                if older:
                    # Calculate momentum (recent vs older performance)
                    recent_score = recent["total_profit"] * recent["win_rate"]
                    older_score = older["total_profit"] * older["win_rate"]

                    momentum = recent_score - older_score
                    momentum_scores[strategy_name] = max(momentum, 0)  # Only positive momentum
                else:
                    # New strategy, give it a chance
                    momentum_scores[strategy_name] = recent["total_profit"] * recent["win_rate"]

            # Allocate based on momentum
            total_momentum = sum(momentum_scores.values())

            allocations = {}
            for strategy, momentum in momentum_scores.items():
                if total_momentum > 0:
                    weight = momentum / total_momentum
                    allocated_capital = self.total_capital * weight
                else:
                    allocated_capital = 0

                allocations[strategy] = round(allocated_capital, 2)

            self._log_allocation(
                "momentum",
                allocations,
                recent_performance[: self.max_strategies],
            )

            return allocations

        except Exception as e:
            print(f"âŒ Error in momentum allocation: {e}")
            return {}

    def rebalance_portfolio(
        self,
        current_allocations: Dict[str, float],
        method: str = "performance",
    ) -> Dict[str, float]:
        """
        Rebalance existing portfolio allocations.

        Args:
            current_allocations: Current capital allocations
            method: Rebalancing method

        Returns:
            New capital allocations
        """
        print(f"ðŸ”„ Rebalancing portfolio using {method} method...")

        if method == "performance":
            new_allocations = self.allocate_by_performance()
        elif method == "risk_parity":
            new_allocations = self.allocate_by_risk_parity()
        elif method == "equal_weight":
            new_allocations = self.allocate_by_equal_weight()
        elif method == "kelly":
            new_allocations = self.allocate_by_kelly_criterion()
        elif method == "momentum":
            new_allocations = self.allocate_by_momentum()
        else:
            new_allocations = self.allocate_by_performance()

        # Calculate rebalancing trades
        rebalancing_trades = {}
        for strategy in set(current_allocations.keys()) | set(new_allocations.keys()):
            current = current_allocations.get(strategy, 0)
            new = new_allocations.get(strategy, 0)
            difference = new - current

            if abs(difference) > 10:  # Only rebalance if difference > $10
                rebalancing_trades[strategy] = {
                    "current": current,
                    "new": new,
                    "difference": round(difference, 2),
                    "action": "buy" if difference > 0 else "sell",
                }

        # Log rebalancing
        self._log_rebalancing(current_allocations, new_allocations, rebalancing_trades)

        return new_allocations

    def _log_allocation(
        self,
        method: str,
        allocations: Dict[str, float],
        strategies: List[Dict[str, Any]],
    ):
        """Log allocation decision."""
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "method": method,
            "total_capital": self.total_capital,
            "allocations": allocations,
            "strategies_count": len(allocations),
            "strategies": [s["strategy"] for s in strategies[:5]],  # Top 5 strategies
        }

        self.allocation_history.append(log_entry)
        self.current_allocations = allocations

        print(f"ðŸ’° Capital allocated using {method}:")
        for strategy, amount in allocations.items():
            percentage = (amount / self.total_capital) * 100
            print(f"   {strategy}: ${amount} ({percentage:.1f}%)")

    def _log_rebalancing(
        self,
        current: Dict[str, float],
        new: Dict[str, float],
        trades: Dict[str, Any],
    ):
        """Log rebalancing decision."""
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": "rebalancing",
            "current_allocations": current,
            "new_allocations": new,
            "rebalancing_trades": trades,
        }

        self.allocation_history.append(log_entry)

        if trades:
            print("ðŸ”„ Rebalancing trades needed:")
            for strategy, trade in trades.items():
                print(f"   {strategy}: {trade['action'].upper()} ${abs(trade['difference'])}")
        else:
            print("âœ… No rebalancing needed")

    def get_allocation_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get allocation history."""
        return self.allocation_history[-limit:]

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get current portfolio summary."""
        if not self.current_allocations:
            return {"total_allocated": 0, "strategies_count": 0}

        total_allocated = sum(self.current_allocations.values())
        unallocated = self.total_capital - total_allocated

        return {
            "total_capital": self.total_capital,
            "total_allocated": round(total_allocated, 2),
            "unallocated": round(unallocated, 2),
            "strategies_count": len(self.current_allocations),
            "allocation_percentage": round((total_allocated / self.total_capital) * 100, 1),
        }


# Convenience functions
def allocate_capital(total_capital: float, method: str = "performance") -> Dict[str, float]:
    """
    Simple capital allocation function.

    Args:
        total_capital: Total capital to allocate
        method: Allocation method

    Returns:
        Capital allocation per strategy
    """
    allocator = CapitalAllocator(total_capital)

    if method == "performance":
        return allocator.allocate_by_performance()
    elif method == "risk_parity":
        return allocator.allocate_by_risk_parity()
    elif method == "equal_weight":
        return allocator.allocate_by_equal_weight()
    elif method == "kelly":
        return allocator.allocate_by_kelly_criterion()
    elif method == "momentum":
        return allocator.allocate_by_momentum()
    else:
        return allocator.allocate_by_performance()


def rebalance_portfolio(
    current_allocations: Dict[str, float],
    total_capital: float,
    method: str = "performance",
) -> Dict[str, float]:
    """
    Rebalance existing portfolio.

    Args:
        current_allocations: Current allocations
        total_capital: Total capital
        method: Rebalancing method

    Returns:
        New allocations
    """
    allocator = CapitalAllocator(total_capital)
    return allocator.rebalance_portfolio(current_allocations, method)


# Example usage
if __name__ == "__main__":
    print("ðŸ’° Smart Capital Allocator")
    print("=" * 40)

    # Test different allocation methods
    total_capital = 10000
    methods = [
        "performance",
        "risk_parity",
        "equal_weight",
        "kelly",
        "momentum",
    ]

    for method in methods:
        print(f"\nðŸ“Š Testing {method.upper()} allocation...")
        try:
            allocations = allocate_capital(total_capital, method)
            if allocations:
                print(f"âœ… {method.upper()} allocation successful")
                for strategy, amount in list(allocations.items())[:3]:  # Show top 3
                    percentage = (amount / total_capital) * 100
                    print(f"   {strategy}: ${amount} ({percentage:.1f}%)")
            else:
                print(f"âš ï¸ No allocations found for {method}")
        except Exception as e:
            print(f"âŒ {method} failed: {e}")

    print("\nðŸŽ¯ Capital allocation testing complete!")

