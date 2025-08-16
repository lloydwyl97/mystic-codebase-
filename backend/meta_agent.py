from strategy_leaderboard import get_strategy_leaderboard
from capital_allocator import allocate_capital
from position_sizer import calculate_position_size
import time


class MetaAgent:
    """Meta-agent that coordinates all trading strategies"""

    def __init__(self, total_capital=10000):
        self.total_capital = total_capital
        self.active_strategies = {}
        self.performance_history = []

    def pick_best_strategy(self):
        """Select the best performing strategy"""
        leaderboard = get_strategy_leaderboard()
        if leaderboard:
            return leaderboard[0]["strategy"]
        return None

    def allocate_capital_to_strategies(self):
        """Allocate capital across strategies"""
        allocation = allocate_capital(self.total_capital)
        self.active_strategies = allocation
        return allocation

    def calculate_optimal_position_size(self, strategy_name, win_rate):
        """Calculate optimal position size for a strategy"""
        return calculate_position_size(self.total_capital, win_rate)

    def monitor_system_health(self):
        """Monitor overall system health"""
        leaderboard = get_strategy_leaderboard()
        total_profit = sum(s["total_profit"] for s in leaderboard)
        avg_win_rate = (
            sum(s["win_rate"] for s in leaderboard) / len(leaderboard) if leaderboard else 0
        )

        health_score = (total_profit / 1000) + (avg_win_rate * 100)

        return {
            "total_profit": total_profit,
            "avg_win_rate": avg_win_rate,
            "health_score": health_score,
            "active_strategies": len(self.active_strategies),
        }

    def execute_meta_decision(self):
        """Execute a meta-level decision"""
        print("[META] Executing meta-level decision...")

        # Get system health
        health = self.monitor_system_health()

        # Allocate capital
        allocation = self.allocate_capital_to_strategies()

        # Pick best strategy
        best_strategy = self.pick_best_strategy()

        decision = {
            "timestamp": time.time(),
            "health": health,
            "allocation": allocation,
            "best_strategy": best_strategy,
            "action": ("continue" if health["health_score"] > 50 else "rebalance"),
        }

        self.performance_history.append(decision)
        return decision

    def get_system_summary(self):
        """Get complete system summary"""
        health = self.monitor_system_health()
        return {
            "meta_agent_status": "active",
            "total_capital": self.total_capital,
            "system_health": health,
            "active_strategies": self.active_strategies,
            "performance_history_count": len(self.performance_history),
        }


def pick_best_strategy():
    """Simple function to pick the best strategy"""
    leaderboard = get_strategy_leaderboard()
    leaderboard.sort(key=lambda x: x["total_profit"], reverse=True)
    return leaderboard[0]["strategy"] if leaderboard else None


def run_meta_agent():
    """Run the meta-agent system"""
    agent = MetaAgent(total_capital=10000)

    print("[META] Starting meta-agent...")

    # Execute initial decision
    decision = agent.execute_meta_decision()

    print(f"[META] Decision: {decision['action']}")
    print(f"[META] Best strategy: {decision['best_strategy']}")
    print(f"[META] System health: {decision['health']['health_score']:.2f}")

    return agent


if __name__ == "__main__":
    run_meta_agent()


