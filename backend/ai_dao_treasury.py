from capital_allocator import allocate_capital
from strategy_leaderboard import get_strategy_leaderboard


def simulate_treasury_growth(treasury_usdt, days, growth_rate_per_day=0.015):
    """Simulate treasury growth over time"""
    for day in range(days):
        treasury_usdt *= 1 + growth_rate_per_day
        print(f"Day {day+1}: ${round(treasury_usdt, 2)}")
    return treasury_usdt


def allocate_dao_treasury(treasury_usdt):
    """Allocate DAO treasury based on strategy performance"""
    get_strategy_leaderboard(hours_back=24)
    allocation = allocate_capital(treasury_usdt)
    print(f"[DAO] Allocating ${treasury_usdt} â†’ {allocation}")
    return allocation


def manage_dao_governance():
    """Manage DAO governance decisions"""
    print("[DAO] Processing governance proposals...")
    # Add governance logic here
    return {"status": "active", "proposals": []}


def execute_dao_decision(decision_type, amount):
    """Execute a DAO decision"""
    print(f"[DAO] Executing {decision_type} with ${amount}")
    return {"executed": True, "type": decision_type, "amount": amount}


