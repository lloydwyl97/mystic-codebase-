import sqlite3
import os
from datetime import datetime, timedelta

DB_FILE = os.getenv("TRADE_LOG_DB", "trades.db")
LOCK_FILE = "strategy_locks.txt"


def get_strategy_leaderboard(hours_back=24):
    """Get strategy leaderboard sorted by total profit"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    since = (datetime.timezone.utcnow() - timedelta(hours=hours_back)).isoformat()
    c.execute(
        """
        SELECT strategy, COUNT(*), AVG(profit_usd), SUM(profit_usd),
            SUM(CASE WHEN profit_usd > 0 THEN 1 ELSE 0 END)
        FROM trades
        WHERE timestamp > ?
        GROUP BY strategy
        ORDER BY SUM(profit_usd) DESC
    """,
        (since,),
    )
    rows = c.fetchall()
    conn.close()

    leaderboard = []
    for row in rows:
        trades = row[1]
        wins = row[4]
        win_rate = wins / trades if trades else 0
        leaderboard.append(
            {
                "strategy": row[0],
                "trades": trades,
                "avg_profit": row[2],
                "total_profit": row[3],
                "win_rate": win_rate,
            }
        )
    return leaderboard


def load_locked_strategies():
    """Load list of locked strategies"""
    if not os.path.exists(LOCK_FILE):
        return set()
    with open(LOCK_FILE) as f:
        return set(line.strip() for line in f.readlines())


def lock_strategy(name):
    """Lock a strategy to prevent further mutations"""
    with open(LOCK_FILE, "a") as f:
        f.write(f"{name}\n")
    print(f"[LEADERBOARD] Locked strategy: {name}")


def auto_evolve():
    """Automatically evolve strategies based on performance"""
    leaderboard = get_strategy_leaderboard(hours_back=24)
    locked = load_locked_strategies()

    for strat in leaderboard:
        name = strat["strategy"]
        if name in locked:
            continue

        # Promote winners
        if strat["win_rate"] > 0.6 and strat["total_profit"] > 0:
            print(f"[PROMOTE] {name} | ${strat['total_profit']:.2f} | WR: {strat['win_rate']:.2%}")
            lock_strategy(name)
            clone_and_mutate(name)

        # Retire losers
        if strat["win_rate"] < 0.3 and strat["total_profit"] < 0:
            print(f"[RETIRE] {name} | Losses: ${strat['total_profit']:.2f}")
            retire_strategy(name)


def clone_and_mutate(base_strategy):
    """Clone and mutate a winning strategy"""
    new_name = base_strategy + "_mutant_" + datetime.timezone.utcnow().strftime("%H%M%S")
    print(f"[MUTATE] {base_strategy} â†’ {new_name}")
    # Copy the strategy logic file or config and apply mutations


def retire_strategy(strategy_name):
    """Retire a losing strategy"""
    print(f"[ARCHIVE] Disabling {strategy_name}")
    # Mark in your configs as inactive or move to /retired folder


def get_top_strategies(limit=5):
    """Get top performing strategies"""
    leaderboard = get_strategy_leaderboard()
    return leaderboard[:limit]


if __name__ == "__main__":
    print("=== STRATEGY LEADERBOARD ===")
    top = get_top_strategies()
    for i, strat in enumerate(top, 1):
        print(
            f"{i}. {strat['strategy']}: ${strat['total_profit']:.2f} | WR: {strat['win_rate']:.2%}"
        )


