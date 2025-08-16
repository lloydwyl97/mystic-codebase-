import sqlite3
from datetime import datetime, timedelta
import os

DB_FILE = os.getenv("TRADE_LOG_DB", "trades.db")


def fetch_recent_strategy_stats(hours_back=24):
    """Fetch recent strategy performance statistics"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    since_time = (datetime.timezone.utcnow() - timedelta(hours=hours_back)).isoformat()

    c.execute(
        """
        SELECT strategy, COUNT(*), AVG(profit_usd), SUM(profit_usd), SUM(CASE WHEN profit_usd > 0 THEN 1 ELSE 0 END)
        FROM trades
        WHERE timestamp > ?
        GROUP BY strategy
    """,
        (since_time,),
    )

    rows = c.fetchall()
    conn.close()

    stats = []
    for row in rows:
        stats.append(
            {
                "strategy": row[0],
                "trade_count": row[1],
                "avg_profit": row[2],
                "total_profit": row[3],
                "win_count": row[4],
                "win_rate": row[4] / row[1] if row[1] else 0,
            }
        )
    return stats


def promote_strategies():
    """Promote strategies based on performance metrics"""
    stats = fetch_recent_strategy_stats(hours_back=24)
    promoted = [s for s in stats if s["win_rate"] > 0.6 and s["total_profit"] > 0]

    for s in promoted:
        print(f"[PROMOTE] {s['strategy']} | P: ${s['total_profit']:.2f} | WR: {s['win_rate']:.2%}")

    return promoted


def evolve_strategies():
    """Evolve strategies using mutation feedback loop"""
    winners = promote_strategies()
    for strat in winners:
        # Inject mutations based on winner template
        mutate_from_template(strat["strategy"])
    return winners


def mutate_from_template(strategy_name):
    """Mutate a strategy from a winning template"""
    print(f"[MUTATE] Cloning {strategy_name} â†’ new version")
    # Add your strategy duplication and mutation logic here
    # This would copy the strategy config and apply random mutations


def get_mutation_candidates():
    """Get strategies that are candidates for mutation"""
    stats = fetch_recent_strategy_stats(hours_back=24)
    candidates = []

    for s in stats:
        if s["trade_count"] >= 5:  # Minimum trades for evaluation
            if s["win_rate"] > 0.5 and s["total_profit"] > 0:
                candidates.append(
                    {
                        "strategy": s["strategy"],
                        "score": s["win_rate"] * s["total_profit"],
                        "type": "promote",
                    }
                )
            elif s["win_rate"] < 0.3 and s["total_profit"] < 0:
                candidates.append(
                    {
                        "strategy": s["strategy"],
                        "score": abs(s["total_profit"]),
                        "type": "retire",
                    }
                )

    return candidates


def run_mutation_cycle():
    """Run a complete mutation cycle"""
    print("[MUTATION] Starting mutation cycle...")

    # Get candidates
    candidates = get_mutation_candidates()

    # Process promotions
    for candidate in candidates:
        if candidate["type"] == "promote":
            print(f"[MUTATION] Promoting {candidate['strategy']} (score: {candidate['score']:.2f})")
            mutate_from_template(candidate["strategy"])
        elif candidate["type"] == "retire":
            print(f"[MUTATION] Retiring {candidate['strategy']} (loss: {candidate['score']:.2f})")

    print(f"[MUTATION] Cycle complete. Processed {len(candidates)} candidates.")


if __name__ == "__main__":
    run_mutation_cycle()


