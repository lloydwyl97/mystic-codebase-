# strategy_reaper.py
"""
Strategy Reaper - Strategy Killer and Cleanup System
Removes underperforming strategies and maintains strategy quality.
Built for Windows 11 Home + PowerShell + Docker.
"""

import json
import logging
import os
import time
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DIR = "./strategy_versions"
INTERVAL = 3600  # 1 hour
PING_FILE = "./logs/strategy_reaper.ping"

# Ensure directories exist
os.makedirs(DIR, exist_ok=True)
os.makedirs("./logs", exist_ok=True)


def create_ping_file(strategies_deleted, total_strategies):
    """Create ping file for dashboard monitoring"""
    try:
        with open(PING_FILE, "w") as f:
            json.dump(
                {
                    "status": "online",
                    "last_update": datetime.timezone.utcnow().isoformat(),
                    "strategies_deleted": strategies_deleted,
                    "total_strategies": total_strategies,
                },
                f,
            )
    except Exception as e:
        print(f"Ping file error: {e}")


def evaluate_strategy(filepath):
    """Evaluate a strategy's performance"""
    try:
        with open(filepath, "r") as f:
            data = json.load(f)

        # Get performance metrics
        winrate = data.get("winrate", 0)
        trades = data.get("trades", 0)
        profit = data.get("total_profit", 0)
        age_days = data.get("age_days", 0)

        # Calculate score (0-100)
        score = 0

        # Winrate component (40% weight)
        if winrate > 0.7:
            score += 40
        elif winrate > 0.6:
            score += 30
        elif winrate > 0.5:
            score += 20
        elif winrate > 0.4:
            score += 10

        # Trade count component (30% weight)
        if trades > 50:
            score += 30
        elif trades > 20:
            score += 20
        elif trades > 10:
            score += 10

        # Profit component (20% weight)
        if profit > 1000:
            score += 20
        elif profit > 500:
            score += 15
        elif profit > 100:
            score += 10
        elif profit > 0:
            score += 5

        # Age component (10% weight) - newer strategies get bonus
        if age_days < 7:
            score += 10
        elif age_days < 30:
            score += 5

        return score, winrate, trades, profit, age_days

    except Exception as e:
        print(f"Error evaluating {filepath}: {e}")
        return 0, 0, 0, 0, 0


def reap_strategies():
    """Remove underperforming strategies"""
    try:
        if not os.path.exists(DIR):
            print(f"[REAPER] Strategy directory not found: {DIR}")
            return 0, 0

        strategies = [f for f in os.listdir(DIR) if f.endswith(".json")]
        total_strategies = len(strategies)
        deleted_count = 0

        print(f"[REAPER] Evaluating {total_strategies} strategies...")

        for filename in strategies:
            filepath = os.path.join(DIR, filename)
            try:
                score, winrate, trades, profit, age_days = evaluate_strategy(filepath)

                # Delete criteria
                should_delete = False
                reason = ""

                if score < 30:
                    should_delete = True
                    reason = f"Low score ({score})"
                elif winrate < 0.3 and trades > 10:
                    should_delete = True
                    reason = f"Poor winrate ({winrate:.3f})"
                elif trades < 5 and age_days > 7:
                    should_delete = True
                    reason = f"Too few trades ({trades})"
                elif profit < -500:
                    should_delete = True
                    reason = f"Large losses (${profit})"

                if should_delete:
                    os.remove(filepath)
                    deleted_count += 1
                    print(f"[REAPER] Deleted {filename}: {reason}")

                    # Log deletion
                    deletion_log = {
                        "timestamp": datetime.timezone.utcnow().isoformat(),
                        "filename": filename,
                        "reason": reason,
                        "score": score,
                        "winrate": winrate,
                        "trades": trades,
                        "profit": profit,
                    }

                    with open("./logs/reaper_log.jsonl", "a") as f:
                        f.write(json.dumps(deletion_log) + "\n")

            except Exception as e:
                print(f"[REAPER] Error processing {filename}: {e}")

        return deleted_count, total_strategies

    except Exception as e:
        print(f"[REAPER] Reaping error: {e}")
        return 0, 0


def main():
    """Main execution loop"""
    print("[REAPER] Strategy Reaper started")
    print(f"[REAPER] Reaping interval: {INTERVAL} seconds")
    print(f"[REAPER] Strategy directory: {DIR}")

    while True:
        try:
            deleted, total = reap_strategies()

            # Create ping file for dashboard
            create_ping_file(deleted, total)

            if deleted > 0:
                print(f"[REAPER] Deleted {deleted} weak strategies")
            else:
                print(f"[REAPER] No strategies deleted. Total: {total}")

        except KeyboardInterrupt:
            print("[REAPER] Shutting down...")
            break
        except Exception as e:
            print(f"[REAPER] Main loop error: {e}")

        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
