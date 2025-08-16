# portfolio_ai_balance.py
"""
Portfolio AI Balance - Health Evaluator and Rebalancing System
Monitors portfolio health and suggests rebalancing actions.
Built for Windows 11 Home + PowerShell + Docker.
"""

import json
import time
import logging
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
PORTFOLIO_FILE = "./data/portfolio.json"
THRESHOLD = 0.7
INTERVAL = 600  # 10 minutes
PING_FILE = "./logs/portfolio_ai_balance.ping"

# Ensure directories exist
os.makedirs("./data", exist_ok=True)
os.makedirs("./logs", exist_ok=True)


def create_ping_file(stablecoin_ratio, total_value, rebalance_needed):
    """Create ping file for dashboard monitoring"""
    try:
        with open(PING_FILE, "w") as f:
            json.dump(
                {
                    "status": "online",
                    "last_update": datetime.timezone.utcnow().isoformat(),
                    "stablecoin_ratio": stablecoin_ratio,
                    "total_value": total_value,
                    "rebalance_needed": rebalance_needed,
                },
                f,
            )
    except Exception as e:
        print(f"Ping file error: {e}")


def load_portfolio():
    """Load portfolio data from file"""
    try:
        if os.path.exists(PORTFOLIO_FILE):
            with open(PORTFOLIO_FILE, "r") as f:
                return json.load(f)
        else:
            # Create default portfolio
            default_portfolio = {
                "coins": {
                    "USDT": 10000,
                    "BTC": 0.5,
                    "ETH": 5.0,
                    "ADA": 1000,
                    "DOT": 50,
                },
                "total_value_usdt": 15000,
                "last_updated": datetime.timezone.utcnow().isoformat(),
                "performance": {
                    "daily_change": 0.02,
                    "weekly_change": 0.08,
                    "monthly_change": 0.15,
                },
            }
            save_portfolio(default_portfolio)
            return default_portfolio
    except Exception as e:
        print(f"Error loading portfolio: {e}")
        return {}


def save_portfolio(portfolio):
    """Save portfolio data to file"""
    try:
        portfolio["last_updated"] = datetime.timezone.utcnow().isoformat()
        with open(PORTFOLIO_FILE, "w") as f:
            json.dump(portfolio, f, indent=2)
    except Exception as e:
        print(f"Error saving portfolio: {e}")


def evaluate_portfolio():
    """Evaluate portfolio balance and suggest rebalancing"""
    try:
        portfolio = load_portfolio()
        coins = portfolio.get("coins", {})

        # Calculate stablecoin ratio
        usdt_value = coins.get("USDT", 0)
        total_value = sum(coins.values())

        if total_value == 0:
            print("[BALANCE] No portfolio value found")
            return 0, 0, False

        stablecoin_ratio = usdt_value / total_value

        # Check if rebalancing is needed
        rebalance_needed = stablecoin_ratio < (1 - THRESHOLD)

        print(f"[BALANCE] Stablecoin ratio: {stablecoin_ratio:.3f}")
        print(f"[BALANCE] Total portfolio value: ${total_value:,.2f}")

        if rebalance_needed:
            print("[BALANCE] âš ï¸  Rebalancing recommended!")
            print(f"[BALANCE] Target stablecoin ratio: {1 - THRESHOLD:.3f}")
            print(f"[BALANCE] Current ratio: {stablecoin_ratio:.3f}")

            # Calculate rebalancing amounts
            target_usdt = total_value * (1 - THRESHOLD)
            usdt_needed = target_usdt - usdt_value

            if usdt_needed > 0:
                print(f"[BALANCE] Need to add ${usdt_needed:,.2f} USDT")
            else:
                print(f"[BALANCE] Need to reduce USDT by ${abs(usdt_needed):,.2f}")
        else:
            print("[BALANCE] âœ… Portfolio is well balanced")

        # Log portfolio evaluation
        balance_log = {
            "timestamp": datetime.timezone.utcnow().isoformat(),
            "stablecoin_ratio": stablecoin_ratio,
            "total_value": total_value,
            "rebalance_needed": rebalance_needed,
            "target_ratio": 1 - THRESHOLD,
        }

        with open("./logs/balance_log.jsonl", "a") as f:
            f.write(json.dumps(balance_log) + "\n")

        return stablecoin_ratio, total_value, rebalance_needed

    except Exception as e:
        print(f"[BALANCE] Portfolio evaluation error: {e}")
        return 0, 0, False


def main():
    """Main execution loop"""
    print("[BALANCE] Portfolio AI Balance started")
    print(f"[BALANCE] Evaluation interval: {INTERVAL} seconds")
    print(f"[BALANCE] Rebalancing threshold: {THRESHOLD}")

    while True:
        try:
            ratio, total, rebalance = evaluate_portfolio()

            # Create ping file for dashboard
            create_ping_file(ratio, total, rebalance)

        except KeyboardInterrupt:
            print("[BALANCE] Shutting down...")
            break
        except Exception as e:
            print(f"[BALANCE] Main loop error: {e}")

        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()


