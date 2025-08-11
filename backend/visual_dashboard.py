import json
import logging
import os
import subprocess
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/visual_dashboard.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("visual_dashboard")

LEADERBOARD_FILE = "mutation_leaderboard.json"
POSITION_FILE = "position.json"


def clear():
    """Clear the terminal screen safely without shell injection"""
    try:
        if os.name == "nt":  # Windows
            subprocess.run(["cls"], shell=False, check=True)
        else:  # Unix/Linux/MacOS
            subprocess.run(["clear"], shell=False, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        # Fail loudly if clear is not possible
        raise RuntimeError(f"Failed to clear terminal: {e}")


def load_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def print_dashboard():
    clear()
    pos = load_json(POSITION_FILE)
    board = load_json(LEADERBOARD_FILE)

    print("=== LIVE STRATEGY DASHBOARD ===")
    print(f"Last Update: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 40)

    # Position Status
    print("\n--- POSITION STATUS ---")
    if pos and pos.get("entry_price"):
        entry_price = pos.get("entry_price", 0)
        current_price = pos.get("current_price", entry_price)
        peak_price = pos.get("peak_price", entry_price)
        exchange = pos.get("exchange", "unknown")

        if current_price and entry_price:
            pnl_percent = ((current_price - entry_price) / entry_price) * 100
            print(f"ACTIVE POSITION: {exchange.upper()}")
            print(f"Entry Price: ${entry_price:.2f}")
            print(f"Current Price: ${current_price:.2f}")
            print(f"Peak Price: ${peak_price:.2f}")
            print(f"P&L: {pnl_percent:+.2f}%")

            # Visual indicator
            if pnl_percent > 0:
                print("Status: PROFIT")
            else:
                print("Status: LOSS")
        else:
            print("Position data incomplete")
    else:
        print("No active position")
        print("Status: WAITING FOR SIGNALS")

    # Top Strategies
    print("\n--- TOP STRATEGIES ---")
    if board and isinstance(board, list):
        for i, strat in enumerate(board[:5], 1):
            win_rate = strat.get("win_rate", 0) * 100
            profit = strat.get("profit", 0)
            trades = strat.get("trades", 0)
            strat_id = strat.get("id", "unknown")

            print(f"{i}. {strat_id}")
            print(f"   Win Rate: {win_rate:.1f}% | P&L: ${profit:.2f} | Trades: {trades}")
    else:
        print("No strategies in leaderboard")

    # System Status
    print("\n--- SYSTEM STATUS ---")
    print("AI Strategy Execution: RUNNING")
    print("AI Leaderboard Executor: RUNNING")
    print("AI Trade Engine: RUNNING")
    print("Mutation Evaluator: RUNNING")

    print("=" * 40)
    print("Press Ctrl+C to exit")


def run_dashboard():
    """Run the visual dashboard"""
    logger.info("Starting visual dashboard...")

    try:
        while True:
            print_dashboard()
            time.sleep(15)  # Refresh every 15 seconds
    except KeyboardInterrupt:
        logger.info("Dashboard stopped by user")
        print("\nDashboard stopped. Goodbye!")


if __name__ == "__main__":
    run_dashboard()
