import json
import logging
import time

from ai_strategy_execution import execute_ai_strategy_signal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/ai_leaderboard_executor.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("ai_leaderboard_executor")

LEADERBOARD_FILE = "mutation_leaderboard.json"
TRADE_SYMBOL_BINANCE = "ETHUSDT"
TRADE_SYMBOL_COINBASE = "ETH-USD"
USD_TRADE_AMOUNT = 50
MIN_WIN_RATE = 0.55
MIN_PROFIT = 10.0


def load_leaderboard():
    try:
        with open(LEADERBOARD_FILE) as f:
            data = json.load(f)
            logger.info(f"Loaded leaderboard with {len(data)} strategies")
            return data
    except FileNotFoundError:
        logger.warning(f"Leaderboard file {LEADERBOARD_FILE} not found, creating empty leaderboard")
        return []
    except Exception as e:
        logger.error(f"Error loading leaderboard: {e}")
        return []


def select_top_strategy():
    leaderboard = load_leaderboard()
    if not leaderboard:
        logger.info("No strategies in leaderboard")
        return None

    # Sort by profit and win rate
    leaderboard.sort(key=lambda x: (x.get("profit", 0), x.get("win_rate", 0)), reverse=True)

    logger.info(f"Top strategies: {[s.get('id', 'unknown') for s in leaderboard[:3]]}")

    for strat in leaderboard:
        win_rate = strat.get("win_rate", 0)
        profit = strat.get("profit", 0)

        if win_rate >= MIN_WIN_RATE and profit >= MIN_PROFIT:
            logger.info(
                f"Selected strategy: {strat.get('id', 'unknown')} (win_rate: {win_rate}, profit: {profit})"
            )
            return strat

    logger.info(
        f"No strategy meets criteria (min_win_rate: {MIN_WIN_RATE}, min_profit: {MIN_PROFIT})"
    )
    return None


def execute_leaderboard_top_strategy():
    top = select_top_strategy()
    if not top:
        logger.info("No suitable strategy found for execution")
        return None

    signal = True  # Always execute when strategy is selected
    logger.info(f"Executing top strategy: {top.get('id', 'unknown')}")

    result = execute_ai_strategy_signal(
        TRADE_SYMBOL_BINANCE, TRADE_SYMBOL_COINBASE, USD_TRADE_AMOUNT, signal
    )

    if result and "error" not in result:
        logger.info(f"Strategy execution successful: {result}")
    else:
        logger.error(f"Strategy execution failed: {result}")

    return result


def create_leaderboard():
    """Create an empty leaderboard for real strategies"""
    empty_leaderboard = []

    try:
        with open(LEADERBOARD_FILE, "w") as f:
            json.dump(empty_leaderboard, f, indent=2)
        logger.info("Created empty leaderboard for real strategies")
    except Exception as e:
        logger.error(f"Failed to create leaderboard: {e}")


def run_continuous_execution():
    """Run continuous leaderboard execution"""
    logger.info("Starting continuous leaderboard execution...")
    logger.info(
        f"Configuration: {TRADE_SYMBOL_BINANCE}/{TRADE_SYMBOL_COINBASE} ${USD_TRADE_AMOUNT}"
    )
    logger.info(f"Min win rate: {MIN_WIN_RATE}, Min profit: {MIN_PROFIT}")

    while True:
        try:
            execute_leaderboard_top_strategy()
            logger.info("Waiting 1 hour before next execution...")
            time.sleep(3600)  # 1 hour
        except KeyboardInterrupt:
            logger.info("Leaderboard execution stopped by user")
            break
        except Exception as e:
            logger.error(f"Error in continuous execution: {e}")
            time.sleep(3600)  # Continue after error


if __name__ == "__main__":
    # Create leaderboard if it doesn't exist
    try:
        with open(LEADERBOARD_FILE) as f:
            pass
    except FileNotFoundError:
        create_leaderboard()

    # Run continuous execution
    run_continuous_execution()


