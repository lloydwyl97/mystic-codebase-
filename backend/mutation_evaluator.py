import json
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/mutation_evaluator.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("mutation_evaluator")

MUTATION_FILE = "mutations.json"
LEADERBOARD_FILE = "mutation_leaderboard.json"
MIN_WIN_RATE = 0.55
MIN_PROFIT = 10.0


def load_file(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, PermissionError, json.JSONDecodeError, IOError) as e:
        logger.warning(f"Failed to load file {path}: {e}")
        return []


def save_file(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def evaluate_mutations():
    mutations = load_file(MUTATION_FILE)
    leaderboard = load_file(LEADERBOARD_FILE)

    promoted_count = 0
    for m in mutations:
        if m.get("promoted"):
            continue
        if m.get("win_rate", 0) >= MIN_WIN_RATE and m.get("profit", 0) >= MIN_PROFIT:
            m["promoted"] = True
            leaderboard.append(m)
            promoted_count += 1
            logger.info(
                f"Promoted strategy: {m.get('id', 'unknown')} (win_rate: {m.get('win_rate')}, profit: {m.get('profit')})"
            )

    if promoted_count > 0:
        leaderboard.sort(
            key=lambda x: (x.get("profit", 0), x.get("win_rate", 0)),
            reverse=True,
        )
        save_file(LEADERBOARD_FILE, leaderboard)
        save_file(MUTATION_FILE, mutations)
        logger.info(f"Promoted {promoted_count} strategies to leaderboard")
    else:
        logger.info("No new strategies promoted")


def create_sample_mutations():
    """Create sample mutations for testing"""
    sample_mutations = [
        {
            "id": "mutation_001_rsi_enhanced",
            "win_rate": 0.62,
            "profit": 18.5,
            "trades": 25,
            "promoted": False,
        },
        {
            "id": "mutation_002_ema_optimized",
            "win_rate": 0.58,
            "profit": 12.3,
            "trades": 32,
            "promoted": False,
        },
        {
            "id": "mutation_003_macd_improved",
            "win_rate": 0.51,
            "profit": 8.7,
            "trades": 19,
            "promoted": False,
        },
    ]

    try:
        save_file(MUTATION_FILE, sample_mutations)
        logger.info(f"Created sample mutations file with {len(sample_mutations)} strategies")
    except Exception as e:
        logger.error(f"Failed to create sample mutations: {e}")


def run_continuous_evaluation():
    """Run continuous mutation evaluation"""
    logger.info("Starting continuous mutation evaluation...")
    logger.info(f"Criteria: Min win rate {MIN_WIN_RATE}, Min profit {MIN_PROFIT}")

    while True:
        try:
            evaluate_mutations()
            logger.info("Waiting 5 minutes before next evaluation...")
            time.sleep(300)  # 5 minutes
        except KeyboardInterrupt:
            logger.info("Mutation evaluation stopped by user")
            break
        except Exception as e:
            logger.error(f"Error in mutation evaluation: {e}")
            time.sleep(300)  # Continue after error


if __name__ == "__main__":
    # Create sample mutations if it doesn't exist
    try:
        with open(MUTATION_FILE, "r") as f:
            pass
    except FileNotFoundError:
        create_sample_mutations()

    # Run continuous evaluation
    run_continuous_evaluation()


