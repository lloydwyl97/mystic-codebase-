# trade_explainer.py
"""
LLM Trade Reason Generator - AI-Powered Trade Analysis
Uses GPT-4 to explain trading decisions and provide insights.
Built for Windows 11 Home + PowerShell + Docker.
"""

import openai
import os
import json
import time
import logging
from typing import Dict, Any, List
from datetime import datetime, timezone

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
LOG_FILE = "./logs/trade_log.jsonl"
EXPLANATION_FILE = "/data/trade_explanations.jsonl"
PING_FILE = "./logs/trade_explainer.ping"
INTERVAL = 300  # 5 minutes

# Ensure logs directory exists
os.makedirs("./logs", exist_ok=True)

# OpenAI configuration
openai.api_key = os.getenv("OPENAI_API_KEY")


def load_trade_log() -> List[Dict[str, Any]]:
    """Load trade log from file."""
    try:
        if not os.path.exists(LOG_FILE):
            logger.info("ðŸ“‚ No trade log found, creating empty file")
            return []

        trades = []
        with open(LOG_FILE, "r") as f:
            for line in f:
                if line.strip():
                    trades.append(json.loads(line))

        logger.info(f"ðŸ“‚ Loaded {len(trades)} trades from log")
        return trades
    except Exception as e:
        logger.error(f"âŒ Error loading trade log: {e}")
        return []


def load_explanations() -> Dict[str, str]:
    """Load existing explanations."""
    try:
        if not os.path.exists(EXPLANATION_FILE):
            return {}

        explanations = {}
        with open(EXPLANATION_FILE, "r") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    explanations[data["trade_id"]] = data["explanation"]

        return explanations
    except Exception as e:
        logger.error(f"âŒ Error loading explanations: {e}")
        return {}


def save_explanation(trade_id: str, explanation: str) -> None:
    """Save explanation to file."""
    try:
        os.makedirs(os.path.dirname(EXPLANATION_FILE), exist_ok=True)

        with open(EXPLANATION_FILE, "a") as f:
            json.dump(
                {
                    "trade_id": trade_id,
                    "explanation": explanation,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                f,
            )
            f.write("\n")

        logger.info(f"ðŸ’¾ Saved explanation for trade {trade_id}")
    except Exception as e:
        logger.error(f"âŒ Error saving explanation: {e}")


def create_ping_file(explanations_generated: int, total_trades: int) -> None:
    """Create ping file for dashboard monitoring"""
    try:
        with open(PING_FILE, "w") as f:
            json.dump(
                {
                    "status": "online",
                    "last_update": datetime.now(timezone.utc).isoformat(),
                    "explanations_generated": explanations_generated,
                    "total_trades": total_trades,
                },
                f,
            )
    except Exception as e:
        print(f"Ping file error: {e}")


def explain_trade(trade: Dict[str, Any]) -> str:
    """Generate AI explanation for a trade"""
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return "No OpenAI API key available"

        # Create a detailed prompt for trade explanation
        prompt = f"Explain why this trade was made:\n{json.dumps(trade)}"

        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Error generating explanation: {e}"


def process_trade_log():
    """Process trade log and add explanations"""
    try:
        if not os.path.exists(LOG_FILE):
            print(f"[EXPLAINER] Trade log not found: {LOG_FILE}")
            return 0, 0

        explanations_generated = 0
        total_trades = 0

        # Read existing trades
        trades = []
        with open(LOG_FILE, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        trade = json.loads(line)
                        trades.append(trade)
                        total_trades += 1
                    except json.JSONDecodeError:
                        continue

        # Process trades that don't have explanations
        updated_trades = []
        for trade in trades:
            if "explanation" not in trade:
                print(
                    f"[EXPLAINER] Generating explanation for trade: {trade.get('symbol', 'Unknown')}"
                )
                trade["explanation"] = explain_trade(trade)
                trade["explanation_timestamp"] = datetime.now(timezone.utc).isoformat()
                explanations_generated += 1
                time.sleep(1)  # Rate limiting for OpenAI API
            updated_trades.append(trade)

        # Write updated trades back to file
        if explanations_generated > 0:
            with open(LOG_FILE, "w") as f:
                for trade in updated_trades:
                    f.write(json.dumps(trade) + "\n")

            print(f"[EXPLAINER] Generated {explanations_generated} explanations")

        return explanations_generated, total_trades

    except Exception as e:
        print(f"[EXPLAINER] Error processing trade log: {e}")
        return 0, 0


def initialize_live_trade_processing() -> None:
    """Initialize live trade processing without sample data."""
    try:
        if os.path.exists(LOG_FILE):
            logger.info("ðŸ“ Trade log already exists - ready for live processing")
            return

        # Create empty trade log file for live data
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        with open(LOG_FILE, "w") as f:
            f.write("")  # Create empty file

        logger.info("ðŸ“ Initialized empty trade log for live data processing")
        logger.info("   Note: Trades will be populated by live trading system")

    except Exception as e:
        logger.error(f"âŒ Error initializing live trade processing: {e}")


def main():
    """Main execution loop"""
    print("[EXPLAINER] Trade Explainer started")
    print(f"[EXPLAINER] Processing interval: {INTERVAL} seconds")

    while True:
        try:
            explanations, total = process_trade_log()

            # Create ping file for dashboard
            create_ping_file(explanations, total)

            if explanations > 0:
                print(f"[EXPLAINER] Processed {explanations} new explanations")
            else:
                print(f"[EXPLAINER] No new trades to explain. Total trades: {total}")

        except KeyboardInterrupt:
            print("[EXPLAINER] Shutting down...")
            break
        except Exception as e:
            print(f"[EXPLAINER] Main loop error: {e}")

        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()


