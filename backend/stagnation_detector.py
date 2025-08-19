import json
import os

from ai_auto_learner import AIAutoLearner
from notifier import send_alert

STATE_FILE = "ai_model_state.json"


def detect_stagnation():
    if not os.path.exists(STATE_FILE):
        return

    with open(STATE_FILE) as f:
        state = json.load(f)

    adjustments = state.get("adjustment_count", 0)
    if adjustments >= 10:
        # Reset state
        state["confidence_threshold"] = 0.75
        state["adjustment_count"] = 0
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
        send_alert("âš ï¸ AI strategy auto-reset due to stagnation.")
        print("[Stagnation] AI auto-reset triggered.")


def check_performance_plateau():
    learner = AIAutoLearner()
    summary = learner.logger.get_summary()

    if summary["total_trades"] > 50:
        avg_profit = summary["avg_profit"]
        if abs(avg_profit) < 0.1:  # Less than 10 cents average profit
            send_alert("ðŸ“‰ AI performance plateau detected. Consider strategy review.")
            print("[Stagnation] Performance plateau detected.")
            return True
    return False


