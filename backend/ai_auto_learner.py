import json
import os
from datetime import datetime

from simulation_logger import SimulationLogger
from datetime import datetime, timezone

MODEL_STATE_FILE = os.getenv("MODEL_STATE_PATH", "ai_model_state.json")


class AIAutoLearner:
    def __init__(self):
        self.logger = SimulationLogger()
        self.state = self._load_state()

    def _load_state(self):
        if os.path.exists(MODEL_STATE_FILE):
            with open(MODEL_STATE_FILE, "r") as f:
                return json.load(f)
        return {
            "version": 1,
            "confidence_threshold": 0.75,
            "avg_profit_threshold": 0.5,
            "adjustment_count": 0,
            "last_update": None,
        }

    def _save_state(self):
        with open(MODEL_STATE_FILE, "w") as f:
            json.dump(self.state, f, indent=2)

    def evaluate_and_adapt(self):
        summary = self.logger.get_summary()
        improved = False

        # Increase confidence threshold if we're winning big
        if summary["avg_profit"] > self.state["avg_profit_threshold"]:
            self.state["confidence_threshold"] = min(
                0.95, self.state["confidence_threshold"] + 0.01
            )
            improved = True

        # Decrease confidence threshold if losing
        elif summary["avg_profit"] < 0:
            self.state["confidence_threshold"] = max(0.5, self.state["confidence_threshold"] - 0.01)
            improved = True

        if improved:
            self.state["adjustment_count"] += 1
            self.state["last_update"] = datetime.now(timezone.utc).isoformat()
            self._save_state()
            print(f"[AutoLearner] AI strategy adjusted: {self.state}")
        else:
            print("[AutoLearner] No strategy change. Performance stable.")

    def get_current_threshold(self):
        return self.state["confidence_threshold"]

