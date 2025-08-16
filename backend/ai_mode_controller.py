import logging
from enum import Enum


class AIMode(str, Enum):
    TRAINING = "training"
    LIVE = "live"
    OFF = "off"


class AITradingController:
    def __init__(self):
        self.mode: AIMode = AIMode.TRAINING
        self.daily_trade_limit: int = 20
        self.trade_counter: int = 0
        self.max_drawdown: float = -0.1  # -10%
        self.total_profit: float = 0.0

    def set_mode(self, mode: str):
        mode_lower = mode.lower()
        if mode_lower in ["training", "live", "off"]:
            self.mode = AIMode(mode_lower)
            logging.info(f"AI mode switched to {self.mode}")
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'training', 'live', or 'off'")

    def should_execute_trade(self, simulated_profit: float) -> bool:
        if self.mode == AIMode.OFF:
            return False

        if self.mode == AIMode.TRAINING:
            logging.debug("Simulating trade in training mode")
            return False

        if self.trade_counter >= self.daily_trade_limit:
            logging.warning("Daily trade limit reached")
            return False

        if self.total_profit < self.max_drawdown:
            logging.warning("Max drawdown reached, stopping trades")
            return False

        return True

    def record_trade(self, profit: float):
        self.trade_counter += 1
        self.total_profit += profit
        logging.info(
            f"Trade recorded: profit={profit}, total={self.total_profit}, count={self.trade_counter}"
        )

    def reset_daily_counter(self):
        self.trade_counter = 0
        logging.info("Daily trade counter reset")

    def get_status(self):
        return {
            "mode": self.mode.value,
            "daily_trades": self.trade_counter,
            "daily_limit": self.daily_trade_limit,
            "total_profit": self.total_profit,
            "max_drawdown": self.max_drawdown,
        }


