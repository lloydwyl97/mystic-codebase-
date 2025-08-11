# risk_optimizer.py
"""
Auto Risk Recalibrator - Dynamic Risk Management System
Automatically adjusts risk parameters based on trading performance.
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
RISK_FILE = "./config/risk.json"
INTERVAL = 900  # 15 minutes
PING_FILE = "./logs/risk_optimizer.ping"

# Ensure directories exist
os.makedirs("./config", exist_ok=True)
os.makedirs("./logs", exist_ok=True)


def create_ping_file(risk_level, winrate):
    """Create ping file for dashboard monitoring"""
    try:
        with open(PING_FILE, "w") as f:
            json.dump(
                {
                    "status": "online",
                    "last_update": datetime.timezone.utcnow().isoformat(),
                    "risk_level": risk_level,
                    "winrate": winrate,
                },
                f,
            )
    except Exception as e:
        print(f"Ping file error: {e}")


def load_risk_config():
    """Load risk configuration from file"""
    try:
        if os.path.exists(RISK_FILE):
            with open(RISK_FILE, "r") as f:
                return json.load(f)
        else:
            # Create default risk config
            default_config = {
                "stop_loss_pct": 0.02,
                "take_profit_pct": 0.04,
                "max_position_size": 0.1,
                "max_daily_loss": 0.05,
                "recent_winrate": 0.6,
                "total_trades": 0,
                "profitable_trades": 0,
                "last_updated": datetime.timezone.utcnow().isoformat(),
            }
            save_risk_config(default_config)
            return default_config
    except Exception as e:
        print(f"Error loading risk config: {e}")
        return {}


def save_risk_config(config):
    """Save risk configuration to file"""
    try:
        config["last_updated"] = datetime.timezone.utcnow().isoformat()
        with open(RISK_FILE, "w") as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        print(f"Error saving risk config: {e}")


def calculate_winrate(config):
    """Calculate current winrate from trade data"""
    total_trades = config.get("total_trades", 0)
    profitable_trades = config.get("profitable_trades", 0)

    if total_trades > 0:
        return profitable_trades / total_trades
    return 0.6  # Default winrate


def adjust_risk():
    """Adjust risk parameters based on performance"""
    try:
        config = load_risk_config()
        winrate = calculate_winrate(config)

        # Update winrate in config
        config["recent_winrate"] = winrate

        # Adjust stop loss based on winrate
        if winrate < 0.4:
            # Poor performance - tighten stop loss
            config["stop_loss_pct"] = 0.01
            risk_level = "conservative"
        elif winrate < 0.5:
            # Below average - moderate stop loss
            config["stop_loss_pct"] = 0.015
            risk_level = "moderate"
        elif winrate < 0.6:
            # Average - standard stop loss
            config["stop_loss_pct"] = 0.02
            risk_level = "standard"
        else:
            # Good performance - can be more aggressive
            config["stop_loss_pct"] = 0.03
            risk_level = "aggressive"

        # Adjust position size based on winrate
        if winrate < 0.5:
            config["max_position_size"] = 0.05  # Reduce position size
        elif winrate > 0.7:
            config["max_position_size"] = 0.15  # Increase position size
        else:
            config["max_position_size"] = 0.1  # Standard position size

        # Save updated config
        save_risk_config(config)

        print(f"[RISK] Winrate: {winrate:.3f}")
        print(f"[RISK] Stop loss: {config['stop_loss_pct']:.3f}")
        print(f"[RISK] Position size: {config['max_position_size']:.3f}")
        print(f"[RISK] Risk level: {risk_level}")

        # Create ping file for dashboard
        create_ping_file(risk_level, winrate)

        # Log risk adjustment
        risk_log = {
            "timestamp": datetime.timezone.utcnow().isoformat(),
            "winrate": winrate,
            "stop_loss_pct": config["stop_loss_pct"],
            "max_position_size": config["max_position_size"],
            "risk_level": risk_level,
        }

        with open("./logs/risk_log.jsonl", "a") as f:
            f.write(json.dumps(risk_log) + "\n")

    except Exception as e:
        print(f"[RISK] Risk adjustment error: {e}")


def main():
    """Main execution loop"""
    print("[RISK] Risk Optimizer started")
    print(f"[RISK] Optimization interval: {INTERVAL} seconds")

    while True:
        try:
            adjust_risk()
        except KeyboardInterrupt:
            print("[RISK] Shutting down...")
            break
        except Exception as e:
            print(f"[RISK] Main loop error: {e}")

        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
