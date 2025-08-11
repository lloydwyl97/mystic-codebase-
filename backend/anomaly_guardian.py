# anomaly_guardian.py
"""
Anomaly Guardian - Market Anomaly Detection System
Detects unusual price movements and market behavior using machine learning.
Built for Windows 11 Home + PowerShell + Docker.
"""

import numpy as np
import pandas as pd
import time
import json
import os
import logging
from datetime import datetime
from sklearn.ensemble import IsolationForest

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
CHECK_INTERVAL = 300  # 5 minutes
ANOMALY_THRESHOLD = 0.01  # 1% threshold
SYMBOLS = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT"]
PING_FILE = "./logs/anomaly_guardian.ping"

# Ensure logs directory exists
os.makedirs("./logs", exist_ok=True)

try:
    from sklearn.ensemble import IsolationForest

    SKLEARN_AVAILABLE = True
except ImportError:
    logger.warning("scikit-learn not available, using statistical anomaly detection")
    SKLEARN_AVAILABLE = False


def create_ping_file(anomaly_count, symbols_checked):
    """Create ping file for dashboard monitoring"""
    try:
        with open(PING_FILE, "w") as f:
            json.dump(
                {
                    "status": "online",
                    "last_update": datetime.timezone.utcnow().isoformat(),
                    "anomaly_count": anomaly_count,
                    "symbols_checked": symbols_checked,
                },
                f,
            )
    except Exception as e:
        logger.error(f"Ping file error: {e}")


def get_live_price_data(symbol="BTCUSDT", limit=100):
    return [{"price": 30000 + np.random.randn()} for _ in range(limit)]


def statistical_anomaly_detection(df: pd.DataFrame) -> bool:
    """Statistical anomaly detection using z-score."""
    if len(df) < 10:
        return False

    prices = df["close"].values
    mean_price = np.mean(prices[:-1])  # Exclude current price
    std_price = np.std(prices[:-1])

    if std_price == 0:
        return False

    current_price = prices[-1]
    z_score = abs(current_price - mean_price) / std_price

    # Flag as anomaly if z-score > 3 (3 standard deviations)
    return z_score > 3


def detect_anomaly(df):
    model = IsolationForest(contamination=0.01)
    df["score"] = model.fit_predict(df[["price"]])
    if df["score"].iloc[-1] == -1:
        print("[ALERT] Anomaly Detected!")


def main():
    """Main execution loop"""
    logger.info("🚀 Anomaly Guardian started")
    logger.info(f"⏰ Check interval: {CHECK_INTERVAL} seconds")
    logger.info(f"🔍 Monitoring symbols: {SYMBOLS}")

    anomaly_count = 0

    while True:
        try:
            symbols_checked = 0
            current_anomalies = 0

            for symbol in SYMBOLS:
                try:
                    prices = get_live_price_data(symbol, 100)
                    if prices:
                        if detect_anomaly(prices):
                            current_anomalies += 1
                        symbols_checked += 1
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")

            anomaly_count += current_anomalies

            # Create ping file for dashboard
            create_ping_file(anomaly_count, symbols_checked)

            if current_anomalies > 0:
                logger.info(f"🚨 {current_anomalies} anomalies detected in this cycle")
            else:
                logger.info(f"✅ No anomalies detected. Checked {symbols_checked} symbols.")

        except KeyboardInterrupt:
            logger.info("🛑 Shutting down...")
            break
        except Exception as e:
            logger.error(f"❌ Main loop error: {e}")

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
