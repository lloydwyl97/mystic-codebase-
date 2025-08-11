import pandas as pd
import time
import requests
from datetime import datetime
from typing import Dict, List
import sqlite3
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Enhanced configuration
ANOMALY_DB = "./data/anomaly_detection.db"
ALERT_THRESHOLD = 0.8
CHECK_INTERVAL = 300  # 5 minutes
LOOKBACK_PERIODS = [1, 4, 24]  # Hours for different timeframes


class AnomalyDatabase:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize anomaly database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS price_anomalies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                anomaly_score REAL NOT NULL,
                anomaly_type TEXT NOT NULL,
                price_change REAL NOT NULL,
                volume_change REAL NOT NULL,
                timeframe TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS volume_anomalies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                volume_ratio REAL NOT NULL,
                avg_volume REAL NOT NULL,
                current_volume REAL NOT NULL,
                anomaly_score REAL NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS pattern_anomalies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                pattern_type TEXT NOT NULL,
                confidence REAL NOT NULL,
                price_level REAL NOT NULL,
                volume_level REAL NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        conn.commit()
        conn.close()

    def save_price_anomaly(self, data: Dict):
        """Save price anomaly to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO price_anomalies
            (timestamp, symbol, anomaly_score, anomaly_type, price_change, volume_change, timeframe, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                data["timestamp"],
                data["symbol"],
                data["anomaly_score"],
                data["anomaly_type"],
                data["price_change"],
                data["volume_change"],
                data["timeframe"],
                data["confidence"],
            ),
        )

        conn.commit()
        conn.close()

    def save_volume_anomaly(self, data: Dict):
        """Save volume anomaly to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO volume_anomalies
            (timestamp, symbol, volume_ratio, avg_volume, current_volume, anomaly_score)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                data["timestamp"],
                data["symbol"],
                data["volume_ratio"],
                data["avg_volume"],
                data["current_volume"],
                data["anomaly_score"],
            ),
        )

        conn.commit()
        conn.close()

    def save_pattern_anomaly(self, data: Dict):
        """Save pattern anomaly to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO pattern_anomalies
            (timestamp, symbol, pattern_type, confidence, price_level, volume_level)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                data["timestamp"],
                data["symbol"],
                data["pattern_type"],
                data["confidence"],
                data["price_level"],
                data["volume_level"],
            ),
        )

        conn.commit()
        conn.close()


def get_historical_data(symbol: str, hours: int) -> pd.DataFrame:
    """Get historical price data from Binance"""
    try:
        # Get klines data
        interval = "1h" if hours <= 24 else "4h"
        limit = min(hours, 1000)  # Binance limit

        url = "https://api.binance.us/api/v3/klines"
        params = {"symbol": symbol, "interval": interval, "limit": limit}

        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()

            df = pd.DataFrame(
                data,
                columns=[
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "close_time",
                    "quote_volume",
                    "trades",
                    "taker_buy_base",
                    "taker_buy_quote",
                    "ignore",
                ],
            )

            # Convert to numeric
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col])

            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            return df

    except Exception as e:
        print(f"Historical data fetch error for {symbol}: {e}")

    return pd.DataFrame()


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators for anomaly detection"""
    if df.empty:
        return df

    try:
        # RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        sma = df["close"].rolling(window=20).mean()
        std = df["close"].rolling(window=20).std()
        df["bb_upper"] = sma + (std * 2)
        df["bb_lower"] = sma - (std * 2)
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["close"]

        # MACD
        ema_fast = df["close"].ewm(span=12).mean()
        ema_slow = df["close"].ewm(span=26).mean()
        df["macd"] = ema_fast - ema_slow
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        df["macd_histogram"] = df["macd"] - df["macd_signal"]

        # Volume indicators
        df["volume_sma"] = df["volume"].rolling(window=20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma"]

        # Price changes
        df["price_change"] = df["close"].pct_change()
        df["price_change_abs"] = df["price_change"].abs()

        # Volatility
        df["volatility"] = df["price_change"].rolling(window=20).std()

    except Exception as e:
        print(f"Technical indicators calculation error: {e}")

    return df


def detect_price_anomalies(df: pd.DataFrame, symbol: str) -> List[Dict]:
    """Detect price anomalies using machine learning"""
    anomalies = []

    if df.empty or len(df) < 50:
        return anomalies

    try:
        # Prepare features for anomaly detection
        features = [
            "price_change",
            "volume_ratio",
            "rsi",
            "bb_width",
            "volatility",
        ]
        feature_data = df[features].dropna()

        if len(feature_data) < 20:
            return anomalies

        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_data)

        # Train isolation forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        iso_forest.fit_predict(scaled_features)

        # Get recent data point
        latest_data = feature_data.iloc[-1]
        latest_score = iso_forest.decision_function(scaled_features[-1:])[0]

        # Check if latest point is anomalous
        if latest_score < -0.5:  # Threshold for anomaly
            anomaly_type = "price_spike" if latest_data["price_change"] > 0.05 else "price_crash"

            anomaly = {
                "timestamp": datetime.timezone.utcnow().isoformat(),
                "symbol": symbol,
                "anomaly_score": abs(latest_score),
                "anomaly_type": anomaly_type,
                "price_change": latest_data["price_change"],
                "volume_change": latest_data["volume_ratio"],
                "timeframe": "1h",
                "confidence": min(abs(latest_score) * 2, 1.0),
            }

            anomalies.append(anomaly)

    except Exception as e:
        print(f"Price anomaly detection error: {e}")

    return anomalies


def detect_volume_anomalies(df: pd.DataFrame, symbol: str) -> List[Dict]:
    """Detect volume anomalies"""
    anomalies = []

    if df.empty or len(df) < 20:
        return anomalies

    try:
        # Calculate volume statistics
        current_volume = df["volume"].iloc[-1]
        avg_volume = df["volume"].rolling(window=20).mean().iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

        # Detect volume spikes
        if volume_ratio > 3.0:  # 3x average volume
            anomaly = {
                "timestamp": datetime.timezone.utcnow().isoformat(),
                "symbol": symbol,
                "volume_ratio": volume_ratio,
                "avg_volume": avg_volume,
                "current_volume": current_volume,
                "anomaly_score": min(volume_ratio / 5, 1.0),
            }
            anomalies.append(anomaly)

    except Exception as e:
        print(f"Volume anomaly detection error: {e}")

    return anomalies


def detect_pattern_anomalies(df: pd.DataFrame, symbol: str) -> List[Dict]:
    """Detect chart pattern anomalies"""
    anomalies = []

    if df.empty or len(df) < 50:
        return anomalies

    try:
        # Detect double tops/bottoms
        highs = df["high"].rolling(window=5).max()
        df["low"].rolling(window=5).min()

        # Check for double top pattern
        if len(highs) >= 20:
            recent_highs = highs.tail(20)
            if len(recent_highs[recent_highs > recent_highs.quantile(0.8)]) >= 2:
                anomaly = {
                    "timestamp": datetime.timezone.utcnow().isoformat(),
                    "symbol": symbol,
                    "pattern_type": "double_top",
                    "confidence": 0.7,
                    "price_level": df["close"].iloc[-1],
                    "volume_level": df["volume"].iloc[-1],
                }
                anomalies.append(anomaly)

        # Check for support/resistance breaks
        current_price = df["close"].iloc[-1]
        support_level = df["low"].tail(20).min()
        resistance_level = df["high"].tail(20).max()

        if current_price < support_level * 0.98:  # Break below support
            anomaly = {
                "timestamp": datetime.timezone.utcnow().isoformat(),
                "symbol": symbol,
                "pattern_type": "support_break",
                "confidence": 0.8,
                "price_level": current_price,
                "volume_level": df["volume"].iloc[-1],
            }
            anomalies.append(anomaly)

        elif current_price > resistance_level * 1.02:  # Break above resistance
            anomaly = {
                "timestamp": datetime.timezone.utcnow().isoformat(),
                "symbol": symbol,
                "pattern_type": "resistance_break",
                "confidence": 0.8,
                "price_level": current_price,
                "volume_level": df["volume"].iloc[-1],
            }
            anomalies.append(anomaly)

    except Exception as e:
        print(f"Pattern anomaly detection error: {e}")

    return anomalies


def monitor_anomalies_enhanced():
    """Enhanced anomaly monitoring with all features"""
    try:
        symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT"]
        db = AnomalyDatabase(ANOMALY_DB)

        all_anomalies = []

        for symbol in symbols:
            # Get historical data for different timeframes
            for hours in LOOKBACK_PERIODS:
                df = get_historical_data(symbol, hours)
                if df.empty:
                    continue

                # Calculate technical indicators
                df = calculate_technical_indicators(df)

                # Detect different types of anomalies
                price_anomalies = detect_price_anomalies(df, symbol)
                volume_anomalies = detect_volume_anomalies(df, symbol)
                pattern_anomalies = detect_pattern_anomalies(df, symbol)

                # Save anomalies to database
                for anomaly in price_anomalies:
                    db.save_price_anomaly(anomaly)
                    all_anomalies.append(anomaly)

                for anomaly in volume_anomalies:
                    db.save_volume_anomaly(anomaly)
                    all_anomalies.append(anomaly)

                for anomaly in pattern_anomalies:
                    db.save_pattern_anomaly(anomaly)
                    all_anomalies.append(anomaly)

        # Print results
        if all_anomalies:
            print(f"[Anomaly] Detected {len(all_anomalies)} anomalies:")
            for anomaly in all_anomalies:
                if "anomaly_type" in anomaly:
                    print(
                        f"[Anomaly] {anomaly['symbol']} - {anomaly['anomaly_type']} (Score: {anomaly['anomaly_score']:.3f})"
                    )
                elif "pattern_type" in anomaly:
                    print(
                        f"[Anomaly] {anomaly['symbol']} - {anomaly['pattern_type']} (Confidence: {anomaly['confidence']:.3f})"
                    )
        else:
            print("[Anomaly] No anomalies detected")

    except Exception as e:
        print(f"[Anomaly] Enhanced monitoring error: {e}")


# Main execution loop
while True:
    monitor_anomalies_enhanced()
    time.sleep(CHECK_INTERVAL)
