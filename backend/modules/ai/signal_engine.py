"""
Signal Engine for Mystic AI Trading Platform
Generates trading signals using technical indicators and machine learning.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timezone
import sys
import os

# Add backend to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from modules.ai.persistent_cache import PersistentCache

logger = logging.getLogger(__name__)

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn not available. Install with: pip install scikit-learn")


class SignalEngine:
    def __init__(self):
        """Initialize signal engine with cache and ML models"""
        self.cache = PersistentCache()
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.ml_model = LogisticRegression(random_state=42) if SKLEARN_AVAILABLE else None

        # Technical indicator parameters
        self.rsi_period = 14
        self.ema_fast = 12
        self.ema_slow = 26
        self.ml_lookback = 100

        # Signal thresholds
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        self.min_confidence = 0.6

        logger.info("✅ SignalEngine initialized")

    def _get_ohlcv_data(self, symbol: str, limit: int = 200) -> Optional[pd.DataFrame]:
        """Get OHLCV data from cache and convert to DataFrame"""
        try:
            # Get price history from cache
            price_history = self.cache.get_price_history('aggregated', symbol, limit=limit)

            if not price_history or len(price_history) < 50:
                logger.warning(f"Insufficient price data for {symbol}: {len(price_history) if price_history else 0} records")
                return None

            # Convert to DataFrame
            df = pd.DataFrame(price_history)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')

            # Ensure we have required columns
            if 'price' not in df.columns:
                logger.error(f"No price data found for {symbol}")
                return None

            # Create OHLCV structure (using price as close for now)
            df['close'] = df['price'].astype(float)
            df['open'] = df['close'].shift(1).fillna(df['close'])
            df['high'] = df['close']
            df['low'] = df['close']
            df['volume'] = df.get('volume', 0).astype(float)

            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].dropna()

        except Exception as e:
            logger.error(f"Failed to get OHLCV data for {symbol}: {e}")
            return None

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI (Relative Strength Index)"""
        try:
            # Calculate price changes
            delta = prices.diff()

            # Separate gains and losses
            gains = delta.where(delta > 0, 0)
            losses = -delta.where(delta < 0, 0)

            # Calculate average gains and losses
            avg_gains = gains.rolling(window=period).mean()
            avg_losses = losses.rolling(window=period).mean()

            # Calculate RS and RSI
            rs = avg_gains / avg_losses
            rsi = 100 - (100 / (1 + rs))

            return rsi

        except Exception as e:
            logger.error(f"Failed to calculate RSI: {e}")
            return pd.Series([np.nan] * len(prices))

    def calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate EMA (Exponential Moving Average)"""
        try:
            return prices.ewm(span=period).mean()
        except Exception as e:
            logger.error(f"Failed to calculate EMA: {e}")
            return pd.Series([np.nan] * len(prices))

    def detect_ema_crossover(self, fast_ema: pd.Series, slow_ema: pd.Series) -> Tuple[str, float]:
        """Detect EMA crossover signals"""
        try:
            if len(fast_ema) < 2 or len(slow_ema) < 2:
                return "HOLD", 0.0

            # Get current and previous values
            fast_current = fast_ema.iloc[-1]
            fast_prev = fast_ema.iloc[-2]
            slow_current = slow_ema.iloc[-1]
            slow_prev = slow_ema.iloc[-2]

            # Check for crossover
            if fast_current > slow_current and fast_prev <= slow_prev:
                # Bullish crossover
                return "BUY", 0.8
            elif fast_current < slow_current and fast_prev >= slow_prev:
                # Bearish crossover
                return "SELL", 0.8
            else:
                return "HOLD", 0.0

        except Exception as e:
            logger.error(f"Failed to detect EMA crossover: {e}")
            return "HOLD", 0.0

    def prepare_ml_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for machine learning model"""
        try:
            if len(df) < self.ml_lookback:
                return np.array([]), np.array([])

            # Calculate technical indicators
            df['rsi'] = self.calculate_rsi(df['close'], self.rsi_period)
            df['ema_fast'] = self.calculate_ema(df['close'], self.ema_fast)
            df['ema_slow'] = self.calculate_ema(df['close'], self.ema_slow)

            # Create features
            features = []
            labels = []

            for i in range(self.ml_lookback, len(df)):
                # Price-based features
                price_change = (df['close'].iloc[i] - df['close'].iloc[i-1]) / df['close'].iloc[i-1]
                price_change_5 = (df['close'].iloc[i] - df['close'].iloc[i-5]) / df['close'].iloc[i-5]
                price_change_10 = (df['close'].iloc[i] - df['close'].iloc[i-10]) / df['close'].iloc[i-10]

                # Technical indicator features
                rsi = df['rsi'].iloc[i]
                ema_fast = df['ema_fast'].iloc[i]
                ema_slow = df['ema_slow'].iloc[i]
                ema_ratio = ema_fast / ema_slow if ema_slow != 0 else 1.0

                # Volume features
                volume_change = (df['volume'].iloc[i] - df['volume'].iloc[i-1]) / max(df['volume'].iloc[i-1], 1)

                # Create feature vector
                feature_vector = [
                    price_change, price_change_5, price_change_10,
                    rsi, ema_ratio, volume_change
                ]

                # Create label (1 for price increase, 0 for decrease)
                future_price = df['close'].iloc[i+1] if i+1 < len(df) else df['close'].iloc[i]
                label = 1 if future_price > df['close'].iloc[i] else 0

                features.append(feature_vector)
                labels.append(label)

            return np.array(features), np.array(labels)

        except Exception as e:
            logger.error(f"Failed to prepare ML features: {e}")
            return np.array([]), np.array([])

    def train_ml_model(self, symbol: str) -> bool:
        """Train machine learning model for a symbol"""
        try:
            if not SKLEARN_AVAILABLE:
                logger.warning("Scikit-learn not available for ML training")
                return False

            # Get data
            df = self._get_ohlcv_data(symbol, limit=500)
            if df is None or len(df) < 200:
                logger.warning(f"Insufficient data for ML training: {symbol}")
                return False

            # Prepare features
            features, labels = self.prepare_ml_features(df)
            if len(features) == 0:
                logger.warning(f"No features prepared for ML training: {symbol}")
                return False

            # Scale features
            features_scaled = self.scaler.fit_transform(features)

            # Train model
            self.ml_model.fit(features_scaled, labels)
            logger.info(f"✅ ML model trained for {symbol}")
            return True

        except Exception as e:
            logger.error(f"Failed to train ML model for {symbol}: {e}")
            return False

    def predict_trend(self, symbol: str) -> Tuple[str, float]:
        """Predict trend using ML model"""
        try:
            if not SKLEARN_AVAILABLE or self.ml_model is None:
                return "HOLD", 0.0

            # Get recent data
            df = self._get_ohlcv_data(symbol, limit=50)
            if df is None or len(df) < 20:
                return "HOLD", 0.0

            # Prepare features for prediction
            features, _ = self.prepare_ml_features(df)
            if len(features) == 0:
                return "HOLD", 0.0

            # Get latest feature vector
            latest_features = features[-1].reshape(1, -1)
            latest_features_scaled = self.scaler.transform(latest_features)

            # Make prediction
            prediction = self.ml_model.predict(latest_features_scaled)[0]
            confidence = self.ml_model.predict_proba(latest_features_scaled)[0].max()

            signal = "BUY" if prediction == 1 else "SELL"
            return signal, confidence

        except Exception as e:
            logger.error(f"Failed to predict trend for {symbol}: {e}")
            return "HOLD", 0.0

    def generate_signal(self, symbol: str) -> Dict[str, Any]:
        """Generate trading signal for a symbol"""
        try:
            # Get OHLCV data
            df = self._get_ohlcv_data(symbol)
            if df is None or len(df) < 50:
                return {
                    "symbol": symbol,
                    "signal": "HOLD",
                    "confidence": 0.0,
                    "error": "Insufficient data",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }

            # Calculate technical indicators
            rsi = self.calculate_rsi(df['close'], self.rsi_period)
            ema_fast = self.calculate_ema(df['close'], self.ema_fast)
            ema_slow = self.calculate_ema(df['close'], self.ema_slow)

            # Get latest values
            current_rsi = rsi.iloc[-1]

            # Initialize signal components
            signals = []
            confidences = []

            # RSI signals
            if current_rsi < self.rsi_oversold:
                signals.append("BUY")
                confidences.append(0.7)
            elif current_rsi > self.rsi_overbought:
                signals.append("SELL")
                confidences.append(0.7)

            # EMA crossover signals
            ema_signal, ema_confidence = self.detect_ema_crossover(ema_fast, ema_slow)
            if ema_signal != "HOLD":
                signals.append(ema_signal)
                confidences.append(ema_confidence)

            # ML prediction
            ml_signal, ml_confidence = self.predict_trend(symbol)
            if ml_signal != "HOLD" and ml_confidence >= self.min_confidence:
                signals.append(ml_signal)
                confidences.append(ml_confidence)

            # Determine final signal
            if not signals:
                final_signal = "HOLD"
                final_confidence = 0.0
            else:
                # Count signal types
                buy_count = signals.count("BUY")
                sell_count = signals.count("SELL")

                if buy_count > sell_count:
                    final_signal = "BUY"
                elif sell_count > buy_count:
                    final_signal = "SELL"
                else:
                    final_signal = "HOLD"

                # Calculate average confidence
                final_confidence = sum(confidences) / len(confidences) if confidences else 0.0

            return {
                "symbol": symbol,
                "signal": final_signal,
                "confidence": round(final_confidence, 3),
                "indicators": {
                    "rsi": round(current_rsi, 2) if not pd.isna(current_rsi) else None,
                    "ema_fast": round(ema_fast.iloc[-1], 2) if not pd.isna(ema_fast.iloc[-1]) else None,
                    "ema_slow": round(ema_slow.iloc[-1], 2) if not pd.isna(ema_slow.iloc[-1]) else None
                },
                "signals_used": signals,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to generate signal for {symbol}: {e}")
            return {
                "symbol": symbol,
                "signal": "HOLD",
                "confidence": 0.0,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    def get_signal(self, exchange: str, symbol: str) -> Dict[str, Any]:
        """Get trading signal for a specific symbol"""
        return self.generate_signal(symbol)

    def get_all_signals(self) -> Dict[str, Any]:
        """Get signals for all monitored symbols"""
        try:
            # Get symbols from cache or use default list
            symbols = [
                "BTC-USD", "ETH-USD", "ADA-USD", "SOL-USD", "DOT-USD",
                "LINK-USD", "MATIC-USD", "AVAX-USD", "UNI-USD", "ATOM-USD"
            ]

            signals = {}
            successful_signals = 0

            for symbol in symbols:
                try:
                    signal = self.generate_signal(symbol)
                    signals[symbol] = signal

                    if signal.get('signal') != 'HOLD':
                        successful_signals += 1

                except Exception as e:
                    logger.error(f"Failed to get signal for {symbol}: {e}")
                    signals[symbol] = {
                        "symbol": symbol,
                        "signal": "HOLD",
                        "confidence": 0.0,
                        "error": str(e)
                    }

            return {
                "success": True,
                "total_symbols": len(symbols),
                "successful_signals": successful_signals,
                "signals": signals,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to get all signals: {e}")
            return {"success": False, "error": str(e)}

    def get_active_signals_count(self) -> int:
        """Get count of active trading signals"""
        try:
            # Get recent signals from cache
            recent_signals = self.cache.get_signals_by_type("TRADING_SIGNAL", limit=100)
            
            # Count active signals (non-HOLD signals)
            active_signals = [
                signal for signal in recent_signals 
                if signal.get("metadata", {}).get("signal") != "HOLD"
            ]
            
            return len(active_signals)
            
        except Exception as e:
            logger.error(f"Failed to get active signals count: {e}")
            return 0

    def get_service_status(self) -> Dict[str, Any]:
        """Get current service status"""
        try:
            return {
                "service": "SignalEngine",
                "status": "active",
                "sklearn_available": SKLEARN_AVAILABLE,
                "ml_model_trained": self.ml_model is not None,
                "configuration": {
                    "rsi_period": self.rsi_period,
                    "ema_fast": self.ema_fast,
                    "ema_slow": self.ema_slow,
                    "ml_lookback": self.ml_lookback,
                    "rsi_oversold": self.rsi_oversold,
                    "rsi_overbought": self.rsi_overbought,
                    "min_confidence": self.min_confidence
                }
            }

        except Exception as e:
            logger.error(f"Failed to get service status: {e}")
            return {"success": False, "error": str(e)}


# Global signal engine instance
signal_engine = SignalEngine()


def get_signal_engine() -> SignalEngine:
    """Get the global signal engine instance"""
    return signal_engine


if __name__ == "__main__":
    # Test the signal engine
    engine = SignalEngine()
    print(f"✅ SignalEngine initialized: {engine}")

    # Test signal generation
    signal = engine.get_signal('coinbase', 'BTC-USD')
    print(f"Signal result: {signal}")

    # Test status
    status = engine.get_service_status()
    print(f"Service status: {status['status']}")
    print(f"Sklearn available: {status['sklearn_available']}")
    print(f"ML model trained: {status['ml_model_trained']}")
