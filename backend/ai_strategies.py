"""
AI Trading Strategies Module

Provides AI-powered trading strategies and machine learning models for market analysis.
"""

import logging
import pickle
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    ARBITRAGE = "arbitrage"
    GRID_TRADING = "grid_trading"
    SCALPING = "scalping"
    SWING_TRADING = "swing_trading"
    CUSTOM = "custom"


class SignalType(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"


@dataclass
class TradingSignal:
    symbol: str
    signal_type: SignalType
    confidence: float
    price: float
    timestamp: datetime
    strategy: str
    indicators: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyConfig:
    name: str
    strategy_type: StrategyType
    symbols: List[str]
    parameters: Dict[str, Any]
    risk_level: str
    enabled: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.timezone.utc))


@dataclass
class MLModel:
    name: str
    model_type: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    last_trained: datetime
    features: List[str]
    parameters: Dict[str, Any]


class TechnicalIndicators:
    """Comprehensive technical analysis indicators."""

    @staticmethod
    def calculate_sma(data: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average."""
        return data.rolling(window=period).mean()

    @staticmethod
    def calculate_ema(data: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average."""
        return data.ewm(span=period).mean()

    @staticmethod
    def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index."""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def calculate_macd(
        data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD (Moving Average Convergence Divergence)."""
        ema_fast = data.ewm(span=fast).mean()
        ema_slow = data.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def calculate_bollinger_bands(
        data: pd.Series, period: int = 20, std_dev: float = 2
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands."""
        sma = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band

    @staticmethod
    def calculate_stochastic(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int = 14,
        d_period: int = 3,
    ) -> Tuple[pd.Series, pd.Series]:
        """Stochastic Oscillator."""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent

    @staticmethod
    def calculate_atr(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        """Average True Range."""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr

    @staticmethod
    def calculate_volume_profile(
        volume: pd.Series, price: pd.Series, bins: int = 50
    ) -> Dict[str, float]:
        """Volume Profile Analysis."""
        price_bins = pd.cut(price, bins=bins)
        volume_profile = volume.groupby(price_bins).sum()
        return volume_profile.to_dict()


class PatternRecognition:
    """Advanced pattern recognition for technical analysis."""

    @staticmethod
    def detect_candlestick_patterns(
        open_prices: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> Dict[str, List[int]]:
        """Detect candlestick patterns using ta library."""
        patterns: Dict[str, List[int]] = {}

        # Note: ta library doesn't have built-in candlestick patterns like TA-Lib
        # We'll implement basic pattern detection manually
        try:
            # Doji pattern (open ≈ close)
            doji_threshold = 0.001
            doji_pattern = abs(open_prices - close) <= (high - low) * doji_threshold
            doji_indices = np.where(doji_pattern)[0]
            if len(doji_indices) > 0:
                patterns["doji"] = doji_indices.tolist()

            # Hammer pattern (long lower shadow, small body)
            body_size = abs(close - open_prices)
            lower_shadow = np.minimum(open_prices, close) - low
            upper_shadow = high - np.maximum(open_prices, close)
            hammer_pattern = (lower_shadow > 2 * body_size) & (upper_shadow < body_size)
            hammer_indices = np.where(hammer_pattern)[0]
            if len(hammer_indices) > 0:
                patterns["hammer"] = hammer_indices.tolist()

            # Shooting star pattern (long upper shadow, small body)
            shooting_star_pattern = (upper_shadow > 2 * body_size) & (lower_shadow < body_size)
            shooting_star_indices = np.where(shooting_star_pattern)[0]
            if len(shooting_star_indices) > 0:
                patterns["shooting_star"] = shooting_star_indices.tolist()

        except Exception as e:
            logger.warning(f"Error detecting candlestick patterns: {str(e)}")

        return patterns

    @staticmethod
    def detect_chart_patterns(
        high: pd.Series, low: pd.Series, close: pd.Series
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Detect chart patterns like triangles, flags, etc."""
        patterns = {}

        # Head and Shoulders pattern
        hns_patterns = PatternRecognition._detect_head_shoulders(high, low, close)
        if hns_patterns:
            patterns["head_shoulders"] = hns_patterns

        # Double Top/Bottom
        double_patterns = PatternRecognition._detect_double_patterns(high, low, close)
        if double_patterns:
            patterns["double_patterns"] = double_patterns

        # Triangle patterns
        triangle_patterns = PatternRecognition._detect_triangles(high, low, close)
        if triangle_patterns:
            patterns["triangles"] = triangle_patterns

        return patterns

    @staticmethod
    def _detect_head_shoulders(
        high: pd.Series, low: pd.Series, close: pd.Series
    ) -> List[Dict[str, Any]]:
        """Detect Head and Shoulders pattern."""
        patterns = []
        window = 20

        for i in range(window, len(high) - window):
            # Look for three peaks with middle peak higher
            left_peak = high.iloc[i - window : i].max()
            middle_peak = high.iloc[i - window : i + window].max()
            right_peak = high.iloc[i : i + window].max()

            if (
                middle_peak > left_peak
                and middle_peak > right_peak
                and abs(left_peak - right_peak) / left_peak < 0.05
            ):  # Shoulders roughly equal
                patterns.append(
                    {
                        "type": "head_shoulders",
                        "left_shoulder": (i - window + high.iloc[i - window : i].idxmax()),
                        "head": (i - window + high.iloc[i - window : i + window].idxmax()),
                        "right_shoulder": (i + high.iloc[i : i + window].idxmax()),
                        "neckline": (left_peak + right_peak) / 2,
                    }
                )

        return patterns

    @staticmethod
    def _detect_double_patterns(
        high: pd.Series, low: pd.Series, close: pd.Series
    ) -> List[Dict[str, Any]]:
        """Detect Double Top and Double Bottom patterns."""
        patterns = []
        window = 15

        for i in range(window, len(high) - window):
            # Double Top
            left_peak = high.iloc[i - window : i].max()
            right_peak = high.iloc[i : i + window].max()

            if abs(left_peak - right_peak) / left_peak < 0.03:  # Peaks roughly equal
                patterns.append(
                    {
                        "type": "double_top",
                        "left_peak": (i - window + high.iloc[i - window : i].idxmax()),
                        "right_peak": i + high.iloc[i : i + window].idxmax(),
                        "resistance": (left_peak + right_peak) / 2,
                    }
                )

            # Double Bottom
            left_trough = low.iloc[i - window : i].min()
            right_trough = low.iloc[i : i + window].min()

            if abs(left_trough - right_trough) / left_trough < 0.03:  # Troughs roughly equal
                patterns.append(
                    {
                        "type": "double_bottom",
                        "left_trough": (i - window + low.iloc[i - window : i].idxmax()),
                        "right_trough": i + low.iloc[i : i + window].idxmax(),
                        "support": (left_trough + right_trough) / 2,
                    }
                )

        return patterns

    @staticmethod
    def _detect_triangles(
        high: pd.Series, low: pd.Series, close: pd.Series
    ) -> List[Dict[str, Any]]:
        """Detect triangle patterns (ascending, descending, symmetrical)."""
        patterns = []
        window = 30

        for i in range(window, len(high) - window):
            highs = high.iloc[i - window : i + window]
            high_slope = np.polyfit(range(len(highs)), highs, 1)[0]
            low_slope = np.polyfit(range(len(low)), low, 1)[0]

            # Determine triangle type
            if high_slope < -0.001 and low_slope > 0.001:
                triangle_type = "ascending"
            elif high_slope < -0.001 and low_slope < -0.001:
                triangle_type = "descending"
            elif abs(high_slope) < 0.001 and abs(low_slope) < 0.001:
                triangle_type = "symmetrical"
            else:
                continue

            patterns.append(
                {
                    "type": triangle_type,
                    "start": i - window,
                    "end": i + window,
                    "high_slope": high_slope,
                    "low_slope": low_slope,
                }
            )

        return patterns


class StrategyBuilder:
    """Visual strategy builder with drag-and-drop interface."""

    def __init__(self):
        self.strategies = {}
        self.available_indicators = {
            "sma": TechnicalIndicators.calculate_sma,
            "ema": TechnicalIndicators.calculate_ema,
            "rsi": TechnicalIndicators.calculate_rsi,
            "macd": TechnicalIndicators.calculate_macd,
            "bollinger_bands": TechnicalIndicators.calculate_bollinger_bands,
            "stochastic": TechnicalIndicators.calculate_stochastic,
            "atr": TechnicalIndicators.calculate_atr,
        }

        self.available_conditions = {
            "crossover": self._crossover_condition,
            "threshold": self._threshold_condition,
            "divergence": self._divergence_condition,
            "pattern": self._pattern_condition,
        }

        self.available_actions = {
            "buy": self._buy_action,
            "sell": self._sell_action,
            "set_stop_loss": self._set_stop_loss_action,
            "set_take_profit": self._set_take_profit_action,
        }

    def create_strategy(self, name: str, description: str, rules: List[Dict[str, Any]]) -> str:
        """Create a new trading strategy."""
        strategy_id = f"strategy_{len(self.strategies) + 1}"

        self.strategies[strategy_id] = {
            "id": strategy_id,
            "name": name,
            "description": description,
            "rules": rules,
            "created_at": datetime.now(timezone.timezone.utc),
            "enabled": True,
            "performance": {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "avg_profit": 0.0,
                "avg_loss": 0.0,
                "profit_factor": 0.0,
            },
        }

        logger.info(f"Created strategy: {name} (ID: {strategy_id})")
        return strategy_id

    def _crossover_condition(self, data1: pd.Series, data2: pd.Series) -> pd.Series:
        """Check if data1 crosses over data2."""
        return (data1 > data2) & (data1.shift(1) <= data2.shift(1))

    def _threshold_condition(self, data: pd.Series, threshold: float, operator: str) -> pd.Series:
        """Check if data meets threshold condition."""
        if operator == ">":
            return data > threshold
        elif operator == "<":
            return data < threshold
        elif operator == ">=":
            return data >= threshold
        elif operator == "<=":
            return data <= threshold
        elif operator == "==":
            return data == threshold
        else:
            return pd.Series([False] * len(data))

    def _divergence_condition(self, price: pd.Series, indicator: pd.Series) -> pd.Series:
        """Detect divergence between price and indicator."""
        # Simplified divergence detection
        price_trend = price.diff().rolling(5).mean()
        indicator_trend = indicator.diff().rolling(5).mean()

        bullish_divergence = (price_trend < 0) & (indicator_trend > 0)
        bearish_divergence = (price_trend > 0) & (indicator_trend < 0)

        return bullish_divergence | bearish_divergence

    def _pattern_condition(self, pattern_data: Dict[str, Any]) -> pd.Series:
        """Check for specific patterns."""
        # This would be implemented based on pattern recognition results
        return pd.Series([False] * len(pattern_data.get("data", [])))

    def _buy_action(self, signal_data: Dict[str, Any]) -> TradingSignal:
        """Generate buy signal."""
        return TradingSignal(
            symbol=signal_data["symbol"],
            signal_type=SignalType.BUY,
            confidence=signal_data.get("confidence", 0.5),
            price=signal_data["price"],
            timestamp=datetime.now(timezone.timezone.utc),
            strategy=signal_data["strategy"],
            indicators=signal_data.get("indicators", {}),
        )

    def _sell_action(self, signal_data: Dict[str, Any]) -> TradingSignal:
        """Generate sell signal."""
        return TradingSignal(
            symbol=signal_data["symbol"],
            signal_type=SignalType.SELL,
            confidence=signal_data.get("confidence", 0.5),
            price=signal_data["price"],
            timestamp=datetime.now(timezone.timezone.utc),
            strategy=signal_data["strategy"],
            indicators=signal_data.get("indicators", {}),
        )

    def _set_stop_loss_action(self, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Set stop loss level."""
        return {
            "action": "set_stop_loss",
            "price": signal_data["stop_loss_price"],
            "type": "stop_loss",
        }

    def _set_take_profit_action(self, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Set take profit level."""
        return {
            "action": "set_take_profit",
            "price": signal_data["take_profit_price"],
            "type": "take_profit",
        }

    async def execute_strategy(
        self, strategy_id: str, market_data: pd.DataFrame
    ) -> List[TradingSignal]:
        """Execute a strategy on market data."""
        if strategy_id not in self.strategies:
            raise ValueError(f"Strategy {strategy_id} not found")

        strategy = self.strategies[strategy_id]
        signals = []

        for rule in strategy["rules"]:
            try:
                # Execute rule conditions
                condition_result = await self._evaluate_rule(rule, market_data)

                if condition_result:
                    # Execute rule actions
                    action_result = await self._execute_rule_actions(rule, market_data)
                    signals.extend(action_result)

            except Exception as e:
                logger.error(f"Error executing rule in strategy {strategy_id}: {str(e)}")

        return signals

    async def _evaluate_rule(self, rule: Dict[str, Any], market_data: pd.DataFrame) -> bool:
        """Evaluate a rule's conditions."""
        conditions = rule.get("conditions", [])

        for condition in conditions:
            condition_type = condition["type"]
            params = condition["parameters"]

            if condition_type == "crossover":
                data1 = market_data[params["indicator1"]]
                data2 = market_data[params["indicator2"]]
                result = self.available_conditions["crossover"](data1, data2)

                if not result.iloc[-1]:  # Check latest value
                    return False

            elif condition_type == "threshold":
                data = market_data[params["indicator"]]
                threshold = params["threshold"]
                operator = params["operator"]
                result = self.available_conditions["threshold"](data, threshold, operator)

                if not result.iloc[-1]:  # Check latest value
                    return False

        return True

    async def _execute_rule_actions(
        self, rule: Dict[str, Any], market_data: pd.DataFrame
    ) -> List[TradingSignal]:
        """Execute rule actions."""
        actions = rule.get("actions", [])
        signals = []

        for action in actions:
            action_type = action["type"]
            params = action["parameters"]

            signal_data = {
                "symbol": params.get("symbol", "BTCUSDT"),
                "price": market_data["close"].iloc[-1],
                "strategy": rule.get("strategy_name", "Unknown"),
                "confidence": params.get("confidence", 0.5),
                "indicators": params.get("indicators", {}),
                "stop_loss_price": params.get("stop_loss_price"),
                "take_profit_price": params.get("take_profit_price"),
            }

            if action_type == "buy":
                signals.append(self.available_actions["buy"](signal_data))
            elif action_type == "sell":
                signals.append(self.available_actions["sell"](signal_data))

        return signals


class PredictiveAnalytics:
    """Machine learning-based predictive analytics."""

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}

    def prepare_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for machine learning models."""
        features = pd.DataFrame()

        # Price-based features
        features["price_change"] = market_data["close"].pct_change()
        features["price_change_5"] = market_data["close"].pct_change(5)
        features["price_change_10"] = market_data["close"].pct_change(10)

        # Volume features
        features["volume_change"] = market_data["volume"].pct_change()
        features["volume_sma_ratio"] = (
            market_data["volume"] / market_data["volume"].rolling(20).mean()
        )

        # Technical indicators
        features["rsi"] = TechnicalIndicators.calculate_rsi(market_data["close"])
        features["sma_20"] = TechnicalIndicators.calculate_sma(market_data["close"], 20)
        features["sma_50"] = TechnicalIndicators.calculate_sma(market_data["close"], 50)
        features["ema_12"] = TechnicalIndicators.calculate_ema(market_data["close"], 12)
        features["ema_26"] = TechnicalIndicators.calculate_ema(market_data["close"], 26)

        # MACD
        macd_line, signal_line, histogram = TechnicalIndicators.calculate_macd(market_data["close"])
        features["macd"] = macd_line
        features["macd_signal"] = signal_line
        features["macd_histogram"] = histogram

        # Bollinger Bands
        upper, middle, lower = TechnicalIndicators.calculate_bollinger_bands(market_data["close"])
        features["bb_upper"] = upper
        features["bb_middle"] = middle
        features["bb_lower"] = lower
        features["bb_position"] = (market_data["close"] - lower) / (upper - lower)

        # Volatility
        features["atr"] = TechnicalIndicators.calculate_atr(
            market_data["high"], market_data["low"], market_data["close"]
        )
        features["volatility"] = market_data["close"].rolling(20).std()

        # Remove NaN values
        features = features.dropna()

        return features

    def create_labels(self, market_data: pd.DataFrame, horizon: int = 5) -> pd.Series:
        """Create labels for supervised learning."""
        future_returns = market_data["close"].shift(-horizon) / market_data["close"] - 1

        # Binary classification: 1 if price goes up, 0 if down
        labels = (future_returns > 0).astype(int)

        # Remove NaN values
        labels = labels.dropna()

        return labels

    def train_model(
        self,
        model_name: str,
        features: pd.DataFrame,
        labels: pd.Series,
        model_type: str = "random_forest",
    ) -> MLModel:
        """Train a machine learning model."""

        # Align features and labels
        common_index = features.index.intersection(labels.index)
        features_aligned = features.loc[common_index]
        labels_aligned = labels.loc[common_index]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features_aligned, labels_aligned, test_size=0.2, random_state=42
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        if model_type == "random_forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == "gradient_boosting":
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        model.fit(X_train_scaled, y_train)

        # Make predictions
        y_pred = model.predict(X_test_scaled)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        # Feature importance
        if hasattr(model, "feature_importances_"):
            feature_importance = dict(zip(features.columns, model.feature_importances_))
        else:
            feature_importance = {}

        # Create MLModel object
        ml_model = MLModel(
            name=model_name,
            model_type=model_type,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            last_trained=datetime.now(timezone.timezone.utc),
            features=list(features.columns),
            parameters=model.get_params(),
        )

        # Store model and scaler
        self.models[model_name] = model
        self.scalers[model_name] = scaler
        self.feature_importance[model_name] = feature_importance

        logger.info(f"Trained model {model_name}: Accuracy={accuracy:.3f}, F1={f1:.3f}")

        return ml_model

    def predict(self, model_name: str, features: pd.DataFrame) -> Tuple[np.ndarray, float]:
        """Make predictions using a trained model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")

        model = self.models[model_name]
        scaler = self.scalers[model_name]

        # Scale features
        features_scaled = scaler.transform(features)

        # Make prediction
        prediction = model.predict(features_scaled)
        confidence = (
            model.predict_proba(features_scaled).max() if hasattr(model, "predict_proba") else 0.5
        )

        return prediction, confidence

    def get_feature_importance(self, model_name: str) -> Dict[str, float]:
        """Get feature importance for a model."""
        return self.feature_importance.get(model_name, {})

    def save_model(self, model_name: str, filepath: str):
        """Save a trained model to disk."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")

        model_data = {
            "model": self.models[model_name],
            "scaler": self.scalers[model_name],
            "feature_importance": self.feature_importance[model_name],
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

        logger.info(f"Saved model {model_name} to {filepath}")

    def load_model(self, model_name: str, filepath: str):
        """Load a trained model from disk."""
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        self.models[model_name] = model_data["model"]
        self.scalers[model_name] = model_data["scaler"]
        self.feature_importance[model_name] = model_data["feature_importance"]

        logger.info(f"Loaded model {model_name} from {filepath}")


# Global instances
strategy_builder = StrategyBuilder()
predictive_analytics = PredictiveAnalytics()
pattern_recognition = PatternRecognition()


class AIStrategies:
    pass
