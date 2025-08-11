"""
Machine Learning Signal Enhancement Service
Uses advanced ML models to increase signal accuracy and win percentage
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class MLPrediction:
    """ML prediction result"""

    probability: float
    confidence: float
    features_importance: Dict[str, float]
    model_version: str
    prediction_time: datetime


class MLSignalEnhancer:
    """Advanced ML signal enhancement for higher win percentage"""

    def __init__(self):
        self.models: Dict[str, Dict[str, Any]] = {}
        self.scalers: Dict[str, Any] = {}
        self.feature_importance: Dict[str, Any] = {}
        self.model_versions: Dict[str, str] = {}
        self.accuracy_history: List[float] = []
        self.min_confidence_threshold = 0.75
        self.ml_predictions: List[Dict[str, Any]] = []

    async def initialize_models(self):
        """Initialize ML models for different timeframes"""
        timeframes = ["1h", "4h", "1d"]

        for timeframe in timeframes:
            # Create ensemble model
            self.models[timeframe] = {
                "rf": RandomForestClassifier(
                    n_estimators=200,
                    max_depth=10,
                    random_state=42,
                    class_weight="balanced",
                ),
                "gb": GradientBoostingClassifier(
                    n_estimators=150,
                    max_depth=8,
                    learning_rate=0.1,
                    random_state=42,
                ),
            }

            self.scalers[timeframe] = StandardScaler()
            self.model_versions[timeframe] = f"v1.0_{datetime.now().strftime('%Y%m%d')}"

        logger.info("ML models initialized successfully")

    def extract_features(self, signal: Dict[str, Any]) -> np.ndarray:
        """Extract advanced features from signal"""
        features: List[float] = []

        # Price-based features
        features.extend(
            [
                signal.get("price", 0),
                signal.get("volume", 0),
                signal.get("market_cap", 0),
                signal.get("price_change_24h", 0),
                signal.get("volume_change_24h", 0),
            ]
        )

        # Technical indicators
        features.extend(
            [
                signal.get("rsi", 50),
                signal.get("macd", 0),
                signal.get("bb_position", 0.5),
                signal.get("ma_20", 0),
                signal.get("ma_50", 0),
                signal.get("ma_200", 0),
            ]
        )

        # Volatility features
        features.extend(
            [
                signal.get("volatility", 0.02),
                signal.get("atr", 0),
                signal.get("bollinger_width", 0),
            ]
        )

        # Volume features
        features.extend(
            [
                signal.get("volume_ratio", 1),
                signal.get("volume_sma_ratio", 1),
                signal.get("obv", 0),
            ]
        )

        # Market sentiment
        features.extend(
            [
                signal.get("fear_greed_index", 50),
                signal.get("social_sentiment", 0.5),
                signal.get("news_sentiment", 0.5),
            ]
        )

        # Time-based features
        current_hour = datetime.now().hour
        features.extend(
            [
                current_hour,
                datetime.now().weekday(),
                signal.get("days_since_ath", 0),
            ]
        )

        # Whale activity
        features.extend(
            [
                1.0 if signal.get("whale_activity", False) else 0.0,
                signal.get("whale_transaction_count", 0),
                signal.get("whale_volume_ratio", 0),
            ]
        )

        return np.array(features, dtype=np.float64).reshape(1, -1)

    async def predict_signal_strength(
        self, signal: Dict[str, Any], timeframe: str = "4h"
    ) -> MLPrediction:
        """Predict signal strength using ML models"""
        try:
            # Extract features
            features = self.extract_features(signal)

            # Scale features
            if timeframe in self.scalers:
                features_scaled = self.scalers[timeframe].transform(features)
            else:
                features_scaled = features

            # Get predictions from ensemble
            predictions: List[float] = []
            confidences: List[float] = []

            for model_name, model in self.models[timeframe].items():
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(features_scaled)[0]
                    predictions.append(float(proba[1]))  # Probability of positive class
                    confidences.append(float(max(proba)))

            # Ensemble prediction
            avg_probability = float(np.mean(predictions))
            avg_confidence = float(np.mean(confidences))

            # Get feature importance
            feature_names = [
                "price",
                "volume",
                "market_cap",
                "price_change_24h",
                "volume_change_24h",
                "rsi",
                "macd",
                "bb_position",
                "ma_20",
                "ma_50",
                "ma_200",
                "volatility",
                "atr",
                "bollinger_width",
                "volume_ratio",
                "volume_sma_ratio",
                "obv",
                "fear_greed_index",
                "social_sentiment",
                "news_sentiment",
                "hour",
                "weekday",
                "days_since_ath",
                "whale_activity",
                "whale_transaction_count",
                "whale_volume_ratio",
            ]

            # Calculate feature importance from Random Forest
            rf_model = self.models[timeframe]["rf"]
            if hasattr(rf_model, "feature_importances_"):
                importance_dict = dict(zip(feature_names, rf_model.feature_importances_))
            else:
                importance_dict = {name: 0.0 for name in feature_names}

            return MLPrediction(
                probability=avg_probability,
                confidence=avg_confidence,
                features_importance=importance_dict,
                model_version=self.model_versions.get(timeframe, "unknown"),
                prediction_time=datetime.now(),
            )

        except Exception as e:
            logger.error(f"Error in ML prediction: {e}")
            # Fallback to basic prediction
            return MLPrediction(
                probability=0.5,
                confidence=0.5,
                features_importance={},
                model_version="fallback",
                prediction_time=datetime.now(),
            )

    async def train_models(self, historical_data: List[Dict[str, Any]]):
        """Train ML models with historical data"""
        try:
            for timeframe in ["1h", "4h", "1d"]:
                # Filter data for timeframe
                timeframe_data = [d for d in historical_data if d.get("timeframe") == timeframe]

                if len(timeframe_data) < 100:
                    logger.warning(f"Insufficient data for {timeframe} model training")
                    continue

                # Prepare features and labels
                X: List[np.ndarray] = []
                y: List[int] = []

                for data_point in timeframe_data:
                    features = self.extract_features(data_point).flatten()
                    X.append(features)

                    # Label: 1 if profitable, 0 if not
                    profit = data_point.get("profit", 0)
                    y.append(1 if profit > 0 else 0)

                X_array = np.array(X)
                y_array = np.array(y)

                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X_array,
                    y_array,
                    test_size=0.2,
                    random_state=42,
                    stratify=y_array,
                )

                # Scale features
                self.scalers[timeframe].fit(X_train)
                X_train_scaled = self.scalers[timeframe].transform(X_train)
                X_test_scaled = self.scalers[timeframe].transform(X_test)

                # Train models
                for model_name, model in self.models[timeframe].items():
                    model.fit(X_train_scaled, y_train)

                    # Evaluate
                    y_pred = model.predict(X_test_scaled)
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, zero_division=0)
                    recall = recall_score(y_test, y_pred, zero_division=0)

                    logger.info(
                        f"{timeframe} {model_name} - Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}"
                    )

                    # Store feature importance for Random Forest
                    if model_name == "rf" and hasattr(model, "feature_importances_"):
                        self.feature_importance[timeframe] = model.feature_importances_

                # Update model version
                self.model_versions[timeframe] = f"v1.1_{datetime.now().strftime('%Y%m%d_%H%M')}"

        except Exception as e:
            logger.error(f"Error training ML models: {e}")

    def get_model_performance(self) -> Dict[str, Any]:
        """Get ML model performance metrics"""
        return {
            "model_versions": self.model_versions,
            "feature_importance": self.feature_importance,
            "accuracy_history": self.accuracy_history,
            "min_confidence_threshold": self.min_confidence_threshold,
        }


# Global ML enhancer instance
ml_enhancer = MLSignalEnhancer()
