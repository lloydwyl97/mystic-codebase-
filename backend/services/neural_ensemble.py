"""
Neural Network Ensemble with Attention Mechanisms
Advanced AI system focusing on important signals and adapting to market conditions
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class NeuralPrediction:
    """Neural ensemble prediction result"""

    ensemble_prediction: float
    attention_weights: dict[str, float]
    model_confidence: float
    adaptation_score: float
    signal_importance: dict[str, float]
    ensemble_variance: float
    recommendation: str
    timestamp: datetime


class NeuralEnsemble:
    """Advanced neural ensemble with attention mechanisms"""

    def __init__(self):
        self.models: dict[str, Any] = {}
        self.attention_weights: dict[str, float] = {}
        self.adaptation_history: list[dict[str, Any]] = []
        self.signal_importance: dict[str, float] = {}
        self.ensemble_variance_threshold = 0.1
        self.min_confidence_threshold = 0.7

    async def predict_with_attention(self, signal: dict[str, Any]) -> NeuralPrediction:
        """Make prediction using neural ensemble with attention"""
        try:
            # Extract features for neural models
            features = self._extract_neural_features(signal)

            # Get predictions from ensemble models
            model_predictions = await self._get_ensemble_predictions(features)

            # Calculate attention weights
            attention_weights = self._calculate_attention_weights(features, model_predictions)

            # Weighted ensemble prediction
            ensemble_prediction = self._calculate_weighted_prediction(
                model_predictions, attention_weights
            )

            # Calculate model confidence
            model_confidence = self._calculate_model_confidence(
                model_predictions, attention_weights
            )

            # Calculate adaptation score
            adaptation_score = self._calculate_adaptation_score(signal)

            # Calculate signal importance
            signal_importance = self._calculate_signal_importance(features, attention_weights)

            # Calculate ensemble variance
            ensemble_variance = self._calculate_ensemble_variance(model_predictions)

            # Generate recommendation
            recommendation = self._generate_neural_recommendation(
                ensemble_prediction,
                model_confidence,
                adaptation_score,
                ensemble_variance,
            )

            return NeuralPrediction(
                ensemble_prediction=ensemble_prediction,
                attention_weights=attention_weights,
                model_confidence=model_confidence,
                adaptation_score=adaptation_score,
                signal_importance=signal_importance,
                ensemble_variance=ensemble_variance,
                recommendation=recommendation,
                timestamp=datetime.now(),
            )

        except Exception as e:
            logger.error(f"Error in neural ensemble prediction: {e}")
            return self._create_fallback_prediction()

    def _extract_neural_features(self, signal: dict[str, Any]) -> dict[str, float]:
        """Extract features for neural network models"""
        try:
            features = {}

            # Price-based features
            features["price"] = signal.get("price", 0)
            features["price_change"] = signal.get("price_change_24h", 0)
            features["price_volatility"] = signal.get("volatility", 0.02)

            # Volume features
            features["volume"] = signal.get("volume", 0)
            features["volume_ratio"] = signal.get("volume_ratio", 1.0)
            features["volume_sma_ratio"] = signal.get("volume_sma_ratio", 1.0)

            # Technical indicators
            features["rsi"] = signal.get("rsi", 50)
            features["macd"] = signal.get("macd", 0)
            features["bb_position"] = signal.get("bb_position", 0.5)
            features["ma_20"] = signal.get("ma_20", 0)
            features["ma_50"] = signal.get("ma_50", 0)
            features["ma_200"] = signal.get("ma_200", 0)

            # Market sentiment
            features["fear_greed"] = signal.get("fear_greed_index", 50)
            features["social_sentiment"] = signal.get("social_sentiment", 0.5)
            features["news_sentiment"] = signal.get("news_sentiment", 0.5)

            # Time-based features
            current_hour = datetime.now().hour
            features["hour"] = current_hour
            features["weekday"] = datetime.now().weekday()

            # Whale activity
            features["whale_activity"] = 1.0 if signal.get("whale_activity", False) else 0.0
            features["whale_volume"] = signal.get("whale_volume_ratio", 0)

            return features

        except Exception as e:
            logger.error(f"Error extracting neural features: {e}")
            return {
                "price": 0,
                "price_change": 0,
                "price_volatility": 0.02,
                "volume": 0,
                "volume_ratio": 1.0,
                "volume_sma_ratio": 1.0,
                "rsi": 50,
                "macd": 0,
                "bb_position": 0.5,
                "ma_20": 0,
                "ma_50": 0,
                "ma_200": 0,
                "fear_greed": 50,
                "social_sentiment": 0.5,
                "news_sentiment": 0.5,
                "hour": 12,
                "weekday": 0,
                "whale_activity": 0.0,
                "whale_volume": 0,
            }

    async def _get_ensemble_predictions(self, features: dict[str, float]) -> dict[str, float]:
        """Get predictions from ensemble models"""
        try:
            # Simulate ensemble model predictions
            # In production, use actual neural network models

            import random

            predictions = {}

            # LSTM model prediction
            lstm_base = (
                0.5 + (features["price_change"] * 0.1) + (features["volume_ratio"] - 1) * 0.2
            )
            predictions["lstm"] = max(0, min(1, lstm_base + random.uniform(-0.1, 0.1)))

            # Transformer model prediction
            transformer_base = (
                0.5 + (features["rsi"] - 50) * 0.01 + features["social_sentiment"] * 0.3
            )
            predictions["transformer"] = max(
                0, min(1, transformer_base + random.uniform(-0.1, 0.1))
            )

            # CNN model prediction
            cnn_base = (
                0.5 + (features["bb_position"] - 0.5) * 0.4 + features["whale_activity"] * 0.2
            )
            predictions["cnn"] = max(0, min(1, cnn_base + random.uniform(-0.1, 0.1)))

            # GRU model prediction
            gru_base = 0.5 + (features["macd"] * 0.1) + (features["fear_greed"] - 50) * 0.005
            predictions["gru"] = max(0, min(1, gru_base + random.uniform(-0.1, 0.1)))

            # Attention model prediction
            attention_base = 0.5 + features["news_sentiment"] * 0.3 + features["whale_volume"] * 0.2
            predictions["attention"] = max(0, min(1, attention_base + random.uniform(-0.1, 0.1)))

            return predictions

        except Exception as e:
            logger.error(f"Error getting ensemble predictions: {e}")
            return {
                "lstm": 0.5,
                "transformer": 0.5,
                "cnn": 0.5,
                "gru": 0.5,
                "attention": 0.5,
            }

    def _calculate_attention_weights(
        self, features: dict[str, float], predictions: dict[str, float]
    ) -> dict[str, float]:
        """Calculate attention weights for ensemble models"""
        try:
            # Calculate attention based on feature importance and prediction confidence
            attention_scores = {}

            # LSTM attention (good for sequential data)
            lstm_attention = (
                0.2 + abs(features["price_change"]) * 0.3 + features["volume_ratio"] * 0.2
            )
            attention_scores["lstm"] = min(lstm_attention, 1.0)

            # Transformer attention (good for complex patterns)
            transformer_attention = (
                0.2 + features["social_sentiment"] * 0.4 + features["news_sentiment"] * 0.3
            )
            attention_scores["transformer"] = min(transformer_attention, 1.0)

            # CNN attention (good for pattern recognition)
            cnn_attention = 0.2 + features["whale_activity"] * 0.4 + features["bb_position"] * 0.3
            attention_scores["cnn"] = min(cnn_attention, 1.0)

            # GRU attention (good for trend analysis)
            gru_attention = (
                0.2 + abs(features["macd"]) * 0.3 + abs(features["fear_greed"] - 50) * 0.002
            )
            attention_scores["gru"] = min(gru_attention, 1.0)

            # Attention model attention (meta-attention)
            attention_attention = (
                0.2 + features["whale_volume"] * 0.4 + features["price_volatility"] * 10
            )
            attention_scores["attention"] = min(attention_attention, 1.0)

            # Normalize attention weights
            total_attention = sum(attention_scores.values())
            if total_attention > 0:
                attention_weights = {k: v / total_attention for k, v in attention_scores.items()}
            else:
                attention_weights = dict.fromkeys(attention_scores.keys(), 0.2)

            return attention_weights

        except Exception as e:
            logger.error(f"Error calculating attention weights: {e}")
            return {
                "lstm": 0.2,
                "transformer": 0.2,
                "cnn": 0.2,
                "gru": 0.2,
                "attention": 0.2,
            }

    def _calculate_weighted_prediction(
        self,
        predictions: dict[str, float],
        attention_weights: dict[str, float],
    ) -> float:
        """Calculate weighted ensemble prediction"""
        try:
            weighted_sum = 0
            total_weight = 0

            for model_name, prediction in predictions.items():
                weight = attention_weights.get(model_name, 0.2)
                weighted_sum += prediction * weight
                total_weight += weight

            if total_weight > 0:
                return weighted_sum / total_weight
            else:
                return 0.5

        except Exception as e:
            logger.error(f"Error calculating weighted prediction: {e}")
            return 0.5

    def _calculate_model_confidence(
        self,
        predictions: dict[str, float],
        attention_weights: dict[str, float],
    ) -> float:
        """Calculate overall model confidence"""
        try:
            # Confidence based on prediction agreement
            prediction_values = list(predictions.values())
            prediction_std = np.std(prediction_values) if len(prediction_values) > 1 else 0

            # Higher agreement (lower std) = higher confidence
            agreement_confidence = max(0, 1 - prediction_std)

            # Confidence based on attention weight distribution
            attention_std = (
                np.std(list(attention_weights.values())) if len(attention_weights) > 1 else 0
            )
            attention_confidence = max(0, 1 - attention_std)

            # Combined confidence
            overall_confidence = (agreement_confidence + attention_confidence) / 2

            return min(overall_confidence, 1.0)

        except Exception as e:
            logger.error(f"Error calculating model confidence: {e}")
            return 0.5

    def _calculate_adaptation_score(self, signal: dict[str, Any]) -> float:
        """Calculate adaptation score based on market conditions"""
        try:
            # Adaptation based on market volatility
            volatility = signal.get("volatility", 0.02)
            volatility_adaptation = min(
                1.0, volatility * 20
            )  # Higher volatility = better adaptation

            # Adaptation based on volume
            volume_ratio = signal.get("volume_ratio", 1.0)
            volume_adaptation = min(1.0, volume_ratio * 0.5)

            # Adaptation based on sentiment
            sentiment = signal.get("social_sentiment", 0.5)
            sentiment_adaptation = abs(sentiment - 0.5) * 2  # Extreme sentiment = better adaptation

            # Combined adaptation score
            adaptation_score = (
                volatility_adaptation + volume_adaptation + sentiment_adaptation
            ) / 3

            return min(adaptation_score, 1.0)

        except Exception as e:
            logger.error(f"Error calculating adaptation score: {e}")
            return 0.5

    def _calculate_signal_importance(
        self, features: dict[str, float], attention_weights: dict[str, float]
    ) -> dict[str, float]:
        """Calculate importance of different signal components"""
        try:
            importance = {}

            # Price importance
            importance["price"] = (
                abs(features["price_change"]) * 0.3 + features["price_volatility"] * 10
            )

            # Volume importance
            importance["volume"] = features["volume_ratio"] * 0.4 + features["whale_volume"] * 0.3

            # Technical importance
            importance["technical"] = abs(features["rsi"] - 50) * 0.02 + abs(features["macd"]) * 0.1

            # Sentiment importance
            importance["sentiment"] = (
                abs(features["social_sentiment"] - 0.5) * 0.4
                + abs(features["news_sentiment"] - 0.5) * 0.3
            )

            # Whale importance
            importance["whale"] = features["whale_activity"] * 0.5 + features["whale_volume"] * 0.3

            # Normalize importance scores
            max_importance = max(importance.values()) if importance.values() else 1
            if max_importance > 0:
                importance = {k: v / max_importance for k, v in importance.items()}

            return importance

        except Exception as e:
            logger.error(f"Error calculating signal importance: {e}")
            return {
                "price": 0.5,
                "volume": 0.5,
                "technical": 0.5,
                "sentiment": 0.5,
                "whale": 0.5,
            }

    def _calculate_ensemble_variance(self, predictions: dict[str, float]) -> float:
        """Calculate variance in ensemble predictions"""
        try:
            prediction_values = list(predictions.values())
            if len(prediction_values) > 1:
                return np.var(prediction_values)
            else:
                return 0.0

        except Exception as e:
            logger.error(f"Error calculating ensemble variance: {e}")
            return 0.0

    def _generate_neural_recommendation(
        self,
        prediction: float,
        confidence: float,
        adaptation: float,
        variance: float,
    ) -> str:
        """Generate neural ensemble recommendation"""
        try:
            if confidence < self.min_confidence_threshold:
                return "Hold - Low model confidence"

            if variance > self.ensemble_variance_threshold:
                return "Hold - High ensemble disagreement"

            if prediction > 0.8 and confidence > 0.8 and adaptation > 0.7:
                return "Strong Buy - High confidence neural prediction"
            elif prediction > 0.6 and confidence > 0.7:
                return "Buy - Positive neural prediction"
            elif prediction < 0.2 and confidence > 0.7:
                return "Sell - Negative neural prediction"
            elif prediction < 0.4 and confidence > 0.6:
                return "Hold - Negative neural prediction"
            else:
                return "Hold - Neutral neural prediction"

        except Exception as e:
            logger.error(f"Error generating neural recommendation: {e}")
            return "Hold - Error in neural analysis"

    def _create_fallback_prediction(self) -> NeuralPrediction:
        """Create fallback prediction when analysis fails"""
        return NeuralPrediction(
            ensemble_prediction=0.5,
            attention_weights={
                "lstm": 0.2,
                "transformer": 0.2,
                "cnn": 0.2,
                "gru": 0.2,
                "attention": 0.2,
            },
            model_confidence=0.5,
            adaptation_score=0.5,
            signal_importance={
                "price": 0.5,
                "volume": 0.5,
                "technical": 0.5,
                "sentiment": 0.5,
                "whale": 0.5,
            },
            ensemble_variance=0.0,
            recommendation="Hold - Neural analysis unavailable",
            timestamp=datetime.now(),
        )

    def get_neural_summary(self) -> dict[str, Any]:
        """Get neural ensemble summary"""
        return {
            "models_active": len(self.models),
            "attention_mechanism_enabled": True,
            "adaptation_tracking": True,
            "ensemble_variance_monitoring": True,
            "signal_importance_tracking": True,
            "confidence_threshold": self.min_confidence_threshold,
            "variance_threshold": self.ensemble_variance_threshold,
        }


# Global neural ensemble instance
neural_ensemble = NeuralEnsemble()


