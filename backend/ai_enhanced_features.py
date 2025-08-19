"""
Enhanced AI Features for Mystic Trading Platform
Advanced AI capabilities with modern LLM integration
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass
import structlog
from datetime import datetime, timezone

# AI/ML imports
try:
    from transformers import pipeline
    from sentence_transformers import SentenceTransformer
    import torch
    from langchain_community.llms import OpenAI
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
except ImportError:
    logging.warning("Some AI libraries not available, using fallback implementations")

logger = structlog.get_logger()


@dataclass
class MarketSentiment:
    symbol: str
    sentiment_score: float  # -1 to 1
    confidence: float
    sources: List[str]
    timestamp: datetime
    news_count: int
    social_volume: int
    fear_greed_index: float


@dataclass
class AIPrediction:
    symbol: str
    prediction_type: str  # price_direction, volatility, volume
    predicted_value: float
    confidence: float
    timeframe: str
    model_version: str
    features_used: List[str]
    timestamp: datetime


@dataclass
class StrategyRecommendation:
    symbol: str
    action: str  # buy, sell, hold
    confidence: float
    reasoning: str
    risk_level: str
    expected_return: float
    time_horizon: str
    stop_loss: Optional[float]
    take_profit: Optional[float]
    timestamp: datetime


class EnhancedSentimentAnalyzer:
    """Advanced sentiment analysis using multiple AI models"""

    def __init__(self):
        self.sentiment_model = None
        self.embedding_model = None
        self.news_sources = [
            "reuters",
            "bloomberg",
            "cnbc",
            "marketwatch",
            "yahoo_finance",
            "seeking_alpha",
            "reddit",
            "twitter",
        ]

        try:
            # Initialize sentiment analysis model
            self.sentiment_model = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                device=0 if torch.cuda.is_available() else -1,
            )

            # Initialize embedding model for semantic analysis
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        except Exception as e:
            logger.warning(f"Could not initialize AI models: {e}")

    async def analyze_market_sentiment(
        self, symbol: str, news_data: List[Dict], social_data: List[Dict]
    ) -> MarketSentiment:
        """Analyze market sentiment from multiple sources"""

        try:
            # Analyze news sentiment
            news_sentiments = []
            for news in news_data[:50]:  # Limit to recent news
                if self.sentiment_model:
                    result = self.sentiment_model(
                        news.get("title", "") + " " + news.get("content", "")
                    )
                    sentiment_score = 1.0 if result[0]["label"] == "POSITIVE" else -1.0
                    sentiment_score *= result[0]["score"]
                    news_sentiments.append(sentiment_score)
                else:
                    # Fallback sentiment analysis
                    text = news.get("title", "") + " " + news.get("content", "")
                    sentiment_score = self._fallback_sentiment_analysis(text)
                    news_sentiments.append(sentiment_score)

            # Analyze social sentiment
            social_sentiments = []
            for post in social_data[:100]:  # Limit to recent posts
                if self.sentiment_model:
                    result = self.sentiment_model(post.get("text", ""))
                    sentiment_score = 1.0 if result[0]["label"] == "POSITIVE" else -1.0
                    sentiment_score *= result[0]["score"]
                    social_sentiments.append(sentiment_score)
                else:
                    sentiment_score = self._fallback_sentiment_analysis(post.get("text", ""))
                    social_sentiments.append(sentiment_score)

            # Calculate weighted sentiment
            news_weight = 0.7
            social_weight = 0.3

            avg_news_sentiment = np.mean(news_sentiments) if news_sentiments else 0.0
            avg_social_sentiment = np.mean(social_sentiments) if social_sentiments else 0.0

            overall_sentiment = (
                news_weight * avg_news_sentiment + social_weight * avg_social_sentiment
            )

            # Calculate confidence based on data volume and consistency
            confidence = min(1.0, (len(news_sentiments) + len(social_sentiments)) / 100)

            # Calculate fear/greed index
            fear_greed = self._calculate_fear_greed_index(overall_sentiment, confidence)

            return MarketSentiment(
                symbol=symbol,
                sentiment_score=overall_sentiment,
                confidence=confidence,
                sources=self.news_sources,
                timestamp=datetime.now(timezone.utc),
                news_count=len(news_data),
                social_volume=len(social_data),
                fear_greed_index=fear_greed,
            )

        except Exception as e:
            logger.error(f"Error analyzing sentiment for {symbol}: {e}")
            return MarketSentiment(
                symbol=symbol,
                sentiment_score=0.0,
                confidence=0.0,
                sources=[],
                timestamp=datetime.now(timezone.utc),
                news_count=0,
                social_volume=0,
                fear_greed_index=50.0,
            )

    def _fallback_sentiment_analysis(self, text: str) -> float:
        """Simple fallback sentiment analysis"""
        positive_words = [
            "bull",
            "bullish",
            "moon",
            "pump",
            "buy",
            "strong",
            "growth",
            "profit",
        ]
        negative_words = [
            "bear",
            "bearish",
            "dump",
            "sell",
            "weak",
            "loss",
            "crash",
            "decline",
        ]

        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        if positive_count == 0 and negative_count == 0:
            return 0.0

        return (positive_count - negative_count) / (positive_count + negative_count)

    def _calculate_fear_greed_index(self, sentiment: float, confidence: float) -> float:
        """Calculate fear/greed index from sentiment"""
        # Convert sentiment (-1 to 1) to fear/greed (0 to 100)
        fear_greed = 50 + (sentiment * 50 * confidence)
        return max(0, min(100, fear_greed))


class AdvancedPredictor:
    """Advanced price prediction using multiple AI models"""

    def __init__(self):
        self.models = {}
        self.feature_importance = {}

    async def predict_price_direction(
        self,
        symbol: str,
        historical_data: pd.DataFrame,
        sentiment_data: MarketSentiment,
        technical_indicators: Dict[str, float],
    ) -> AIPrediction:
        """Predict price direction using ensemble of models"""

        try:
            # Prepare features
            features = self._prepare_features(historical_data, sentiment_data, technical_indicators)

            # Ensemble prediction
            predictions = []
            weights = []

            # LSTM prediction
            lstm_pred = self._lstm_prediction(historical_data)
            predictions.append(lstm_pred)
            weights.append(0.3)

            # Transformer prediction
            transformer_pred = self._transformer_prediction(features)
            predictions.append(transformer_pred)
            weights.append(0.3)

            # Sentiment-based prediction
            sentiment_pred = self._sentiment_prediction(sentiment_data)
            predictions.append(sentiment_pred)
            weights.append(0.2)

            # Technical prediction
            technical_pred = self._technical_prediction(technical_indicators)
            predictions.append(technical_pred)
            weights.append(0.2)

            # Weighted ensemble
            ensemble_prediction = np.average(predictions, weights=weights)

            # Calculate confidence
            confidence = self._calculate_prediction_confidence(predictions, weights)

            return AIPrediction(
                symbol=symbol,
                prediction_type="price_direction",
                predicted_value=ensemble_prediction,
                confidence=confidence,
                timeframe="1h",
                model_version="ensemble_v2",
                features_used=list(features.keys()),
                timestamp=datetime.now(timezone.utc),
            )

        except Exception as e:
            logger.error(f"Error predicting price direction for {symbol}: {e}")
            return AIPrediction(
                symbol=symbol,
                prediction_type="price_direction",
                predicted_value=0.0,
                confidence=0.0,
                timeframe="1h",
                model_version="fallback",
                features_used=[],
                timestamp=datetime.now(timezone.utc),
            )

    def _prepare_features(
        self,
        historical_data: pd.DataFrame,
        sentiment_data: MarketSentiment,
        technical_indicators: Dict[str, float],
    ) -> Dict[str, float]:
        """Prepare features for prediction"""

        features = {}

        # Price-based features
        if len(historical_data) > 0:
            features["price_change_1h"] = historical_data["close"].pct_change().iloc[-1]
            features["price_change_24h"] = historical_data["close"].pct_change(24).iloc[-1]
            features["volatility"] = historical_data["close"].pct_change().std()
            features["volume_ratio"] = (
                historical_data["volume"].iloc[-1] / historical_data["volume"].mean()
            )

        # Sentiment features
        features["sentiment_score"] = sentiment_data.sentiment_score
        features["fear_greed_index"] = sentiment_data.fear_greed_index
        features["news_volume"] = sentiment_data.news_count
        features["social_volume"] = sentiment_data.social_volume

        # Technical features
        features.update(technical_indicators)

        return features

    def _lstm_prediction(self, historical_data: pd.DataFrame) -> float:
        """LSTM-based price prediction"""
        # Simplified LSTM prediction
        if len(historical_data) < 10:
            return 0.0

        # Use recent price momentum as proxy
        recent_returns = historical_data["close"].pct_change().tail(10)
        return recent_returns.mean()

    def _transformer_prediction(self, features: Dict[str, float]) -> float:
        """Transformer-based prediction"""
        # Simplified transformer prediction
        sentiment_weight = 0.4
        technical_weight = 0.6

        sentiment_score = features.get("sentiment_score", 0.0)
        technical_score = features.get("rsi", 50) / 100 - 0.5  # Normalize RSI

        return sentiment_weight * sentiment_score + technical_weight * technical_score

    def _sentiment_prediction(self, sentiment_data: MarketSentiment) -> float:
        """Sentiment-based prediction"""
        return sentiment_data.sentiment_score * 0.5  # Scale down sentiment impact

    def _technical_prediction(self, technical_indicators: Dict[str, float]) -> float:
        """Technical indicator-based prediction"""
        # Combine multiple technical indicators
        rsi = technical_indicators.get("rsi", 50)
        macd = technical_indicators.get("macd", 0.0)
        bollinger_position = technical_indicators.get("bollinger_position", 0.5)

        # Normalize and combine
        rsi_signal = (rsi - 50) / 50  # -1 to 1
        macd_signal = np.tanh(macd)  # Normalize MACD
        bb_signal = (bollinger_position - 0.5) * 2  # -1 to 1

        return (rsi_signal + macd_signal + bb_signal) / 3

    def _calculate_prediction_confidence(
        self, predictions: List[float], weights: List[float]
    ) -> float:
        """Calculate confidence based on model agreement"""
        if not predictions:
            return 0.0

        # Calculate weighted variance
        weighted_mean = np.average(predictions, weights=weights)
        weighted_variance = np.average(
            [(p - weighted_mean) ** 2 for p in predictions], weights=weights
        )

        # Convert to confidence (lower variance = higher confidence)
        confidence = 1.0 / (1.0 + weighted_variance)
        return min(1.0, confidence)


class AIStrategyGenerator:
    """AI-powered strategy generation using LLMs"""

    def __init__(self):
        self.llm_chain = None
        self.strategy_templates = self._load_strategy_templates()

        try:
            # Initialize LLM chain
            prompt_template = PromptTemplate(
                input_variables=[
                    "market_data",
                    "sentiment",
                    "technical_indicators",
                    "risk_profile",
                ],
                template="""
                Based on the following market data, generate a trading strategy:

                Market Data: {market_data}
                Sentiment: {sentiment}
                Technical Indicators: {technical_indicators}
                Risk Profile: {risk_profile}

                Generate a detailed trading strategy with:
                1. Entry conditions
                2. Exit conditions
                3. Risk management rules
                4. Position sizing
                5. Expected otimezone.utcomes

                Strategy:
                """,
            )

            # Use OpenAI if available, otherwise use fallback
            try:
                self.llm_chain = LLMChain(llm=OpenAI(temperature=0.7), prompt=prompt_template)
            except (ImportError, ValueError, ConnectionError, Exception) as e:
                logger.warning(f"OpenAI not available ({e}), using fallback strategy generation")

        except Exception as e:
            logger.warning(f"Could not initialize LLM chain: {e}")

    def _load_strategy_templates(self) -> Dict[str, str]:
        """Load strategy templates"""
        return {
            "momentum": (
                """
            Strategy: Momentum Breakout
            Entry: Price breaks above resistance with high volume
            Exit: Stop loss at support level or take profit at 2:1 risk/reward
            Risk: 2% per trade
            """
            ),
            "mean_reversion": (
                """
            Strategy: Mean Reversion
            Entry: Price oversold (RSI < 30) or overbought (RSI > 70)
            Exit: Return to mean or opposite extreme
            Risk: 1.5% per trade
            """
            ),
            "trend_following": (
                """
            Strategy: Trend Following
            Entry: Price above moving averages with momentum
            Exit: Trend reversal or trailing stop
            Risk: 2.5% per trade
            """
            ),
        }

    async def generate_strategy(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        sentiment_data: MarketSentiment,
        technical_indicators: Dict[str, float],
        risk_profile: str = "moderate",
    ) -> StrategyRecommendation:
        """Generate AI-powered trading strategy"""

        try:
            # Prepare market data summary
            market_summary = self._summarize_market_data(market_data)
            sentiment_summary = f"Sentiment: {sentiment_data.sentiment_score:.2f}, Fear/Greed: {sentiment_data.fear_greed_index:.1f}"
            technical_summary = self._summarize_technical_indicators(technical_indicators)

            # Generate strategy using LLM or fallback
            if self.llm_chain:
                strategy_text = await self.llm_chain.arun(
                    market_data=market_summary,
                    sentiment=sentiment_summary,
                    technical_indicators=technical_summary,
                    risk_profile=risk_profile,
                )
            else:
                strategy_text = self._fallback_strategy_generation(
                    sentiment_data, technical_indicators, risk_profile
                )

            # Parse strategy and create recommendation
            recommendation = self._parse_strategy_to_recommendation(
                symbol, strategy_text, sentiment_data, technical_indicators
            )

            return recommendation

        except Exception as e:
            logger.error(f"Error generating strategy for {symbol}: {e}")
            return self._create_fallback_recommendation(symbol)

    def _summarize_market_data(self, market_data: Dict[str, Any]) -> str:
        """Summarize market data for LLM"""
        return f"Price: {market_data.get('current_price', 0):.2f}, Volume: {market_data.get('volume', 0)}, 24h Change: {market_data.get('change_24h', 0):.2%}"

    def _summarize_technical_indicators(self, indicators: Dict[str, float]) -> str:
        """Summarize technical indicators"""
        summary = []
        for name, value in indicators.items():
            summary.append(f"{name}: {value:.2f}")
        return ", ".join(summary)

    def _fallback_strategy_generation(
        self,
        sentiment_data: MarketSentiment,
        technical_indicators: Dict[str, float],
        risk_profile: str,
    ) -> str:
        """Fallback strategy generation"""

        # Simple rule-based strategy generation
        sentiment_score = sentiment_data.sentiment_score
        rsi = technical_indicators.get("rsi", 50)

        if sentiment_score > 0.3 and rsi < 70:
            return self.strategy_templates["momentum"]
        elif abs(sentiment_score) < 0.2 and (rsi < 30 or rsi > 70):
            return self.strategy_templates["mean_reversion"]
        else:
            return self.strategy_templates["trend_following"]

    def _parse_strategy_to_recommendation(
        self,
        symbol: str,
        strategy_text: str,
        sentiment_data: MarketSentiment,
        technical_indicators: Dict[str, float],
    ) -> StrategyRecommendation:
        """Parse strategy text to recommendation"""

        # Simple parsing logic
        if "buy" in strategy_text.lower() or "long" in strategy_text.lower():
            action = "buy"
            confidence = min(0.8, sentiment_data.confidence + 0.2)
        elif "sell" in strategy_text.lower() or "short" in strategy_text.lower():
            action = "sell"
            confidence = min(0.8, sentiment_data.confidence + 0.2)
        else:
            action = "hold"
            confidence = 0.5

        return StrategyRecommendation(
            symbol=symbol,
            action=action,
            confidence=confidence,
            reasoning=strategy_text,
            risk_level="moderate",
            expected_return=0.02 if action != "hold" else 0.0,
            time_horizon="1d",
            stop_loss=None,
            take_profit=None,
            timestamp=datetime.now(timezone.utc),
        )

    def _create_fallback_recommendation(self, symbol: str) -> StrategyRecommendation:
        """Create fallback recommendation"""
        return StrategyRecommendation(
            symbol=symbol,
            action="hold",
            confidence=0.0,
            reasoning="Insufficient data for recommendation",
            risk_level="low",
            expected_return=0.0,
            time_horizon="1d",
            stop_loss=None,
            take_profit=None,
            timestamp=datetime.now(timezone.utc),
        )


class EnhancedAITrading:
    """Main enhanced AI trading system"""

    def __init__(self):
        self.sentiment_analyzer = EnhancedSentimentAnalyzer()
        self.predictor = AdvancedPredictor()
        self.strategy_generator = AIStrategyGenerator()
        self.active_predictions: Dict[str, AIPrediction] = {}
        self.active_recommendations: Dict[str, StrategyRecommendation] = {}

    async def analyze_symbol(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        news_data: List[Dict],
        social_data: List[Dict],
        technical_indicators: Dict[str, float],
    ) -> Dict[str, Any]:
        """Complete AI analysis for a symbol"""

        try:
            # Sentiment analysis
            sentiment = await self.sentiment_analyzer.analyze_market_sentiment(
                symbol, news_data, social_data
            )

            # Price prediction
            historical_df = pd.DataFrame(market_data.get("historical", []))
            prediction = await self.predictor.predict_price_direction(
                symbol, historical_df, sentiment, technical_indicators
            )

            # Strategy generation
            recommendation = await self.strategy_generator.generate_strategy(
                symbol, market_data, sentiment, technical_indicators
            )

            # Store results
            self.active_predictions[symbol] = prediction
            self.active_recommendations[symbol] = recommendation

            return {
                "symbol": symbol,
                "sentiment": sentiment,
                "prediction": prediction,
                "recommendation": recommendation,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error(f"Error in AI analysis for {symbol}: {e}")
            return {
                "symbol": symbol,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    async def get_portfolio_insights(self, portfolio_symbols: List[str]) -> Dict[str, Any]:
        """Get AI insights for entire portfolio"""

        insights = {
            "overall_sentiment": 0.0,
            "risk_assessment": "moderate",
            "recommendations": [],
            "high_confidence_signals": [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Aggregate sentiment
        sentiments = []
        for symbol in portfolio_symbols:
            if symbol in self.active_predictions:
                pred = self.active_predictions[symbol]
                sentiments.append(pred.predicted_value)

        if sentiments:
            insights["overall_sentiment"] = np.mean(sentiments)

        # Risk assessment
        if insights["overall_sentiment"] < -0.3:
            insights["risk_assessment"] = "high"
        elif insights["overall_sentiment"] > 0.3:
            insights["risk_assessment"] = "low"

        # High confidence signals
        for symbol, rec in self.active_recommendations.items():
            if rec.confidence > 0.7:
                insights["high_confidence_signals"].append(
                    {
                        "symbol": symbol,
                        "action": rec.action,
                        "confidence": rec.confidence,
                        "reasoning": rec.reasoning,
                    }
                )

        return insights

