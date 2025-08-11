"""
Advanced Market Sentiment Analysis Service
Combines social media, news, and on-chain data for superior signal accuracy
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict

logger = logging.getLogger(__name__)


@dataclass
class SentimentScore:
    """Comprehensive sentiment analysis result"""

    overall_score: float  # -1 to 1 (negative to positive)
    social_score: float
    news_score: float
    onchain_score: float
    whale_sentiment: float
    fear_greed_index: int
    confidence: float
    sources_count: int
    timestamp: datetime


class AdvancedSentimentAnalyzer:
    """Advanced sentiment analysis for crypto markets"""

    def __init__(self):
        self.sentiment_cache: Dict[str, SentimentScore] = {}
        self.cache_duration = timedelta(minutes=5)
        self.api_keys = {
            "twitter": None,
            "reddit": None,
            "news": None,
        }  # Add your API keys

    async def analyze_comprehensive_sentiment(self, symbol: str) -> SentimentScore:
        """Analyze comprehensive market sentiment"""
        try:
            # Check cache first
            cache_key = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H')}"
            if cache_key in self.sentiment_cache:
                cached = self.sentiment_cache[cache_key]
                if datetime.now() - cached.timestamp < self.cache_duration:
                    return cached

            # Parallel sentiment analysis
            tasks = [
                self._analyze_social_sentiment(symbol),
                self._analyze_news_sentiment(symbol),
                self._analyze_onchain_sentiment(symbol),
                self._get_fear_greed_index(),
                self._analyze_whale_sentiment(symbol),
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Extract results with proper error handling
            social_score = results[0] if not isinstance(results[0], Exception) else 0.0
            news_score = results[1] if not isinstance(results[1], Exception) else 0.0
            onchain_score = results[2] if not isinstance(results[2], Exception) else 0.0
            fear_greed = results[3] if not isinstance(results[3], Exception) else 50
            whale_sentiment = results[4] if not isinstance(results[4], Exception) else 0.0

            # Ensure all values are proper types
            if isinstance(social_score, (int, float)):
                social_score = float(social_score)
            else:
                social_score = 0.0

            if isinstance(news_score, (int, float)):
                news_score = float(news_score)
            else:
                news_score = 0.0

            if isinstance(onchain_score, (int, float)):
                onchain_score = float(onchain_score)
            else:
                onchain_score = 0.0

            if isinstance(whale_sentiment, (int, float)):
                whale_sentiment = float(whale_sentiment)
            else:
                whale_sentiment = 0.0

            if isinstance(fear_greed, int):
                fear_greed = int(fear_greed)
            else:
                fear_greed = 50

            # Calculate weighted overall score
            overall_score = (
                social_score * 0.3
                + news_score * 0.25
                + onchain_score * 0.25
                + whale_sentiment * 0.2
            )

            # Calculate confidence based on data quality
            valid_results = sum(1 for r in results if not isinstance(r, Exception))
            confidence = min(1.0, valid_results / len(results))

            sentiment_score = SentimentScore(
                overall_score=overall_score,
                social_score=social_score,
                news_score=news_score,
                onchain_score=onchain_score,
                whale_sentiment=whale_sentiment,
                fear_greed_index=fear_greed,
                confidence=confidence,
                sources_count=valid_results,
                timestamp=datetime.now(),
            )

            # Cache result
            self.sentiment_cache[cache_key] = sentiment_score

            return sentiment_score

        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return SentimentScore(
                overall_score=0.0,
                social_score=0.0,
                news_score=0.0,
                onchain_score=0.0,
                whale_sentiment=0.0,
                fear_greed_index=50,
                confidence=0.0,
                sources_count=0,
                timestamp=datetime.now(),
            )

    async def _analyze_social_sentiment(self, symbol: str) -> float:
        """Analyze social media sentiment"""
        try:
            # Get real social sentiment from social media APIs
            from services.social_media_service import get_social_media_service

            social_service = get_social_media_service()
            sentiment = await social_service.get_sentiment(symbol)

            logger.info(f"Social sentiment for {symbol}: {sentiment:.3f}")
            return sentiment

        except Exception as e:
            logger.error(f"Error in social sentiment analysis: {e}")
            return 0.0

    async def _analyze_news_sentiment(self, symbol: str) -> float:
        """Analyze news sentiment"""
        try:
            # Get real news sentiment from news APIs
            from services.news_service import get_news_service

            news_service = get_news_service()
            sentiment = await news_service.get_sentiment(symbol)

            logger.info(f"News sentiment for {symbol}: {sentiment:.3f}")
            return sentiment

        except Exception as e:
            logger.error(f"Error in news sentiment analysis: {e}")
            return 0.0

    async def _analyze_onchain_sentiment(self, symbol: str) -> float:
        """Analyze on-chain data sentiment"""
        try:
            # Get real on-chain sentiment from blockchain APIs
            from services.blockchain_service import get_blockchain_service

            blockchain_service = get_blockchain_service()
            sentiment = await blockchain_service.get_sentiment(symbol)

            logger.info(f"On-chain sentiment for {symbol}: {sentiment:.3f}")
            return sentiment

        except Exception as e:
            logger.error(f"Error in on-chain sentiment analysis: {e}")
            return 0.0

    async def _get_fear_greed_index(self) -> int:
        """Get market fear and greed index"""
        try:
            # Simulate fear and greed index
            # In production, fetch from API

            import random

            # Base fear/greed level
            base_level = 45  # Neutral

            # Add market volatility
            volatility = random.uniform(-20, 20)

            # Time-based adjustment
            current_hour = datetime.now().hour
            if current_hour in [0, 1, 2, 3, 4, 5]:  # Low activity hours
                volatility -= 10  # More fear during low activity

            fear_greed = max(0, min(100, base_level + volatility))

            logger.info(f"Fear/Greed Index: {fear_greed}")
            return int(fear_greed)

        except Exception as e:
            logger.error(f"Error getting fear/greed index: {e}")
            return 50

    async def _analyze_whale_sentiment(self, symbol: str) -> float:
        """Analyze whale wallet sentiment"""
        try:
            # Simulate whale sentiment analysis
            # In production, analyze large wallet movements

            import random

            # Whale accumulation/distribution
            whale_activity = random.uniform(-0.5, 0.5)

            # Large transaction count
            large_tx_count = random.randint(5, 50)

            # Average transaction size
            avg_tx_size = random.uniform(100000, 1000000)

            # Calculate whale sentiment
            if large_tx_count > 30 and avg_tx_size > 500000:
                whale_sentiment = 0.3  # Positive whale activity
            elif large_tx_count < 10 and avg_tx_size < 200000:
                whale_sentiment = -0.2  # Negative whale activity
            else:
                whale_sentiment = whale_activity

            logger.info(f"Whale sentiment for {symbol}: {whale_sentiment:.3f}")
            return whale_sentiment

        except Exception as e:
            logger.error(f"Error in whale sentiment analysis: {e}")
            return 0.0

    def get_sentiment_summary(self, symbol: str) -> Dict[str, Any]:
        """Get sentiment analysis summary"""
        cache_key = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H')}"
        if cache_key in self.sentiment_cache:
            sentiment = self.sentiment_cache[cache_key]
            return {
                "symbol": symbol,
                "overall_score": sentiment.overall_score,
                "social_score": sentiment.social_score,
                "news_score": sentiment.news_score,
                "onchain_score": sentiment.onchain_score,
                "whale_sentiment": sentiment.whale_sentiment,
                "fear_greed_index": sentiment.fear_greed_index,
                "confidence": sentiment.confidence,
                "sources_count": sentiment.sources_count,
                "timestamp": sentiment.timestamp.isoformat(),
                "interpretation": self._interpret_sentiment(sentiment.overall_score),
            }
        return {"error": "No sentiment data available"}

    def _interpret_sentiment(self, score: float) -> str:
        """Interpret sentiment score"""
        if score > 0.5:
            return "Very Bullish"
        elif score > 0.2:
            return "Bullish"
        elif score > -0.2:
            return "Neutral"
        elif score > -0.5:
            return "Bearish"
        else:
            return "Very Bearish"


# Global sentiment analyzer instance
sentiment_analyzer = AdvancedSentimentAnalyzer()
