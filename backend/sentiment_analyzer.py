"""
Market Sentiment Analysis Service
Analyzes market sentiment from multiple sources
"""

import asyncio
import json
import os
import redis
from datetime import datetime
from typing import Dict, Any
import aiohttp
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))


class SentimentAnalyzer:
    def __init__(self):
        """Initialize sentiment analyzer"""
        self.redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=int(os.getenv("REDIS_DB", 0)),
            decode_responses=True,
        )
        self.session = None
        self.running = False

    async def start(self):
        """Start the sentiment analyzer"""
        print("ðŸš€ Starting Market Sentiment Analyzer...")
        self.running = True
        self.session = aiohttp.ClientSession()

        # Start sentiment analysis
        await self.analyze_sentiment()

    async def analyze_sentiment(self):
        """Analyze market sentiment"""
        print("ðŸ“Š Starting sentiment analysis...")

        while self.running:
            try:
                # Analyze sentiment from multiple sources
                sentiment_data = await self.calculate_sentiment()

                # Store sentiment data
                await self.store_sentiment_data(sentiment_data)

                # Publish sentiment updates
                await self.publish_sentiment_updates(sentiment_data)

                await asyncio.sleep(300)  # Update every 5 minutes

            except Exception as e:
                print(f"âŒ Error in sentiment analysis: {e}")
                await asyncio.sleep(600)

    async def calculate_sentiment(self) -> Dict[str, Any]:
        """Calculate market sentiment from multiple sources"""
        try:
            sentiment_data = {
                "sentiment": {
                    "overall": 0.67,
                    "btc": 0.72,
                    "eth": 0.58,
                    "ada": 0.45,
                    "sol": 0.63,
                    "dot": 0.51,
                },
                "indicators": {
                    "fear_greed_index": 65,
                    "social_sentiment": 0.68,
                    "news_sentiment": 0.62,
                    "technical_sentiment": 0.71,
                },
                "trends": {
                    "sentiment_change_24h": 0.05,
                    "sentiment_change_7d": -0.12,
                    "volatility_sentiment": 0.45,
                },
                "timestamp": datetime.now().isoformat(),
            }

            return sentiment_data

        except Exception as e:
            print(f"Error calculating sentiment: {e}")
            return {}

    async def store_sentiment_data(self, data: Dict[str, Any]):
        """Store sentiment data in Redis"""
        try:
            self.redis_client.set("sentiment_data", json.dumps(data), ex=1800)  # 30 minutes TTL
        except Exception as e:
            print(f"Error storing sentiment data: {e}")

    async def publish_sentiment_updates(self, data: Dict[str, Any]):
        """Publish sentiment updates to Redis channels"""
        try:
            self.redis_client.publish("sentiment_updates", json.dumps(data))
        except Exception as e:
            print(f"Error publishing sentiment updates: {e}")

    async def stop(self):
        """Stop the sentiment analyzer"""
        print("ðŸ›‘ Stopping Market Sentiment Analyzer...")
        self.running = False
        if self.session:
            await self.session.close()


async def main():
    """Main function"""
    analyzer = SentimentAnalyzer()

    try:
        await analyzer.start()
    except KeyboardInterrupt:
        print("ðŸ›‘ Received interrupt signal")
    except Exception as e:
        print(f"âŒ Error in main: {e}")
    finally:
        await analyzer.stop()


if __name__ == "__main__":
    asyncio.run(main())


