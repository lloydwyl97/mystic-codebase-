# sentiment_monitor.py
"""
Sentiment AI Feed Parser - Real-time Crypto News Sentiment Analysis
Monitors crypto news feeds and analyzes sentiment for trading signals.
Built for Windows 11 Home + PowerShell + Docker.
"""

import requests
import time
import json
import os
import logging
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
NEWS_FEED = "https://cryptopanic.com/api/v1/posts/?auth_token=demo&public=true"
INTERVAL = 600  # 10 minutes
PING_FILE = "./logs/sentiment_monitor.ping"
SENTIMENT_THRESHOLD = 0.3

# Ensure logs directory exists
os.makedirs("./logs", exist_ok=True)

# Initialize sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()
    VADER_AVAILABLE = True
except ImportError:
    logger.warning("vaderSentiment not available, using simple sentiment analysis")
    VADER_AVAILABLE = False


def simple_sentiment_analysis(text: str) -> float:
    """Simple sentiment analysis using keyword matching."""
    positive_words = [
        "bull",
        "bullish",
        "moon",
        "pump",
        "surge",
        "rally",
        "gain",
        "profit",
    ]
    negative_words = [
        "bear",
        "bearish",
        "crash",
        "dump",
        "drop",
        "fall",
        "loss",
        "sell",
    ]

    text_lower = text.lower()
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)

    if positive_count == 0 and negative_count == 0:
        return 0.0

    return (positive_count - negative_count) / (positive_count + negative_count)


def create_ping_file(sentiment_score, headline_count):
    """Create ping file for dashboard monitoring"""
    try:
        with open(PING_FILE, "w") as f:
            json.dump(
                {
                    "status": "online",
                    "last_update": datetime.timezone.utcnow().isoformat(),
                    "sentiment_score": sentiment_score,
                    "headline_count": headline_count,
                },
                f,
            )
    except Exception as e:
        logger.error(f"Ping file error: {e}")


def fetch_headlines():
    """Fetch crypto news headlines from multiple sources"""
    headlines = []

    try:
        # Primary source: CryptoPanic
        response = requests.get(NEWS_FEED, timeout=10)
        if response.status_code == 200:
            data = response.json()
            headlines.extend([x["title"] for x in data.get("results", [])])

        # Secondary source: CoinGecko news (if available)
        try:
            cg_response = requests.get("https://api.coingecko.com/api/v3/news", timeout=5)
            if cg_response.status_code == 200:
                cg_data = cg_response.json()
                headlines.extend([item.get("title", "") for item in cg_data.get("data", [])])
        except Exception as cg_error:
            logger.warning(f"CoinGecko news fetch failed: {cg_error}")
            pass

    except Exception as e:
        logger.error(f"Headline fetch error: {e}")

    return headlines


def analyze_sentiment():
    """Analyze sentiment of crypto news headlines"""
    try:
        headlines = fetch_headlines()

        if not headlines:
            logger.warning("No headlines fetched")
            create_ping_file(0, 0)
            return

        # Analyze each headline
        sentiments = []
        for headline in headlines:
            if headline and len(headline.strip()) > 10:  # Filter out empty/short headlines
                scores = analyzer.polarity_scores(headline)
                sentiments.append(scores["compound"])

        if sentiments:
            avg_sentiment = sum(sentiments) / len(sentiments)
            logger.info(f"ðŸ“Š Sentiment: {avg_sentiment:.3f} ({len(headlines)} headlines)")
            logger.info(f"ðŸ“Š Positive headlines: {len([s for s in sentiments if s > 0.1])}")
            logger.info(f"ðŸ“Š Negative headlines: {len([s for s in sentiments if s < -0.1])}")

            # Create ping file for dashboard
            create_ping_file(avg_sentiment, len(headlines))

            # Log sentiment data
            sentiment_log = {
                "timestamp": datetime.timezone.utcnow().isoformat(),
                "sentiment_score": avg_sentiment,
                "headline_count": len(headlines),
                "positive_count": len([s for s in sentiments if s > 0.1]),
                "negative_count": len([s for s in sentiments if s < -0.1]),
            }

            with open("./logs/sentiment_log.jsonl", "a") as f:
                f.write(json.dumps(sentiment_log) + "\n")

        else:
            logger.warning("No valid headlines to analyze")
            create_ping_file(0, 0)

    except Exception as e:
        logger.error(f"[Sentiment] Analysis error: {e}")
        create_ping_file(0, 0)


def main():
    """Main execution loop"""
    logger.info("ðŸš€ Sentiment Monitor starting...")
    logger.info(f"â° Analysis interval: {INTERVAL} seconds")

    while True:
        try:
            analyze_sentiment()
        except KeyboardInterrupt:
            logger.info("ðŸ›‘ Sentiment monitor stopped")
            break
        except Exception as e:
            logger.error(f"âŒ Main loop error: {e}")

        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()


