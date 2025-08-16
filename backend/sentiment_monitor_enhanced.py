import requests
import json
import sqlite3
import asyncio
import aiohttp
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime
from typing import Dict, List
import re

# Enhanced configuration
NEWS_SOURCES = {
    "cryptopanic": ("https://cryptopanic.com/api/v1/posts/?auth_token=demo&public=true"),
    "coingecko": "https://api.coingecko.com/api/v3/news",
    "twitter": ("https://api.twitter.com/2/tweets/search/recent?query=crypto"),  # Placeholder
    "reddit": "https://www.reddit.com/r/cryptocurrency/hot.json",
}

INTERVAL = 600
SENTIMENT_DB = "./data/sentiment_history.db"
ALERT_THRESHOLD = 0.5  # Extreme sentiment threshold

analyzer = SentimentIntensityAnalyzer()


class SentimentDatabase:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize sentiment database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS sentiment_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                source TEXT NOT NULL,
                sentiment_score REAL NOT NULL,
                headline_count INTEGER NOT NULL,
                positive_count INTEGER NOT NULL,
                negative_count INTEGER NOT NULL,
                neutral_count INTEGER NOT NULL,
                market_data TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS sentiment_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                sentiment_score REAL NOT NULL,
                message TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        conn.commit()
        conn.close()

    def save_sentiment(self, data: Dict):
        """Save sentiment data to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO sentiment_data
            (timestamp, source, sentiment_score, headline_count, positive_count, negative_count, neutral_count, market_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                data["timestamp"],
                data["source"],
                data["sentiment_score"],
                data["headline_count"],
                data["positive_count"],
                data["negative_count"],
                data["neutral_count"],
                json.dumps(data.get("market_data", {})),
            ),
        )

        conn.commit()
        conn.close()

    def save_alert(self, alert_type: str, sentiment_score: float, message: str):
        """Save sentiment alert"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO sentiment_alerts (timestamp, alert_type, sentiment_score, message)
            VALUES (?, ?, ?, ?)
        """,
            (
                datetime.timezone.utcnow().isoformat(),
                alert_type,
                sentiment_score,
                message,
            ),
        )

        conn.commit()
        conn.close()


async def get_market_data() -> Dict:
    """Get current market data for correlation"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("https://api.binance.us/api/v3/ticker/24hr", timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    data = await response.json()
                    market_data = {}
                    for item in data:
                        if item["symbol"] in ["BTCUSDT", "ETHUSDT"]:
                            market_data[item["symbol"]] = {
                                "price": float(item["lastPrice"]),
                                "change": float(item["priceChangePercent"]),
                                "volume": float(item["volume"]),
                            }
                    return market_data
    except Exception as e:
        print(f"Market data fetch error: {e}")
    return {}


def fetch_headlines_enhanced() -> Dict[str, List[str]]:
    """Fetch headlines from multiple sources"""
    headlines = {}

    # CryptoPanic
    try:
        response = requests.get(NEWS_SOURCES["cryptopanic"], timeout=10)
        if response.status_code == 200:
            data = response.json()
            headlines["cryptopanic"] = [x["title"] for x in data.get("results", [])]
    except Exception as e:
        print(f"CryptoPanic fetch error: {e}")
        headlines["cryptopanic"] = []

    # CoinGecko
    try:
        response = requests.get(NEWS_SOURCES["coingecko"], timeout=10)
        if response.status_code == 200:
            data = response.json()
            headlines["coingecko"] = [item.get("title", "") for item in data.get("data", [])]
    except Exception as e:
        print(f"CoinGecko fetch error: {e}")
        headlines["coingecko"] = []

    # Reddit (simplified)
    try:
        response = requests.get(NEWS_SOURCES["reddit"], timeout=10)
        if response.status_code == 200:
            data = response.json()
            headlines["reddit"] = [
                post["data"]["title"] for post in data.get("data", {}).get("children", [])
            ]
    except Exception as e:
        print(f"Reddit fetch error: {e}")
        headlines["reddit"] = []

    return headlines


def detect_language(text: str) -> str:
    """Simple language detection"""
    # Basic language detection using character patterns
    if re.search(r"[Ð°-ÑÑ‘]", text, re.IGNORECASE):
        return "russian"
    elif re.search(r"[ä¸€-é¾¯]", text):
        return "chinese"
    elif re.search(r"[ê°€-íž£]", text):
        return "korean"
    elif re.search(r"[ã‚-ã‚“]", text):
        return "japanese"
    else:
        return "english"


def translate_sentiment_keywords(text: str, language: str) -> str:
    """Translate common sentiment keywords for better analysis"""
    translations = {
        "russian": {
            "bull": "Ð±Ñ‹Ðº",
            "bear": "Ð¼ÐµÐ´Ð²ÐµÐ´ÑŒ",
            "moon": "Ð»ÑƒÐ½Ð°",
            "pump": "Ð½Ð°ÑÐ¾Ñ",
            "dump": "ÑÐ²Ð°Ð»ÐºÐ°",
            "crash": "ÐºÑ€Ð°Ñ…",
            "rally": "Ñ€Ð°Ð»Ð»Ð¸",
        },
        "chinese": {
            "bull": "ç‰›å¸‚",
            "bear": "ç†Šå¸‚",
            "moon": "æœˆäº®",
            "pump": "æ³µ",
            "dump": "å€¾å€’",
            "crash": "å´©æºƒ",
            "rally": "é›†ä¼š",
        },
    }

    if language in translations:
        for eng, trans in translations[language].items():
            text = text.replace(trans, eng)

    return text


async def analyze_sentiment_enhanced():
    """Enhanced sentiment analysis with all features"""
    try:
        # Fetch headlines from multiple sources
        all_headlines = fetch_headlines_enhanced()
        market_data = await get_market_data()

        total_sentiment = 0
        total_headlines = 0
        source_results = {}

        for source, headlines in all_headlines.items():
            if not headlines:
                continue

            source_sentiments = []
            positive_count = 0
            negative_count = 0
            neutral_count = 0

            for headline in headlines:
                if not headline or len(headline.strip()) < 10:
                    continue

                # Language detection and translation
                language = detect_language(headline)
                processed_headline = translate_sentiment_keywords(headline, language)

                # Analyze sentiment
                scores = analyzer.polarity_scores(processed_headline)
                compound_score = scores["compound"]
                source_sentiments.append(compound_score)

                # Categorize
                if compound_score > 0.1:
                    positive_count += 1
                elif compound_score < -0.1:
                    negative_count += 1
                else:
                    neutral_count += 1

            if source_sentiments:
                avg_sentiment = sum(source_sentiments) / len(source_sentiments)
                source_results[source] = {
                    "sentiment_score": avg_sentiment,
                    "headline_count": len(headlines),
                    "positive_count": positive_count,
                    "negative_count": negative_count,
                    "neutral_count": neutral_count,
                }

                total_sentiment += avg_sentiment * len(headlines)
                total_headlines += len(headlines)

        # Calculate overall sentiment
        overall_sentiment = total_sentiment / total_headlines if total_headlines > 0 else 0

        # Prepare data for database
        sentiment_data = {
            "timestamp": datetime.timezone.utcnow().isoformat(),
            "source": "multi_source",
            "sentiment_score": overall_sentiment,
            "headline_count": total_headlines,
            "positive_count": sum(r["positive_count"] for r in source_results.values()),
            "negative_count": sum(r["negative_count"] for r in source_results.values()),
            "neutral_count": sum(r["neutral_count"] for r in source_results.values()),
            "market_data": market_data,
        }

        # Save to database
        db = SentimentDatabase(SENTIMENT_DB)
        db.save_sentiment(sentiment_data)

        # Check for alerts
        if abs(overall_sentiment) > ALERT_THRESHOLD:
            alert_type = "extreme_positive" if overall_sentiment > 0 else "extreme_negative"
            message = (
                f"Extreme sentiment detected: {overall_sentiment:.3f} ({total_headlines} headlines)"
            )
            db.save_alert(alert_type, overall_sentiment, message)
            print(f"[ALERT] {message}")

        # Print results
        print(f"[Sentiment] Overall Score: {overall_sentiment:.3f} ({total_headlines} headlines)")
        for source, results in source_results.items():
            print(
                f"[Sentiment] {source}: {results['sentiment_score']:.3f} ({results['headline_count']} headlines)"
            )

        # Sentiment correlation analysis
        if market_data:
            btc_change = market_data.get("BTCUSDT", {}).get("change", 0)
            correlation_message = (
                "positive"
                if (overall_sentiment > 0 and btc_change > 0)
                or (overall_sentiment < 0 and btc_change < 0)
                else "negative"
            )
            print(
                f"[Sentiment] BTC correlation: {correlation_message} (BTC: {btc_change:+.2f}%, Sentiment: {overall_sentiment:+.3f})"
            )

    except Exception as e:
        print(f"[Sentiment] Enhanced analysis error: {e}")


# Main execution loop
async def main():
    while True:
        await analyze_sentiment_enhanced()
        await asyncio.sleep(INTERVAL)

if __name__ == "__main__":
    asyncio.run(main())


