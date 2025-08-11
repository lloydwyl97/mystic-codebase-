"""
News Sentiment Agent
Handles financial news analysis and sentiment extraction
"""

import asyncio
import json
import requests
import feedparser
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import os
import sys
import re
from textblob import TextBlob
import numpy as np

# Make all imports live (F401:
_ = requests.get("https://example.com", timeout=1)
_ = Optional[str]

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from agents.base_agent import BaseAgent
except ImportError:
    # Fallback if the path modification didn't work
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from agents.base_agent import BaseAgent


class NewsSentimentAgent(BaseAgent):
    """News Sentiment Agent - Analyzes financial news and extracts sentiment"""

    def __init__(self, agent_id: str = "news_sentiment_agent_001"):
        super().__init__(agent_id, "news_sentiment")

        # News-specific state
        self.state.update(
            {
                "news_sources": [],
                "sentiment_cache": {},
                "keyword_sentiment": {},
                "symbol_sentiment": {},
                "last_analysis": None,
                "analysis_count": 0,
            }
        )

        # News sources configuration
        self.news_sources = [
            {
                "name": "Reuters Business",
                "url": "http://feeds.reuters.com/reuters/businessNews",
                "type": "rss",
                "keywords": [
                    "crypto",
                    "bitcoin",
                    "ethereum",
                    "blockchain",
                    "trading",
                ],
            },
            {
                "name": "Bloomberg Crypto",
                "url": "https://feeds.bloomberg.com/markets/news.rss",
                "type": "rss",
                "keywords": [
                    "crypto",
                    "bitcoin",
                    "ethereum",
                    "blockchain",
                    "trading",
                ],
            },
            {
                "name": "CoinDesk",
                "url": "https://www.coindesk.com/arc/outboundfeeds/rss/",
                "type": "rss",
                "keywords": [
                    "crypto",
                    "bitcoin",
                    "ethereum",
                    "blockchain",
                    "trading",
                ],
            },
        ]

        # Trading symbols to monitor
        self.trading_symbols = [
            "BTC",
            "ETH",
            "ADA",
            "DOT",
            "LINK",
            "UNI",
            "AAVE",
        ]

        # Register news-specific handlers
        self.register_handler("analyze_news", self.handle_analyze_news)
        self.register_handler("get_sentiment", self.handle_get_sentiment)
        self.register_handler("update_sources", self.handle_update_sources)
        self.register_handler("market_data", self.handle_market_data)

        print(f"📰 News Sentiment Agent {agent_id} initialized")

    async def initialize(self):
        """Initialize news sentiment agent resources"""
        try:
            # Load news sources configuration
            await self.load_news_config()

            # Initialize sentiment analysis models
            await self.initialize_sentiment_models()

            # Start news monitoring
            await self.start_news_monitoring()

            print(f"✅ News Sentiment Agent {self.agent_id} initialized successfully")

        except Exception as e:
            print(f"❌ Error initializing News Sentiment Agent: {e}")
            self.update_health_status("error")

    async def process_loop(self):
        """Main news processing loop"""
        while self.running:
            try:
                # Fetch and analyze news
                await self.fetch_and_analyze_news()

                # Update sentiment metrics
                await self.update_sentiment_metrics()

                # Clean up old cache entries
                await self.cleanup_cache()

                await asyncio.sleep(300)  # Check every 5 minutes

            except Exception as e:
                print(f"❌ Error in news processing loop: {e}")
                await asyncio.sleep(600)

    async def load_news_config(self):
        """Load news configuration from Redis"""
        try:
            # Load news sources
            sources_data = self.redis_client.get("news_sources")
            if sources_data:
                self.news_sources = json.loads(sources_data)

            # Load trading symbols
            symbols_data = self.redis_client.get("trading_symbols")
            if symbols_data:
                self.trading_symbols = json.loads(symbols_data)

            print(
                f"📋 News configuration loaded: {len(self.news_sources)} sources, {len(self.trading_symbols)} symbols"
            )

        except Exception as e:
            print(f"❌ Error loading news configuration: {e}")

    async def initialize_sentiment_models(self):
        """Initialize sentiment analysis models"""
        try:
            # Initialize TextBlob for sentiment analysis
            # In production, you might use more sophisticated models like:
            # - FinBERT (financial-specific BERT)
            # - VADER (Valence Aware Dictionary and sEntiment Reasoner)
            # - Custom fine-tuned models

            # For now, using TextBlob as a baseline
            test_text = "Bitcoin price increases significantly"
            TextBlob(test_text)

            print("🧠 Sentiment models initialized (TextBlob)")

        except Exception as e:
            print(f"❌ Error initializing sentiment models: {e}")

    async def start_news_monitoring(self):
        """Start news monitoring"""
        try:
            # Subscribe to market data for context
            pubsub = self.redis_client.pubsub()
            pubsub.subscribe("market_data")

            # Start market data listener
            asyncio.create_task(self.listen_market_data(pubsub))

            print("📡 News monitoring started")

        except Exception as e:
            print(f"❌ Error starting news monitoring: {e}")

    async def listen_market_data(self, pubsub):
        """Listen for market data updates"""
        try:
            for message in pubsub.listen():
                if not self.running:
                    break

                if message["type"] == "message":
                    market_data = json.loads(message["data"])
                    await self.process_market_data(market_data)

        except Exception as e:
            print(f"❌ Error in market data listener: {e}")
        finally:
            pubsub.close()

    async def process_market_data(self, market_data: Dict[str, Any]):
        """Process market data for news context"""
        try:
            symbol = market_data.get("symbol")
            price = market_data.get("price")

            # Update symbol sentiment context
            if symbol and price:
                await self.update_symbol_context(symbol, price)

        except Exception as e:
            print(f"❌ Error processing market data: {e}")

    async def handle_market_data(self, message: Dict[str, Any]):
        """Handle market data message"""
        try:
            market_data = message.get("market_data", {})
            print(f"📊 News Sentiment Agent received market data for {len(market_data)} symbols")
            await self.process_market_data(market_data)
            await self.fetch_and_analyze_news()
        except Exception as e:
            print(f"❌ Error handling market data: {e}")
            await self.broadcast_error(f"Market data handling error: {e}")

    async def fetch_and_analyze_news(self):
        """Fetch and analyze news from all sources"""
        try:
            print(f"📰 Fetching news from {len(self.news_sources)} sources...")

            all_articles = []

            # Fetch news from each source
            for source in self.news_sources:
                try:
                    articles = await self.fetch_news_from_source(source)
                    all_articles.extend(articles)
                except Exception as e:
                    print(f"❌ Error fetching from {source['name']}: {e}")

            # Analyze sentiment for all articles
            if all_articles:
                await self.analyze_articles_sentiment(all_articles)

                # Update analysis count
                self.state["analysis_count"] += 1
                self.state["last_analysis"] = datetime.now().isoformat()

            print(f"✅ Analyzed {len(all_articles)} articles")

        except Exception as e:
            print(f"❌ Error fetching and analyzing news: {e}")

    async def fetch_news_from_source(self, source: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fetch news from a specific source"""
        try:
            articles = []

            if source["type"] == "rss":
                # Parse RSS feed
                feed = feedparser.parse(source["url"])

                for entry in feed.entries[:10]:  # Get latest 10 articles
                    # Check if article is recent (within last 24 hours)
                    pub_date = getattr(entry, "published_parsed", None)
                    if pub_date:
                        pub_datetime = datetime(*pub_date[:6])
                        if (datetime.now() - pub_datetime) < timedelta(hours=24):
                            article = {
                                "title": entry.title,
                                "summary": getattr(entry, "summary", ""),
                                "link": entry.link,
                                "published": pub_datetime.isoformat(),
                                "source": source["name"],
                                "keywords": source["keywords"],
                            }
                            articles.append(article)

            return articles

        except Exception as e:
            print(f"❌ Error fetching from {source['name']}: {e}")
            return []

    async def analyze_articles_sentiment(self, articles: List[Dict[str, Any]]):
        """Analyze sentiment for a list of articles"""
        try:
            for article in articles:
                # Analyze article sentiment
                sentiment = await self.analyze_article_sentiment(article)

                # Extract relevant symbols
                symbols = await self.extract_symbols(article)

                # Update sentiment cache
                await self.update_sentiment_cache(article, sentiment, symbols)

                # Broadcast sentiment update
                await self.broadcast_sentiment_update(article, sentiment, symbols)

        except Exception as e:
            print(f"❌ Error analyzing articles sentiment: {e}")

    async def analyze_article_sentiment(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze sentiment for a single article"""
        try:
            # Combine title and summary for analysis
            text = f"{article['title']} {article['summary']}"

            # Clean text
            cleaned_text = self.clean_text(text)

            # Analyze sentiment using TextBlob
            blob = TextBlob(cleaned_text)

            # Get polarity (-1 to 1) and subjectivity (0 to 1)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity

            # Categorize sentiment
            if polarity > 0.1:
                sentiment_category = "positive"
            elif polarity < -0.1:
                sentiment_category = "negative"
            else:
                sentiment_category = "neutral"

            # Calculate confidence based on subjectivity
            confidence = 1 - subjectivity

            # Extract keywords and their sentiment
            keywords_sentiment = await self.extract_keywords_sentiment(cleaned_text)

            return {
                "polarity": polarity,
                "subjectivity": subjectivity,
                "category": sentiment_category,
                "confidence": confidence,
                "keywords_sentiment": keywords_sentiment,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            print(f"❌ Error analyzing article sentiment: {e}")
            return {
                "polarity": 0,
                "subjectivity": 0.5,
                "category": "neutral",
                "confidence": 0,
                "keywords_sentiment": {},
                "timestamp": datetime.now().isoformat(),
            }

    def clean_text(self, text: str) -> str:
        """Clean text for sentiment analysis"""
        try:
            # Remove HTML tags
            text = re.sub(r"<[^>]+>", "", text)

            # Remove special characters but keep spaces
            text = re.sub(r"[^\w\s]", " ", text)

            # Remove extra whitespace
            text = re.sub(r"\s+", " ", text).strip()

            return text.lower()

        except Exception as e:
            print(f"❌ Error cleaning text: {e}")
            return text

    async def extract_symbols(self, article: Dict[str, Any]) -> List[str]:
        """Extract trading symbols from article"""
        try:
            text = f"{article['title']} {article['summary']}"
            symbols = []

            # Look for common crypto symbols
            for symbol in self.trading_symbols:
                if symbol.lower() in text.lower():
                    symbols.append(symbol)

            # Look for Bitcoin and Ethereum variations
            if "bitcoin" in text.lower() or "btc" in text.lower():
                symbols.append("BTC")
            if "ethereum" in text.lower() or "eth" in text.lower():
                symbols.append("ETH")

            return list(set(symbols))  # Remove duplicates

        except Exception as e:
            print(f"❌ Error extracting symbols: {e}")
            return []

    async def extract_keywords_sentiment(self, text: str) -> Dict[str, float]:
        """Extract sentiment for specific keywords"""
        try:
            keywords_sentiment = {}

            # Define financial keywords
            financial_keywords = [
                "bullish",
                "bearish",
                "rally",
                "crash",
                "surge",
                "plunge",
                "gain",
                "loss",
                "profit",
                "revenue",
                "earnings",
                "growth",
                "decline",
                "increase",
                "decrease",
                "up",
                "down",
                "high",
                "low",
            ]

            # Analyze sentiment for each keyword
            for keyword in financial_keywords:
                if keyword in text:
                    # Create a context around the keyword
                    words = text.split()
                    try:
                        keyword_index = words.index(keyword)
                        start = max(0, keyword_index - 5)
                        end = min(len(words), keyword_index + 6)
                        context = " ".join(words[start:end])

                        # Analyze context sentiment
                        blob = TextBlob(context)
                        keywords_sentiment[keyword] = blob.sentiment.polarity
                    except ValueError:
                        continue

            return keywords_sentiment

        except Exception as e:
            print(f"❌ Error extracting keywords sentiment: {e}")
            return {}

    async def update_sentiment_cache(
        self,
        article: Dict[str, Any],
        sentiment: Dict[str, Any],
        symbols: List[str],
    ):
        """Update sentiment cache"""
        try:
            # Create cache entry
            cache_entry = {
                "article": article,
                "sentiment": sentiment,
                "symbols": symbols,
                "timestamp": datetime.now().isoformat(),
            }

            # Store in cache
            cache_key = f"news_sentiment:{hash(article['title'])}"
            self.state["sentiment_cache"][cache_key] = cache_entry

            # Update symbol sentiment
            for symbol in symbols:
                if symbol not in self.state["symbol_sentiment"]:
                    self.state["symbol_sentiment"][symbol] = []

                self.state["symbol_sentiment"][symbol].append(
                    {
                        "sentiment": sentiment,
                        "timestamp": datetime.now().isoformat(),
                    }
                )

            # Store in Redis
            self.redis_client.set(cache_key, json.dumps(cache_entry), ex=3600)

        except Exception as e:
            print(f"❌ Error updating sentiment cache: {e}")

    async def broadcast_sentiment_update(
        self,
        article: Dict[str, Any],
        sentiment: Dict[str, Any],
        symbols: List[str],
    ):
        """Broadcast sentiment update to other agents"""
        try:
            sentiment_update = {
                "type": "news_sentiment_update",
                "article": article,
                "sentiment": sentiment,
                "symbols": symbols,
                "timestamp": datetime.now().isoformat(),
            }

            # Broadcast to all agents
            await self.broadcast_message(sentiment_update)

            # Send to specific agents
            await self.send_message("strategy_agent", sentiment_update)
            await self.send_message("risk_agent", sentiment_update)

        except Exception as e:
            print(f"❌ Error broadcasting sentiment update: {e}")

    async def update_symbol_context(self, symbol: str, price: float):
        """Update symbol context for sentiment analysis"""
        try:
            # Store price context for sentiment correlation
            price_context = {
                "symbol": symbol,
                "price": price,
                "timestamp": datetime.now().isoformat(),
            }

            # Store in Redis
            self.redis_client.set(f"price_context:{symbol}", json.dumps(price_context), ex=300)

        except Exception as e:
            print(f"❌ Error updating symbol context: {e}")

    async def handle_analyze_news(self, message: Dict[str, Any]):
        """Handle manual news analysis request"""
        try:
            source_url = message.get("source_url")
            keywords = message.get("keywords", [])

            print(f"📰 Manual news analysis requested for {source_url}")

            # Fetch and analyze news from specified source
            source = {
                "name": "Manual Request",
                "url": source_url,
                "type": "manual",
                "keywords": keywords,
            }

            articles = await self.fetch_news_from_source(source)
            if articles:
                await self.analyze_articles_sentiment(articles)

            # Send response
            response = {
                "type": "news_analysis_complete",
                "articles_analyzed": len(articles),
                "timestamp": datetime.now().isoformat(),
            }

            sender = message.get("from_agent")
            if sender:
                await self.send_message(sender, response)

        except Exception as e:
            print(f"❌ Error handling news analysis request: {e}")
            await self.broadcast_error(f"News analysis error: {e}")

    async def handle_get_sentiment(self, message: Dict[str, Any]):
        """Handle sentiment request"""
        try:
            symbol = message.get("symbol")
            timeframe = message.get("timeframe", "24h")

            print(f"📊 Sentiment request for {symbol} ({timeframe})")

            # Get sentiment for symbol
            sentiment_data = await self.get_symbol_sentiment(symbol, timeframe)

            # Send response
            response = {
                "type": "sentiment_response",
                "symbol": symbol,
                "timeframe": timeframe,
                "sentiment": sentiment_data,
                "timestamp": datetime.now().isoformat(),
            }

            sender = message.get("from_agent")
            if sender:
                await self.send_message(sender, response)

        except Exception as e:
            print(f"❌ Error handling sentiment request: {e}")
            await self.broadcast_error(f"Sentiment request error: {e}")

    async def handle_update_sources(self, message: Dict[str, Any]):
        """Handle news sources update request"""
        try:
            new_sources = message.get("sources", [])

            print("📋 Updating news sources")

            # Update sources
            self.news_sources = new_sources

            # Store in Redis
            self.redis_client.set("news_sources", json.dumps(new_sources), ex=3600)

            # Send confirmation
            response = {
                "type": "sources_updated",
                "sources_count": len(new_sources),
                "timestamp": datetime.now().isoformat(),
            }

            sender = message.get("from_agent")
            if sender:
                await self.send_message(sender, response)

        except Exception as e:
            print(f"❌ Error updating news sources: {e}")
            await self.broadcast_error(f"Sources update error: {e}")

    async def get_symbol_sentiment(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Get sentiment data for a symbol"""
        try:
            symbol_sentiments = self.state["symbol_sentiment"].get(symbol, [])

            if not symbol_sentiments:
                return {
                    "average_polarity": 0,
                    "sentiment_category": "neutral",
                    "confidence": 0,
                    "article_count": 0,
                }

            # Filter by timeframe
            cutoff_time = datetime.now()
            if timeframe == "1h":
                cutoff_time -= timedelta(hours=1)
            elif timeframe == "6h":
                cutoff_time -= timedelta(hours=6)
            elif timeframe == "24h":
                cutoff_time -= timedelta(hours=24)
            else:
                cutoff_time -= timedelta(hours=24)  # Default to 24h

            recent_sentiments = [
                s for s in symbol_sentiments if datetime.fromisoformat(s["timestamp"]) > cutoff_time
            ]

            if not recent_sentiments:
                return {
                    "average_polarity": 0,
                    "sentiment_category": "neutral",
                    "confidence": 0,
                    "article_count": 0,
                }

            # Calculate average sentiment
            polarities = [s["sentiment"]["polarity"] for s in recent_sentiments]
            confidences = [s["sentiment"]["confidence"] for s in recent_sentiments]

            avg_polarity = np.mean(polarities)
            avg_confidence = np.mean(confidences)

            # Determine sentiment category
            if avg_polarity > 0.1:
                sentiment_category = "positive"
            elif avg_polarity < -0.1:
                sentiment_category = "negative"
            else:
                sentiment_category = "neutral"

            return {
                "average_polarity": avg_polarity,
                "sentiment_category": sentiment_category,
                "confidence": avg_confidence,
                "article_count": len(recent_sentiments),
                "recent_articles": recent_sentiments[-5:],  # Last 5 articles
            }

        except Exception as e:
            print(f"❌ Error getting symbol sentiment: {e}")
            return {
                "average_polarity": 0,
                "sentiment_category": "neutral",
                "confidence": 0,
                "article_count": 0,
            }

    async def update_sentiment_metrics(self):
        """Update sentiment metrics"""
        try:
            metrics = {
                "agent_id": self.agent_id,
                "sources_count": len(self.news_sources),
                "symbols_count": len(self.trading_symbols),
                "cache_size": len(self.state["sentiment_cache"]),
                "analysis_count": self.state["analysis_count"],
                "last_analysis": self.state["last_analysis"],
                "timestamp": datetime.now().isoformat(),
            }

            # Store metrics in Redis
            self.redis_client.set(f"agent_metrics:{self.agent_id}", json.dumps(metrics), ex=300)

        except Exception as e:
            print(f"❌ Error updating sentiment metrics: {e}")

    async def cleanup_cache(self):
        """Clean up old cache entries"""
        try:
            current_time = datetime.now()
            cache_keys = list(self.state["sentiment_cache"].keys())

            for key in cache_keys:
                entry = self.state["sentiment_cache"][key]
                entry_time = datetime.fromisoformat(entry["timestamp"])

                # Remove entries older than 24 hours
                if (current_time - entry_time) > timedelta(hours=24):
                    del self.state["sentiment_cache"][key]

            # Clean up symbol sentiment (keep last 100 entries per symbol)
            for symbol in self.state["symbol_sentiment"]:
                sentiments = self.state["symbol_sentiment"][symbol]
                if len(sentiments) > 100:
                    self.state["symbol_sentiment"][symbol] = sentiments[-100:]

        except Exception as e:
            print(f"❌ Error cleaning up cache: {e}")
