"""
Social Media Agent
Handles social media sentiment analysis and monitoring
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List
import os
import sys
import re
from textblob import TextBlob
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.agents.base_agent import BaseAgent


class SocialMediaAgent(BaseAgent):
    """Social Media Agent - Monitors social media sentiment"""

    def __init__(self, agent_id: str = "social_media_agent_001"):
        super().__init__(agent_id, "social_media")

        # Social media-specific state
        self.state.update(
            {
                "platforms": [],
                "sentiment_cache": {},
                "trending_topics": {},
                "influencer_sentiment": {},
                "last_analysis": None,
                "analysis_count": 0,
            }
        )

        # Social media platforms configuration
        self.platforms = [
            {
                "name": "Twitter",
                "type": "twitter",
                "keywords": [
                    "#bitcoin",
                    "#crypto",
                    "#btc",
                    "#eth",
                    "#trading",
                ],
                "api_endpoint": ("https://api.twitter.com/2/tweets/search/recent"),
                "enabled": False,  # Requires API keys
            },
            {
                "name": "Reddit",
                "type": "reddit",
                "subreddits": [
                    "cryptocurrency",
                    "bitcoin",
                    "ethereum",
                    "CryptoMarkets",
                ],
                "keywords": ["bitcoin", "crypto", "btc", "eth", "trading"],
                "enabled": True,
            },
            {
                "name": "Telegram",
                "type": "telegram",
                "channels": ["@binance", "@coinbase", "@cryptocom"],
                "keywords": ["bitcoin", "crypto", "btc", "eth", "trading"],
                "enabled": False,  # Requires bot token
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

        # Influencer accounts to monitor
        self.influencers = [
            "elonmusk",
            "cz_binance",
            "VitalikButerin",
            "SBF_FTX",
            "michael_saylor",
            "jack",
            "chamath",
        ]

        # Register social media-specific handlers
        self.register_handler("analyze_social", self.handle_analyze_social)
        self.register_handler("get_sentiment", self.handle_get_sentiment)
        self.register_handler("monitor_influencer", self.handle_monitor_influencer)
        self.register_handler("market_data", self.handle_market_data)

        print(f"ðŸ“± Social Media Agent {agent_id} initialized")

    async def initialize(self):
        """Initialize social media agent resources"""
        try:
            # Load social media configuration
            await self.load_social_config()

            # Initialize sentiment analysis models
            await self.initialize_sentiment_models()

            # Start social media monitoring
            await self.start_social_monitoring()

            print(f"âœ… Social Media Agent {self.agent_id} initialized successfully")

        except Exception as e:
            print(f"âŒ Error initializing Social Media Agent: {e}")
            self.update_health_status("error")

    async def process_loop(self):
        """Main social media processing loop"""
        while self.running:
            try:
                # Fetch and analyze social media posts
                await self.fetch_and_analyze_social()

                # Update trending topics
                await self.update_trending_topics()

                # Monitor influencers
                await self.monitor_influencers()

                # Update sentiment metrics
                await self.update_sentiment_metrics()

                # Clean up old cache entries
                await self.cleanup_cache()

                await asyncio.sleep(180)  # Check every 3 minutes

            except Exception as e:
                print(f"âŒ Error in social media processing loop: {e}")
                await asyncio.sleep(300)

    async def load_social_config(self):
        """Load social media configuration from Redis"""
        try:
            # Load platforms configuration
            platforms_data = self.redis_client.get("social_platforms")
            if platforms_data:
                self.platforms = json.loads(platforms_data)

            # Load trading symbols
            symbols_data = self.redis_client.get("trading_symbols")
            if symbols_data:
                self.trading_symbols = json.loads(symbols_data)

            # Load influencers
            influencers_data = self.redis_client.get("social_influencers")
            if influencers_data:
                self.influencers = json.loads(influencers_data)

            print(
                f"ðŸ“‹ Social media configuration loaded: {len(self.platforms)} platforms, {len(self.trading_symbols)} symbols"
            )

        except Exception as e:
            print(f"âŒ Error loading social media configuration: {e}")

    async def initialize_sentiment_models(self):
        """Initialize sentiment analysis models"""
        try:
            # Initialize TextBlob for sentiment analysis
            # In production, you might use more sophisticated models like:
            # - BERT for social media sentiment
            # - RoBERTa for Twitter sentiment
            # - Custom models trained on crypto social data

            # For now, using TextBlob as a baseline
            test_text = "Bitcoin is going to the moon! ðŸš€"
            TextBlob(test_text)

            print("ðŸ§  Social sentiment models initialized (TextBlob)")

        except Exception as e:
            print(f"âŒ Error initializing sentiment models: {e}")

    async def start_social_monitoring(self):
        """Start social media monitoring"""
        try:
            # Subscribe to market data for context
            pubsub = self.redis_client.pubsub()
            pubsub.subscribe("market_data")

            # Start market data listener
            asyncio.create_task(self.listen_market_data(pubsub))

            print("ðŸ“¡ Social media monitoring started")

        except Exception as e:
            print(f"âŒ Error starting social media monitoring: {e}")

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
            print(f"âŒ Error in market data listener: {e}")
        finally:
            pubsub.close()

    async def process_market_data(self, market_data: Dict[str, Any]):
        """Process market data for social media context"""
        try:
            symbol = market_data.get("symbol")
            price = market_data.get("price")

            # Update symbol context for social sentiment correlation
            if symbol and price:
                await self.update_symbol_context(symbol, price)

        except Exception as e:
            print(f"âŒ Error processing market data: {e}")

    async def fetch_and_analyze_social(self):
        """Fetch and analyze social media posts"""
        try:
            print(f"ðŸ“± Fetching social media posts from {len(self.platforms)} platforms...")

            all_posts = []

            # Fetch posts from each platform
            for platform in self.platforms:
                if platform.get("enabled", False):
                    try:
                        posts = await self.fetch_posts_from_platform(platform)
                        all_posts.extend(posts)
                    except Exception as e:
                        print(f"âŒ Error fetching from {platform['name']}: {e}")

            # Analyze sentiment for all posts
            if all_posts:
                await self.analyze_posts_sentiment(all_posts)

                # Update analysis count
                self.state["analysis_count"] += 1
                self.state["last_analysis"] = datetime.now().isoformat()

            print(f"âœ… Analyzed {len(all_posts)} social media posts")

        except Exception as e:
            print(f"âŒ Error fetching and analyzing social media: {e}")

    async def fetch_posts_from_platform(self, platform: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fetch posts from a specific platform"""
        try:
            posts = []

            if platform["type"] == "reddit":
                posts = await self.fetch_reddit_posts(platform)
            elif platform["type"] == "twitter":
                posts = await self.fetch_twitter_posts(platform)
            elif platform["type"] == "telegram":
                posts = await self.fetch_telegram_posts(platform)

            return posts

        except Exception as e:
            print(f"âŒ Error fetching from {platform['name']}: {e}")
            return []

    async def fetch_reddit_posts(self, platform: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fetch posts from Reddit"""
        try:
            posts = []

            # Simulate Reddit API calls (in production, use praw library)
            # For now, create mock posts for demonstration

            subreddits = platform.get("subreddits", [])
            keywords = platform.get("keywords", [])

            for subreddit in subreddits:
                # Mock posts for demonstration
                mock_posts = [
                    {
                        "id": f"reddit_{subreddit}_{i}",
                        "platform": "reddit",
                        "subreddit": subreddit,
                        "title": f"Bitcoin price analysis - {subreddit}",
                        "content": (
                            "Bitcoin is showing strong momentum. What do you think about the current price action?"
                        ),
                        "author": f"user_{i}",
                        "score": np.random.randint(1, 100),
                        "comments": np.random.randint(0, 50),
                        "created_utc": datetime.now().timestamp(),
                        "keywords": keywords,
                    }
                    for i in range(5)  # 5 mock posts per subreddit
                ]

                posts.extend(mock_posts)

            return posts

        except Exception as e:
            print(f"âŒ Error fetching Reddit posts: {e}")
            return []

    async def fetch_twitter_posts(self, platform: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fetch posts from Twitter"""
        try:
            posts = []

            # Simulate Twitter API calls (in production, use tweepy library)
            # For now, create mock posts for demonstration

            keywords = platform.get("keywords", [])

            # Mock tweets for demonstration
            mock_tweets = [
                {
                    "id": f"twitter_{i}",
                    "platform": "twitter",
                    "text": (f"Bitcoin is looking bullish! #{keywords[0]} #crypto"),
                    "author": f"crypto_user_{i}",
                    "followers_count": np.random.randint(100, 10000),
                    "retweet_count": np.random.randint(0, 100),
                    "like_count": np.random.randint(0, 500),
                    "created_at": datetime.now().isoformat(),
                    "keywords": keywords,
                }
                for i in range(10)  # 10 mock tweets
            ]

            posts.extend(mock_tweets)

            return posts

        except Exception as e:
            print(f"âŒ Error fetching Twitter posts: {e}")
            return []

    async def fetch_telegram_posts(self, platform: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fetch posts from Telegram"""
        try:
            posts = []

            # Simulate Telegram API calls (in production, use python-telegram-bot library)
            # For now, create mock posts for demonstration

            channels = platform.get("channels", [])
            keywords = platform.get("keywords", [])

            for channel in channels:
                # Mock posts for demonstration
                mock_posts = [
                    {
                        "id": f"telegram_{channel}_{i}",
                        "platform": "telegram",
                        "channel": channel,
                        "text": (f"Breaking: Bitcoin reaches new highs! {keywords[0]}"),
                        "author": channel,
                        "views": np.random.randint(1000, 50000),
                        "created_at": datetime.now().isoformat(),
                        "keywords": keywords,
                    }
                    for i in range(3)  # 3 mock posts per channel
                ]

                posts.extend(mock_posts)

            return posts

        except Exception as e:
            print(f"âŒ Error fetching Telegram posts: {e}")
            return []

    async def analyze_posts_sentiment(self, posts: List[Dict[str, Any]]):
        """Analyze sentiment for a list of posts"""
        try:
            for post in posts:
                # Analyze post sentiment
                sentiment = await self.analyze_post_sentiment(post)

                # Extract relevant symbols
                symbols = await self.extract_symbols(post)

                # Update sentiment cache
                await self.update_sentiment_cache(post, sentiment, symbols)

                # Broadcast sentiment update
                await self.broadcast_sentiment_update(post, sentiment, symbols)

        except Exception as e:
            print(f"âŒ Error analyzing posts sentiment: {e}")

    async def analyze_post_sentiment(self, post: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze sentiment for a single post"""
        try:
            # Combine title and content for analysis
            text = f"{post.get('title', '')} {post.get('content', '')} {post.get('text', '')}"

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

            # Calculate confidence based on subjectivity and engagement
            engagement_score = self.calculate_engagement_score(post)
            confidence = (1 - subjectivity) * 0.7 + engagement_score * 0.3

            # Extract emojis and their sentiment
            emoji_sentiment = await self.extract_emoji_sentiment(cleaned_text)

            # Extract hashtags
            hashtags = self.extract_hashtags(cleaned_text)

            return {
                "polarity": polarity,
                "subjectivity": subjectivity,
                "category": sentiment_category,
                "confidence": confidence,
                "engagement_score": engagement_score,
                "emoji_sentiment": emoji_sentiment,
                "hashtags": hashtags,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            print(f"âŒ Error analyzing post sentiment: {e}")
            return {
                "polarity": 0,
                "subjectivity": 0.5,
                "category": "neutral",
                "confidence": 0,
                "engagement_score": 0,
                "emoji_sentiment": {},
                "hashtags": [],
                "timestamp": datetime.now().isoformat(),
            }

    def clean_text(self, text: str) -> str:
        """Clean text for sentiment analysis"""
        try:
            # Remove URLs
            text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

            # Remove special characters but keep emojis and hashtags
            text = re.sub(r"[^\w\s#@]", " ", text)

            # Remove extra whitespace
            text = re.sub(r"\s+", " ", text).strip()

            return text.lower()

        except Exception as e:
            print(f"âŒ Error cleaning text: {e}")
            return text

    def calculate_engagement_score(self, post: Dict[str, Any]) -> float:
        """Calculate engagement score for a post"""
        try:
            score = 0.0

            # Reddit engagement
            if post.get("platform") == "reddit":
                score = post.get("score", 0) / 1000  # Normalize score
                score += post.get("comments", 0) / 100  # Add comment weight

            # Twitter engagement
            elif post.get("platform") == "twitter":
                followers = post.get("followers_count", 0)
                retweets = post.get("retweet_count", 0)
                likes = post.get("like_count", 0)

                if followers > 0:
                    engagement_rate = (retweets + likes) / followers
                    score = min(engagement_rate * 10, 1.0)  # Cap at 1.0

            # Telegram engagement
            elif post.get("platform") == "telegram":
                views = post.get("views", 0)
                score = min(views / 10000, 1.0)  # Normalize views

            return min(score, 1.0)  # Ensure score is between 0 and 1

        except Exception as e:
            print(f"âŒ Error calculating engagement score: {e}")
            return 0.0

    async def extract_emoji_sentiment(self, text: str) -> Dict[str, float]:
        """Extract sentiment from emojis"""
        try:
            emoji_sentiment = {}

            # Define emoji sentiment mappings
            emoji_mappings = {
                "ðŸš€": 0.8,
                "ðŸ“ˆ": 0.7,
                "ðŸ’Ž": 0.6,
                "ðŸ”¥": 0.5,
                "âœ…": 0.4,
                "ðŸ“‰": -0.7,
                "ðŸ’©": -0.6,
                "ðŸ˜±": -0.5,
                "âŒ": -0.4,
                "ðŸ˜­": -0.3,
                "ðŸ¤”": 0.0,
                "ðŸ’­": 0.0,
                "ðŸ“Š": 0.1,
                "ðŸ’°": 0.3,
                "ðŸŽ¯": 0.4,
            }

            # Find emojis in text
            for emoji, sentiment in emoji_mappings.items():
                if emoji in text:
                    emoji_sentiment[emoji] = sentiment

            return emoji_sentiment

        except Exception as e:
            print(f"âŒ Error extracting emoji sentiment: {e}")
            return {}

    def extract_hashtags(self, text: str) -> List[str]:
        """Extract hashtags from text"""
        try:
            hashtags = re.findall(r"#\w+", text)
            return [tag.lower() for tag in hashtags]

        except Exception as e:
            print(f"âŒ Error extracting hashtags: {e}")
            return []

    async def extract_symbols(self, post: Dict[str, Any]) -> List[str]:
        """Extract trading symbols from post"""
        try:
            text = f"{post.get('title', '')} {post.get('content', '')} {post.get('text', '')}"
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
            print(f"âŒ Error extracting symbols: {e}")
            return []

    async def update_sentiment_cache(
        self,
        post: Dict[str, Any],
        sentiment: Dict[str, Any],
        symbols: List[str],
    ):
        """Update sentiment cache"""
        try:
            # Create cache entry
            cache_entry = {
                "post": post,
                "sentiment": sentiment,
                "symbols": symbols,
                "timestamp": datetime.now().isoformat(),
            }

            # Store in cache
            cache_key = f"social_sentiment:{hash(post['id'])}"
            self.state["sentiment_cache"][cache_key] = cache_entry

            # Update symbol sentiment
            for symbol in symbols:
                if symbol not in self.state["symbol_sentiment"]:
                    self.state["symbol_sentiment"][symbol] = []

                self.state["symbol_sentiment"][symbol].append(
                    {
                        "sentiment": sentiment,
                        "platform": post.get("platform"),
                        "timestamp": datetime.now().isoformat(),
                    }
                )

            # Store in Redis
            self.redis_client.set(cache_key, json.dumps(cache_entry), ex=1800)  # 30 minutes

        except Exception as e:
            print(f"âŒ Error updating sentiment cache: {e}")

    async def broadcast_sentiment_update(
        self,
        post: Dict[str, Any],
        sentiment: Dict[str, Any],
        symbols: List[str],
    ):
        """Broadcast sentiment update to other agents"""
        try:
            sentiment_update = {
                "type": "social_sentiment_update",
                "post": post,
                "sentiment": sentiment,
                "symbols": symbols,
                "timestamp": datetime.now().isoformat(),
            }

            # Broadcast to all agents
            await self.broadcast_message(sentiment_update)

            # Send to specific agents
            await self.send_message("strategy_agent", sentiment_update)
            await self.send_message("market_sentiment_agent", sentiment_update)

        except Exception as e:
            print(f"âŒ Error broadcasting sentiment update: {e}")

    async def update_trending_topics(self):
        """Update trending topics"""
        try:
            # Analyze recent posts for trending topics
            recent_posts = []
            current_time = datetime.now()

            for cache_entry in self.state["sentiment_cache"].values():
                post_time = datetime.fromisoformat(cache_entry["timestamp"])
                if (current_time - post_time) < timedelta(hours=1):
                    recent_posts.append(cache_entry["post"])

            # Extract trending hashtags and keywords
            trending_hashtags = {}
            trending_keywords = {}

            for post in recent_posts:
                text = f"{post.get('title', '')} {post.get('content', '')} {post.get('text', '')}"
                hashtags = self.extract_hashtags(text)

                for hashtag in hashtags:
                    trending_hashtags[hashtag] = trending_hashtags.get(hashtag, 0) + 1

            # Update trending topics
            self.state["trending_topics"] = {
                "hashtags": dict(
                    sorted(
                        trending_hashtags.items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )[:10]
                ),
                "keywords": trending_keywords,
                "timestamp": datetime.now().isoformat(),
            }

            # Store in Redis
            self.redis_client.set(
                "trending_topics",
                json.dumps(self.state["trending_topics"]),
                ex=1800,
            )

        except Exception as e:
            print(f"âŒ Error updating trending topics: {e}")

    async def monitor_influencers(self):
        """Monitor influencer sentiment"""
        try:
            # Simulate influencer monitoring
            # In production, this would fetch real influencer posts

            for influencer in self.influencers:
                # Mock influencer sentiment
                sentiment = {
                    "polarity": np.random.uniform(-0.5, 0.8),
                    "subjectivity": np.random.uniform(0.3, 0.8),
                    "category": "neutral",
                    "confidence": np.random.uniform(0.6, 0.9),
                    "timestamp": datetime.now().isoformat(),
                }

                # Categorize sentiment
                if sentiment["polarity"] > 0.1:
                    sentiment["category"] = "positive"
                elif sentiment["polarity"] < -0.1:
                    sentiment["category"] = "negative"

                self.state["influencer_sentiment"][influencer] = sentiment

            # Store in Redis
            self.redis_client.set(
                "influencer_sentiment",
                json.dumps(self.state["influencer_sentiment"]),
                ex=3600,
            )

        except Exception as e:
            print(f"âŒ Error monitoring influencers: {e}")

    async def update_symbol_context(self, symbol: str, price: float):
        """Update symbol context for social sentiment correlation"""
        try:
            # Store price context for sentiment correlation
            price_context = {
                "symbol": symbol,
                "price": price,
                "timestamp": datetime.now().isoformat(),
            }

            # Store in Redis
            self.redis_client.set(
                f"social_price_context:{symbol}",
                json.dumps(price_context),
                ex=300,
            )

        except Exception as e:
            print(f"âŒ Error updating symbol context: {e}")

    async def handle_analyze_social(self, message: Dict[str, Any]):
        """Handle manual social media analysis request"""
        try:
            platform = message.get("platform")
            keywords = message.get("keywords", [])

            print(f"ðŸ“± Manual social media analysis requested for {platform}")

            # Find platform configuration
            platform_config = next(
                (p for p in self.platforms if p["name"].lower() == platform.lower()),
                None,
            )

            if platform_config:
                posts = await self.fetch_posts_from_platform(platform_config)
                if posts:
                    await self.analyze_posts_sentiment(posts)

            # Send response
            response = {
                "type": "social_analysis_complete",
                "platform": platform,
                "posts_analyzed": len(posts) if "posts" in locals() else 0,
                "timestamp": datetime.now().isoformat(),
            }

            sender = message.get("from_agent")
            if sender:
                await self.send_message(sender, response)

        except Exception as e:
            print(f"âŒ Error handling social media analysis request: {e}")
            await self.broadcast_error(f"Social media analysis error: {e}")

    async def handle_get_sentiment(self, message: Dict[str, Any]):
        """Handle sentiment request"""
        try:
            symbol = message.get("symbol")
            platform = message.get("platform", "all")
            timeframe = message.get("timeframe", "1h")

            print(f"ðŸ“Š Social sentiment request for {symbol} on {platform} ({timeframe})")

            # Get sentiment for symbol
            sentiment_data = await self.get_symbol_sentiment(symbol, platform, timeframe)

            # Send response
            response = {
                "type": "social_sentiment_response",
                "symbol": symbol,
                "platform": platform,
                "timeframe": timeframe,
                "sentiment": sentiment_data,
                "timestamp": datetime.now().isoformat(),
            }

            sender = message.get("from_agent")
            if sender:
                await self.send_message(sender, response)

        except Exception as e:
            print(f"âŒ Error handling sentiment request: {e}")
            await self.broadcast_error(f"Sentiment request error: {e}")

    async def handle_monitor_influencer(self, message: Dict[str, Any]):
        """Handle influencer monitoring request"""
        try:
            influencer = message.get("influencer")

            print(f"ðŸ‘¤ Monitoring influencer: {influencer}")

            # Add influencer to monitoring list
            if influencer not in self.influencers:
                self.influencers.append(influencer)

                # Store in Redis
                self.redis_client.set("social_influencers", json.dumps(self.influencers), ex=3600)

            # Send confirmation
            response = {
                "type": "influencer_monitoring_started",
                "influencer": influencer,
                "timestamp": datetime.now().isoformat(),
            }

            sender = message.get("from_agent")
            if sender:
                await self.send_message(sender, response)

        except Exception as e:
            print(f"âŒ Error monitoring influencer: {e}")
            await self.broadcast_error(f"Influencer monitoring error: {e}")

    async def get_symbol_sentiment(
        self, symbol: str, platform: str = "all", timeframe: str = "1h"
    ) -> Dict[str, Any]:
        """Get sentiment data for a symbol"""
        try:
            symbol_sentiments = self.state["symbol_sentiment"].get(symbol, [])

            if not symbol_sentiments:
                return {
                    "average_polarity": 0,
                    "sentiment_category": "neutral",
                    "confidence": 0,
                    "post_count": 0,
                    "platform_breakdown": {},
                }

            # Filter by timeframe
            cutoff_time = datetime.now()
            if timeframe == "30m":
                cutoff_time -= timedelta(minutes=30)
            elif timeframe == "1h":
                cutoff_time -= timedelta(hours=1)
            elif timeframe == "6h":
                cutoff_time -= timedelta(hours=6)
            elif timeframe == "24h":
                cutoff_time -= timedelta(hours=24)
            else:
                cutoff_time -= timedelta(hours=1)  # Default to 1h

            recent_sentiments = [
                s for s in symbol_sentiments if datetime.fromisoformat(s["timestamp"]) > cutoff_time
            ]

            if not recent_sentiments:
                return {
                    "average_polarity": 0,
                    "sentiment_category": "neutral",
                    "confidence": 0,
                    "post_count": 0,
                    "platform_breakdown": {},
                }

            # Filter by platform if specified
            if platform != "all":
                recent_sentiments = [s for s in recent_sentiments if s.get("platform") == platform]

            if not recent_sentiments:
                return {
                    "average_polarity": 0,
                    "sentiment_category": "neutral",
                    "confidence": 0,
                    "post_count": 0,
                    "platform_breakdown": {},
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

            # Calculate platform breakdown
            platform_breakdown = {}
            for sentiment in recent_sentiments:
                platform = sentiment.get("platform", "unknown")
                if platform not in platform_breakdown:
                    platform_breakdown[platform] = {
                        "count": 0,
                        "avg_polarity": 0,
                        "polarities": [],
                    }

                platform_breakdown[platform]["count"] += 1
                platform_breakdown[platform]["polarities"].append(
                    sentiment["sentiment"]["polarity"]
                )

            # Calculate averages for each platform
            for platform_data in platform_breakdown.values():
                platform_data["avg_polarity"] = np.mean(platform_data["polarities"])
                del platform_data["polarities"]  # Remove raw data

            return {
                "average_polarity": avg_polarity,
                "sentiment_category": sentiment_category,
                "confidence": avg_confidence,
                "post_count": len(recent_sentiments),
                "platform_breakdown": platform_breakdown,
                "recent_posts": recent_sentiments[-10:],  # Last 10 posts
            }

        except Exception as e:
            print(f"âŒ Error getting symbol sentiment: {e}")
            return {
                "average_polarity": 0,
                "sentiment_category": "neutral",
                "confidence": 0,
                "post_count": 0,
                "platform_breakdown": {},
            }

    async def update_sentiment_metrics(self):
        """Update sentiment metrics"""
        try:
            metrics = {
                "agent_id": self.agent_id,
                "platforms_count": len(self.platforms),
                "symbols_count": len(self.trading_symbols),
                "influencers_count": len(self.influencers),
                "cache_size": len(self.state["sentiment_cache"]),
                "analysis_count": self.state["analysis_count"],
                "last_analysis": self.state["last_analysis"],
                "trending_topics": self.state["trending_topics"],
                "timestamp": datetime.now().isoformat(),
            }

            # Store metrics in Redis
            self.redis_client.set(f"agent_metrics:{self.agent_id}", json.dumps(metrics), ex=300)

        except Exception as e:
            print(f"âŒ Error updating sentiment metrics: {e}")

    async def cleanup_cache(self):
        """Clean up old cache entries"""
        try:
            current_time = datetime.now()
            cache_keys = list(self.state["sentiment_cache"].keys())

            for key in cache_keys:
                entry = self.state["sentiment_cache"][key]
                entry_time = datetime.fromisoformat(entry["timestamp"])

                # Remove entries older than 2 hours
                if (current_time - entry_time) > timedelta(hours=2):
                    del self.state["sentiment_cache"][key]

            # Clean up symbol sentiment (keep last 200 entries per symbol)
            for symbol in self.state["symbol_sentiment"]:
                sentiments = self.state["symbol_sentiment"][symbol]
                if len(sentiments) > 200:
                    self.state["symbol_sentiment"][symbol] = sentiments[-200:]

        except Exception as e:
            print(f"âŒ Error cleaning up cache: {e}")


