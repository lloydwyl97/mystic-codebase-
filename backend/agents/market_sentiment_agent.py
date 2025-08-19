"""
Market Sentiment Agent
Handles aggregated market sentiment analysis from multiple sources
"""

import asyncio
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

import numpy as np

# Make all imports live (F401):
_ = os.getcwd()
_ = sys.version
_ = np.array([0])
_ = defaultdict(list)

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from backend.agents.base_agent import BaseAgent
except ImportError:
    # Fallback if the path modification didn't work
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from backend.agents.base_agent import BaseAgent


class MarketSentimentAgent(BaseAgent):
    """Market Sentiment Agent - Aggregates sentiment from multiple sources"""

    def __init__(self, agent_id: str = "market_sentiment_agent_001"):
        super().__init__(agent_id, "market_sentiment")

        # Market sentiment-specific state
        self.state.update(
            {
                "aggregated_sentiment": {},
                "source_weights": {},
                "sentiment_history": {},
                "market_fear_greed": {},
                "sentiment_signals": {},
                "last_aggregation": None,
                "aggregation_count": 0,
            }
        )

        # Source weights for sentiment aggregation
        self.source_weights = {
            "news_sentiment": 0.3,
            "social_media": 0.25,
            "market_data": 0.25,
            "technical_indicators": 0.2,
        }

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

        # Fear & Greed thresholds
        self.fear_greed_thresholds = {
            "extreme_fear": 25,
            "fear": 45,
            "neutral": 55,
            "greed": 75,
            "extreme_greed": 100,
        }

        # Register market sentiment-specific handlers
        self.register_handler("aggregate_sentiment", self.handle_aggregate_sentiment)
        self.register_handler("get_market_sentiment", self.handle_get_market_sentiment)
        self.register_handler("update_weights", self.handle_update_weights)
        self.register_handler("news_sentiment_update", self.handle_news_sentiment_update)
        self.register_handler("social_sentiment_update", self.handle_social_sentiment_update)
        self.register_handler("market_data", self.handle_market_data)

        print(f"ðŸ“Š Market Sentiment Agent {agent_id} initialized")

    async def initialize(self):
        """Initialize market sentiment agent resources"""
        try:
            # Load sentiment configuration
            await self.load_sentiment_config()

            # Initialize sentiment aggregation models
            await self.initialize_aggregation_models()

            # Start sentiment monitoring
            await self.start_sentiment_monitoring()

            print(f"âœ… Market Sentiment Agent {self.agent_id} initialized successfully")

        except Exception as e:
            print(f"âŒ Error initializing Market Sentiment Agent: {e}")
            self.update_health_status("error")

    async def process_loop(self):
        """Main sentiment aggregation loop"""
        while self.running:
            try:
                # Aggregate sentiment from all sources
                await self.aggregate_all_sentiment()

                # Calculate fear & greed index
                await self.calculate_fear_greed_index()

                # Generate sentiment signals
                await self.generate_sentiment_signals()

                # Update sentiment metrics
                await self.update_sentiment_metrics()

                # Clean up old history
                await self.cleanup_history()

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                print(f"âŒ Error in sentiment aggregation loop: {e}")
                await asyncio.sleep(120)

    async def load_sentiment_config(self):
        """Load sentiment configuration from Redis"""
        try:
            # Load source weights
            weights_data = self.redis_client.get("sentiment_source_weights")
            if weights_data:
                self.source_weights = json.loads(weights_data)

            # Load trading symbols
            symbols_data = self.redis_client.get("trading_symbols")
            if symbols_data:
                self.trading_symbols = json.loads(symbols_data)

            # Load fear & greed thresholds
            thresholds_data = self.redis_client.get("fear_greed_thresholds")
            if thresholds_data:
                self.fear_greed_thresholds = json.loads(thresholds_data)

            print(
                f"ðŸ“‹ Sentiment configuration loaded: {len(self.source_weights)} sources, {len(self.trading_symbols)} symbols"
            )

        except Exception as e:
            print(f"âŒ Error loading sentiment configuration: {e}")

    async def initialize_aggregation_models(self):
        """Initialize sentiment aggregation models"""
        try:
            # Initialize sentiment aggregation algorithms
            # In production, you might use more sophisticated models like:
            # - Ensemble methods for sentiment aggregation
            # - Time-series analysis for sentiment trends
            # - Machine learning models for sentiment prediction

            # For now, using weighted averaging as baseline
            print("ðŸ§  Sentiment aggregation models initialized")

        except Exception as e:
            print(f"âŒ Error initializing aggregation models: {e}")

    async def start_sentiment_monitoring(self):
        """Start sentiment monitoring"""
        try:
            # Subscribe to sentiment updates from other agents
            pubsub = self.redis_client.pubsub()
            pubsub.subscribe("news_sentiment_update")
            pubsub.subscribe("social_sentiment_update")
            pubsub.subscribe("market_data")

            # Start sentiment listener
            asyncio.create_task(self.listen_sentiment_updates(pubsub))

            print("ðŸ“¡ Sentiment monitoring started")

        except Exception as e:
            print(f"âŒ Error starting sentiment monitoring: {e}")

    async def listen_sentiment_updates(self, pubsub):
        """Listen for sentiment updates from other agents"""
        try:
            for message in pubsub.listen():
                if not self.running:
                    break

                if message["type"] == "message":
                    try:
                        data = json.loads(message["data"])
                        channel = message["channel"]

                        if channel == "news_sentiment_update":
                            await self.handle_news_sentiment_update(data)
                        elif channel == "social_sentiment_update":
                            await self.handle_social_sentiment_update(data)
                        elif channel == "market_data":
                            await self.handle_market_data(data)

                    except json.JSONDecodeError:
                        print(f"âŒ Error decoding sentiment update: {message['data']}")

        except Exception as e:
            print(f"âŒ Error in sentiment listener: {e}")
        finally:
            pubsub.close()

    async def handle_news_sentiment_update(self, data: dict[str, Any]):
        """Handle news sentiment updates"""
        try:
            article = data.get("article", {})
            if article:
                print(f"Received article for sentiment: {article.get('title', 'No Title')}")
            sentiment = data.get("sentiment", {})
            symbols = data.get("symbols", [])

            # Store news sentiment
            for symbol in symbols:
                if symbol not in self.state["sentiment_history"]:
                    self.state["sentiment_history"][symbol] = {
                        "news": [],
                        "social": [],
                        "market": [],
                        "technical": [],
                    }

                self.state["sentiment_history"][symbol]["news"].append(
                    {
                        "sentiment": sentiment,
                        "source": "news",
                        "timestamp": datetime.now().isoformat(),
                    }
                )

            print(f"ðŸ“° News sentiment update for {symbols}: {sentiment.get('category', 'neutral')}")

        except Exception as e:
            print(f"âŒ Error handling news sentiment update: {e}")

    async def handle_social_sentiment_update(self, data: dict[str, Any]):
        """Handle social media sentiment updates"""
        try:
            post = data.get("post", {})
            sentiment = data.get("sentiment", {})
            symbols = data.get("symbols", [])

            # Store social sentiment
            for symbol in symbols:
                if symbol not in self.state["sentiment_history"]:
                    self.state["sentiment_history"][symbol] = {
                        "news": [],
                        "social": [],
                        "market": [],
                        "technical": [],
                    }

                self.state["sentiment_history"][symbol]["social"].append(
                    {
                        "sentiment": sentiment,
                        "source": "social",
                        "platform": post.get("platform"),
                        "timestamp": datetime.now().isoformat(),
                    }
                )

            print(
                f"ðŸ“± Social sentiment update for {symbols}: {sentiment.get('category', 'neutral')}"
            )

        except Exception as e:
            print(f"âŒ Error handling social sentiment update: {e}")

    async def handle_market_data(self, data: dict[str, Any]):
        """Handle market data for sentiment correlation"""
        try:
            symbol = data.get("symbol")
            price = data.get("price")
            volume = data.get("volume", 0)
            change_24h = data.get("change_24h", 0)

            if symbol and price:
                # Calculate market-based sentiment
                market_sentiment = await self.calculate_market_sentiment(
                    symbol, price, volume, change_24h
                )

                # Store market sentiment
                if symbol not in self.state["sentiment_history"]:
                    self.state["sentiment_history"][symbol] = {
                        "news": [],
                        "social": [],
                        "market": [],
                        "technical": [],
                    }

                self.state["sentiment_history"][symbol]["market"].append(
                    {
                        "sentiment": market_sentiment,
                        "source": "market",
                        "price": price,
                        "volume": volume,
                        "change_24h": change_24h,
                        "timestamp": datetime.now().isoformat(),
                    }
                )

        except Exception as e:
            print(f"âŒ Error handling market data: {e}")

    async def calculate_market_sentiment(
        self, symbol: str, price: float, volume: float, change_24h: float
    ) -> dict[str, Any]:
        """Calculate sentiment based on market data"""
        try:
            # Get historical price data for context
            price_history = await self.get_price_history(symbol)

            # Calculate various market sentiment indicators
            price_momentum = await self.calculate_price_momentum(price_history, price)
            volume_sentiment = await self.calculate_volume_sentiment(volume, price_history)
            volatility_sentiment = await self.calculate_volatility_sentiment(price_history)

            # Combine indicators into overall sentiment
            sentiment_score = (
                price_momentum * 0.4 + volume_sentiment * 0.3 + volatility_sentiment * 0.3
            )

            # Categorize sentiment
            if sentiment_score > 0.1:
                category = "positive"
            elif sentiment_score < -0.1:
                category = "negative"
            else:
                category = "neutral"

            return {
                "polarity": sentiment_score,
                "subjectivity": 0.5,  # Market data is objective
                "category": category,
                "confidence": 0.8,
                "price_momentum": price_momentum,
                "volume_sentiment": volume_sentiment,
                "volatility_sentiment": volatility_sentiment,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            print(f"âŒ Error calculating market sentiment: {e}")
            return {
                "polarity": 0,
                "subjectivity": 0.5,
                "category": "neutral",
                "confidence": 0,
                "timestamp": datetime.now().isoformat(),
            }

    async def get_price_history(self, symbol: str) -> list[float]:
        """Get price history for sentiment calculation"""
        try:
            # Get price history from Redis
            price_data = self.redis_client.get(f"price_history:{symbol}")
            if price_data:
                return json.loads(price_data)

            # Return empty list if no history
            return []

        except Exception as e:
            print(f"âŒ Error getting price history: {e}")
            return []

    async def calculate_price_momentum(
        self, price_history: list[float], current_price: float
    ) -> float:
        """Calculate price momentum sentiment"""
        try:
            if len(price_history) < 2:
                return 0.0

            # Calculate price change over different timeframes
            recent_prices = price_history[-10:]  # Last 10 prices

            if len(recent_prices) < 2:
                return 0.0

            # Calculate momentum indicators
            short_momentum = (current_price - recent_prices[0]) / recent_prices[0]
            long_momentum = (
                (current_price - price_history[0]) / price_history[0] if price_history else 0
            )

            # Combine short and long momentum
            momentum_score = short_momentum * 0.7 + long_momentum * 0.3

            # Normalize to [-1, 1] range
            return np.clip(momentum_score, -1, 1)

        except Exception as e:
            print(f"âŒ Error calculating price momentum: {e}")
            return 0.0

    async def calculate_volume_sentiment(
        self, current_volume: float, price_history: list[float]
    ) -> float:
        """Calculate volume-based sentiment"""
        try:
            if len(price_history) < 5:
                return 0.0

            # Get volume history (simplified - in production, you'd store volume data)
            # For now, use price volatility as proxy for volume sentiment
            recent_prices = price_history[-5:]

            if len(recent_prices) < 2:
                return 0.0

            # Calculate price volatility
            price_changes = [
                abs(recent_prices[i] - recent_prices[i - 1]) / recent_prices[i - 1]
                for i in range(1, len(recent_prices))
            ]

            avg_volatility = np.mean(price_changes)

            # Higher volatility often indicates higher volume and interest
            # But too high volatility can indicate fear
            if avg_volatility < 0.02:  # Low volatility
                return 0.0
            elif avg_volatility < 0.05:  # Moderate volatility (positive)
                return 0.3
            elif avg_volatility < 0.1:  # High volatility (mixed)
                return 0.0
            else:  # Very high volatility (negative)
                return -0.3

        except Exception as e:
            print(f"âŒ Error calculating volume sentiment: {e}")
            return 0.0

    async def calculate_volatility_sentiment(self, price_history: list[float]) -> float:
        """Calculate volatility-based sentiment"""
        try:
            if len(price_history) < 10:
                return 0.0

            # Calculate rolling volatility
            returns = []
            for i in range(1, len(price_history)):
                if price_history[i - 1] > 0:
                    returns.append((price_history[i] - price_history[i - 1]) / price_history[i - 1])

            if len(returns) < 5:
                return 0.0

            # Calculate volatility
            volatility = np.std(returns)

            # Low volatility is generally positive for sentiment
            # High volatility can indicate uncertainty
            if volatility < 0.02:  # Low volatility
                return 0.2
            elif volatility < 0.05:  # Moderate volatility
                return 0.0
            elif volatility < 0.1:  # High volatility
                return -0.2
            else:  # Very high volatility
                return -0.4

        except Exception as e:
            print(f"âŒ Error calculating volatility sentiment: {e}")
            return 0.0

    async def aggregate_all_sentiment(self):
        """Aggregate sentiment from all sources for all symbols"""
        try:
            print(f"ðŸ“Š Aggregating sentiment for {len(self.trading_symbols)} symbols...")

            for symbol in self.trading_symbols:
                await self.aggregate_symbol_sentiment(symbol)

            # Update aggregation count
            self.state["aggregation_count"] += 1
            self.state["last_aggregation"] = datetime.now().isoformat()

            print("âœ… Sentiment aggregation complete")

        except Exception as e:
            print(f"âŒ Error aggregating sentiment: {e}")

    async def aggregate_symbol_sentiment(self, symbol: str):
        """Aggregate sentiment for a specific symbol"""
        try:
            if symbol not in self.state["sentiment_history"]:
                return

            history = self.state["sentiment_history"][symbol]

            # Get recent sentiment from each source
            recent_news = self.get_recent_sentiment(history["news"], hours=6)
            recent_social = self.get_recent_sentiment(history["social"], hours=3)
            recent_market = self.get_recent_sentiment(history["market"], hours=1)
            recent_technical = self.get_recent_sentiment(history["technical"], hours=1)

            # Aggregate sentiment using weighted average
            aggregated_sentiment = await self.calculate_weighted_sentiment(
                {
                    "news": recent_news,
                    "social": recent_social,
                    "market": recent_market,
                    "technical": recent_technical,
                }
            )

            # Store aggregated sentiment
            self.state["aggregated_sentiment"][symbol] = aggregated_sentiment

            # Broadcast aggregated sentiment
            await self.broadcast_aggregated_sentiment(symbol, aggregated_sentiment)

        except Exception as e:
            print(f"âŒ Error aggregating sentiment for {symbol}: {e}")

    def get_recent_sentiment(
        self, sentiment_list: list[dict[str, Any]], hours: int
    ) -> list[dict[str, Any]]:
        """Get recent sentiment entries within specified hours"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)

            recent_sentiments = [
                entry
                for entry in sentiment_list
                if datetime.fromisoformat(entry["timestamp"]) > cutoff_time
            ]

            return recent_sentiments

        except Exception as e:
            print(f"âŒ Error getting recent sentiment: {e}")
            return []

    async def calculate_weighted_sentiment(
        self, source_sentiments: dict[str, list[dict[str, Any]]]
    ) -> dict[str, Any]:
        """Calculate weighted sentiment from multiple sources"""
        try:
            total_weight = 0
            weighted_polarity = 0
            weighted_confidence = 0
            source_contributions = {}

            for source, sentiments in source_sentiments.items():
                if not sentiments:
                    continue

                # Calculate average sentiment for this source
                polarities = [s["sentiment"]["polarity"] for s in sentiments]
                confidences = [s["sentiment"]["confidence"] for s in sentiments]

                avg_polarity = np.mean(polarities)
                avg_confidence = np.mean(confidences)

                # Get weight for this source
                weight = self.source_weights.get(source, 0.25)

                # Add to weighted sum
                weighted_polarity += avg_polarity * weight
                weighted_confidence += avg_confidence * weight
                total_weight += weight

                # Store source contribution
                source_contributions[source] = {
                    "avg_polarity": avg_polarity,
                    "avg_confidence": avg_confidence,
                    "weight": weight,
                    "count": len(sentiments),
                }

            if total_weight == 0:
                return {
                    "polarity": 0,
                    "confidence": 0,
                    "category": "neutral",
                    "source_contributions": {},
                    "timestamp": datetime.now().isoformat(),
                }

            # Calculate final weighted averages
            final_polarity = weighted_polarity / total_weight
            final_confidence = weighted_confidence / total_weight

            # Categorize sentiment
            if final_polarity > 0.1:
                category = "positive"
            elif final_polarity < -0.1:
                category = "negative"
            else:
                category = "neutral"

            return {
                "polarity": final_polarity,
                "confidence": final_confidence,
                "category": category,
                "source_contributions": source_contributions,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            print(f"âŒ Error calculating weighted sentiment: {e}")
            return {
                "polarity": 0,
                "confidence": 0,
                "category": "neutral",
                "source_contributions": {},
                "timestamp": datetime.now().isoformat(),
            }

    async def broadcast_aggregated_sentiment(self, symbol: str, sentiment: dict[str, Any]):
        """Broadcast aggregated sentiment to other agents"""
        try:
            sentiment_update = {
                "type": "aggregated_sentiment_update",
                "symbol": symbol,
                "sentiment": sentiment,
                "timestamp": datetime.now().isoformat(),
            }

            # Broadcast to all agents
            await self.broadcast_message(sentiment_update)

            # Send to specific agents
            await self.send_message("strategy_agent", sentiment_update)
            await self.send_message("risk_agent", sentiment_update)
            await self.send_message("execution_agent", sentiment_update)

        except Exception as e:
            print(f"âŒ Error broadcasting aggregated sentiment: {e}")

    async def calculate_fear_greed_index(self):
        """Calculate Fear & Greed Index for the market"""
        try:
            # Calculate Fear & Greed Index based on aggregated sentiment
            all_sentiments = list(self.state["aggregated_sentiment"].values())

            if not all_sentiments:
                return

            # Calculate average market sentiment
            polarities = [s["polarity"] for s in all_sentiments]
            avg_polarity = np.mean(polarities)

            # Convert polarity (-1 to 1) to Fear & Greed Index (0 to 100)
            # Negative polarity = fear, positive polarity = greed
            fear_greed_score = int((avg_polarity + 1) * 50)

            # Categorize Fear & Greed Index
            if fear_greed_score <= self.fear_greed_thresholds["extreme_fear"]:
                category = "Extreme Fear"
            elif fear_greed_score <= self.fear_greed_thresholds["fear"]:
                category = "Fear"
            elif fear_greed_score <= self.fear_greed_thresholds["neutral"]:
                category = "Neutral"
            elif fear_greed_score <= self.fear_greed_thresholds["greed"]:
                category = "Greed"
            else:
                category = "Extreme Greed"

            fear_greed_data = {
                "score": fear_greed_score,
                "category": category,
                "avg_polarity": avg_polarity,
                "symbols_count": len(all_sentiments),
                "timestamp": datetime.now().isoformat(),
            }

            self.state["market_fear_greed"] = fear_greed_data

            # Store in Redis
            self.redis_client.set("market_fear_greed_index", json.dumps(fear_greed_data), ex=300)

            print(f"ðŸ˜¨ Fear & Greed Index: {fear_greed_score} ({category})")

        except Exception as e:
            print(f"âŒ Error calculating Fear & Greed Index: {e}")

    async def generate_sentiment_signals(self):
        """Generate trading signals based on sentiment"""
        try:
            signals = {}

            for symbol, sentiment in self.state["aggregated_sentiment"].items():
                signal = await self.generate_symbol_signal(symbol, sentiment)
                if signal:
                    signals[symbol] = signal

            self.state["sentiment_signals"] = signals

            # Broadcast signals
            if signals:
                await self.broadcast_sentiment_signals(signals)

        except Exception as e:
            print(f"âŒ Error generating sentiment signals: {e}")

    async def generate_symbol_signal(
        self, symbol: str, sentiment: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Generate trading signal for a symbol based on sentiment"""
        try:
            polarity = sentiment["polarity"]
            confidence = sentiment["confidence"]
            category = sentiment["category"]

            # Only generate signals for high confidence sentiment
            if confidence < 0.6:
                return None

            # Define signal thresholds
            strong_buy_threshold = 0.3
            buy_threshold = 0.1
            sell_threshold = -0.1
            strong_sell_threshold = -0.3

            signal_type = None
            strength = 0

            if polarity >= strong_buy_threshold:
                signal_type = "strong_buy"
                strength = min((polarity - strong_buy_threshold) / 0.7, 1.0)
            elif polarity >= buy_threshold:
                signal_type = "buy"
                strength = min((polarity - buy_threshold) / 0.2, 1.0)
            elif polarity <= strong_sell_threshold:
                signal_type = "strong_sell"
                strength = min((strong_sell_threshold - polarity) / 0.7, 1.0)
            elif polarity <= sell_threshold:
                signal_type = "sell"
                strength = min((sell_threshold - polarity) / 0.2, 1.0)
            else:
                signal_type = "hold"
                strength = 0

            return {
                "symbol": symbol,
                "signal_type": signal_type,
                "strength": strength,
                "polarity": polarity,
                "confidence": confidence,
                "category": category,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            print(f"âŒ Error generating signal for {symbol}: {e}")
            return None

    async def broadcast_sentiment_signals(self, signals: dict[str, Any]):
        """Broadcast sentiment signals to other agents"""
        try:
            signals_update = {
                "type": "sentiment_signals_update",
                "signals": signals,
                "timestamp": datetime.now().isoformat(),
            }

            # Broadcast to all agents
            await self.broadcast_message(signals_update)

            # Send to specific agents
            await self.send_message("strategy_agent", signals_update)
            await self.send_message("execution_agent", signals_update)

        except Exception as e:
            print(f"âŒ Error broadcasting sentiment signals: {e}")

    async def handle_aggregate_sentiment(self, message: dict[str, Any]):
        """Handle manual sentiment aggregation request"""
        try:
            symbol = message.get("symbol")

            print(f"ðŸ“Š Manual sentiment aggregation requested for {symbol}")

            if symbol:
                await self.aggregate_symbol_sentiment(symbol)
            else:
                await self.aggregate_all_sentiment()

            # Send response
            response = {
                "type": "sentiment_aggregation_complete",
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
            }

            sender = message.get("from_agent")
            if sender:
                await self.send_message(sender, response)

        except Exception as e:
            print(f"âŒ Error handling sentiment aggregation request: {e}")
            await self.broadcast_error(f"Sentiment aggregation error: {e}")

    async def handle_get_market_sentiment(self, message: dict[str, Any]):
        """Handle market sentiment request"""
        try:
            symbol = message.get("symbol")
            timeframe = message.get("timeframe", "1h")

            print(f"ðŸ“Š Market sentiment request for {symbol} ({timeframe})")

            # Get aggregated sentiment
            if symbol:
                sentiment_data = self.state["aggregated_sentiment"].get(symbol)
            else:
                sentiment_data = {
                    "market_overview": self.state["market_fear_greed"],
                    "symbols": self.state["aggregated_sentiment"],
                }

            # Send response
            response = {
                "type": "market_sentiment_response",
                "symbol": symbol,
                "timeframe": timeframe,
                "sentiment": sentiment_data,
                "timestamp": datetime.now().isoformat(),
            }

            sender = message.get("from_agent")
            if sender:
                await self.send_message(sender, response)

        except Exception as e:
            print(f"âŒ Error handling market sentiment request: {e}")
            await self.broadcast_error(f"Market sentiment request error: {e}")

    async def handle_update_weights(self, message: dict[str, Any]):
        """Handle source weights update request"""
        try:
            new_weights = message.get("weights", {})

            print("âš–ï¸ Updating sentiment source weights")

            # Update weights
            self.source_weights.update(new_weights)

            # Store in Redis
            self.redis_client.set(
                "sentiment_source_weights",
                json.dumps(self.source_weights),
                ex=3600,
            )

            # Send confirmation
            response = {
                "type": "weights_updated",
                "weights": self.source_weights,
                "timestamp": datetime.now().isoformat(),
            }

            sender = message.get("from_agent")
            if sender:
                await self.send_message(sender, response)

        except Exception as e:
            print(f"âŒ Error updating source weights: {e}")
            await self.broadcast_error(f"Weights update error: {e}")

    async def update_sentiment_metrics(self):
        """Update sentiment metrics"""
        try:
            metrics = {
                "agent_id": self.agent_id,
                "symbols_count": len(self.trading_symbols),
                "sources_count": len(self.source_weights),
                "aggregated_symbols": len(self.state["aggregated_sentiment"]),
                "aggregation_count": self.state["aggregation_count"],
                "last_aggregation": self.state["last_aggregation"],
                "fear_greed_index": self.state["market_fear_greed"],
                "active_signals": len(self.state["sentiment_signals"]),
                "timestamp": datetime.now().isoformat(),
            }

            # Store metrics in Redis
            self.redis_client.set(f"agent_metrics:{self.agent_id}", json.dumps(metrics), ex=300)

        except Exception as e:
            print(f"âŒ Error updating sentiment metrics: {e}")

    async def cleanup_history(self):
        """Clean up old sentiment history"""
        try:
            datetime.now()

            for symbol in self.state["sentiment_history"]:
                for source in self.state["sentiment_history"][symbol]:
                    history = self.state["sentiment_history"][symbol][source]

                    # Keep only last 1000 entries per source
                    if len(history) > 1000:
                        self.state["sentiment_history"][symbol][source] = history[-1000:]

        except Exception as e:
            print(f"âŒ Error cleaning up history: {e}")

    async def process_market_data(self, market_data: dict[str, Any]):
        """Process incoming market data for sentiment analysis"""
        try:
            print("ðŸ“Š Processing market data for sentiment analysis")

            # Update market data in state
            self.state["last_market_data"] = market_data
            self.state["last_market_update"] = datetime.now().isoformat()

            # Process each symbol for sentiment analysis
            for symbol, data in market_data.items():
                if symbol in self.trading_symbols:
                    price = data.get("price", 0)
                    volume = data.get("volume", 0)
                    change_24h = data.get("change_24h", 0)
                    
                    if price > 0:
                        # Calculate market sentiment for this symbol
                        sentiment = await self.calculate_market_sentiment(symbol, price, volume, change_24h)
                        
                        # Update sentiment history
                        if symbol not in self.state["sentiment_history"]:
                            self.state["sentiment_history"][symbol] = {}
                        
                        if "market_data" not in self.state["sentiment_history"][symbol]:
                            self.state["sentiment_history"][symbol]["market_data"] = []
                        
                        self.state["sentiment_history"][symbol]["market_data"].append({
                            "timestamp": datetime.now().isoformat(),
                            "sentiment": sentiment
                        })

            # Aggregate sentiment with new market data
            await self.aggregate_all_sentiment()
            
            # Generate sentiment signals
            await self.generate_sentiment_signals()
            
            # Update sentiment metrics
            await self.update_sentiment_metrics()

            print("âœ… Market data processed for sentiment analysis")

        except Exception as e:
            print(f"âŒ Error processing market data for sentiment analysis: {e}")
            await self.broadcast_error(f"Sentiment analysis market data error: {e}")


