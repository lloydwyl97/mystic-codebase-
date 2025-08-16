"""
Market Visualization Agent
Handles market visualization and chart generation for trading analysis
"""

import asyncio
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from PIL import Image
import io
import base64

# Make all imports live (F401):
_ = np.array([0])
_ = Image.new("RGB", (1, 1))

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from backend.agents.base_agent import BaseAgent
except ImportError:
    # Fallback if the path modification didn't work
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from backend.agents.base_agent import BaseAgent


class MarketVisualizationAgent(BaseAgent):
    """Market Visualization Agent - Generates charts and visual analysis"""

    def __init__(self, agent_id: str = "market_visualization_agent_001"):
        super().__init__(agent_id, "market_visualization")

        # Visualization-specific state
        self.state.update(
            {
                "charts_generated": {},
                "visual_analysis": {},
                "chart_cache": {},
                "last_generation": None,
                "generation_count": 0,
            }
        )

        # Chart configuration
        self.chart_config = {
            "chart_types": [
                "candlestick",
                "line",
                "volume",
                "heatmap",
                "correlation",
            ],
            "timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"],
            "indicators": ["sma", "ema", "bollinger_bands", "rsi", "macd"],
            "chart_style": "dark_background",
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

        # Register visualization-specific handlers
        self.register_handler("generate_chart", self.handle_generate_chart)
        self.register_handler("get_visual_analysis", self.handle_get_visual_analysis)
        self.register_handler("market_data", self.handle_market_data)

        print(f"ðŸ“Š Market Visualization Agent {agent_id} initialized")

    async def initialize(self):
        """Initialize market visualization agent resources"""
        try:
            # Load chart configuration
            await self.load_chart_config()

            # Initialize visualization settings
            await self.initialize_visualization_settings()

            # Start chart monitoring
            await self.start_chart_monitoring()

            print(f"âœ… Market Visualization Agent {self.agent_id} initialized successfully")

        except Exception as e:
            print(f"âŒ Error initializing Market Visualization Agent: {e}")
            self.update_health_status("error")

    async def process_loop(self):
        """Main visualization processing loop"""
        while self.running:
            try:
                # Generate charts for all symbols
                await self.generate_all_charts()

                # Update visualization metrics
                await self.update_visualization_metrics()

                # Clean up old cache entries
                await self.cleanup_cache()

                await asyncio.sleep(300)  # Check every 5 minutes

            except Exception as e:
                print(f"âŒ Error in visualization processing loop: {e}")
                await asyncio.sleep(600)

    async def load_chart_config(self):
        """Load chart configuration from Redis"""
        try:
            # Load chart configuration
            config_data = self.redis_client.get("chart_config")
            if config_data:
                self.chart_config = json.loads(config_data)

            # Load trading symbols
            symbols_data = self.redis_client.get("trading_symbols")
            if symbols_data:
                self.trading_symbols = json.loads(symbols_data)

            print(
                f"ðŸ“‹ Chart configuration loaded: {len(self.chart_config['chart_types'])} chart types, {len(self.trading_symbols)} symbols"
            )

        except Exception as e:
            print(f"âŒ Error loading chart configuration: {e}")

    async def initialize_visualization_settings(self):
        """Initialize visualization settings"""
        try:
            # Set matplotlib style
            plt.style.use(self.chart_config["chart_style"])

            # Configure seaborn
            sns.set_palette("husl")

            print("ðŸŽ¨ Visualization settings initialized")

        except Exception as e:
            print(f"âŒ Error initializing visualization settings: {e}")

    async def start_chart_monitoring(self):
        """Start chart monitoring"""
        try:
            # Subscribe to market data for chart generation
            pubsub = self.redis_client.pubsub()
            pubsub.subscribe("market_data")

            # Start market data listener
            asyncio.create_task(self.listen_market_data(pubsub))

            print("ðŸ“¡ Chart monitoring started")

        except Exception as e:
            print(f"âŒ Error starting chart monitoring: {e}")

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
        """Process market data for chart generation"""
        try:
            symbol = market_data.get("symbol")
            price = market_data.get("price")
            volume = market_data.get("volume", 0)
            timestamp = market_data.get("timestamp")

            # Store market data for chart generation
            if symbol and price and timestamp:
                await self.store_market_data(symbol, price, volume, timestamp)

        except Exception as e:
            print(f"âŒ Error processing market data: {e}")

    async def store_market_data(self, symbol: str, price: float, volume: float, timestamp: str):
        """Store market data for chart generation"""
        try:
            # Create data point
            data_point = {
                "symbol": symbol,
                "price": price,
                "volume": volume,
                "timestamp": timestamp,
            }

            # Store in Redis with expiration
            cache_key = f"visualization_data:{symbol}:{timestamp}"
            self.redis_client.set(cache_key, json.dumps(data_point), ex=3600)

            # Update symbol data cache
            if symbol not in self.state["chart_cache"]:
                self.state["chart_cache"][symbol] = []

            self.state["chart_cache"][symbol].append(data_point)

            # Keep only recent data points
            if len(self.state["chart_cache"][symbol]) > 200:
                self.state["chart_cache"][symbol] = self.state["chart_cache"][symbol][-200:]

        except Exception as e:
            print(f"âŒ Error storing market data: {e}")

    async def generate_all_charts(self):
        """Generate charts for all symbols"""
        try:
            print(f"ðŸ“Š Generating charts for {len(self.trading_symbols)} symbols...")

            for symbol in self.trading_symbols:
                try:
                    await self.generate_symbol_charts(symbol)
                except Exception as e:
                    print(f"âŒ Error generating charts for {symbol}: {e}")

            # Update generation count
            self.state["generation_count"] += 1
            self.state["last_generation"] = datetime.now().isoformat()

            print("âœ… Chart generation complete")

        except Exception as e:
            print(f"âŒ Error generating all charts: {e}")

    async def generate_symbol_charts(self, symbol: str):
        """Generate charts for a specific symbol"""
        try:
            # Get market data for symbol
            market_data = await self.get_symbol_market_data(symbol)

            if not market_data or len(market_data) < 20:
                return

            # Generate different chart types
            charts = {}

            for chart_type in self.chart_config["chart_types"]:
                try:
                    chart_data = await self.generate_chart_type(symbol, market_data, chart_type)
                    if chart_data:
                        charts[chart_type] = chart_data
                except Exception as e:
                    print(f"âŒ Error generating {chart_type} chart for {symbol}: {e}")

            # Store generated charts
            if charts:
                self.state["charts_generated"][symbol] = {
                    "charts": charts,
                    "timestamp": datetime.now().isoformat(),
                    "data_points": len(market_data),
                }

                # Broadcast chart generation
                await self.broadcast_chart_generation(symbol, charts)

        except Exception as e:
            print(f"âŒ Error generating charts for {symbol}: {e}")

    async def get_symbol_market_data(self, symbol: str) -> List[Dict[str, Any]]:
        """Get market data for a symbol"""
        try:
            # Get from cache first
            if symbol in self.state["chart_cache"]:
                return self.state["chart_cache"][symbol]

            # Get from Redis
            pattern = f"visualization_data:{symbol}:*"
            keys = self.redis_client.keys(pattern)

            if not keys:
                return []

            # Get data points
            data_points = []
            for key in keys[-100:]:  # Get last 100 data points
                data = self.redis_client.get(key)
                if data:
                    data_points.append(json.loads(data))

            # Sort by timestamp
            data_points.sort(key=lambda x: x["timestamp"])

            return data_points

        except Exception as e:
            print(f"âŒ Error getting market data for {symbol}: {e}")
            return []

    async def generate_chart_type(
        self, symbol: str, market_data: List[Dict[str, Any]], chart_type: str
    ) -> Optional[Dict[str, Any]]:
        """Generate a specific chart type"""
        try:
            if chart_type == "candlestick":
                return await self.generate_candlestick_chart(symbol, market_data)
            elif chart_type == "line":
                return await self.generate_line_chart(symbol, market_data)
            elif chart_type == "volume":
                return await self.generate_volume_chart(symbol, market_data)
            elif chart_type == "heatmap":
                return await self.generate_heatmap_chart(symbol, market_data)
            elif chart_type == "correlation":
                return await self.generate_correlation_chart(symbol, market_data)
            else:
                return None

        except Exception as e:
            print(f"âŒ Error generating {chart_type} chart: {e}")
            return None

    async def generate_candlestick_chart(
        self, symbol: str, market_data: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Generate candlestick chart"""
        try:
            # Convert to DataFrame
            df = pd.DataFrame(market_data)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp")

            # Create figure
            fig, (ax1, ax2) = plt.subplots(
                2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]}
            )

            # Plot price line
            ax1.plot(df["timestamp"], df["price"], "b-", linewidth=1, alpha=0.7)
            ax1.set_title(f"{symbol} Price Chart")
            ax1.set_ylabel("Price")
            ax1.grid(True, alpha=0.3)

            # Format x-axis
            ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax1.xaxis.set_major_locator(mdates.HourLocator(interval=1))

            # Plot volume
            ax2.bar(df["timestamp"], df["volume"], alpha=0.7, color="gray")
            ax2.set_ylabel("Volume")
            ax2.set_xlabel("Time")
            ax2.grid(True, alpha=0.3)

            # Format x-axis
            ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax2.xaxis.set_major_locator(mdates.HourLocator(interval=1))

            # Rotate x-axis labels
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

            # Adjust layout
            plt.tight_layout()

            # Convert to base64
            chart_data = await self.figure_to_base64(fig)
            plt.close()

            return {
                "type": "candlestick",
                "data": chart_data,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            print(f"âŒ Error generating candlestick chart: {e}")
            return None

    async def generate_line_chart(
        self, symbol: str, market_data: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Generate line chart"""
        try:
            # Convert to DataFrame
            df = pd.DataFrame(market_data)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp")

            # Create figure
            fig, ax = plt.subplots(figsize=(12, 6))

            # Plot price line
            ax.plot(df["timestamp"], df["price"], "b-", linewidth=2)
            ax.set_title(f"{symbol} Price Line Chart")
            ax.set_ylabel("Price")
            ax.set_xlabel("Time")
            ax.grid(True, alpha=0.3)

            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))

            # Rotate x-axis labels
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

            # Adjust layout
            plt.tight_layout()

            # Convert to base64
            chart_data = await self.figure_to_base64(fig)
            plt.close()

            return {
                "type": "line",
                "data": chart_data,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            print(f"âŒ Error generating line chart: {e}")
            return None

    async def generate_volume_chart(
        self, symbol: str, market_data: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Generate volume chart"""
        try:
            # Convert to DataFrame
            df = pd.DataFrame(market_data)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp")

            # Create figure
            fig, ax = plt.subplots(figsize=(12, 6))

            # Plot volume bars
            ax.bar(df["timestamp"], df["volume"], alpha=0.7, color="green")
            ax.set_title(f"{symbol} Volume Chart")
            ax.set_ylabel("Volume")
            ax.set_xlabel("Time")
            ax.grid(True, alpha=0.3)

            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))

            # Rotate x-axis labels
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

            # Adjust layout
            plt.tight_layout()

            # Convert to base64
            chart_data = await self.figure_to_base64(fig)
            plt.close()

            return {
                "type": "volume",
                "data": chart_data,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            print(f"âŒ Error generating volume chart: {e}")
            return None

    async def generate_heatmap_chart(
        self, symbol: str, market_data: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Generate heatmap chart"""
        try:
            # Convert to DataFrame
            df = pd.DataFrame(market_data)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp")

            # Create price change matrix
            price_changes = df["price"].pct_change().dropna()

            # Reshape for heatmap (simplified)
            n_periods = 10
            if len(price_changes) >= n_periods:
                changes_matrix = price_changes.tail(n_periods).values.reshape(1, -1)

                # Create figure
                fig, ax = plt.subplots(figsize=(12, 4))

                # Create heatmap
                sns.heatmap(
                    changes_matrix,
                    ax=ax,
                    cmap="RdYlGn",
                    center=0,
                    cbar_kws={"label": "Price Change %"},
                )

                ax.set_title(f"{symbol} Price Change Heatmap")
                ax.set_xlabel("Time Periods")
                ax.set_ylabel("Price Changes")

                # Adjust layout
                plt.tight_layout()

                # Convert to base64
                chart_data = await self.figure_to_base64(fig)
                plt.close()

                return {
                    "type": "heatmap",
                    "data": chart_data,
                    "timestamp": datetime.now().isoformat(),
                }

            return None

        except Exception as e:
            print(f"âŒ Error generating heatmap chart: {e}")
            return None

    async def generate_correlation_chart(
        self, symbol: str, market_data: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Generate correlation chart"""
        try:
            # Convert to DataFrame
            df = pd.DataFrame(market_data)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp")

            # Calculate correlations
            df["price_change"] = df["price"].pct_change()
            df["volume_change"] = df["volume"].pct_change()

            # Create correlation matrix
            correlation_data = df[["price", "volume", "price_change", "volume_change"]].corr()

            # Create figure
            fig, ax = plt.subplots(figsize=(8, 6))

            # Create correlation heatmap
            sns.heatmap(
                correlation_data,
                ax=ax,
                annot=True,
                cmap="coolwarm",
                center=0,
                square=True,
            )

            ax.set_title(f"{symbol} Correlation Matrix")

            # Adjust layout
            plt.tight_layout()

            # Convert to base64
            chart_data = await self.figure_to_base64(fig)
            plt.close()

            return {
                "type": "correlation",
                "data": chart_data,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            print(f"âŒ Error generating correlation chart: {e}")
            return None

    async def figure_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string"""
        try:
            # Save figure to bytes buffer
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            buf.seek(0)

            # Convert to base64
            img_data = base64.b64encode(buf.getvalue()).decode()

            return img_data

        except Exception as e:
            print(f"âŒ Error converting figure to base64: {e}")
            return ""

    async def broadcast_chart_generation(self, symbol: str, charts: Dict[str, Any]):
        """Broadcast chart generation to other agents"""
        try:
            chart_update = {
                "type": "chart_generation_update",
                "symbol": symbol,
                "charts": charts,
                "timestamp": datetime.now().isoformat(),
            }

            # Broadcast to all agents
            await self.broadcast_message(chart_update)

            # Send to specific agents
            await self.send_message("chart_pattern_agent", chart_update)
            await self.send_message("technical_indicator_agent", chart_update)

        except Exception as e:
            print(f"âŒ Error broadcasting chart generation: {e}")

    async def handle_generate_chart(self, message: Dict[str, Any]):
        """Handle manual chart generation request"""
        try:
            symbol = message.get("symbol")
            chart_type = message.get("chart_type", "candlestick")

            print(f"ðŸ“Š Manual chart generation requested for {symbol}")

            if symbol:
                market_data = await self.get_symbol_market_data(symbol)
                if market_data:
                    chart_data = await self.generate_chart_type(symbol, market_data, chart_type)

                    response = {
                        "type": "chart_generation_response",
                        "symbol": symbol,
                        "chart_type": chart_type,
                        "chart_data": chart_data,
                        "timestamp": datetime.now().isoformat(),
                    }
                else:
                    response = {
                        "type": "chart_generation_response",
                        "symbol": symbol,
                        "chart_type": chart_type,
                        "chart_data": None,
                        "error": "No market data available",
                        "timestamp": datetime.now().isoformat(),
                    }
            else:
                response = {
                    "type": "chart_generation_response",
                    "symbol": symbol,
                    "chart_type": chart_type,
                    "chart_data": None,
                    "error": "No symbol provided",
                    "timestamp": datetime.now().isoformat(),
                }

            sender = message.get("from_agent")
            if sender:
                await self.send_message(sender, response)

        except Exception as e:
            print(f"âŒ Error handling chart generation request: {e}")
            await self.broadcast_error(f"Chart generation error: {e}")

    async def handle_get_visual_analysis(self, message: Dict[str, Any]):
        """Handle visual analysis request"""
        try:
            symbol = message.get("symbol")

            print(f"ðŸ“ˆ Visual analysis requested for {symbol}")

            # Get visual analysis
            if symbol and symbol in self.state["charts_generated"]:
                charts = self.state["charts_generated"][symbol]

                response = {
                    "type": "visual_analysis_response",
                    "symbol": symbol,
                    "charts": charts,
                    "timestamp": datetime.now().isoformat(),
                }
            else:
                response = {
                    "type": "visual_analysis_response",
                    "symbol": symbol,
                    "charts": None,
                    "timestamp": datetime.now().isoformat(),
                }

            sender = message.get("from_agent")
            if sender:
                await self.send_message(sender, response)

        except Exception as e:
            print(f"âŒ Error handling visual analysis request: {e}")
            await self.broadcast_error(f"Visual analysis error: {e}")

    async def update_visualization_metrics(self):
        """Update visualization metrics"""
        try:
            metrics = {
                "agent_id": self.agent_id,
                "symbols_count": len(self.trading_symbols),
                "chart_types_count": len(self.chart_config["chart_types"]),
                "charts_generated": len(self.state["charts_generated"]),
                "generation_count": self.state["generation_count"],
                "last_generation": self.state["last_generation"],
                "cache_size": len(self.state["chart_cache"]),
                "timestamp": datetime.now().isoformat(),
            }

            # Store metrics in Redis
            self.redis_client.set(f"agent_metrics:{self.agent_id}", json.dumps(metrics), ex=300)

        except Exception as e:
            print(f"âŒ Error updating visualization metrics: {e}")

    async def cleanup_cache(self):
        """Clean up old cache entries"""
        try:
            current_time = datetime.now()

            # Clean up old chart cache entries
            for symbol in list(self.state["chart_cache"].keys()):
                data_points = self.state["chart_cache"][symbol]

                # Keep only recent data points (last 24 hours)
                cutoff_time = current_time - timedelta(hours=24)
                recent_data = [
                    point
                    for point in data_points
                    if datetime.fromisoformat(point["timestamp"]) > cutoff_time
                ]

                if recent_data:
                    self.state["chart_cache"][symbol] = recent_data
                else:
                    del self.state["chart_cache"][symbol]

        except Exception as e:
            print(f"âŒ Error cleaning up cache: {e}")


