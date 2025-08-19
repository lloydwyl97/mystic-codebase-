"""
Chart Pattern Agent
Handles chart pattern recognition and technical analysis using computer vision
"""

import asyncio
import io
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Any

import cv2
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.agents.base_agent import BaseAgent


class ChartPatternAgent(BaseAgent):
    """Chart Pattern Agent - Analyzes price charts and identifies patterns"""

    def __init__(self, agent_id: str = "chart_pattern_agent_001"):
        super().__init__(agent_id, "chart_pattern")

        # Chart pattern-specific state
        self.state.update(
            {
                "patterns_detected": {},
                "chart_cache": {},
                "pattern_templates": {},
                "analysis_history": {},
                "last_analysis": None,
                "analysis_count": 0,
            }
        )

        # Pattern detection configuration
        self.pattern_config = {
            "supported_patterns": [
                "head_and_shoulders",
                "inverse_head_and_shoulders",
                "double_top",
                "double_bottom",
                "triangle_ascending",
                "triangle_descending",
                "triangle_symmetrical",
                "flag_bullish",
                "flag_bearish",
                "wedge_rising",
                "wedge_falling",
                "channel_horizontal",
                "channel_ascending",
                "channel_descending",
            ],
            "confidence_threshold": 0.7,
            "min_pattern_size": 10,  # minimum candles for pattern
            "max_pattern_size": 100,  # maximum candles for pattern
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

        # Chart generation settings
        self.chart_settings = {
            "timeframe": "1h",
            "period": 100,  # number of candles
            "chart_type": "candlestick",
            "indicators": ["sma_20", "sma_50", "volume"],
        }

        # Register chart pattern-specific handlers
        self.register_handler("analyze_chart", self.handle_analyze_chart)
        self.register_handler("detect_patterns", self.handle_detect_patterns)
        self.register_handler("get_pattern_signals", self.handle_get_pattern_signals)
        self.register_handler("market_data", self.handle_market_data)

        print(f"ðŸ“Š Chart Pattern Agent {agent_id} initialized")

    async def initialize(self):
        """Initialize chart pattern agent resources"""
        try:
            # Load pattern configuration
            await self.load_pattern_config()

            # Initialize pattern templates
            await self.initialize_pattern_templates()

            # Initialize computer vision models
            await self.initialize_cv_models()

            # Start chart monitoring
            await self.start_chart_monitoring()

            print(f"âœ… Chart Pattern Agent {self.agent_id} initialized successfully")

        except Exception as e:
            print(f"âŒ Error initializing Chart Pattern Agent: {e}")
            self.update_health_status("error")

    async def process_loop(self):
        """Main chart pattern processing loop"""
        while self.running:
            try:
                # Analyze charts for all symbols
                await self.analyze_all_charts()

                # Update pattern detection models
                await self.update_pattern_models()

                # Generate pattern signals
                await self.generate_pattern_signals()

                # Update pattern metrics
                await self.update_pattern_metrics()

                # Clean up old cache entries
                await self.cleanup_cache()

                await asyncio.sleep(300)  # Check every 5 minutes

            except Exception as e:
                print(f"âŒ Error in chart pattern processing loop: {e}")
                await asyncio.sleep(600)

    async def load_pattern_config(self):
        """Load pattern configuration from Redis"""
        try:
            # Load pattern configuration
            config_data = self.redis_client.get("chart_pattern_config")
            if config_data:
                self.pattern_config = json.loads(config_data)

            # Load trading symbols
            symbols_data = self.redis_client.get("trading_symbols")
            if symbols_data:
                self.trading_symbols = json.loads(symbols_data)

            # Load chart settings
            chart_data = self.redis_client.get("chart_settings")
            if chart_data:
                self.chart_settings = json.loads(chart_data)

            print(
                f"ðŸ“‹ Pattern configuration loaded: "
                f"{len(self.pattern_config['supported_patterns'])} patterns, "
                f"{len(self.trading_symbols)} symbols"
            )

        except Exception as e:
            print(f"âŒ Error loading pattern configuration: {e}")

    async def initialize_pattern_templates(self):
        """Initialize pattern templates for detection"""
        try:
            # Create pattern templates for each supported pattern
            for pattern in self.pattern_config["supported_patterns"]:
                template = await self.create_pattern_template(pattern)
                self.state["pattern_templates"][pattern] = template

            print(
                f"ðŸŽ¯ Pattern templates initialized: {len(self.state['pattern_templates'])} templates"
            )

        except Exception as e:
            print(f"âŒ Error initializing pattern templates: {e}")

    async def create_pattern_template(self, pattern_name: str) -> dict[str, Any]:
        """Create a template for pattern detection"""
        try:
            # Define pattern characteristics
            pattern_templates = {
                "head_and_shoulders": {
                    "type": "reversal",
                    "direction": "bearish",
                    "key_points": 5,
                    "shape": "m_wave",
                    "confidence_factors": ["symmetry", "neckline", "volume"],
                },
                "inverse_head_and_shoulders": {
                    "type": "reversal",
                    "direction": "bullish",
                    "key_points": 5,
                    "shape": "w_wave",
                    "confidence_factors": ["symmetry", "neckline", "volume"],
                },
                "double_top": {
                    "type": "reversal",
                    "direction": "bearish",
                    "key_points": 3,
                    "shape": "m_pattern",
                    "confidence_factors": [
                        "resistance_level",
                        "volume_decline",
                    ],
                },
                "double_bottom": {
                    "type": "reversal",
                    "direction": "bullish",
                    "key_points": 3,
                    "shape": "w_pattern",
                    "confidence_factors": ["support_level", "volume_increase"],
                },
                "triangle_ascending": {
                    "type": "continuation",
                    "direction": "bullish",
                    "key_points": 4,
                    "shape": "triangle_up",
                    "confidence_factors": ["breakout_volume", "support_line"],
                },
                "triangle_descending": {
                    "type": "continuation",
                    "direction": "bearish",
                    "key_points": 4,
                    "shape": "triangle_down",
                    "confidence_factors": [
                        "breakdown_volume",
                        "resistance_line",
                    ],
                },
                "triangle_symmetrical": {
                    "type": "continuation",
                    "direction": "neutral",
                    "key_points": 4,
                    "shape": "triangle_sym",
                    "confidence_factors": [
                        "breakout_direction",
                        "volume_confirmation",
                    ],
                },
                "flag_bullish": {
                    "type": "continuation",
                    "direction": "bullish",
                    "key_points": 3,
                    "shape": "flag_up",
                    "confidence_factors": [
                        "pole_height",
                        "flag_consolidation",
                    ],
                },
                "flag_bearish": {
                    "type": "continuation",
                    "direction": "bearish",
                    "key_points": 3,
                    "shape": "flag_down",
                    "confidence_factors": [
                        "pole_height",
                        "flag_consolidation",
                    ],
                },
                "wedge_rising": {
                    "type": "reversal",
                    "direction": "bearish",
                    "key_points": 4,
                    "shape": "wedge_up",
                    "confidence_factors": ["wedge_angle", "volume_decline"],
                },
                "wedge_falling": {
                    "type": "reversal",
                    "direction": "bullish",
                    "key_points": 4,
                    "shape": "wedge_down",
                    "confidence_factors": ["wedge_angle", "volume_increase"],
                },
                "channel_horizontal": {
                    "type": "continuation",
                    "direction": "neutral",
                    "key_points": 4,
                    "shape": "channel_horiz",
                    "confidence_factors": ["channel_width", "bounce_count"],
                },
                "channel_ascending": {
                    "type": "continuation",
                    "direction": "bullish",
                    "key_points": 4,
                    "shape": "channel_up",
                    "confidence_factors": ["channel_slope", "support_touches"],
                },
                "channel_descending": {
                    "type": "continuation",
                    "direction": "bearish",
                    "key_points": 4,
                    "shape": "channel_down",
                    "confidence_factors": [
                        "channel_slope",
                        "resistance_touches",
                    ],
                },
            }

            return pattern_templates.get(
                pattern_name,
                {
                    "type": "unknown",
                    "direction": "neutral",
                    "key_points": 3,
                    "shape": "unknown",
                    "confidence_factors": ["basic_analysis"],
                },
            )

        except Exception as e:
            print(f"âŒ Error creating pattern template for {pattern_name}: {e}")
            return {
                "type": "unknown",
                "direction": "neutral",
                "key_points": 3,
                "shape": "unknown",
                "confidence_factors": ["basic_analysis"],
            }

    async def initialize_cv_models(self):
        """Initialize computer vision models"""
        try:
            # Initialize OpenCV for image processing
            # In production, you might use more sophisticated models like:
            # - TensorFlow/Keras for deep learning pattern recognition
            # - PyTorch for custom CNN models
            # - Pre-trained models for chart analysis

            # For now, using OpenCV for basic pattern detection
            print("ðŸ‘ï¸ Computer vision models initialized (OpenCV)")

        except Exception as e:
            print(f"âŒ Error initializing CV models: {e}")

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

    async def process_market_data(self, market_data: dict[str, Any]):
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
            cache_key = f"market_data:{symbol}:{timestamp}"
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

    async def analyze_all_charts(self):
        """Analyze charts for all symbols"""
        try:
            print(f"ðŸ“Š Analyzing charts for {len(self.trading_symbols)} symbols...")

            for symbol in self.trading_symbols:
                try:
                    await self.analyze_symbol_chart(symbol)
                except Exception as e:
                    print(f"âŒ Error analyzing chart for {symbol}: {e}")

            # Update analysis count
            self.state["analysis_count"] += 1
            self.state["last_analysis"] = datetime.now().isoformat()

            print("âœ… Chart analysis complete")

        except Exception as e:
            print(f"âŒ Error analyzing all charts: {e}")

    async def analyze_symbol_chart(self, symbol: str):
        """Analyze chart for a specific symbol"""
        try:
            # Get market data for symbol
            market_data = await self.get_symbol_market_data(symbol)

            if not market_data or len(market_data) < self.pattern_config["min_pattern_size"]:
                return

            # Generate chart image
            chart_image = await self.generate_chart_image(symbol, market_data)

            if chart_image is None:
                return

            # Detect patterns in chart
            patterns = await self.detect_patterns_in_chart(chart_image, symbol, market_data)

            # Store detected patterns
            if patterns:
                self.state["patterns_detected"][symbol] = patterns

                # Broadcast pattern detection
                await self.broadcast_pattern_detection(symbol, patterns)

        except Exception as e:
            print(f"âŒ Error analyzing chart for {symbol}: {e}")

    async def get_symbol_market_data(self, symbol: str) -> list[dict[str, Any]]:
        """Get market data for a symbol"""
        try:
            # Get from cache first
            if symbol in self.state["chart_cache"]:
                return self.state["chart_cache"][symbol]

            # Get from Redis
            pattern = f"market_data:{symbol}:*"
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

    async def generate_chart_image(
        self, symbol: str, market_data: list[dict[str, Any]]
    ) -> np.ndarray | None:
        """Generate chart image from market data"""
        try:
            if not market_data:
                return None

            # Extract price and volume data
            timestamps = [datetime.fromisoformat(point["timestamp"]) for point in market_data]
            prices = [point["price"] for point in market_data]
            volumes = [point["volume"] for point in market_data]

            # Create matplotlib figure
            fig, (ax1, ax2) = plt.subplots(
                2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]}
            )

            # Plot price chart
            ax1.plot(timestamps, prices, "b-", linewidth=1)
            ax1.set_title(f"{symbol} Price Chart")
            ax1.set_ylabel("Price")
            ax1.grid(True, alpha=0.3)

            # Format x-axis
            ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax1.xaxis.set_major_locator(mdates.HourLocator(interval=1))

            # Plot volume
            ax2.bar(timestamps, volumes, alpha=0.7, color="gray")
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

            # Convert to numpy array
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            buf.seek(0)

            # Read image
            img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            plt.close()

            return img

        except Exception as e:
            print(f"âŒ Error generating chart image for {symbol}: {e}")
            return None

    async def detect_patterns_in_chart(
        self,
        chart_image: np.ndarray,
        symbol: str,
        market_data: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Detect patterns in chart image"""
        try:
            detected_patterns = []

            # Convert to grayscale for processing
            gray = cv2.cvtColor(chart_image, cv2.COLOR_BGR2GRAY)

            # Detect patterns using template matching and feature detection
            for pattern_name in self.pattern_config["supported_patterns"]:
                confidence = await self.detect_specific_pattern(gray, pattern_name, market_data)

                if confidence >= self.pattern_config["confidence_threshold"]:
                    pattern_info = {
                        "pattern": pattern_name,
                        "confidence": confidence,
                        "symbol": symbol,
                        "timestamp": datetime.now().isoformat(),
                        "template": (self.state["pattern_templates"].get(pattern_name, {})),
                    }

                    detected_patterns.append(pattern_info)

            return detected_patterns

        except Exception as e:
            print(f"âŒ Error detecting patterns in chart: {e}")
            return []

    async def detect_specific_pattern(
        self,
        gray_image: np.ndarray,
        pattern_name: str,
        market_data: list[dict[str, Any]],
    ) -> float:
        """Detect a specific pattern in the chart"""
        try:
            # Extract price data for pattern analysis
            prices = [point["price"] for point in market_data]

            if len(prices) < 10:
                return 0.0

            # Pattern-specific detection logic
            if pattern_name == "head_and_shoulders":
                return await self.detect_head_and_shoulders(prices)
            elif pattern_name == "inverse_head_and_shoulders":
                return await self.detect_inverse_head_and_shoulders(prices)
            elif pattern_name == "double_top":
                return await self.detect_double_top(prices)
            elif pattern_name == "double_bottom":
                return await self.detect_double_bottom(prices)
            elif pattern_name == "triangle_ascending":
                return await self.detect_triangle_ascending(prices)
            elif pattern_name == "triangle_descending":
                return await self.detect_triangle_descending(prices)
            elif pattern_name == "triangle_symmetrical":
                return await self.detect_triangle_symmetrical(prices)
            else:
                # Generic pattern detection using image processing
                return await self.detect_generic_pattern(gray_image, pattern_name)

        except Exception as e:
            print(f"âŒ Error detecting {pattern_name}: {e}")
            return 0.0

    async def detect_head_and_shoulders(self, prices: list[float]) -> float:
        """Detect head and shoulders pattern"""
        try:
            if len(prices) < 20:
                return 0.0

            # Find peaks in price data
            peaks = self.find_peaks(prices)

            if len(peaks) < 3:
                return 0.0

            # Look for H&S pattern: left shoulder, head, right shoulder
            for i in range(len(peaks) - 2):
                left_shoulder = peaks[i]
                head = peaks[i + 1]
                right_shoulder = peaks[i + 2]

                # Check H&S characteristics
                if (
                    head > left_shoulder
                    and head > right_shoulder
                    and abs(left_shoulder - right_shoulder) / head < 0.1
                ):  # shoulders roughly equal

                    # Calculate confidence based on pattern quality
                    confidence = 0.8

                    # Adjust based on pattern symmetry
                    symmetry = 1 - abs(left_shoulder - right_shoulder) / head
                    confidence *= symmetry

                    return min(confidence, 1.0)

            return 0.0

        except Exception as e:
            print(f"âŒ Error detecting head and shoulders: {e}")
            return 0.0

    async def detect_inverse_head_and_shoulders(self, prices: list[float]) -> float:
        """Detect inverse head and shoulders pattern"""
        try:
            if len(prices) < 20:
                return 0.0

            # Find troughs in price data
            troughs = self.find_troughs(prices)

            if len(troughs) < 3:
                return 0.0

            # Look for inverse H&S pattern: left shoulder, head, right shoulder
            for i in range(len(troughs) - 2):
                left_shoulder = troughs[i]
                head = troughs[i + 1]
                right_shoulder = troughs[i + 2]

                # Check inverse H&S characteristics
                if (
                    head < left_shoulder
                    and head < right_shoulder
                    and abs(left_shoulder - right_shoulder) / abs(head) < 0.1
                ):  # shoulders roughly equal

                    # Calculate confidence based on pattern quality
                    confidence = 0.8

                    # Adjust based on pattern symmetry
                    symmetry = 1 - abs(left_shoulder - right_shoulder) / abs(head)
                    confidence *= symmetry

                    return min(confidence, 1.0)

            return 0.0

        except Exception as e:
            print(f"âŒ Error detecting inverse head and shoulders: {e}")
            return 0.0

    async def detect_double_top(self, prices: list[float]) -> float:
        """Detect double top pattern"""
        try:
            if len(prices) < 15:
                return 0.0

            # Find peaks in price data
            peaks = self.find_peaks(prices)

            if len(peaks) < 2:
                return 0.0

            # Look for double top pattern
            for i in range(len(peaks) - 1):
                peak1 = peaks[i]
                peak2 = peaks[i + 1]

                # Check double top characteristics
                if abs(peak1 - peak2) / peak1 < 0.05:  # peaks roughly equal

                    # Calculate confidence based on pattern quality
                    confidence = 0.7

                    # Adjust based on peak similarity
                    similarity = 1 - abs(peak1 - peak2) / peak1
                    confidence *= similarity

                    return min(confidence, 1.0)

            return 0.0

        except Exception as e:
            print(f"âŒ Error detecting double top: {e}")
            return 0.0

    async def detect_double_bottom(self, prices: list[float]) -> float:
        """Detect double bottom pattern"""
        try:
            if len(prices) < 15:
                return 0.0

            # Find troughs in price data
            troughs = self.find_troughs(prices)

            if len(troughs) < 2:
                return 0.0

            # Look for double bottom pattern
            for i in range(len(troughs) - 1):
                trough1 = troughs[i]
                trough2 = troughs[i + 1]

                # Check double bottom characteristics
                if abs(trough1 - trough2) / abs(trough1) < 0.05:  # troughs roughly equal

                    # Calculate confidence based on pattern quality
                    confidence = 0.7

                    # Adjust based on trough similarity
                    similarity = 1 - abs(trough1 - trough2) / abs(trough1)
                    confidence *= similarity

                    return min(confidence, 1.0)

            return 0.0

        except Exception as e:
            print(f"âŒ Error detecting double bottom: {e}")
            return 0.0

    async def detect_triangle_ascending(self, prices: list[float]) -> float:
        """Detect ascending triangle pattern"""
        try:
            if len(prices) < 20:
                return 0.0

            # Find peaks and troughs
            peaks = self.find_peaks(prices)
            troughs = self.find_troughs(prices)

            if len(peaks) < 2 or len(troughs) < 2:
                return 0.0

            # Check for ascending triangle: horizontal resistance, rising support
            # This is a simplified check - in production, you'd use more sophisticated analysis

            # Calculate trend of troughs (should be rising)
            if len(troughs) >= 3:
                trough_trend = np.polyfit(range(len(troughs)), troughs, 1)[0]

                # Calculate resistance level (should be roughly horizontal)
                resistance_variance = np.var(peaks)
                resistance_mean = np.mean(peaks)

                if trough_trend > 0 and resistance_variance / resistance_mean < 0.1:
                    return 0.6

            return 0.0

        except Exception as e:
            print(f"âŒ Error detecting ascending triangle: {e}")
            return 0.0

    async def detect_triangle_descending(self, prices: list[float]) -> float:
        """Detect descending triangle pattern"""
        try:
            if len(prices) < 20:
                return 0.0

            # Find peaks and troughs
            peaks = self.find_peaks(prices)
            troughs = self.find_troughs(prices)

            if len(peaks) < 2 or len(troughs) < 2:
                return 0.0

            # Check for descending triangle: falling resistance, horizontal support
            # This is a simplified check - in production, you'd use more sophisticated analysis

            # Calculate trend of peaks (should be falling)
            if len(peaks) >= 3:
                peak_trend = np.polyfit(range(len(peaks)), peaks, 1)[0]

                # Calculate support level (should be roughly horizontal)
                support_variance = np.var(troughs)
                support_mean = np.mean(troughs)

                if peak_trend < 0 and support_variance / support_mean < 0.1:
                    return 0.6

            return 0.0

        except Exception as e:
            print(f"âŒ Error detecting descending triangle: {e}")
            return 0.0

    async def detect_triangle_symmetrical(self, prices: list[float]) -> float:
        """Detect symmetrical triangle pattern"""
        try:
            if len(prices) < 20:
                return 0.0

            # Find peaks and troughs
            peaks = self.find_peaks(prices)
            troughs = self.find_troughs(prices)

            if len(peaks) < 2 or len(troughs) < 2:
                return 0.0

            # Check for symmetrical triangle: converging lines
            # This is a simplified check - in production, you'd use more sophisticated analysis

            # Calculate trends
            if len(peaks) >= 3 and len(troughs) >= 3:
                peak_trend = np.polyfit(range(len(peaks)), peaks, 1)[0]
                trough_trend = np.polyfit(range(len(troughs)), troughs, 1)[0]

                # Check if lines are converging (opposite trends)
                if (peak_trend < 0 and trough_trend > 0) or (peak_trend > 0 and trough_trend < 0):
                    return 0.5

            return 0.0

        except Exception as e:
            print(f"âŒ Error detecting symmetrical triangle: {e}")
            return 0.0

    async def detect_generic_pattern(self, gray_image: np.ndarray, pattern_name: str) -> float:
        """Detect generic pattern using image processing"""
        try:
            # Use edge detection to find pattern features
            edges = cv2.Canny(gray_image, 50, 150)

            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return 0.0

            # Analyze contours for pattern characteristics
            # This is a simplified approach - in production, you'd use more sophisticated analysis

            # Calculate basic confidence based on contour analysis
            confidence = 0.3

            # Adjust based on pattern-specific characteristics
            if "triangle" in pattern_name:
                # Look for triangular shapes
                for contour in contours:
                    if len(contour) >= 3:
                        # Approximate contour to polygon
                        epsilon = 0.02 * cv2.arcLength(contour, True)
                        approx = cv2.approxPolyDP(contour, epsilon, True)

                        if len(approx) == 3:  # Triangle
                            confidence = 0.6

            return confidence

        except Exception as e:
            print(f"âŒ Error detecting generic pattern {pattern_name}: {e}")
            return 0.0

    def find_peaks(self, prices: list[float]) -> list[float]:
        """Find peaks in price data"""
        try:
            peaks = []
            for i in range(1, len(prices) - 1):
                if prices[i] > prices[i - 1] and prices[i] > prices[i + 1]:
                    peaks.append(prices[i])
            return peaks
        except Exception as e:
            print(f"âŒ Error finding peaks: {e}")
            return []

    def find_troughs(self, prices: list[float]) -> list[float]:
        """Find troughs in price data"""
        try:
            troughs = []
            for i in range(1, len(prices) - 1):
                if prices[i] < prices[i - 1] and prices[i] < prices[i + 1]:
                    troughs.append(prices[i])
            return troughs
        except Exception as e:
            print(f"âŒ Error finding troughs: {e}")
            return []

    async def broadcast_pattern_detection(self, symbol: str, patterns: list[dict[str, Any]]):
        """Broadcast pattern detection to other agents"""
        try:
            pattern_update = {
                "type": "pattern_detection_update",
                "symbol": symbol,
                "patterns": patterns,
                "timestamp": datetime.now().isoformat(),
            }

            # Broadcast to all agents
            await self.broadcast_message(pattern_update)

            # Send to specific agents
            await self.send_message("strategy_agent", pattern_update)
            await self.send_message("risk_agent", pattern_update)

        except Exception as e:
            print(f"âŒ Error broadcasting pattern detection: {e}")

    async def update_pattern_models(self):
        """Update pattern detection models"""
        try:
            # Update pattern templates based on recent detections
            # In production, you'd implement machine learning model updates here

            print("ðŸ”„ Pattern models updated")

        except Exception as e:
            print(f"âŒ Error updating pattern models: {e}")

    async def generate_pattern_signals(self):
        """Generate trading signals based on detected patterns"""
        try:
            signals = {}

            for symbol, patterns in self.state["patterns_detected"].items():
                if patterns:
                    signal = await self.generate_symbol_pattern_signal(symbol, patterns)
                    if signal:
                        signals[symbol] = signal

            # Broadcast signals
            if signals:
                await self.broadcast_pattern_signals(signals)

        except Exception as e:
            print(f"âŒ Error generating pattern signals: {e}")

    async def generate_symbol_pattern_signal(
        self, symbol: str, patterns: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        """Generate trading signal for a symbol based on patterns"""
        try:
            if not patterns:
                return None

            # Find the highest confidence pattern
            best_pattern = max(patterns, key=lambda x: x["confidence"])

            pattern_name = best_pattern["pattern"]
            confidence = best_pattern["confidence"]
            template = best_pattern["template"]

            # Generate signal based on pattern type and direction
            signal_type = "hold"
            strength = confidence

            if template.get("type") == "reversal" or template.get("type") == "continuation":
                if template.get("direction") == "bullish":
                    signal_type = "buy"
                elif template.get("direction") == "bearish":
                    signal_type = "sell"

            return {
                "symbol": symbol,
                "signal_type": signal_type,
                "strength": strength,
                "pattern": pattern_name,
                "confidence": confidence,
                "template": template,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            print(f"âŒ Error generating pattern signal for {symbol}: {e}")
            return None

    async def broadcast_pattern_signals(self, signals: dict[str, Any]):
        """Broadcast pattern signals to other agents"""
        try:
            signals_update = {
                "type": "pattern_signals_update",
                "signals": signals,
                "timestamp": datetime.now().isoformat(),
            }

            # Broadcast to all agents
            await self.broadcast_message(signals_update)

            # Send to specific agents
            await self.send_message("strategy_agent", signals_update)
            await self.send_message("execution_agent", signals_update)

        except Exception as e:
            print(f"âŒ Error broadcasting pattern signals: {e}")

    async def handle_analyze_chart(self, message: dict[str, Any]):
        """Handle manual chart analysis request"""
        try:
            symbol = message.get("symbol")

            print(f"ðŸ“Š Manual chart analysis requested for {symbol}")

            if symbol:
                await self.analyze_symbol_chart(symbol)

            # Send response
            response = {
                "type": "chart_analysis_complete",
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
            }

            sender = message.get("from_agent")
            if sender:
                await self.send_message(sender, response)

        except Exception as e:
            print(f"âŒ Error handling chart analysis request: {e}")
            await self.broadcast_error(f"Chart analysis error: {e}")

    async def handle_detect_patterns(self, message: dict[str, Any]):
        """Handle pattern detection request"""
        try:
            symbol = message.get("symbol")
            pattern_types = message.get("patterns", self.pattern_config["supported_patterns"])

            print(f"ðŸŽ¯ Pattern detection requested for {symbol}")

            if symbol:
                # Get market data and detect specific patterns
                market_data = await self.get_symbol_market_data(symbol)
                if market_data:
                    chart_image = await self.generate_chart_image(symbol, market_data)
                    if chart_image:
                        patterns = await self.detect_patterns_in_chart(
                            chart_image, symbol, market_data
                        )

                        # Filter by requested pattern types
                        filtered_patterns = [p for p in patterns if p["pattern"] in pattern_types]

                        # Send response
                        response = {
                            "type": "pattern_detection_response",
                            "symbol": symbol,
                            "patterns": filtered_patterns,
                            "timestamp": datetime.now().isoformat(),
                        }

                        sender = message.get("from_agent")
                        if sender:
                            await self.send_message(sender, response)

        except Exception as e:
            print(f"âŒ Error handling pattern detection request: {e}")
            await self.broadcast_error(f"Pattern detection error: {e}")

    async def handle_get_pattern_signals(self, message: dict[str, Any]):
        """Handle pattern signals request"""
        try:
            symbol = message.get("symbol")

            print(f"ðŸ“ˆ Pattern signals requested for {symbol}")

            # Get pattern signals
            if symbol and symbol in self.state["patterns_detected"]:
                patterns = self.state["patterns_detected"][symbol]
                signal = await self.generate_symbol_pattern_signal(symbol, patterns)

                response = {
                    "type": "pattern_signals_response",
                    "symbol": symbol,
                    "signal": signal,
                    "timestamp": datetime.now().isoformat(),
                }
            else:
                response = {
                    "type": "pattern_signals_response",
                    "symbol": symbol,
                    "signal": None,
                    "timestamp": datetime.now().isoformat(),
                }

            sender = message.get("from_agent")
            if sender:
                await self.send_message(sender, response)

        except Exception as e:
            print(f"âŒ Error handling pattern signals request: {e}")
            await self.broadcast_error(f"Pattern signals error: {e}")

    async def update_pattern_metrics(self):
        """Update pattern metrics"""
        try:
            metrics = {
                "agent_id": self.agent_id,
                "symbols_count": len(self.trading_symbols),
                "patterns_count": len(self.pattern_config["supported_patterns"]),
                "detected_patterns": len(self.state["patterns_detected"]),
                "analysis_count": self.state["analysis_count"],
                "last_analysis": self.state["last_analysis"],
                "cache_size": len(self.state["chart_cache"]),
                "timestamp": datetime.now().isoformat(),
            }

            # Store metrics in Redis
            self.redis_client.set(f"agent_metrics:{self.agent_id}", json.dumps(metrics), ex=300)

        except Exception as e:
            print(f"âŒ Error updating pattern metrics: {e}")

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


