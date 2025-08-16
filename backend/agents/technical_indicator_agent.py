"""
Technical Indicator Agent
Handles technical indicator analysis and signal generation
"""

import asyncio
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.agents.base_agent import BaseAgent


class TechnicalIndicatorAgent(BaseAgent):
    """Technical Indicator Agent - Analyzes technical indicators and generates signals"""

    def __init__(self, agent_id: str = "technical_indicator_agent_001"):
        super().__init__(agent_id, "technical_indicator")

        # Technical indicator-specific state
        self.state.update(
            {
                "indicators_calculated": {},
                "signals_generated": {},
                "indicator_cache": {},
                "analysis_history": {},
                "last_analysis": None,
                "analysis_count": 0,
            }
        )

        # Indicator configuration
        self.indicator_config = {
            "supported_indicators": [
                "sma",
                "ema",
                "rsi",
                "macd",
                "bollinger_bands",
                "stochastic",
                "williams_r",
                "cci",
                "adx",
                "atr",
                "obv",
                "volume_sma",
                "price_channels",
                "parabolic_sar",
            ],
            "signal_thresholds": {
                "rsi_oversold": 30,
                "rsi_overbought": 70,
                "stochastic_oversold": 20,
                "stochastic_overbought": 80,
                "williams_r_oversold": -80,
                "williams_r_overbought": -20,
                "cci_oversold": -100,
                "cci_overbought": 100,
            },
            "crossover_threshold": (0.001),  # minimum change for crossover detection
            "signal_confidence": {
                "strong_buy": 0.8,
                "buy": 0.6,
                "neutral": 0.4,
                "sell": 0.6,
                "strong_sell": 0.8,
            },
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

        # Timeframe settings
        self.timeframe_settings = {
            "short_term": {"period": 14, "multiplier": 1},
            "medium_term": {"period": 20, "multiplier": 2},
            "long_term": {"period": 50, "multiplier": 3},
        }

        # Register technical indicator-specific handlers
        self.register_handler("calculate_indicators", self.handle_calculate_indicators)
        self.register_handler("get_indicator_signals", self.handle_get_indicator_signals)
        self.register_handler("analyze_crossovers", self.handle_analyze_crossovers)
        self.register_handler("market_data", self.handle_market_data)

        print(f"ðŸ“ˆ Technical Indicator Agent {agent_id} initialized")

    async def initialize(self):
        """Initialize technical indicator agent resources"""
        try:
            # Load indicator configuration
            await self.load_indicator_config()

            # Initialize indicator calculation functions
            await self.initialize_indicator_functions()

            # Start indicator monitoring
            await self.start_indicator_monitoring()

            print(f"âœ… Technical Indicator Agent {self.agent_id} initialized successfully")

        except Exception as e:
            print(f"âŒ Error initializing Technical Indicator Agent: {e}")
            self.update_health_status("error")

    async def process_loop(self):
        """Main technical indicator processing loop"""
        while self.running:
            try:
                # Calculate indicators for all symbols
                await self.calculate_all_indicators()

                # Generate indicator signals
                await self.generate_indicator_signals()

                # Analyze crossovers
                await self.analyze_all_crossovers()

                # Update indicator metrics
                await self.update_indicator_metrics()

                # Clean up old cache entries
                await self.cleanup_cache()

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                print(f"âŒ Error in technical indicator processing loop: {e}")
                await asyncio.sleep(120)

    async def load_indicator_config(self):
        """Load indicator configuration from Redis"""
        try:
            # Load indicator configuration
            config_data = self.redis_client.get("technical_indicator_config")
            if config_data:
                self.indicator_config = json.loads(config_data)

            # Load trading symbols
            symbols_data = self.redis_client.get("trading_symbols")
            if symbols_data:
                self.trading_symbols = json.loads(symbols_data)

            # Load timeframe settings
            timeframe_data = self.redis_client.get("timeframe_settings")
            if timeframe_data:
                self.timeframe_settings = json.loads(timeframe_data)

            print(
                f"ðŸ“‹ Indicator configuration loaded: {len(self.indicator_config['supported_indicators'])} indicators, {len(self.trading_symbols)} symbols"
            )

        except Exception as e:
            print(f"âŒ Error loading indicator configuration: {e}")

    async def initialize_indicator_functions(self):
        """Initialize indicator calculation functions"""
        try:
            # Initialize pandas-ta functions for each indicator
            self.indicator_functions = {
                "sma": lambda df, period: df.ta.sma(length=period),
                "ema": lambda df, period: df.ta.ema(length=period),
                "rsi": lambda df, period: df.ta.rsi(length=period),
                "macd": lambda df, fast=12, slow=26, signal=9: df.ta.macd(
                    fast=fast, slow=slow, signal=signal
                ),
                "bollinger_bands": lambda df, period=20, std=2: df.ta.bbands(
                    length=period, std=std
                ),
                "stochastic": lambda df, k=14, d=3: df.ta.stoch(k=k, d=d),
                "williams_r": lambda df, period=14: df.ta.willr(length=period),
                "cci": lambda df, period=20: df.ta.cci(length=period),
                "adx": lambda df, period=14: df.ta.adx(length=period),
                "atr": lambda df, period=14: df.ta.atr(length=period),
                "obv": lambda df: df.ta.obv(),
                "parabolic_sar": lambda df, af0=0.02, af=0.02, max_af=0.2: df.ta.psar(
                    af0=af0, af=af, max_af=max_af
                ),
            }

            print(f"ðŸ“Š Indicator functions initialized: {len(self.indicator_functions)} functions")

        except Exception as e:
            print(f"âŒ Error initializing indicator functions: {e}")

    async def start_indicator_monitoring(self):
        """Start indicator monitoring"""
        try:
            # Subscribe to market data for indicator calculation
            pubsub = self.redis_client.pubsub()
            pubsub.subscribe("market_data")

            # Start market data listener
            asyncio.create_task(self.listen_market_data(pubsub))

            print("ðŸ“¡ Indicator monitoring started")

        except Exception as e:
            print(f"âŒ Error starting indicator monitoring: {e}")

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
        """Process market data for indicator calculation"""
        try:
            symbol = market_data.get("symbol")
            price = market_data.get("price")
            volume = market_data.get("volume", 0)
            timestamp = market_data.get("timestamp")

            # Store market data for indicator calculation
            if symbol and price and timestamp:
                await self.store_market_data(symbol, price, volume, timestamp)

        except Exception as e:
            print(f"âŒ Error processing market data: {e}")

    async def store_market_data(self, symbol: str, price: float, volume: float, timestamp: str):
        """Store market data for indicator calculation"""
        try:
            # Create data point
            data_point = {
                "symbol": symbol,
                "price": price,
                "volume": volume,
                "timestamp": timestamp,
            }

            # Store in Redis with expiration
            cache_key = f"indicator_data:{symbol}:{timestamp}"
            self.redis_client.set(cache_key, json.dumps(data_point), ex=3600)

            # Update symbol data cache
            if symbol not in self.state["indicator_cache"]:
                self.state["indicator_cache"][symbol] = []

            self.state["indicator_cache"][symbol].append(data_point)

            # Keep only recent data points
            if len(self.state["indicator_cache"][symbol]) > 200:
                self.state["indicator_cache"][symbol] = self.state["indicator_cache"][symbol][-200:]

        except Exception as e:
            print(f"âŒ Error storing market data: {e}")

    async def calculate_all_indicators(self):
        """Calculate indicators for all symbols"""
        try:
            print(f"ðŸ“Š Calculating indicators for {len(self.trading_symbols)} symbols...")

            for symbol in self.trading_symbols:
                try:
                    await self.calculate_symbol_indicators(symbol)
                except Exception as e:
                    print(f"âŒ Error calculating indicators for {symbol}: {e}")

            # Update analysis count
            self.state["analysis_count"] += 1
            self.state["last_analysis"] = datetime.now().isoformat()

            print("âœ… Indicator calculation complete")

        except Exception as e:
            print(f"âŒ Error calculating all indicators: {e}")

    async def calculate_symbol_indicators(self, symbol: str):
        """Calculate indicators for a specific symbol"""
        try:
            # Get market data for symbol
            market_data = await self.get_symbol_market_data(symbol)

            if not market_data or len(market_data) < 50:  # Need minimum data for indicators
                return

            # Convert to pandas DataFrame
            df = pd.DataFrame(market_data)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp")

            # Calculate all supported indicators
            indicators = {}

            for indicator_name in self.indicator_config["supported_indicators"]:
                try:
                    indicator_value = await self.calculate_indicator(df, indicator_name)
                    if indicator_value is not None:
                        indicators[indicator_name] = indicator_value
                except Exception as e:
                    print(f"âŒ Error calculating {indicator_name} for {symbol}: {e}")

            # Store calculated indicators
            if indicators:
                self.state["indicators_calculated"][symbol] = {
                    "indicators": indicators,
                    "timestamp": datetime.now().isoformat(),
                    "data_points": len(market_data),
                }

                # Broadcast indicator update
                await self.broadcast_indicator_update(symbol, indicators)

        except Exception as e:
            print(f"âŒ Error calculating indicators for {symbol}: {e}")

    async def get_symbol_market_data(self, symbol: str) -> List[Dict[str, Any]]:
        """Get market data for a symbol"""
        try:
            # Get from cache first
            if symbol in self.state["indicator_cache"]:
                return self.state["indicator_cache"][symbol]

            # Get from Redis
            pattern = f"indicator_data:{symbol}:*"
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

    async def calculate_indicator(
        self, df: pd.DataFrame, indicator_name: str
    ) -> Optional[Dict[str, Any]]:
        """Calculate a specific indicator using pandas-ta"""
        try:
            if indicator_name not in self.indicator_functions:
                return None

            func = self.indicator_functions[indicator_name]

            # Prepare DataFrame with required columns for pandas-ta
            df_ta = df.copy()
            df_ta["open"] = df_ta["price"]  # Use price as open for single price data
            df_ta["high"] = df_ta["price"]  # Use price as high for single price data
            df_ta["low"] = df_ta["price"]  # Use price as low for single price data
            df_ta["close"] = df_ta["price"]  # Use price as close for single price data
            df_ta["volume"] = df_ta["volume"]

            if indicator_name == "sma":
                # Calculate SMA for different periods
                sma_20 = func(df_ta, 20)
                sma_50 = func(df_ta, 50)
                return {
                    "sma_20": (float(sma_20.iloc[-1]) if not pd.isna(sma_20.iloc[-1]) else None),
                    "sma_50": (float(sma_50.iloc[-1]) if not pd.isna(sma_50.iloc[-1]) else None),
                    "values": {
                        "sma_20": sma_20.dropna().tolist(),
                        "sma_50": sma_50.dropna().tolist(),
                    },
                }

            elif indicator_name == "ema":
                # Calculate EMA for different periods
                ema_12 = func(df_ta, 12)
                ema_26 = func(df_ta, 26)
                return {
                    "ema_12": (float(ema_12.iloc[-1]) if not pd.isna(ema_12.iloc[-1]) else None),
                    "ema_26": (float(ema_26.iloc[-1]) if not pd.isna(ema_26.iloc[-1]) else None),
                    "values": {
                        "ema_12": ema_12.dropna().tolist(),
                        "ema_26": ema_26.dropna().tolist(),
                    },
                }

            elif indicator_name == "rsi":
                rsi = func(df_ta, 14)
                return {
                    "current": (float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None),
                    "values": rsi.dropna().tolist(),
                    "oversold": self.indicator_config["signal_thresholds"]["rsi_oversold"],
                    "overbought": self.indicator_config["signal_thresholds"]["rsi_overbought"],
                }

            elif indicator_name == "macd":
                macd_result = func(df_ta)
                if isinstance(macd_result, pd.DataFrame) and len(macd_result.columns) >= 3:
                    macd = macd_result.iloc[:, 0]  # MACD line
                    signal = macd_result.iloc[:, 1]  # Signal line
                    histogram = macd_result.iloc[:, 2]  # Histogram
                else:
                    return None

                return {
                    "macd": (float(macd.iloc[-1]) if not pd.isna(macd.iloc[-1]) else None),
                    "signal": (float(signal.iloc[-1]) if not pd.isna(signal.iloc[-1]) else None),
                    "histogram": (
                        float(histogram.iloc[-1]) if not pd.isna(histogram.iloc[-1]) else None
                    ),
                    "values": {
                        "macd": macd.dropna().tolist(),
                        "signal": signal.dropna().tolist(),
                        "histogram": histogram.dropna().tolist(),
                    },
                }

            elif indicator_name == "bollinger_bands":
                bb_result = func(df_ta)
                if isinstance(bb_result, pd.DataFrame) and len(bb_result.columns) >= 3:
                    upper = bb_result.iloc[:, 0]  # Upper band
                    middle = bb_result.iloc[:, 1]  # Middle band (SMA)
                    lower = bb_result.iloc[:, 2]  # Lower band
                else:
                    return None

                return {
                    "upper": (float(upper.iloc[-1]) if not pd.isna(upper.iloc[-1]) else None),
                    "middle": (float(middle.iloc[-1]) if not pd.isna(middle.iloc[-1]) else None),
                    "lower": (float(lower.iloc[-1]) if not pd.isna(lower.iloc[-1]) else None),
                    "values": {
                        "upper": upper.dropna().tolist(),
                        "middle": middle.dropna().tolist(),
                        "lower": lower.dropna().tolist(),
                    },
                }

            elif indicator_name == "stochastic":
                stoch_result = func(df_ta)
                if isinstance(stoch_result, pd.DataFrame) and len(stoch_result.columns) >= 2:
                    k = stoch_result.iloc[:, 0]  # %K
                    d = stoch_result.iloc[:, 1]  # %D
                else:
                    return None

                return {
                    "k": (float(k.iloc[-1]) if not pd.isna(k.iloc[-1]) else None),
                    "d": (float(d.iloc[-1]) if not pd.isna(d.iloc[-1]) else None),
                    "values": {
                        "k": k.dropna().tolist(),
                        "d": d.dropna().tolist(),
                    },
                    "oversold": self.indicator_config["signal_thresholds"]["stochastic_oversold"],
                    "overbought": self.indicator_config["signal_thresholds"][
                        "stochastic_overbought"
                    ],
                }

            elif indicator_name == "williams_r":
                willr = func(df_ta, 14)
                return {
                    "current": (float(willr.iloc[-1]) if not pd.isna(willr.iloc[-1]) else None),
                    "values": willr.dropna().tolist(),
                    "oversold": self.indicator_config["signal_thresholds"]["williams_r_oversold"],
                    "overbought": self.indicator_config["signal_thresholds"][
                        "williams_r_overbought"
                    ],
                }

            elif indicator_name == "cci":
                cci = func(df_ta, 20)
                return {
                    "current": (float(cci.iloc[-1]) if not pd.isna(cci.iloc[-1]) else None),
                    "values": cci.dropna().tolist(),
                    "oversold": self.indicator_config["signal_thresholds"]["cci_oversold"],
                    "overbought": self.indicator_config["signal_thresholds"]["cci_overbought"],
                }

            elif indicator_name == "adx":
                adx = func(df_ta, 14)
                return {
                    "current": (float(adx.iloc[-1]) if not pd.isna(adx.iloc[-1]) else None),
                    "values": adx.dropna().tolist(),
                    "strong_trend": 25,  # ADX > 25 indicates strong trend
                }

            elif indicator_name == "atr":
                atr = func(df_ta, 14)
                return {
                    "current": (float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else None),
                    "values": atr.dropna().tolist(),
                }

            elif indicator_name == "obv":
                obv = func(df_ta)
                return {
                    "current": (float(obv.iloc[-1]) if not pd.isna(obv.iloc[-1]) else None),
                    "values": obv.dropna().tolist(),
                }

            elif indicator_name == "parabolic_sar":
                sar = func(df_ta)
                return {
                    "current": (float(sar.iloc[-1]) if not pd.isna(sar.iloc[-1]) else None),
                    "values": sar.dropna().tolist(),
                }

            else:
                # Generic indicator calculation
                result = func(df_ta)
                if isinstance(result, pd.DataFrame):
                    return {
                        "current": (
                            float(result.iloc[-1, 0]) if not pd.isna(result.iloc[-1, 0]) else None
                        ),
                        "values": result.dropna().values.tolist(),
                    }
                elif isinstance(result, pd.Series):
                    return {
                        "current": (
                            float(result.iloc[-1]) if not pd.isna(result.iloc[-1]) else None
                        ),
                        "values": result.dropna().tolist(),
                    }
                else:
                    return None

        except Exception as e:
            print(f"âŒ Error calculating {indicator_name}: {e}")
            return None

    async def broadcast_indicator_update(self, symbol: str, indicators: Dict[str, Any]):
        """Broadcast indicator update to other agents"""
        try:
            indicator_update = {
                "type": "indicator_update",
                "symbol": symbol,
                "indicators": indicators,
                "timestamp": datetime.now().isoformat(),
            }

            # Broadcast to all agents
            await self.broadcast_message(indicator_update)

            # Send to specific agents
            await self.send_message("strategy_agent", indicator_update)
            await self.send_message("risk_agent", indicator_update)

        except Exception as e:
            print(f"âŒ Error broadcasting indicator update: {e}")

    async def generate_indicator_signals(self):
        """Generate trading signals based on indicators"""
        try:
            signals = {}

            for symbol, indicator_data in self.state["indicators_calculated"].items():
                if indicator_data and "indicators" in indicator_data:
                    signal = await self.generate_symbol_indicator_signal(
                        symbol, indicator_data["indicators"]
                    )
                    if signal:
                        signals[symbol] = signal

            # Store generated signals
            self.state["signals_generated"] = signals

            # Broadcast signals
            if signals:
                await self.broadcast_indicator_signals(signals)

        except Exception as e:
            print(f"âŒ Error generating indicator signals: {e}")

    async def generate_symbol_indicator_signal(
        self, symbol: str, indicators: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Generate trading signal for a symbol based on indicators"""
        try:
            if not indicators:
                return None

            # Analyze each indicator for signals
            signal_analysis = {}
            total_confidence = 0
            signal_count = 0

            # RSI Analysis
            if "rsi" in indicators:
                rsi_signal = await self.analyze_rsi_signal(indicators["rsi"])
                if rsi_signal:
                    signal_analysis["rsi"] = rsi_signal
                    total_confidence += rsi_signal["confidence"]
                    signal_count += 1

            # MACD Analysis
            if "macd" in indicators:
                macd_signal = await self.analyze_macd_signal(indicators["macd"])
                if macd_signal:
                    signal_analysis["macd"] = macd_signal
                    total_confidence += macd_signal["confidence"]
                    signal_count += 1

            # Bollinger Bands Analysis
            if "bollinger_bands" in indicators:
                bb_signal = await self.analyze_bollinger_signal(indicators["bollinger_bands"])
                if bb_signal:
                    signal_analysis["bollinger_bands"] = bb_signal
                    total_confidence += bb_signal["confidence"]
                    signal_count += 1

            # Stochastic Analysis
            if "stochastic" in indicators:
                stoch_signal = await self.analyze_stochastic_signal(indicators["stochastic"])
                if stoch_signal:
                    signal_analysis["stochastic"] = stoch_signal
                    total_confidence += stoch_signal["confidence"]
                    signal_count += 1

            # Williams %R Analysis
            if "williams_r" in indicators:
                willr_signal = await self.analyze_williams_r_signal(indicators["williams_r"])
                if willr_signal:
                    signal_analysis["williams_r"] = willr_signal
                    total_confidence += willr_signal["confidence"]
                    signal_count += 1

            # CCI Analysis
            if "cci" in indicators:
                cci_signal = await self.analyze_cci_signal(indicators["cci"])
                if cci_signal:
                    signal_analysis["cci"] = cci_signal
                    total_confidence += cci_signal["confidence"]
                    signal_count += 1

            # Generate overall signal
            if signal_count > 0:
                avg_confidence = total_confidence / signal_count

                # Determine signal type based on analysis
                signal_type = await self.determine_signal_type(signal_analysis)

                return {
                    "symbol": symbol,
                    "signal_type": signal_type,
                    "confidence": avg_confidence,
                    "analysis": signal_analysis,
                    "timestamp": datetime.now().isoformat(),
                }

            return None

        except Exception as e:
            print(f"âŒ Error generating indicator signal for {symbol}: {e}")
            return None

    async def analyze_rsi_signal(self, rsi_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze RSI for trading signals"""
        try:
            current_rsi = rsi_data.get("current")
            if current_rsi is None:
                return None

            oversold = rsi_data.get("oversold", 30)
            overbought = rsi_data.get("overbought", 70)

            if current_rsi <= oversold:
                return {
                    "signal": "buy",
                    "confidence": 0.7,
                    "reason": f"RSI oversold ({current_rsi:.2f})",
                }
            elif current_rsi >= overbought:
                return {
                    "signal": "sell",
                    "confidence": 0.7,
                    "reason": f"RSI overbought ({current_rsi:.2f})",
                }

            return None

        except Exception as e:
            print(f"âŒ Error analyzing RSI signal: {e}")
            return None

    async def analyze_macd_signal(self, macd_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze MACD for trading signals"""
        try:
            macd = macd_data.get("macd")
            signal = macd_data.get("signal")
            histogram = macd_data.get("histogram")

            if macd is None or signal is None:
                return None

            # MACD crossover signals
            if macd > signal and histogram > 0:
                return {
                    "signal": "buy",
                    "confidence": 0.6,
                    "reason": "MACD bullish crossover",
                }
            elif macd < signal and histogram < 0:
                return {
                    "signal": "sell",
                    "confidence": 0.6,
                    "reason": "MACD bearish crossover",
                }

            return None

        except Exception as e:
            print(f"âŒ Error analyzing MACD signal: {e}")
            return None

    async def analyze_bollinger_signal(self, bb_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze Bollinger Bands for trading signals"""
        try:
            upper = bb_data.get("upper")
            middle = bb_data.get("middle")
            lower = bb_data.get("lower")

            if upper is None or middle is None or lower is None:
                return None

            # Get current price (approximate from middle band)
            current_price = middle

            # Bollinger Bands signals
            if current_price <= lower:
                return {
                    "signal": "buy",
                    "confidence": 0.6,
                    "reason": "Price at lower Bollinger Band",
                }
            elif current_price >= upper:
                return {
                    "signal": "sell",
                    "confidence": 0.6,
                    "reason": "Price at upper Bollinger Band",
                }

            return None

        except Exception as e:
            print(f"âŒ Error analyzing Bollinger Bands signal: {e}")
            return None

    async def analyze_stochastic_signal(
        self, stoch_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Analyze Stochastic for trading signals"""
        try:
            k = stoch_data.get("k")
            d = stoch_data.get("d")

            if k is None or d is None:
                return None

            oversold = stoch_data.get("oversold", 20)
            overbought = stoch_data.get("overbought", 80)

            if k <= oversold and d <= oversold:
                return {
                    "signal": "buy",
                    "confidence": 0.6,
                    "reason": f"Stochastic oversold (K:{k:.2f}, D:{d:.2f})",
                }
            elif k >= overbought and d >= overbought:
                return {
                    "signal": "sell",
                    "confidence": 0.6,
                    "reason": f"Stochastic overbought (K:{k:.2f}, D:{d:.2f})",
                }

            return None

        except Exception as e:
            print(f"âŒ Error analyzing Stochastic signal: {e}")
            return None

    async def analyze_williams_r_signal(
        self, willr_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Analyze Williams %R for trading signals"""
        try:
            current = willr_data.get("current")
            if current is None:
                return None

            oversold = willr_data.get("oversold", -80)
            overbought = willr_data.get("overbought", -20)

            if current <= oversold:
                return {
                    "signal": "buy",
                    "confidence": 0.6,
                    "reason": f"Williams %R oversold ({current:.2f})",
                }
            elif current >= overbought:
                return {
                    "signal": "sell",
                    "confidence": 0.6,
                    "reason": f"Williams %R overbought ({current:.2f})",
                }

            return None

        except Exception as e:
            print(f"âŒ Error analyzing Williams %R signal: {e}")
            return None

    async def analyze_cci_signal(self, cci_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze CCI for trading signals"""
        try:
            current = cci_data.get("current")
            if current is None:
                return None

            oversold = cci_data.get("oversold", -100)
            overbought = cci_data.get("overbought", 100)

            if current <= oversold:
                return {
                    "signal": "buy",
                    "confidence": 0.6,
                    "reason": f"CCI oversold ({current:.2f})",
                }
            elif current >= overbought:
                return {
                    "signal": "sell",
                    "confidence": 0.6,
                    "reason": f"CCI overbought ({current:.2f})",
                }

            return None

        except Exception as e:
            print(f"âŒ Error analyzing CCI signal: {e}")
            return None

    async def determine_signal_type(self, signal_analysis: Dict[str, Any]) -> str:
        """Determine overall signal type based on indicator analysis"""
        try:
            buy_signals = 0
            sell_signals = 0

            for indicator, analysis in signal_analysis.items():
                if analysis["signal"] == "buy":
                    buy_signals += 1
                elif analysis["signal"] == "sell":
                    sell_signals += 1

            if buy_signals > sell_signals:
                return "buy"
            elif sell_signals > buy_signals:
                return "sell"
            else:
                return "neutral"

        except Exception as e:
            print(f"âŒ Error determining signal type: {e}")
            return "neutral"

    async def broadcast_indicator_signals(self, signals: Dict[str, Any]):
        """Broadcast indicator signals to other agents"""
        try:
            signals_update = {
                "type": "indicator_signals_update",
                "signals": signals,
                "timestamp": datetime.now().isoformat(),
            }

            # Broadcast to all agents
            await self.broadcast_message(signals_update)

            # Send to specific agents
            await self.send_message("strategy_agent", signals_update)
            await self.send_message("execution_agent", signals_update)

        except Exception as e:
            print(f"âŒ Error broadcasting indicator signals: {e}")

    async def analyze_all_crossovers(self):
        """Analyze all crossovers for all symbols"""
        try:
            print(f"ðŸ”„ Analyzing crossovers for {len(self.trading_symbols)} symbols...")

            for symbol in self.trading_symbols:
                try:
                    await self.analyze_symbol_crossovers(symbol)
                except Exception as e:
                    print(f"âŒ Error analyzing crossovers for {symbol}: {e}")

            print("âœ… Crossover analysis complete")

        except Exception as e:
            print(f"âŒ Error analyzing all crossovers: {e}")

    async def analyze_symbol_crossovers(self, symbol: str):
        """Analyze crossovers for a specific symbol"""
        try:
            if symbol not in self.state["indicators_calculated"]:
                return

            indicator_data = self.state["indicators_calculated"][symbol]
            if not indicator_data or "indicators" not in indicator_data:
                return

            indicators = indicator_data["indicators"]
            crossovers = {}

            # Analyze SMA crossovers
            if "sma" in indicators:
                sma_crossover = await self.analyze_sma_crossover(indicators["sma"])
                if sma_crossover:
                    crossovers["sma"] = sma_crossover

            # Analyze EMA crossovers
            if "ema" in indicators:
                ema_crossover = await self.analyze_ema_crossover(indicators["ema"])
                if ema_crossover:
                    crossovers["ema"] = ema_crossover

            # Analyze MACD crossovers
            if "macd" in indicators:
                macd_crossover = await self.analyze_macd_crossover(indicators["macd"])
                if macd_crossover:
                    crossovers["macd"] = macd_crossover

            # Store crossovers
            if crossovers:
                if "crossovers" not in self.state["analysis_history"]:
                    self.state["analysis_history"]["crossovers"] = {}

                self.state["analysis_history"]["crossovers"][symbol] = crossovers

                # Broadcast crossover analysis
                await self.broadcast_crossover_analysis(symbol, crossovers)

        except Exception as e:
            print(f"âŒ Error analyzing crossovers for {symbol}: {e}")

    async def analyze_sma_crossover(self, sma_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze SMA crossover"""
        try:
            sma_20 = sma_data.get("sma_20")
            sma_50 = sma_data.get("sma_50")

            if sma_20 is None or sma_50 is None:
                return None

            # Check for crossover
            if sma_20 > sma_50:
                return {
                    "type": "bullish",
                    "signal": "buy",
                    "confidence": 0.6,
                    "reason": "SMA 20 crossed above SMA 50",
                }
            elif sma_20 < sma_50:
                return {
                    "type": "bearish",
                    "signal": "sell",
                    "confidence": 0.6,
                    "reason": "SMA 20 crossed below SMA 50",
                }

            return None

        except Exception as e:
            print(f"âŒ Error analyzing SMA crossover: {e}")
            return None

    async def analyze_ema_crossover(self, ema_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze EMA crossover"""
        try:
            ema_12 = ema_data.get("ema_12")
            ema_26 = ema_data.get("ema_26")

            if ema_12 is None or ema_26 is None:
                return None

            # Check for crossover
            if ema_12 > ema_26:
                return {
                    "type": "bullish",
                    "signal": "buy",
                    "confidence": 0.6,
                    "reason": "EMA 12 crossed above EMA 26",
                }
            elif ema_12 < ema_26:
                return {
                    "type": "bearish",
                    "signal": "sell",
                    "confidence": 0.6,
                    "reason": "EMA 12 crossed below EMA 26",
                }

            return None

        except Exception as e:
            print(f"âŒ Error analyzing EMA crossover: {e}")
            return None

    async def analyze_macd_crossover(self, macd_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze MACD crossover"""
        try:
            macd = macd_data.get("macd")
            signal = macd_data.get("signal")

            if macd is None or signal is None:
                return None

            # Check for crossover
            if macd > signal:
                return {
                    "type": "bullish",
                    "signal": "buy",
                    "confidence": 0.6,
                    "reason": "MACD crossed above signal line",
                }
            elif macd < signal:
                return {
                    "type": "bearish",
                    "signal": "sell",
                    "confidence": 0.6,
                    "reason": "MACD crossed below signal line",
                }

            return None

        except Exception as e:
            print(f"âŒ Error analyzing MACD crossover: {e}")
            return None

    async def broadcast_crossover_analysis(self, symbol: str, crossovers: Dict[str, Any]):
        """Broadcast crossover analysis to other agents"""
        try:
            crossover_update = {
                "type": "crossover_analysis_update",
                "symbol": symbol,
                "crossovers": crossovers,
                "timestamp": datetime.now().isoformat(),
            }

            # Broadcast to all agents
            await self.broadcast_message(crossover_update)

            # Send to specific agents
            await self.send_message("strategy_agent", crossover_update)
            await self.send_message("risk_agent", crossover_update)

        except Exception as e:
            print(f"âŒ Error broadcasting crossover analysis: {e}")

    async def handle_calculate_indicators(self, message: Dict[str, Any]):
        """Handle manual indicator calculation request"""
        try:
            symbol = message.get("symbol")
            indicators = message.get("indicators", self.indicator_config["supported_indicators"])

            print(f"ðŸ“Š Manual indicator calculation requested for {symbol}")

            if symbol:
                await self.calculate_symbol_indicators(symbol)

            # Send response
            response = {
                "type": "indicator_calculation_complete",
                "symbol": symbol,
                "indicators": indicators,
                "timestamp": datetime.now().isoformat(),
            }

            sender = message.get("from_agent")
            if sender:
                await self.send_message(sender, response)

        except Exception as e:
            print(f"âŒ Error handling indicator calculation request: {e}")
            await self.broadcast_error(f"Indicator calculation error: {e}")

    async def handle_get_indicator_signals(self, message: Dict[str, Any]):
        """Handle indicator signals request"""
        try:
            symbol = message.get("symbol")

            print(f"ðŸ“ˆ Indicator signals requested for {symbol}")

            # Get indicator signals
            if symbol and symbol in self.state["signals_generated"]:
                signal = self.state["signals_generated"][symbol]

                response = {
                    "type": "indicator_signals_response",
                    "symbol": symbol,
                    "signal": signal,
                    "timestamp": datetime.now().isoformat(),
                }
            else:
                response = {
                    "type": "indicator_signals_response",
                    "symbol": symbol,
                    "signal": None,
                    "timestamp": datetime.now().isoformat(),
                }

            sender = message.get("from_agent")
            if sender:
                await self.send_message(sender, response)

        except Exception as e:
            print(f"âŒ Error handling indicator signals request: {e}")
            await self.broadcast_error(f"Indicator signals error: {e}")

    async def handle_analyze_crossovers(self, message: Dict[str, Any]):
        """Handle crossover analysis request"""
        try:
            symbol = message.get("symbol")

            print(f"ðŸ”„ Crossover analysis requested for {symbol}")

            if symbol:
                await self.analyze_symbol_crossovers(symbol)

            # Send response
            response = {
                "type": "crossover_analysis_complete",
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
            }

            sender = message.get("from_agent")
            if sender:
                await self.send_message(sender, response)

        except Exception as e:
            print(f"âŒ Error handling crossover analysis request: {e}")
            await self.broadcast_error(f"Crossover analysis error: {e}")

    async def update_indicator_metrics(self):
        """Update indicator metrics"""
        try:
            metrics = {
                "agent_id": self.agent_id,
                "symbols_count": len(self.trading_symbols),
                "indicators_count": len(self.indicator_config["supported_indicators"]),
                "calculated_symbols": len(self.state["indicators_calculated"]),
                "signals_generated": len(self.state["signals_generated"]),
                "analysis_count": self.state["analysis_count"],
                "last_analysis": self.state["last_analysis"],
                "cache_size": len(self.state["indicator_cache"]),
                "timestamp": datetime.now().isoformat(),
            }

            # Store metrics in Redis
            self.redis_client.set(f"agent_metrics:{self.agent_id}", json.dumps(metrics), ex=300)

        except Exception as e:
            print(f"âŒ Error updating indicator metrics: {e}")

    async def cleanup_cache(self):
        """Clean up old cache entries"""
        try:
            current_time = datetime.now()

            # Clean up old indicator cache entries
            for symbol in list(self.state["indicator_cache"].keys()):
                data_points = self.state["indicator_cache"][symbol]

                # Keep only recent data points (last 24 hours)
                cutoff_time = current_time - timedelta(hours=24)
                recent_data = [
                    point
                    for point in data_points
                    if datetime.fromisoformat(point["timestamp"]) > cutoff_time
                ]

                if recent_data:
                    self.state["indicator_cache"][symbol] = recent_data
                else:
                    del self.state["indicator_cache"][symbol]

        except Exception as e:
            print(f"âŒ Error cleaning up cache: {e}")

    async def handle_market_data(self, message: Dict[str, Any]):
        """Handle market data message"""
        try:
            market_data = message.get("market_data", {})
            print(f"ðŸ“Š Technical Indicator Agent received market data for {len(market_data)} symbols")
            
            # Process market data
            await self.process_market_data(market_data)
            
            # Store market data for each symbol
            for symbol, data in market_data.items():
                if symbol in self.trading_symbols:
                    price = data.get("price", 0)
                    volume = data.get("volume", 0)
                    timestamp = data.get("timestamp", datetime.now().isoformat())
                    
                    await self.store_market_data(symbol, price, volume, timestamp)
            
            # Calculate indicators with new data
            await self.calculate_all_indicators()
            
            # Generate signals
            await self.generate_indicator_signals()
            
        except Exception as e:
            print(f"âŒ Error handling market data: {e}")
            await self.broadcast_error(f"Market data handling error: {e}")


