#!/usr/bin/env python3
"""
Strategy System for Mystic Trading Platform

Handles trading strategies and signal generation.
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from ai_mutation.strategy_locker import get_live_strategy

# Use absolute imports
from crypto_autoengine_config import get_config
from services.websocket_manager import websocket_manager
from shared_cache import CoinCache, SharedCache

logger = logging.getLogger(__name__)


class StrategySignal:
    """Individual strategy signal"""

    def __init__(
        self,
        name: str,
        signal: str,
        confidence: float,
        strength: float,
        description: str,
    ):
        self.name = name
        self.signal = signal  # 'buy', 'sell', 'hold'
        self.confidence = confidence  # 0.0 to 1.0
        self.strength = strength  # 0.0 to 1.0
        self.description = description
        self.timestamp = datetime.now(timezone.timezone.utc).isoformat()


class StrategyController:
    """Controller for individual coin strategies with AI mutation integration"""

    def __init__(self, symbol: str, cache: SharedCache):
        self.symbol = symbol
        self.cache = cache
        self.config = get_config()
        self.coin_config = self.config.get_coin_by_symbol(symbol)
        self.ai_strategy = None
        self.last_ai_strategy_load = 0
        self.ai_strategy_cache_ttl = 300  # 5 minutes

    def _load_ai_strategy(self) -> Optional[Dict[str, Any]]:
        """Load the latest promoted AI strategy from the mutation system"""
        current_time = time.time()

        # Check cache first
        if (
            self.ai_strategy
            and current_time - self.last_ai_strategy_load < self.ai_strategy_cache_ttl
        ):
            return self.ai_strategy

        try:
            # Get the current live strategy from the AI mutation system
            live_strategy_file = get_live_strategy()

            if not live_strategy_file:
                logger.debug("No live AI strategy available")
                return None

            # Load the strategy file
            strategy_path = os.path.join("strategies", live_strategy_file)
            if not os.path.exists(strategy_path):
                # Try mutated_strategies directory
                strategy_path = os.path.join("mutated_strategies", live_strategy_file)

            if os.path.exists(strategy_path):
                with open(strategy_path, "r") as f:
                    self.ai_strategy = json.load(f)
                self.last_ai_strategy_load = current_time
                logger.info(f"Loaded AI strategy: {live_strategy_file}")
                return self.ai_strategy
            else:
                logger.warning(f"AI strategy file not found: {strategy_path}")
                return None

        except Exception as e:
            logger.error(f"Error loading AI strategy: {e}")
            return None

    def _apply_ai_strategy(self, coin_data: CoinCache) -> List[StrategySignal]:
        """Apply the latest AI strategy to generate signals"""
        signals = []
        ai_strategy = self._load_ai_strategy()

        if not ai_strategy:
            return signals

        try:
            strategy_type = ai_strategy.get("strategy_type", "unknown")
            params = ai_strategy.get("parameters", {})

            # Apply strategy based on type
            if strategy_type == "breakout":
                signals.extend(self._apply_breakout_strategy(coin_data, params))
            elif strategy_type == "ema_crossover":
                signals.extend(self._apply_ema_crossover_strategy(coin_data, params))
            elif strategy_type == "rsi_threshold":
                signals.extend(self._apply_rsi_strategy(coin_data, params))
            elif strategy_type == "bollinger_bands":
                signals.extend(self._apply_bollinger_strategy(coin_data, params))
            elif strategy_type == "macd":
                signals.extend(self._apply_macd_strategy(coin_data, params))
            elif strategy_type == "stochastic":
                signals.extend(self._apply_stochastic_strategy(coin_data, params))
            elif strategy_type == "volume_price":
                signals.extend(self._apply_volume_price_strategy(coin_data, params))
            elif strategy_type == "momentum":
                signals.extend(self._apply_momentum_strategy(coin_data, params))

            # Add AI strategy metadata
            for signal in signals:
                signal.name = f"AI_{signal.name}"
                signal.description = f"AI Generated: {signal.description}"
                # Broadcast each signal as a dict
                asyncio.create_task(
                    websocket_manager.broadcast_json({"type": "signal", "data": signal.__dict__})
                )

        except Exception as e:
            logger.error(f"Error applying AI strategy: {e}")

        return signals

    def _apply_breakout_strategy(
        self, coin_data: CoinCache, params: Dict[str, Any]
    ) -> List[StrategySignal]:
        """Apply breakout strategy from AI parameters"""
        signals = []
        history = coin_data.price_history or []

        if len(history) < 20:
            return signals

        lookback = int(params.get("lookback_period", 20))
        entry_threshold = float(params.get("entry_threshold", 1.5))
        exit_threshold = float(params.get("exit_threshold", 0.8))

        if len(history) >= lookback:
            recent_prices = [float(h["price"]) for h in history[-lookback:]]
            current_price = float(history[-1]["price"])
            avg_price = sum(recent_prices) / len(recent_prices)

            # Entry signal
            if current_price > avg_price * entry_threshold:
                s = StrategySignal(
                    "Breakout Entry",
                    "buy",
                    0.8,
                    0.7,
                    f"Price {current_price} above {entry_threshold}x average {avg_price}",
                )
                signals.append(s)
                asyncio.create_task(
                    websocket_manager.broadcast_json({"type": "signal", "data": s.__dict__})
                )
            # Exit signal
            elif current_price < avg_price * exit_threshold:
                s = StrategySignal(
                    "Breakout Exit",
                    "sell",
                    0.8,
                    0.7,
                    f"Price {current_price} below {exit_threshold}x average {avg_price}",
                )
                signals.append(s)
                asyncio.create_task(
                    websocket_manager.broadcast_json({"type": "signal", "data": s.__dict__})
                )

        return signals

    def _apply_ema_crossover_strategy(
        self, coin_data: CoinCache, params: Dict[str, Any]
    ) -> List[StrategySignal]:
        """Apply EMA crossover strategy from AI parameters"""
        signals = []
        history = coin_data.price_history or []

        if len(history) < 50:
            return signals

        fast_ema = int(params.get("fast_ema", 12))
        slow_ema = int(params.get("slow_ema", 26))

        if len(history) >= slow_ema:
            prices = [float(h["price"]) for h in history]
            ema_fast = self._calculate_ema(prices, fast_ema)
            ema_slow = self._calculate_ema(prices, slow_ema)

            if len(ema_fast) > 0 and len(ema_slow) > 0:
                current_fast = ema_fast[-1]
                current_slow = ema_slow[-1]
                prev_fast = ema_fast[-2] if len(ema_fast) > 1 else current_fast
                prev_slow = ema_slow[-2] if len(ema_slow) > 1 else current_slow

                # Crossover signals
                if current_fast > current_slow and prev_fast <= prev_slow:
                    signals.append(
                        StrategySignal(
                            "EMA Crossover Buy",
                            "buy",
                            0.75,
                            0.65,
                            f"Fast EMA {current_fast:.2f} crossed above slow EMA {current_slow:.2f}",
                        )
                    )
                elif current_fast < current_slow and prev_fast >= prev_slow:
                    signals.append(
                        StrategySignal(
                            "EMA Crossover Sell",
                            "sell",
                            0.75,
                            0.65,
                            f"Fast EMA {current_fast:.2f} crossed below slow EMA {current_slow:.2f}",
                        )
                    )

        return signals

    def _apply_rsi_strategy(
        self, coin_data: CoinCache, params: Dict[str, Any]
    ) -> List[StrategySignal]:
        """Apply RSI strategy from AI parameters"""
        signals = []
        history = coin_data.price_history or []

        if len(history) < 30:
            return signals

        rsi_period = int(params.get("rsi_period", 14))
        rsi_buy = int(params.get("rsi_buy", 30))
        rsi_sell = int(params.get("rsi_sell", 70))

        if len(history) >= rsi_period:
            prices = [float(h["price"]) for h in history]
            rsi = self._calculate_rsi(prices, rsi_period)

            if rsi is not None:
                if rsi < rsi_buy:
                    signals.append(
                        StrategySignal(
                            "RSI Oversold",
                            "buy",
                            0.7,
                            0.6,
                            f"RSI {rsi:.1f} below buy threshold {rsi_buy}",
                        )
                    )
                elif rsi > rsi_sell:
                    signals.append(
                        StrategySignal(
                            "RSI Overbought",
                            "sell",
                            0.7,
                            0.6,
                            f"RSI {rsi:.1f} above sell threshold {rsi_sell}",
                        )
                    )

        return signals

    def _apply_bollinger_strategy(
        self, coin_data: CoinCache, params: Dict[str, Any]
    ) -> List[StrategySignal]:
        """Apply Bollinger Bands strategy from AI parameters"""
        signals = []
        history = coin_data.price_history or []

        if len(history) < 20:
            return signals

        period = int(params.get("period", 20))
        std_dev = float(params.get("std_dev", 2.0))

        if len(history) >= period:
            prices = [float(h["price"]) for h in history[-period:]]
            current_price = prices[-1]
            sma = sum(prices) / len(prices)

            # Calculate standard deviation
            variance = sum((p - sma) ** 2 for p in prices) / len(prices)
            std = variance**0.5

            upper_band = sma + (std_dev * std)
            lower_band = sma - (std_dev * std)

            if current_price <= lower_band:
                signals.append(
                    StrategySignal(
                        "Bollinger Lower Band",
                        "buy",
                        0.8,
                        0.7,
                        f"Price {current_price:.2f} at lower band {lower_band:.2f}",
                    )
                )
            elif current_price >= upper_band:
                signals.append(
                    StrategySignal(
                        "Bollinger Upper Band",
                        "sell",
                        0.8,
                        0.7,
                        f"Price {current_price:.2f} at upper band {upper_band:.2f}",
                    )
                )

        return signals

    def _apply_macd_strategy(
        self, coin_data: CoinCache, params: Dict[str, Any]
    ) -> List[StrategySignal]:
        """Apply MACD strategy from AI parameters"""
        signals = []
        history = coin_data.price_history or []

        if len(history) < 50:
            return signals

        int(params.get("fast_period", 12))
        slow = int(params.get("slow_period", 26))
        signal = int(params.get("signal_period", 9))

        if len(history) >= slow + signal:
            prices = [float(h["price"]) for h in history]
            macd_data = self._calculate_macd(prices)

            if macd_data and "macd_line" in macd_data and "signal_line" in macd_data:
                macd_line = macd_data["macd_line"]
                signal_line = macd_data["signal_line"]

                if macd_line > signal_line:
                    signals.append(
                        StrategySignal(
                            "MACD Bullish",
                            "buy",
                            0.7,
                            0.6,
                            f"MACD {macd_line:.6f} above signal {signal_line:.6f}",
                        )
                    )
                else:
                    signals.append(
                        StrategySignal(
                            "MACD Bearish",
                            "sell",
                            0.7,
                            0.6,
                            f"MACD {macd_line:.6f} below signal {signal_line:.6f}",
                        )
                    )

        return signals

    def _apply_stochastic_strategy(
        self, coin_data: CoinCache, params: Dict[str, Any]
    ) -> List[StrategySignal]:
        """Apply Stochastic strategy from AI parameters"""
        signals = []
        history = coin_data.price_history or []

        if len(history) < 20:
            return signals

        k_period = int(params.get("k_period", 14))
        int(params.get("d_period", 3))

        if len(history) >= k_period:
            prices = [float(h["price"]) for h in history]

            # Calculate %K
            recent_prices = prices[-k_period:]
            high = max(recent_prices)
            low = min(recent_prices)
            current = recent_prices[-1]

            if high != low:
                k_percent = ((current - low) / (high - low)) * 100

                if k_percent < 20:
                    signals.append(
                        StrategySignal(
                            "Stochastic Oversold",
                            "buy",
                            0.7,
                            0.6,
                            f"Stochastic %K {k_percent:.1f} oversold",
                        )
                    )
                elif k_percent > 80:
                    signals.append(
                        StrategySignal(
                            "Stochastic Overbought",
                            "sell",
                            0.7,
                            0.6,
                            f"Stochastic %K {k_percent:.1f} overbought",
                        )
                    )

        return signals

    def _apply_volume_price_strategy(
        self, coin_data: CoinCache, params: Dict[str, Any]
    ) -> List[StrategySignal]:
        """Apply Volume-Price strategy from AI parameters"""
        signals = []
        history = coin_data.price_history or []

        if len(history) < 20:
            return signals

        volume_multiplier = float(params.get("volume_multiplier", 1.5))

        if len(history) >= 20:
            recent_history = history[-20:]
            avg_volume = sum(float(h.get("volume", 0)) for h in recent_history) / len(
                recent_history
            )
            current_volume = float(history[-1].get("volume", 0))

            if current_volume > avg_volume * volume_multiplier:
                current_price = float(history[-1]["price"])
                prev_price = float(history[-2]["price"])

                if current_price > prev_price:
                    signals.append(
                        StrategySignal(
                            "Volume Price Breakout",
                            "buy",
                            0.8,
                            0.7,
                            f"High volume {current_volume:.0f} with price increase",
                        )
                    )
                else:
                    signals.append(
                        StrategySignal(
                            "Volume Price Breakdown",
                            "sell",
                            0.8,
                            0.7,
                            f"High volume {current_volume:.0f} with price decrease",
                        )
                    )

        return signals

    def _apply_momentum_strategy(
        self, coin_data: CoinCache, params: Dict[str, Any]
    ) -> List[StrategySignal]:
        """Apply Momentum strategy from AI parameters"""
        signals = []
        history = coin_data.price_history or []

        if len(history) < 20:
            return signals

        momentum_period = int(params.get("momentum_period", 10))
        momentum_threshold = float(params.get("momentum_threshold", 0.05))

        if len(history) >= momentum_period:
            prices = [float(h["price"]) for h in history]
            recent_prices = prices[-momentum_period:]

            momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]

            if momentum > momentum_threshold:
                signals.append(
                    StrategySignal(
                        "Momentum Bullish",
                        "buy",
                        0.7,
                        0.6,
                        f"Positive momentum {momentum:.2%}",
                    )
                )
            elif momentum < -momentum_threshold:
                signals.append(
                    StrategySignal(
                        "Momentum Bearish",
                        "sell",
                        0.7,
                        0.6,
                        f"Negative momentum {momentum:.2%}",
                    )
                )

        return signals

    def run_all_strategies(self) -> Dict[str, Any]:
        """Run all 65+ strategies for this coin plus AI-generated strategies"""
        if not self.coin_config:
            return {}

        coin_data: Optional[CoinCache] = self.cache.get_coin_cache(self.symbol)
        if not coin_data:
            return {}

        signals: List[StrategySignal] = []

        # Price-based strategies (1-20)
        signals.extend(self._price_based_strategies(coin_data))

        # Volume-based strategies (21-35)
        signals.extend(self._volume_based_strategies(coin_data))

        # Technical indicator strategies (36-50)
        signals.extend(self._technical_indicator_strategies(coin_data))

        # Momentum strategies (51-60)
        signals.extend(self._momentum_strategies(coin_data))

        # Volatility strategies (61-65)
        signals.extend(self._volatility_strategies(coin_data))

        # Cosmic/mystic strategies (66+)
        signals.extend(self._cosmic_strategies(coin_data))

        # AI-generated strategies (from mutation system)
        ai_signals = self._apply_ai_strategy(coin_data)
        signals.extend(ai_signals)

        # Aggregate signals (AI strategies get higher weight)
        aggregated = self._aggregate_signals(signals)

        return {
            "signals": [self._signal_to_dict(s) for s in signals],
            "aggregated": aggregated,
            "timestamp": datetime.now(timezone.timezone.utc).isoformat(),
            "ai_strategy_loaded": self.ai_strategy is not None,
            "ai_signals_count": len(ai_signals),
        }

    def _price_based_strategies(self, coin_data: CoinCache) -> List[StrategySignal]:
        """Price-based strategies (1-20)"""
        signals: List[StrategySignal] = []
        price: float = coin_data.price
        history: List[Dict[str, Union[float, str]]] = coin_data.price_history or []

        if len(history) < 10:
            return signals

        # Strategy 1: Price Breakout
        if len(history) >= 20:
            recent_high = max([float(h["price"]) for h in history[-20:]])
            if price > recent_high * 1.02:  # 2% breakout
                signals.append(
                    StrategySignal(
                        "Price Breakout",
                        "buy",
                        0.8,
                        0.7,
                        f"Price {price} broke above recent high {recent_high}",
                    )
                )

        # Strategy 2: Price Breakdown
        if len(history) >= 20:
            recent_low = min([float(h["price"]) for h in history[-20:]])
            if price < recent_low * 0.98:  # 2% breakdown
                signals.append(
                    StrategySignal(
                        "Price Breakdown",
                        "sell",
                        0.8,
                        0.7,
                        f"Price {price} broke below recent low {recent_low}",
                    )
                )

        # Strategy 3: Price Momentum
        if len(history) >= 10:
            recent_prices = [float(h["price"]) for h in history[-10:]]
            momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
            if momentum > 0.05:  # 5% positive momentum
                signals.append(
                    StrategySignal(
                        "Price Momentum",
                        "buy",
                        0.7,
                        0.6,
                        f"Positive momentum: {momentum:.2%}",
                    )
                )
            elif momentum < -0.05:  # 5% negative momentum
                signals.append(
                    StrategySignal(
                        "Price Momentum",
                        "sell",
                        0.7,
                        0.6,
                        f"Negative momentum: {momentum:.2%}",
                    )
                )

        # Strategy 4: Price Reversal
        if len(history) >= 15:
            prices_15 = [float(h["price"]) for h in history[-15:]]
            prices_5 = [float(h["price"]) for h in history[-5:]]

            trend_15 = (prices_15[-1] - prices_15[0]) / prices_15[0]
            trend_5 = (prices_5[-1] - prices_5[0]) / prices_5[0]

            if trend_15 < -0.03 and trend_5 > 0.02:  # Reversal from down to up
                signals.append(
                    StrategySignal(
                        "Price Reversal",
                        "buy",
                        0.75,
                        0.65,
                        "Reversal detected: 15-period down, 5-period up",
                    )
                )
            elif trend_15 > 0.03 and trend_5 < -0.02:  # Reversal from up to down
                signals.append(
                    StrategySignal(
                        "Price Reversal",
                        "sell",
                        0.75,
                        0.65,
                        "Reversal detected: 15-period up, 5-period down",
                    )
                )

        # Strategy 5: Support/Resistance
        if len(history) >= 30:
            all_prices = [float(h["price"]) for h in history]
            support = min(all_prices[-30:])
            resistance = max(all_prices[-30:])

            if price <= support * 1.01:  # Near support
                signals.append(
                    StrategySignal(
                        "Support Level",
                        "buy",
                        0.8,
                        0.7,
                        f"Price near support: {support}",
                    )
                )
            elif price >= resistance * 0.99:  # Near resistance
                signals.append(
                    StrategySignal(
                        "Resistance Level",
                        "sell",
                        0.8,
                        0.7,
                        f"Price near resistance: {resistance}",
                    )
                )

        # Continue with more price strategies...
        # Strategies 6-20 would include more sophisticated price analysis

        return signals

    def _volume_based_strategies(self, coin_data: CoinCache) -> List[StrategySignal]:
        """Volume-based strategies (21-35)"""
        signals: List[StrategySignal] = []
        volume: float = coin_data.volume_24h
        price: float = coin_data.price
        history: List[Dict[str, Union[float, str]]] = coin_data.price_history or []

        if not volume or len(history) < 10:
            return signals

        # Strategy 21: Volume Spike
        if len(history) >= 20:
            recent_volumes = [float(h.get("volume", volume)) for h in history[-20:]]
            avg_volume = sum(recent_volumes) / len(recent_volumes)

            if volume > avg_volume * 2:  # 2x average volume
                price_change = (price - float(history[-2]["price"])) / float(history[-2]["price"])
                if price_change > 0:
                    signals.append(
                        StrategySignal(
                            "Volume Spike Bullish",
                            "buy",
                            0.8,
                            0.7,
                            f"High volume with price increase: {volume:.0f}",
                        )
                    )
                else:
                    signals.append(
                        StrategySignal(
                            "Volume Spike Bearish",
                            "sell",
                            0.8,
                            0.7,
                            f"High volume with price decrease: {volume:.0f}",
                        )
                    )

        # Strategy 22: Volume Divergence
        if len(history) >= 15:
            prices_5 = [float(h["price"]) for h in history[-5:]]
            volumes_5 = [float(h.get("volume", volume)) for h in history[-5:]]

            price_trend = (prices_5[-1] - prices_5[0]) / prices_5[0]
            volume_trend = (volumes_5[-1] - volumes_5[0]) / volumes_5[0]

            if price_trend > 0.03 and volume_trend < -0.2:  # Price up, volume down
                signals.append(
                    StrategySignal(
                        "Volume Divergence Bearish",
                        "sell",
                        0.7,
                        0.6,
                        "Price rising but volume declining",
                    )
                )
            elif price_trend < -0.03 and volume_trend > 0.2:  # Price down, volume up
                signals.append(
                    StrategySignal(
                        "Volume Divergence Bullish",
                        "buy",
                        0.7,
                        0.6,
                        "Price falling but volume increasing",
                    )
                )

        # Continue with more volume strategies...
        # Strategies 23-35 would include more volume analysis

        return signals

    def _technical_indicator_strategies(self, coin_data: CoinCache) -> List[StrategySignal]:
        """Technical indicator strategies (36-50)"""
        signals: List[StrategySignal] = []
        rsi: float = coin_data.rsi
        macd: Dict[str, float] = coin_data.macd or {}
        price: float = coin_data.price
        history: List[Dict[str, Union[float, str]]] = coin_data.price_history or []

        # Strategy 36: RSI Oversold/Overbought
        if rsi < 30:
            signals.append(
                StrategySignal("RSI Oversold", "buy", 0.8, 0.7, f"RSI oversold: {rsi:.1f}")
            )
        elif rsi > 70:
            signals.append(
                StrategySignal(
                    "RSI Overbought",
                    "sell",
                    0.8,
                    0.7,
                    f"RSI overbought: {rsi:.1f}",
                )
            )

        # Strategy 37: RSI Divergence
        if len(history) >= 20:
            recent_rsi = self._calculate_rsi([float(h["price"]) for h in history[-10:]])
            if rsi < 40 and recent_rsi > rsi:  # RSI divergence
                signals.append(
                    StrategySignal(
                        "RSI Bullish Divergence",
                        "buy",
                        0.75,
                        0.65,
                        f"RSI divergence detected: {rsi:.1f}",
                    )
                )

        # Strategy 38: MACD Crossover
        if macd["macd_line"] > macd["signal_line"] and macd["histogram"] > 0:
            signals.append(
                StrategySignal(
                    "MACD Bullish Crossover",
                    "buy",
                    0.8,
                    0.7,
                    f"MACD above signal line: {macd['macd_line']:.4f}",
                )
            )
        elif macd["macd_line"] < macd["signal_line"] and macd["histogram"] < 0:
            signals.append(
                StrategySignal(
                    "MACD Bearish Crossover",
                    "sell",
                    0.8,
                    0.7,
                    f"MACD below signal line: {macd['macd_line']:.4f}",
                )
            )

        # Strategy 39: MACD Divergence
        if len(history) >= 30:
            prices_30 = [float(h["price"]) for h in history[-30:]]
            macd_30 = self._calculate_macd(prices_30)

            if price > prices_30[0] and macd["macd_line"] < macd_30["macd_line"]:
                signals.append(
                    StrategySignal(
                        "MACD Bearish Divergence",
                        "sell",
                        0.75,
                        0.65,
                        "Price higher but MACD lower",
                    )
                )
            elif price < prices_30[0] and macd["macd_line"] > macd_30["macd_line"]:
                signals.append(
                    StrategySignal(
                        "MACD Bullish Divergence",
                        "buy",
                        0.75,
                        0.65,
                        "Price lower but MACD higher",
                    )
                )

        # Continue with more technical indicator strategies...
        # Strategies 40-50 would include more indicator analysis

        return signals

    def _momentum_strategies(self, coin_data: CoinCache) -> List[StrategySignal]:
        """Momentum strategies (51-60)"""
        signals: List[StrategySignal] = []
        change_1m: float = coin_data.change_1m
        change_5m: float = coin_data.change_5m
        change_15m: float = coin_data.change_15m
        change_1h: float = coin_data.change_1h

        # Strategy 51: Short-term Momentum
        if change_1m > 2:  # 2% in 1 minute
            signals.append(
                StrategySignal(
                    "Short-term Bullish Momentum",
                    "buy",
                    0.7,
                    0.6,
                    f"1-minute change: {change_1m:.2f}%",
                )
            )
        elif change_1m < -2:  # -2% in 1 minute
            signals.append(
                StrategySignal(
                    "Short-term Bearish Momentum",
                    "sell",
                    0.7,
                    0.6,
                    f"1-minute change: {change_1m:.2f}%",
                )
            )

        # Strategy 52: Medium-term Momentum
        if change_5m > 5:  # 5% in 5 minutes
            signals.append(
                StrategySignal(
                    "Medium-term Bullish Momentum",
                    "buy",
                    0.8,
                    0.7,
                    f"5-minute change: {change_5m:.2f}%",
                )
            )
        elif change_5m < -5:  # -5% in 5 minutes
            signals.append(
                StrategySignal(
                    "Medium-term Bearish Momentum",
                    "sell",
                    0.8,
                    0.7,
                    f"5-minute change: {change_5m:.2f}%",
                )
            )

        # Strategy 53: Momentum Convergence
        if change_1m > 1 and change_5m > 3 and change_15m > 5:
            signals.append(
                StrategySignal(
                    "Momentum Convergence Bullish",
                    "buy",
                    0.9,
                    0.8,
                    "All timeframes showing bullish momentum",
                )
            )
        elif change_1m < -1 and change_5m < -3 and change_15m < -5:
            signals.append(
                StrategySignal(
                    "Momentum Convergence Bearish",
                    "sell",
                    0.9,
                    0.8,
                    "All timeframes showing bearish momentum",
                )
            )

        # Strategy 54: Long-term Momentum (1-hour)
        if change_1h > 8:  # 8% in 1 hour
            signals.append(
                StrategySignal(
                    "Long-term Bullish Momentum",
                    "buy",
                    0.85,
                    0.75,
                    f"1-hour change: {change_1h:.2f}%",
                )
            )
        elif change_1h < -8:  # -8% in 1 hour
            signals.append(
                StrategySignal(
                    "Long-term Bearish Momentum",
                    "sell",
                    0.85,
                    0.75,
                    f"1-hour change: {change_1h:.2f}%",
                )
            )

        # Continue with more momentum strategies...
        # Strategies 55-60 would include more momentum analysis

        return signals

    def _volatility_strategies(self, coin_data: CoinCache) -> List[StrategySignal]:
        """Volatility strategies (61-65)"""
        signals: List[StrategySignal] = []
        volatility: float = coin_data.volatility_index
        price: float = coin_data.price
        history: List[Dict[str, Union[float, str]]] = coin_data.price_history or []

        # Strategy 61: High Volatility Breakout
        if volatility > 5:  # High volatility
            if len(history) >= 10:
                recent_avg = sum([float(h["price"]) for h in history[-10:]]) / 10
                if price > recent_avg * 1.02:  # Breakout above average
                    signals.append(
                        StrategySignal(
                            "High Volatility Bullish Breakout",
                            "buy",
                            0.8,
                            0.7,
                            f"High volatility breakout: {volatility:.2f}%",
                        )
                    )
                elif price < recent_avg * 0.98:  # Breakdown below average
                    signals.append(
                        StrategySignal(
                            "High Volatility Bearish Breakdown",
                            "sell",
                            0.8,
                            0.7,
                            f"High volatility breakdown: {volatility:.2f}%",
                        )
                    )

        # Strategy 62: Low Volatility Consolidation
        if volatility < 1:  # Low volatility
            signals.append(
                StrategySignal(
                    "Low Volatility Consolidation",
                    "hold",
                    0.6,
                    0.5,
                    f"Low volatility period: {volatility:.2f}%",
                )
            )

        # Continue with more volatility strategies...
        # Strategies 63-65 would include more volatility analysis

        return signals

    def _cosmic_strategies(self, coin_data: CoinCache) -> List[StrategySignal]:
        """Cosmic/mystic strategies (66+)"""
        signals: List[StrategySignal] = []
        cosmic_data: Dict[str, Any] = self.cache.cosmic_data

        if not cosmic_data:
            return signals

        # Strategy 66: Solar Flare Impact
        solar_data = cosmic_data.get("solar_flare_index", {})
        if solar_data:
            flare_class = solar_data.get("flare_class", "A")
            if flare_class in ["M", "X"]:  # Major solar flares
                signals.append(
                    StrategySignal(
                        "Solar Flare Impact",
                        "sell",
                        0.6,
                        0.5,
                        f"Major solar flare detected: {flare_class}",
                    )
                )

        # Strategy 67: Schumann Resonance
        schumann_data = cosmic_data.get("schumann_resonance", {})
        if schumann_data:
            frequency = schumann_data.get("frequency", 7.83)
            if frequency > 8.0:  # Elevated Schumann resonance
                signals.append(
                    StrategySignal(
                        "Elevated Schumann Resonance",
                        "buy",
                        0.5,
                        0.4,
                        f"Elevated Schumann frequency: {frequency:.2f} Hz",
                    )
                )

        # Strategy 68: Lunar Phase
        cosmic_alignment = cosmic_data.get("cosmic_alignment", {})
        if cosmic_alignment:
            lunar_phase = cosmic_alignment.get("lunar_phase", "")
            if lunar_phase in ["new_moon", "full_moon"]:
                signals.append(
                    StrategySignal(
                        "Lunar Phase Influence",
                        "hold",
                        0.4,
                        0.3,
                        f"Lunar phase: {lunar_phase}",
                    )
                )

        # Continue with more cosmic strategies...
        # Strategies 69+ would include more cosmic analysis

        return signals

    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate RSI for strategy use"""
        if len(prices) < period + 1:
            return 50.0

        gains: List[float] = []
        losses: List[float] = []

        for i in range(1, len(prices)):
            change = prices[i] - prices[i - 1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))

        if len(gains) < period:
            return 50.0

        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_macd(self, prices: List[float]) -> Dict[str, float]:
        """Calculate MACD for strategy use"""
        if len(prices) < 26:
            return {"macd_line": 0.0, "signal_line": 0.0, "histogram": 0.0}

        # Calculate EMAs
        ema12 = self._calculate_ema(prices, 12)
        ema26 = self._calculate_ema(prices, 26)

        macd_line = ema12 - ema26

        # Calculate signal line
        if len(prices) >= 35:
            macd_values: List[float] = []
            for i in range(26, len(prices)):
                ema12_i = self._calculate_ema(prices[: i + 1], 12)
                ema26_i = self._calculate_ema(prices[: i + 1], 26)
                macd_values.append(ema12_i - ema26_i)

            if len(macd_values) >= 9:
                signal_line = self._calculate_ema(macd_values, 9)
            else:
                signal_line = macd_line
        else:
            signal_line = macd_line

        histogram = macd_line - signal_line

        return {
            "macd_line": macd_line,
            "signal_line": signal_line,
            "histogram": histogram,
        }

    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """Calculate EMA for strategy use"""
        if len(prices) < period:
            return prices[-1] if prices else 0.0

        multiplier = 2 / (period + 1)
        ema = sum(prices[:period]) / period

        for price in prices[period:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))

        return ema

    def _aggregate_signals(self, signals: List[StrategySignal]) -> Dict[str, Any]:
        """Aggregate all signals into final decision with AI strategies getting higher weight"""
        if not signals:
            return {
                "decision": "hold",
                "confidence": 0.0,
                "strength": 0.0,
                "buy_signals": 0,
                "sell_signals": 0,
                "hold_signals": 0,
            }

        buy_signals = [s for s in signals if s.signal == "buy"]
        sell_signals = [s for s in signals if s.signal == "sell"]
        hold_signals = [s for s in signals if s.signal == "hold"]

        # Calculate weighted scores with AI strategies getting 2x weight
        def calculate_weighted_score(signal_list):
            total_score = 0.0
            for signal in signal_list:
                # AI strategies get 2x weight
                weight = 2.0 if signal.name.startswith("AI_") else 1.0
                total_score += signal.confidence * signal.strength * weight
            return total_score

        buy_score = calculate_weighted_score(buy_signals)
        sell_score = calculate_weighted_score(sell_signals)
        hold_score = calculate_weighted_score(hold_signals)

        # Count AI signals separately
        ai_buy_signals = [s for s in buy_signals if s.name.startswith("AI_")]
        ai_sell_signals = [s for s in sell_signals if s.name.startswith("AI_")]

        # Determine decision with AI bias
        if buy_score > sell_score and buy_score > hold_score:
            decision = "buy"
            confidence = (
                buy_score / (len(buy_signals) + len(ai_buy_signals)) if buy_signals else 0.0
            )
            strength = max(s.strength for s in buy_signals) if buy_signals else 0.0
        elif sell_score > buy_score and sell_score > hold_score:
            decision = "sell"
            confidence = (
                sell_score / (len(sell_signals) + len(ai_sell_signals)) if sell_signals else 0.0
            )
            strength = max(s.strength for s in sell_signals) if sell_signals else 0.0
        else:
            decision = "hold"
            confidence = hold_score / len(hold_signals) if hold_signals else 0.0
            strength = max(s.strength for s in hold_signals) if hold_signals else 0.0

        return {
            "decision": decision,
            "confidence": min(confidence, 1.0),
            "strength": min(strength, 1.0),
            "buy_signals": len(buy_signals),
            "sell_signals": len(sell_signals),
            "hold_signals": len(hold_signals),
            "total_signals": len(signals),
            "ai_buy_signals": len(ai_buy_signals),
            "ai_sell_signals": len(ai_sell_signals),
            "ai_signals_total": len(ai_buy_signals) + len(ai_sell_signals),
        }

    def _signal_to_dict(self, signal: StrategySignal) -> Dict[str, Any]:
        """Convert signal to dictionary"""
        return {
            "name": signal.name,
            "signal": signal.signal,
            "confidence": signal.confidence,
            "strength": signal.strength,
            "description": signal.description,
            "timestamp": signal.timestamp,
        }


class StrategyManager:
    """Manages all strategy controllers"""

    def __init__(self, cache: SharedCache):
        self.cache = cache
        self.config = get_config()
        self.controllers: Dict[str, StrategyController] = {}

        # Initialize controllers for all coins
        for coin_config in self.config.get_enabled_coins():
            self.controllers[coin_config.symbol] = StrategyController(coin_config.symbol, cache)

        logger.info(f"Strategy manager initialized with {len(self.controllers)} controllers")

    def run_all_strategies(self) -> Dict[str, Any]:
        """Run strategies for all coins"""
        results: Dict[str, Any] = {}

        for symbol, controller in self.controllers.items():
            try:
                result = controller.run_all_strategies()
                if result:
                    results[symbol] = result
                    # Update cache with strategy signals
                    self.cache.update_strategy_signals(symbol, result)
            except Exception as e:
                logger.error(f"Error running strategies for {symbol}: {e}")

        return results

    def run_coin_strategies(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Run strategies for specific coin"""
        controller = self.controllers.get(symbol)
        if not controller:
            return None

        try:
            result = controller.run_all_strategies()
            if result:
                self.cache.update_strategy_signals(symbol, result)
            return result
        except Exception as e:
            logger.error(f"Error running strategies for {symbol}: {e}")
            return None

    def get_strategy_status(self) -> Dict[str, Any]:
        """Get status of all strategy controllers"""
        return {
            "total_controllers": len(self.controllers),
            "active_controllers": len([c for c in self.controllers.values() if c]),
            "symbols": list(self.controllers.keys()),
        }
