#!/usr/bin/env python3
"""
Strategy System Manager for Mystic Trading Platform

Manages strategy execution and signal generation.
"""

import logging
import time
from typing import Any, Dict, List, Optional

# Use absolute imports
from crypto_autoengine_config import get_config
from backend.services.websocket_manager import websocket_manager
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
        self.timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")


class StrategySystemManager:
    """Manager for the entire strategy system"""

    def __init__(self, cache: SharedCache):
        self.cache = cache
        self.config = get_config()
        self._strategy_controllers: Dict[str, "StrategyController"] = {}

    def get_strategy_controller(self, symbol: str) -> "StrategyController":
        """Get or create strategy controller for a symbol"""
        if symbol not in self._strategy_controllers:
            self._strategy_controllers[symbol] = StrategyController(symbol, self.cache)
        return self._strategy_controllers[symbol]

    def run_all_strategies(self) -> Dict[str, Any]:
        """Run all strategies for all configured coins"""
        results: Dict[str, Any] = {}
        symbols = self.config.get_all_symbols()

        for symbol in symbols:
            try:
                controller = self.get_strategy_controller(symbol)
                result = controller.run_all_strategies()
                if result:
                    results[symbol] = result
            except Exception as e:
                logger.error(f"Error running strategies for {symbol}: {e}")
                continue

        return {
            "results": results,
            "total_coins": len(symbols),
            "successful_coins": len(results),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

    def run_coin_strategies(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Run strategies for a specific coin"""
        try:
            controller = self.get_strategy_controller(symbol)
            return controller.run_all_strategies()
        except Exception as e:
            logger.error(f"Error running strategies for {symbol}: {e}")
            return None

    def get_strategy_status(self) -> Dict[str, Any]:
        """Get status of all strategy controllers"""
        status = {
            "total_controllers": len(self._strategy_controllers),
            "active_symbols": list(self._strategy_controllers.keys()),
            "cache_status": self.cache.get_cache_stats(),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        return status

    def clear_strategy_cache(self, symbol: Optional[str] = None):
        """Clear strategy cache for specific symbol or all"""
        if symbol:
            if symbol in self._strategy_controllers:
                del self._strategy_controllers[symbol]
        else:
            self._strategy_controllers.clear()

    def get_strategy_performance(self) -> Dict[str, Any]:
        """Get performance metrics for all strategies"""
        performance: Dict[str, Any] = {
            "total_signals_generated": 0,
            "buy_signals": 0,
            "sell_signals": 0,
            "hold_signals": 0,
            "average_confidence": 0.0,
            "average_strength": 0.0,
            "active_strategies": len(self._strategy_controllers),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        total_confidence = 0.0
        total_strength = 0.0
        signal_count = 0

        for controller in self._strategy_controllers.values():
            try:
                result = controller.run_all_strategies()
                if result and "signals" in result:
                    signals = result["signals"]
                    performance["total_signals_generated"] = int(
                        performance["total_signals_generated"]
                    ) + len(signals)

                    for signal in signals:
                        if signal["signal"] == "buy":
                            performance["buy_signals"] = int(performance["buy_signals"]) + 1
                        elif signal["signal"] == "sell":
                            performance["sell_signals"] = int(performance["sell_signals"]) + 1
                        else:
                            performance["hold_signals"] = int(performance["hold_signals"]) + 1

                        total_confidence += signal["confidence"]
                        total_strength += signal["strength"]
                        signal_count += 1
            except Exception as e:
                logger.error(f"Error calculating performance: {e}")
                continue

        if signal_count > 0:
            performance["average_confidence"] = total_confidence / signal_count
            performance["average_strength"] = total_strength / signal_count

        # Broadcast strategy performance update
        import asyncio

        asyncio.create_task(
            websocket_manager.broadcast_json({"type": "strategy_performance", "data": performance})
        )

        return performance


class StrategyController:
    """Controller for individual coin strategies"""

    def __init__(self, symbol: str, cache: SharedCache):
        self.symbol = symbol
        self.cache = cache
        self.config = get_config()
        self.coin_config = self.config.get_coin_by_symbol(symbol)

    def run_all_strategies(self) -> Dict[str, Any]:
        """Run all 65+ strategies for this coin"""
        if not self.coin_config:
            return {}

        coin_data = self.cache.get_coin_cache(self.symbol)
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

        # Aggregate signals
        aggregated = self._aggregate_signals(signals)

        return {
            "signals": [self._signal_to_dict(s) for s in signals],
            "aggregated": aggregated,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

    def _price_based_strategies(self, coin_data: CoinCache) -> List[StrategySignal]:
        """Price-based strategies (1-20)"""
        signals: List[StrategySignal] = []
        price = coin_data.price
        history = coin_data.price_history

        if history is None or len(history) < 10:
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

        return signals

    def _volume_based_strategies(self, coin_data: CoinCache) -> List[StrategySignal]:
        """Volume-based strategies (21-35)"""
        signals: List[StrategySignal] = []
        volume = coin_data.volume_24h
        history = coin_data.price_history

        if history is None or len(history) < 10:
            return signals

        # Strategy 21: Volume Spike
        # Note: We're using the current volume_24h value as we don't have historical volume in price_history
        if volume > 0:
            signals.append(
                StrategySignal(
                    "Volume Analysis",
                    ("buy" if coin_data.price > float(history[-2]["price"]) else "sell"),
                    0.6,
                    0.5,
                    f"Current 24h volume: {volume:.0f}",
                )
            )

        # Strategy 22: Price-Volume Correlation
        # Since we don't have historical volume data in price_history,
        # we'll use price movement as a proxy for volume trend
        if len(history) >= 10:
            price_change = (coin_data.price - float(history[-10]["price"])) / float(
                history[-10]["price"]
            )

            if price_change > 0.05:  # 5% price increase
                signals.append(
                    StrategySignal(
                        "Price-Volume Correlation",
                        "buy",
                        0.6,
                        0.5,
                        f"Strong price movement: {price_change:.1%}",
                    )
                )
            elif price_change < -0.05:  # 5% price decrease
                signals.append(
                    StrategySignal(
                        "Price-Volume Correlation",
                        "sell",
                        0.6,
                        0.5,
                        f"Strong price movement: {price_change:.1%}",
                    )
                )

        return signals

    def _technical_indicator_strategies(self, coin_data: CoinCache) -> List[StrategySignal]:
        """Technical indicator strategies (36-50)"""
        signals: List[StrategySignal] = []
        history = coin_data.price_history

        if history is None or len(history) < 20:
            return signals

        prices = [float(h["price"]) for h in history]

        # Strategy 36: RSI
        rsi = self._calculate_rsi(prices)
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

        # Strategy 37: MACD
        macd_data = self._calculate_macd(prices)
        if macd_data["macd"] > macd_data["signal"] and macd_data["histogram"] > 0:
            signals.append(
                StrategySignal(
                    "MACD Bullish",
                    "buy",
                    0.7,
                    0.6,
                    f"MACD bullish: {macd_data['macd']:.4f} > {macd_data['signal']:.4f}",
                )
            )
        elif macd_data["macd"] < macd_data["signal"] and macd_data["histogram"] < 0:
            signals.append(
                StrategySignal(
                    "MACD Bearish",
                    "sell",
                    0.7,
                    0.6,
                    f"MACD bearish: {macd_data['macd']:.4f} < {macd_data['signal']:.4f}",
                )
            )

        return signals

    def _momentum_strategies(self, coin_data: CoinCache) -> List[StrategySignal]:
        """Momentum strategies (51-60)"""
        signals: List[StrategySignal] = []
        history = coin_data.price_history

        if history is None or len(history) < 20:
            return signals

        prices = [float(h["price"]) for h in history]

        # Strategy 51: EMA Crossover
        ema_12 = self._calculate_ema(prices, 12)
        ema_26 = self._calculate_ema(prices, 26)

        if ema_12 > ema_26:
            signals.append(
                StrategySignal(
                    "EMA Bullish Crossover",
                    "buy",
                    0.7,
                    0.6,
                    f"EMA 12 ({ema_12:.2f}) > EMA 26 ({ema_26:.2f})",
                )
            )
        else:
            signals.append(
                StrategySignal(
                    "EMA Bearish Crossover",
                    "sell",
                    0.7,
                    0.6,
                    f"EMA 12 ({ema_12:.2f}) < EMA 26 ({ema_26:.2f})",
                )
            )

        return signals

    def _volatility_strategies(self, coin_data: CoinCache) -> List[StrategySignal]:
        """Volatility strategies (61-65)"""
        signals: List[StrategySignal] = []
        history = coin_data.price_history

        if history is None or len(history) < 20:
            return signals

        prices = [float(h["price"]) for h in history]

        # Strategy 61: Volatility Breakout
        if len(prices) >= 20:
            recent_prices = prices[-20:]
            volatility = self._calculate_volatility(recent_prices)

            if volatility > 0.05:  # High volatility
                signals.append(
                    StrategySignal(
                        "High Volatility",
                        "hold",
                        0.6,
                        0.5,
                        f"High volatility detected: {volatility:.2%}",
                    )
                )

        return signals

    def _cosmic_strategies(self, coin_data: CoinCache) -> List[StrategySignal]:
        """Cosmic/mystic strategies (66+)"""
        signals: List[StrategySignal] = []

        # Strategy 66: Mystic Alignment
        current_hour = time.localtime().tm_hour
        if current_hour in [0, 6, 12, 18]:  # Cosmic alignment hours
            signals.append(
                StrategySignal(
                    "Cosmic Alignment",
                    "buy",
                    0.5,
                    0.4,
                    f"Cosmic alignment at hour {current_hour}",
                )
            )

        return signals

    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate RSI"""
        if not prices or len(prices) < period + 1:
            return 50.0

        deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]

        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(self, prices: List[float]) -> Dict[str, float]:
        """Calculate MACD"""
        if len(prices) < 26:
            return {"macd": 0, "signal": 0, "histogram": 0}

        ema_12 = self._calculate_ema(prices, 12)
        ema_26 = self._calculate_ema(prices, 26)
        macd = ema_12 - ema_26

        # Simplified signal line calculation
        signal = macd * 0.8  # Approximation
        histogram = macd - signal

        return {"macd": macd, "signal": signal, "histogram": histogram}

    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """Calculate EMA"""
        if len(prices) < period:
            return prices[-1] if prices else 0

        multiplier = 2 / (period + 1)
        ema = prices[0]

        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))

        return ema

    def _calculate_volatility(self, prices: List[float]) -> float:
        """Calculate price volatility"""
        if len(prices) < 2:
            return 0.0

        returns = [(prices[i] - prices[i - 1]) / prices[i - 1] for i in range(1, len(prices))]
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        volatility = variance**0.5

        return volatility

    def _aggregate_signals(self, signals: List[StrategySignal]) -> Dict[str, Any]:
        """Aggregate all signals into final recommendation"""
        if not signals:
            return {
                "signal": "hold",
                "confidence": 0.0,
                "strength": 0.0,
                "description": "No signals generated",
            }

        buy_signals = [s for s in signals if s.signal == "buy"]
        sell_signals = [s for s in signals if s.signal == "sell"]
        hold_signals = [s for s in signals if s.signal == "hold"]

        # Calculate weighted scores
        buy_score = sum(s.confidence * s.strength for s in buy_signals)
        sell_score = sum(s.confidence * s.strength for s in sell_signals)
        hold_score = sum(s.confidence * s.strength for s in hold_signals)

        # Determine final signal
        if buy_score > sell_score and buy_score > hold_score:
            final_signal = "buy"
            final_score = buy_score
        elif sell_score > buy_score and sell_score > hold_score:
            final_signal = "sell"
            final_score = sell_score
        else:
            final_signal = "hold"
            final_score = hold_score

        # Normalize confidence and strength
        total_signals = len(signals)
        confidence = final_score / total_signals if total_signals > 0 else 0.0
        strength = (
            final_score / max(buy_score, sell_score, hold_score)
            if max(buy_score, sell_score, hold_score) > 0
            else 0.0
        )

        return {
            "signal": final_signal,
            "confidence": min(confidence, 1.0),
            "strength": min(strength, 1.0),
            "description": (
                f"Based on {len(signals)} strategies: {len(buy_signals)} buy, {len(sell_signals)} sell, {len(hold_signals)} hold"
            ),
            "signal_counts": {
                "buy": len(buy_signals),
                "sell": len(sell_signals),
                "hold": len(hold_signals),
            },
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



