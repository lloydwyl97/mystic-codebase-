#!/usr/bin/env python3
"""
Tier 2: Tactical Strategy Indicators
Handles tactical strategy signals every 2-3 minutes for trade timing and decision confidence
Optimized for 10 Binance + 10 Coinbase coins (20 total)
"""

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TechnicalIndicator:
    symbol: str
    rsi: float
    macd: Dict[str, float]
    volume_24h: float
    volatility_index: float
    change_5m: float
    change_15m: float
    change_1h: float
    timestamp: str
    api_source: str


class IndicatorsFetcher:
    def __init__(self, redis_client: Any):
        self.redis_client = redis_client
        self.is_running = False

        # Tier 2 Configuration - OPTIMIZED FOR 20 COINS
        self.config = {
            "rsi_fetch_interval": 120,  # 2 minutes per coin
            "volume_fetch_interval": 180,  # 3 minutes per coin
            "volatility_fetch_interval": 300,  # 5 minutes per coin
            "cache_ttl": 600,  # 10 minutes
            "rsi_period": 14,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
        }

        # Track last fetch times for throttling
        self.last_fetch_times: Dict[str, float] = {}

        # Price history for calculations
        self.price_history: Dict[str, List[Dict[str, Any]]] = {}

        # OPTIMIZED COIN LISTS - Updated with user specified pairs
        self.binance_coins = [
            "BTCUSDT",
            "ETHUSDT",
            "ADAUSDT",
            "SOLUSDT",
            "DOTUSDT",
            "LINKUSDT",
            "MATICUSDT",
            "AVAXUSDT",
            "UNIUSDT",
            "ATOMUSDT",
        ]
        self.coinbase_coins = [
            "BTC-USD",
            "ETH-USD",
            "ADA-USD",
            "SOL-USD",
            "DOT-USD",
            "LINK-USD",
            "MATIC-USD",
            "AVAX-USD",
            "UNI-USD",
            "ATOM-USD",
        ]

        logger.info(
            f"Indicators Fetcher initialized with {len(self.binance_coins)} Binance + {len(self.coinbase_coins)} Coinbase coins"
        )

    def _should_fetch(self, indicator_type: str, symbol: str) -> bool:
        """Check if we should fetch based on throttling rules"""
        now = time.time()
        key = f"{indicator_type}_{symbol}"

        if key not in self.last_fetch_times:
            return True

        last_fetch = self.last_fetch_times[key]
        interval = self.config[f"{indicator_type}_fetch_interval"]

        return (now - last_fetch) >= interval

    def _update_fetch_time(self, indicator_type: str, symbol: str):
        """Update the last fetch time for throttling"""
        key = f"{indicator_type}_{symbol}"
        self.last_fetch_times[key] = time.time()

    async def calculate_rsi(self, symbol: str) -> Optional[float]:
        """Calculate RSI for a symbol (2 minute frequency per coin)"""
        if not self._should_fetch("rsi", symbol):
            return None

        try:
            # Get price history from cache
            price_history = await self._get_price_history(symbol)
            if len(price_history) < self.config["rsi_period"] + 1:
                return None

            # Calculate RSI
            gains: List[float] = []
            losses: List[float] = []

            for i in range(1, len(price_history)):
                change = price_history[i]["price"] - price_history[i - 1]["price"]
                if change > 0:
                    gains.append(change)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(abs(change))

            if len(gains) < self.config["rsi_period"]:
                return None

            # Calculate average gains and losses
            avg_gain = sum(gains[-self.config["rsi_period"] :]) / self.config["rsi_period"]
            avg_loss = sum(losses[-self.config["rsi_period"] :]) / self.config["rsi_period"]

            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))

            self._update_fetch_time("rsi", symbol)
            return round(rsi, 2)

        except Exception as e:
            logger.error(f"Error calculating RSI for {symbol}: {e}")
            return None

    async def calculate_macd(self, symbol: str) -> Optional[Dict[str, float]]:
        """Calculate MACD for a symbol (2 minute frequency per coin)"""
        if not self._should_fetch("rsi", symbol):  # Use same interval as RSI
            return None

        try:
            # Get price history from cache
            price_history = await self._get_price_history(symbol)
            if len(price_history) < self.config["macd_slow"] + self.config["macd_signal"]:
                return None

            prices = [p["price"] for p in price_history]

            # Calculate EMA for fast and slow periods
            ema_fast = self._calculate_ema(prices, self.config["macd_fast"])
            ema_slow = self._calculate_ema(prices, self.config["macd_slow"])

            if len(ema_fast) == 0 or len(ema_slow) == 0:
                return None

            # Calculate MACD line
            macd_line = ema_fast[-1] - ema_slow[-1]

            # Calculate signal line (EMA of MACD line)
            macd_values = [ema_fast[i] - ema_slow[i] for i in range(len(ema_slow))]
            signal_line = self._calculate_ema(macd_values, self.config["macd_signal"])

            if len(signal_line) == 0:
                return None

            # Calculate histogram
            histogram = macd_line - signal_line[-1]

            self._update_fetch_time("rsi", symbol)  # Use same interval as RSI
            return {
                "macd_line": round(macd_line, 6),
                "signal_line": round(signal_line[-1], 6),
                "histogram": round(histogram, 6),
            }

        except Exception as e:
            logger.error(f"Error calculating MACD for {symbol}: {e}")
            return None

    def _calculate_ema(self, prices: List[float], period: int) -> List[float]:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return []

        ema_values: List[float] = []
        multiplier = 2 / (period + 1)

        # First EMA is SMA
        sma = sum(prices[:period]) / period
        ema_values.append(sma)

        # Calculate EMA for remaining prices
        for i in range(period, len(prices)):
            ema = (prices[i] * multiplier) + (ema_values[-1] * (1 - multiplier))
            ema_values.append(ema)

        return ema_values

    async def fetch_24h_volume(self, symbol: str) -> Optional[float]:
        """Fetch 24h volume data (3 minute frequency per coin)"""
        if not self._should_fetch("volume", symbol):
            return None

        try:
            # Get from Tier 1 cache or calculate from price history
            tier1_data = self.redis_client.get("tier1_signals")
            if tier1_data:
                tier1_signals = json.loads(tier1_data)
                if symbol in tier1_signals.get("prices", {}):
                    # Use volume from price signal if available
                    volume = tier1_signals["prices"][symbol].get("volume_1m", 0.0)
                    # Convert 1m volume to 24h estimate (simplified)
                    volume_24h = volume * 1440  # 24 hours * 60 minutes

                    self._update_fetch_time("volume", symbol)
                    return volume_24h

            # Fallback: calculate from price history
            price_history = await self._get_price_history(symbol)
            if len(price_history) >= 24:  # Need at least 24 data points
                # Calculate volume from price changes (simplified)
                total_volume = sum(
                    abs(p["price"] - price_history[i - 1]["price"])
                    for i, p in enumerate(price_history[1:], 1)
                )
                volume_24h = total_volume * 60  # Scale to 24h

                self._update_fetch_time("volume", symbol)
                return volume_24h

        except Exception as e:
            logger.error(f"Error fetching 24h volume for {symbol}: {e}")

        return None

    async def calculate_volatility_index(self, symbol: str) -> Optional[float]:
        """Calculate custom volatility index (5 minute frequency per coin)"""
        if not self._should_fetch("volatility", symbol):
            return None

        try:
            price_history = await self._get_price_history(symbol)
            if len(price_history) < 20:  # Need at least 20 data points
                return None

            prices = [p["price"] for p in price_history[-20:]]  # Last 20 prices

            # Calculate price changes
            changes: List[float] = []
            for i in range(1, len(prices)):
                change = ((prices[i] - prices[i - 1]) / prices[i - 1]) * 100
                changes.append(abs(change))

            # Calculate volatility as average of absolute changes
            volatility = sum(changes) / len(changes)

            # Normalize to 0-100 scale
            volatility_index = min(100, volatility * 10)

            self._update_fetch_time("volatility", symbol)
            return round(volatility_index, 2)

        except Exception as e:
            logger.error(f"Error calculating volatility index for {symbol}: {e}")
            return None

    async def calculate_time_changes(self, symbol: str) -> Dict[str, float]:
        """Calculate price changes over different time periods"""
        try:
            price_history = await self._get_price_history(symbol)
            if len(price_history) < 60:  # Need at least 60 data points
                return {"change_5m": 0.0, "change_15m": 0.0, "change_1h": 0.0}

            current_price = price_history[-1]["price"]

            # Calculate changes for different periods
            changes: Dict[str, float] = {}

            # 5-minute change (assuming 10-second intervals)
            if len(price_history) >= 30:  # 5 minutes = 30 intervals
                price_5m = price_history[-30]["price"]
                changes["change_5m"] = ((current_price - price_5m) / price_5m) * 100

            # 15-minute change
            if len(price_history) >= 90:  # 15 minutes = 90 intervals
                price_15m = price_history[-90]["price"]
                changes["change_15m"] = ((current_price - price_15m) / price_15m) * 100

            # 1-hour change
            if len(price_history) >= 360:  # 1 hour = 360 intervals
                price_1h = price_history[-360]["price"]
                changes["change_1h"] = ((current_price - price_1h) / price_1h) * 100

            return {k: round(v, 2) for k, v in changes.items()}

        except Exception as e:
            logger.error(f"Error calculating time changes for {symbol}: {e}")
            return {"change_5m": 0.0, "change_15m": 0.0, "change_1h": 0.0}

    async def _get_price_history(self, symbol: str) -> List[Dict[str, Any]]:
        """Get price history from cache"""
        try:
            # Get from Tier 1 cache
            tier1_data = self.redis_client.get("tier1_signals")
            if tier1_data:
                tier1_signals = json.loads(tier1_data)
                if symbol in tier1_signals.get("prices", {}):
                    price_data = tier1_signals["prices"][symbol]
                    return [
                        {
                            "price": price_data["price"],
                            "timestamp": price_data["timestamp"],
                        }
                    ]

            # Get from price history cache
            history_key = f"price_history_{symbol}"
            history_data = self.redis_client.get(history_key)
            if history_data:
                return json.loads(history_data)

            return []

        except Exception as e:
            logger.error(f"Error getting price history for {symbol}: {e}")
            return []

    async def _cache_indicator_data(self, symbol: str, data: Dict[str, Any]):
        """Cache indicator data"""
        try:
            key = f"indicators_{symbol}"
            self.redis_client.setex(key, self.config["cache_ttl"], json.dumps(data))
        except Exception as e:
            logger.error(f"Error caching indicator data: {e}")

    async def fetch_all_tier2_indicators(self) -> Dict[str, Any]:
        """Fetch all Tier 2 indicators for all 20 coins"""
        results: Dict[str, Any] = {
            "indicators": {},
            "timestamp": datetime.now(timezone.timezone.utc).isoformat(),
        }

        # Get all symbols from both exchanges
        all_symbols = self.binance_coins + self.coinbase_coins

        for symbol in all_symbols:
            try:
                # Calculate all indicators
                rsi = await self.calculate_rsi(symbol)
                macd = await self.calculate_macd(symbol)
                volume_24h = await self.fetch_24h_volume(symbol)
                volatility_index = await self.calculate_volatility_index(symbol)
                time_changes = await self.calculate_time_changes(symbol)

                # Determine API source
                api_source = "binance" if symbol in self.binance_coins else "coinbase"

                if any([rsi, macd, volume_24h, volatility_index]):
                    indicator_data = TechnicalIndicator(
                        symbol=symbol,
                        rsi=rsi or 50.0,
                        macd=macd
                        or {
                            "macd_line": 0.0,
                            "signal_line": 0.0,
                            "histogram": 0.0,
                        },
                        volume_24h=volume_24h or 0.0,
                        volatility_index=volatility_index or 0.0,
                        change_5m=time_changes.get("change_5m", 0.0),
                        change_15m=time_changes.get("change_15m", 0.0),
                        change_1h=time_changes.get("change_1h", 0.0),
                        timestamp=datetime.now(timezone.timezone.utc).isoformat(),
                        api_source=api_source,
                    )

                    results["indicators"][symbol] = asdict(indicator_data)
                    await self._cache_indicator_data(symbol, asdict(indicator_data))

            except Exception as e:
                logger.error(f"Error processing indicators for {symbol}: {e}")
                continue

        # Cache the complete Tier 2 data
        await self._cache_tier2_data(results)

        return results

    async def _cache_tier2_data(self, data: Dict[str, Any]):
        """Cache complete Tier 2 data"""
        try:
            self.redis_client.setex("tier2_indicators", self.config["cache_ttl"], json.dumps(data))
        except Exception as e:
            logger.error(f"Error caching Tier 2 data: {e}")

    async def run(self):
        """Main indicators fetcher loop - OPTIMIZED FOR 20 COINS"""
        logger.info("Starting Tier 2 Indicators Fetcher (20 coins)...")
        self.is_running = True

        try:
            while self.is_running:
                try:
                    # Fetch all Tier 2 indicators
                    indicators = await self.fetch_all_tier2_indicators()

                    logger.debug(
                        f"Calculated indicators for {len(indicators['indicators'])} symbols"
                    )

                    # Wait for next cycle (use shortest interval - 2 minutes)
                    await asyncio.sleep(self.config["rsi_fetch_interval"])

                except Exception as e:
                    logger.error(f"Error in indicators fetcher loop: {e}")
                    await asyncio.sleep(30)  # Wait 30 seconds on error

        except Exception as e:
            logger.error(f"Fatal error in indicators fetcher: {e}")
        finally:
            self.is_running = False

    def get_status(self) -> Dict[str, Any]:
        """Get indicators fetcher status"""
        return {
            "status": "running" if self.is_running else "stopped",
            "config": self.config,
            "last_fetch_times": self.last_fetch_times,
            "price_history_count": len(self.price_history),
            "supported_coins": {
                "binance": self.binance_coins,
                "coinbase": self.coinbase_coins,
                "total": len(self.binance_coins) + len(self.coinbase_coins),
            },
        }
