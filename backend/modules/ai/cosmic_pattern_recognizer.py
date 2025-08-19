"""
Cosmic Pattern Recognizer for Mystic AI Trading Platform
Analyzes market data for cosmic resonance patterns and generates anomaly flags.
"""

import logging
import math
import os
import sys
from datetime import datetime, timezone
from typing import Any

import numpy as np

# Add backend to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.modules.ai.persistent_cache import PersistentCache

logger = logging.getLogger(__name__)


class CosmicPatternRecognizer:
    def __init__(self):
        """Initialize cosmic pattern recognizer with analysis parameters"""
        self.cache = PersistentCache()

        # Cosmic analysis parameters
        self.lunar_cycle_days = 29.53
        self.min_data_points = 100
        self.volatility_window = 24  # hours
        self.resonance_threshold = 0.7
        self.discordance_threshold = 0.3

        # Frequency analysis parameters
        self.fft_window_size = 64
        self.min_frequency = 0.001  # Hz
        self.max_frequency = 0.1    # Hz

        # Pattern detection parameters
        self.micro_cycle_min_period = 4   # hours
        self.micro_cycle_max_period = 48  # hours
        self.lunar_correlation_threshold = 0.6
        
        # Dashboard tracking attributes
        self.patterns_detected = 0
        self.active_patterns = 0
        self.pattern_accuracy = 0.0
        self.last_pattern_time = datetime.now(timezone.utc).isoformat()

        logger.info("âœ… CosmicPatternRecognizer initialized")

    def _get_market_data(self, symbol: str, hours: int = 168) -> list[dict[str, Any]]:
        """Get recent market data from cache"""
        try:
            # Get price history from cache
            price_history = self.cache.get_price_history('aggregated', symbol, limit=hours)

            if not price_history or len(price_history) < self.min_data_points:
                logger.warning(f"Insufficient market data for {symbol}: {len(price_history) if price_history else 0} records")
                return []

            # Convert to structured format
            market_data = []
            for data in price_history:
                try:
                    timestamp = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
                    price = float(data['price'])
                    volume = float(data.get('volume', 0))

                    market_data.append({
                        'timestamp': timestamp,
                        'price': price,
                        'volume': volume
                    })
                except (ValueError, KeyError) as e:
                    logger.warning(f"Skipping invalid data point: {e}")
                    continue

            # Sort by timestamp
            market_data.sort(key=lambda x: x['timestamp'])

            return market_data

        except Exception as e:
            logger.error(f"Failed to get market data for {symbol}: {e}")
            return []

    def _calculate_volatility(self, prices: list[float], window: int = 24) -> list[float]:
        """Calculate rolling volatility"""
        try:
            if len(prices) < window:
                return []

            volatilities = []
            for i in range(window, len(prices)):
                window_prices = prices[i-window:i]
                returns = [math.log(prices[j] / prices[j-1]) for j in range(1, len(window_prices))]

                if returns:
                    volatility = np.std(returns) * math.sqrt(24)  # Annualized
                    volatilities.append(volatility)
                else:
                    volatilities.append(0.0)

            return volatilities

        except Exception as e:
            logger.error(f"Failed to calculate volatility: {e}")
            return []

    def _detect_micro_cycles(self, timestamps: list[datetime], prices: list[float]) -> dict[str, Any]:
        """Detect micro-cycles in price data"""
        try:
            if len(prices) < 50:
                return {"cycles": [], "dominant_period": None}

            # Calculate price changes
            price_changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]

            # Apply FFT to detect periodic patterns
            fft_data = np.fft.fft(price_changes)
            frequencies = np.fft.fftfreq(len(price_changes))

            # Filter relevant frequencies (exclude DC and very high frequencies)
            relevant_indices = np.where((frequencies > 0) & (frequencies < 0.1))[0]

            if len(relevant_indices) == 0:
                return {"cycles": [], "dominant_period": None}

            # Find dominant frequency
            power_spectrum = np.abs(fft_data[relevant_indices]) ** 2
            dominant_idx = relevant_indices[np.argmax(power_spectrum)]
            dominant_frequency = frequencies[dominant_idx]

            # Convert frequency to period (hours)
            if dominant_frequency > 0:
                dominant_period = 1 / dominant_frequency
            else:
                dominant_period = None

            # Detect cycles within reasonable range
            cycles = []
            for i, freq in enumerate(frequencies[relevant_indices]):
                if freq > 0:
                    period = 1 / freq
                    if self.micro_cycle_min_period <= period <= self.micro_cycle_max_period:
                        power = power_spectrum[i]
                        cycles.append({
                            "period": period,
                            "frequency": freq,
                            "power": power
                        })

            return {
                "cycles": cycles,
                "dominant_period": dominant_period,
                "power_spectrum": power_spectrum.tolist()
            }

        except Exception as e:
            logger.error(f"Failed to detect micro-cycles: {e}")
            return {"cycles": [], "dominant_period": None}

    def _calculate_lunar_correlation(self, timestamps: list[datetime], prices: list[float]) -> float:
        """Calculate correlation with lunar cycle"""
        try:
            if len(timestamps) < 30:
                return 0.0

            # Calculate lunar phase for each timestamp
            lunar_phases = []
            for timestamp in timestamps:
                # Calculate days since a known new moon (approximate)
                known_new_moon = datetime(2024, 1, 11, tzinfo=timezone.utc)
                days_since = (timestamp - known_new_moon).total_seconds() / (24 * 3600)

                # Calculate lunar phase (0-1, where 0 is new moon)
                lunar_phase = (days_since % self.lunar_cycle_days) / self.lunar_cycle_days
                lunar_phases.append(lunar_phase)

            # Calculate correlation between price changes and lunar phase
            if len(prices) > 1:
                price_changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
                lunar_phases_changes = [lunar_phases[i] - lunar_phases[i-1] for i in range(1, len(lunar_phases))]

                if len(price_changes) == len(lunar_phases_changes):
                    correlation = np.corrcoef(price_changes, lunar_phases_changes)[0, 1]
                    return correlation if not np.isnan(correlation) else 0.0

            return 0.0

        except Exception as e:
            logger.error(f"Failed to calculate lunar correlation: {e}")
            return 0.0

    def _analyze_high_frequency_oscillations(self, timestamps: list[datetime], prices: list[float]) -> dict[str, Any]:
        """Analyze high-frequency oscillations in price data"""
        try:
            if len(prices) < 20:
                return {"oscillation_strength": 0.0, "dominant_frequency": None}

            # Calculate price changes
            price_changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]

            # Apply FFT for frequency analysis
            fft_data = np.fft.fft(price_changes)
            frequencies = np.fft.fftfreq(len(price_changes))

            # Focus on high-frequency components
            high_freq_indices = np.where((frequencies > 0.01) & (frequencies < 0.1))[0]

            if len(high_freq_indices) == 0:
                return {"oscillation_strength": 0.0, "dominant_frequency": None}

            # Calculate oscillation strength
            power_spectrum = np.abs(fft_data[high_freq_indices]) ** 2
            oscillation_strength = np.mean(power_spectrum)

            # Find dominant high-frequency component
            dominant_idx = high_freq_indices[np.argmax(power_spectrum)]
            dominant_frequency = frequencies[dominant_idx]

            return {
                "oscillation_strength": float(oscillation_strength),
                "dominant_frequency": float(dominant_frequency),
                "power_spectrum": power_spectrum.tolist()
            }

        except Exception as e:
            logger.error(f"Failed to analyze high-frequency oscillations: {e}")
            return {"oscillation_strength": 0.0, "dominant_frequency": None}

    def _calculate_cosmic_resonance_index(self, micro_cycles: dict[str, Any],
                                        lunar_correlation: float,
                                        oscillation_analysis: dict[str, Any]) -> float:
        """Calculate Cosmic Resonance Index (CRI)"""
        try:
            cri_components = []

            # Micro-cycle component (0-1)
            if micro_cycles.get("dominant_period"):
                cycle_strength = min(1.0, len(micro_cycles["cycles"]) / 10.0)
                cri_components.append(cycle_strength * 0.4)
            else:
                cri_components.append(0.0)

            # Lunar correlation component (0-1)
            lunar_strength = abs(lunar_correlation)
            cri_components.append(lunar_strength * 0.3)

            # High-frequency oscillation component (0-1)
            oscillation_strength = min(1.0, oscillation_analysis.get("oscillation_strength", 0.0) / 100.0)
            cri_components.append(oscillation_strength * 0.3)

            # Calculate weighted CRI
            cri = sum(cri_components)

            return min(1.0, cri)

        except Exception as e:
            logger.error(f"Failed to calculate CRI: {e}")
            return 0.0

    def _determine_anomaly_flag(self, cri: float, lunar_correlation: float,
                               oscillation_strength: float) -> str:
        """Determine anomaly flag based on analysis"""
        try:
            # High resonance conditions
            if cri > self.resonance_threshold and abs(lunar_correlation) > 0.5:
                return "HIGH_RESONANCE"

            # Discordant conditions
            if cri < self.discordance_threshold or (abs(lunar_correlation) < 0.1 and oscillation_strength > 50):
                return "DISCORDANT"

            # Normal conditions
            return "NORMAL"

        except Exception as e:
            logger.error(f"Failed to determine anomaly flag: {e}")
            return "NORMAL"

    def analyze_pattern(self, exchange: str, symbol: str) -> dict[str, Any]:
        """Analyze cosmic patterns for a symbol"""
        try:
            logger.info(f"ðŸ”® Analyzing cosmic patterns for {symbol} on {exchange}")

            # Get market data
            market_data = self._get_market_data(symbol)
            if not market_data:
                return {
                    "exchange": exchange,
                    "symbol": symbol,
                    "success": False,
                    "reason": "Insufficient market data",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }

            # Extract data
            timestamps = [data['timestamp'] for data in market_data]
            prices = [data['price'] for data in market_data]
            [data['volume'] for data in market_data]

            # Calculate volatility
            volatilities = self._calculate_volatility(prices)

            # Detect micro-cycles
            micro_cycles = self._detect_micro_cycles(timestamps, prices)

            # Calculate lunar correlation
            lunar_correlation = self._calculate_lunar_correlation(timestamps, prices)

            # Analyze high-frequency oscillations
            oscillation_analysis = self._analyze_high_frequency_oscillations(timestamps, prices)

            # Calculate Cosmic Resonance Index
            cri = self._calculate_cosmic_resonance_index(micro_cycles, lunar_correlation, oscillation_analysis)

            # Determine anomaly flag
            anomaly_flag = self._determine_anomaly_flag(
                cri, lunar_correlation, oscillation_analysis.get("oscillation_strength", 0.0)
            )

            # Create analysis result
            analysis_result = {
                "exchange": exchange,
                "symbol": symbol,
                "success": True,
                "cosmic_resonance_index": cri,
                "anomaly_flag": anomaly_flag,
                "lunar_correlation": lunar_correlation,
                "micro_cycles": micro_cycles,
                "oscillation_analysis": oscillation_analysis,
                "volatility_analysis": {
                    "current_volatility": volatilities[-1] if volatilities else 0.0,
                    "avg_volatility": np.mean(volatilities) if volatilities else 0.0,
                    "volatility_trend": "increasing" if len(volatilities) > 1 and volatilities[-1] > volatilities[-2] else "decreasing"
                },
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            # Store analysis in cache
            analysis_id = f"cosmic_{symbol}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            self.cache.store_signal(
                signal_id=analysis_id,
                symbol=symbol,
                signal_type="COSMIC_ANALYSIS",
                confidence=cri,
                strategy="cosmic_pattern_recognition",
                metadata=analysis_result
            )

            logger.info(f"âœ… Cosmic analysis complete: {anomaly_flag} (CRI: {cri:.3f})")

            return analysis_result

        except Exception as e:
            logger.error(f"Failed to analyze cosmic pattern: {e}")
            return {
                "exchange": exchange,
                "symbol": symbol,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    def get_latest_pattern(self, exchange: str, symbol: str) -> dict[str, Any]:
        """Get the latest cosmic pattern analysis"""
        try:
            # Get recent cosmic analysis from cache
            signals = self.cache.get_signals_by_type("COSMIC_ANALYSIS", limit=1)

            if signals:
                latest_analysis = signals[0]
                return {
                    "exchange": exchange,
                    "symbol": symbol,
                    "success": True,
                    "analysis": latest_analysis,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            else:
                # Perform new analysis if no cached data
                return self.analyze_pattern(exchange, symbol)

        except Exception as e:
            logger.error(f"Failed to get latest pattern: {e}")
            return {
                "exchange": exchange,
                "symbol": symbol,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    def get_pattern_history(self, symbol: str, limit: int = 10) -> list[dict[str, Any]]:
        """Get cosmic pattern analysis history"""
        try:
            # Get recent cosmic analyses from cache
            signals = self.cache.get_signals_by_type("COSMIC_ANALYSIS", limit=limit)

            # Filter by symbol
            symbol_analyses = [
                signal for signal in signals
                if signal.get("symbol") == symbol
            ]
            return symbol_analyses
        except Exception as e:
            logger.error(f"Failed to get pattern history: {e}")
            return []

    def get_cosmic_status(self) -> dict[str, Any]:
        """Get current cosmic pattern recognizer status"""
        try:
            return {
                "service": "CosmicPatternRecognizer",
                "status": "active",
                "parameters": {
                    "lunar_cycle_days": self.lunar_cycle_days,
                    "min_data_points": self.min_data_points,
                    "volatility_window": self.volatility_window,
                    "resonance_threshold": self.resonance_threshold,
                    "discordance_threshold": self.discordance_threshold
                },
                "analysis_capabilities": {
                    "micro_cycle_detection": True,
                    "lunar_correlation": True,
                    "high_frequency_analysis": True,
                    "cosmic_resonance_index": True
                }
            }

        except Exception as e:
            logger.error(f"Failed to get cosmic status: {e}")
            return {"success": False, "error": str(e)}

    def update_pattern_stats(self, pattern_detected: bool = True, accuracy: float = 0.0):
        """Update pattern statistics for dashboard"""
        try:
            if pattern_detected:
                self.patterns_detected += 1
                self.active_patterns = min(10, self.active_patterns + 1)  # Cap at 10
                self.pattern_accuracy = accuracy
                self.last_pattern_time = datetime.now(timezone.utc).isoformat()
            else:
                self.active_patterns = max(0, self.active_patterns - 1)
                
        except Exception as e:
            logger.error(f"Failed to update pattern stats: {e}")

    def get_pattern_stats(self) -> dict[str, Any]:
        """Get pattern statistics for dashboard"""
        try:
            return {
                "patterns_detected": self.patterns_detected,
                "active_patterns": self.active_patterns,
                "pattern_accuracy": self.pattern_accuracy,
                "last_pattern_time": self.last_pattern_time
            }
        except Exception as e:
            logger.error(f"Failed to get pattern stats: {e}")
            return {
                "patterns_detected": 0,
                "active_patterns": 0,
                "pattern_accuracy": 0.0,
                "last_pattern_time": datetime.now(timezone.utc).isoformat()
            }


# Global cosmic pattern recognizer instance
cosmic_pattern_recognizer = CosmicPatternRecognizer()


def get_cosmic_pattern_recognizer() -> CosmicPatternRecognizer:
    """Get the global cosmic pattern recognizer instance"""
    return cosmic_pattern_recognizer


if __name__ == "__main__":
    # Test the cosmic pattern recognizer
    recognizer = CosmicPatternRecognizer()
    print(f"âœ… CosmicPatternRecognizer initialized: {recognizer}")

    # Test pattern analysis
    analysis = recognizer.analyze_pattern('coinbase', 'BTC-USD')
    print(f"Cosmic analysis: {analysis}")

    # Test status
    status = recognizer.get_cosmic_status()
    print(f"Cosmic status: {status['status']}")



