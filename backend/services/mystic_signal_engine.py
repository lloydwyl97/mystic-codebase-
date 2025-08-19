"""
Mystic Signal Engine
Combines various mystic factors to generate trading signals
including Tesla 369, Faerie Star, Lagos alignment, and other esoteric patterns.
"""

import logging
import math
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from mystic_config import mystic_config

from backend.services.mystic_integration_service import mystic_integration_service

logger = logging.getLogger(__name__)


class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    STRONG_BUY = "STRONG_BUY"
    STRONG_SELL = "STRONG_SELL"


@dataclass
class MysticSignal:
    """Comprehensive mystic signal for trading decisions"""

    signal_type: SignalType
    confidence: float
    strength: float
    factors: dict[str, Any]
    timestamp: datetime
    reasoning: list[str]


class Tesla369Engine:
    """Tesla 369 frequency engine for trading signals"""

    def __init__(self):
        self.base_frequency = 369  # Tesla's sacred number
        self.harmonics = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30]

    def calculate_tesla_signal(self, current_time: datetime) -> dict[str, Any]:
        """Calculate Tesla 369 influence on trading"""
        # Tesla's time-based frequency calculations
        hour_factor = (current_time.hour % 12) / 12.0
        minute_factor = (current_time.minute % 60) / 60.0
        second_factor = (current_time.second % 60) / 60.0

        # 369 resonance calculation
        resonance = (hour_factor * 3 + minute_factor * 6 + second_factor * 9) / 18

        # Tesla's vortex mathematics
        vortex_strength = math.sin(resonance * math.pi) * math.cos(resonance * math.pi * 2)

        # Determine signal direction based on Tesla's principles
        if vortex_strength > 0.5:
            signal_direction = "BUY"
            strength = min(1.0, vortex_strength)
        elif vortex_strength < -0.5:
            signal_direction = "SELL"
            strength = min(1.0, abs(vortex_strength))
        else:
            signal_direction = "HOLD"
            strength = 0.3

        return {
            "direction": signal_direction,
            "strength": strength,
            "resonance": resonance,
            "vortex_strength": vortex_strength,
            "frequency": self.base_frequency * (1 + resonance),
        }


class FaerieStarEngine:
    """Faerie Star alignment engine for mystical trading signals"""

    def __init__(self):
        self.star_phases = ["Dawn", "Noon", "Dusk", "Midnight"]
        self.elemental_forces = ["Fire", "Water", "Earth", "Air", "Ether"]

    def calculate_faerie_signal(self, current_time: datetime) -> dict[str, Any]:
        """Calculate Faerie Star influence on trading"""
        # Faerie time cycles (based on lunar and solar alignments)
        lunar_cycle = (current_time.day % 29.5) / 29.5
        solar_cycle = (current_time.hour % 24) / 24

        # Faerie star phase determination
        phase_index = int((lunar_cycle + solar_cycle) * 2) % 4
        current_phase = self.star_phases[phase_index]

        # Elemental force calculation
        elemental_index = int(current_time.hour / 4.8) % 5
        current_element = self.elemental_forces[elemental_index]

        # Faerie magic strength
        magic_strength = math.sin(lunar_cycle * math.pi * 2) * math.cos(solar_cycle * math.pi)

        # Trading signal based on Faerie wisdom
        if current_phase in ["Dawn", "Noon"] and magic_strength > 0.3:
            signal_direction = "BUY"
            strength = min(1.0, magic_strength + 0.3)
        elif current_phase in ["Dusk", "Midnight"] and magic_strength < -0.3:
            signal_direction = "SELL"
            strength = min(1.0, abs(magic_strength) + 0.3)
        else:
            signal_direction = "HOLD"
            strength = 0.4

        return {
            "direction": signal_direction,
            "strength": strength,
            "phase": current_phase,
            "element": current_element,
            "magic_strength": magic_strength,
            "lunar_cycle": lunar_cycle,
            "solar_cycle": solar_cycle,
        }


class LagosAlignmentEngine:
    """Lagos alignment engine for cosmic trading signals"""

    def __init__(self):
        self.cosmic_cycles = ["Harmony", "Chaos", "Balance", "Transformation"]
        self.energy_levels = ["Low", "Medium", "High", "Peak"]

    def calculate_lagos_signal(self, current_time: datetime) -> dict[str, Any]:
        """Calculate Lagos alignment influence on trading"""
        # Cosmic time cycles
        cosmic_cycle = (current_time.hour + current_time.minute / 60) / 24
        energy_cycle = (current_time.minute + current_time.second / 60) / 60

        # Lagos alignment strength
        alignment_strength = math.sin(cosmic_cycle * math.pi * 4) * math.cos(
            energy_cycle * math.pi * 2
        )

        # Cosmic cycle determination
        cycle_index = int(cosmic_cycle * 4) % 4
        current_cycle = self.cosmic_cycles[cycle_index]

        # Energy level determination
        energy_index = int(energy_cycle * 4) % 4
        current_energy = self.energy_levels[energy_index]

        # Trading signal based on Lagos alignment
        if current_cycle in ["Harmony", "Balance"] and alignment_strength > 0.4:
            signal_direction = "BUY"
            strength = min(1.0, alignment_strength + 0.2)
        elif current_cycle in ["Chaos", "Transformation"] and alignment_strength < -0.4:
            signal_direction = "SELL"
            strength = min(1.0, abs(alignment_strength) + 0.2)
        else:
            signal_direction = "HOLD"
            strength = 0.5

        return {
            "direction": signal_direction,
            "strength": strength,
            "cycle": current_cycle,
            "energy": current_energy,
            "alignment_strength": alignment_strength,
            "cosmic_cycle": cosmic_cycle,
            "energy_cycle": energy_cycle,
        }


class MysticSignalEngine:
    """Main mystic signal engine that combines all factors"""

    def __init__(self) -> None:
        self.config = mystic_config
        self.tesla_engine = Tesla369Engine()
        self.faerie_engine = FaerieStarEngine()
        self.lagos_engine = LagosAlignmentEngine()
        self.cache: dict[str, Any] = {}
        self.cache_ttl: int = 60  # 1 minute cache

    async def generate_comprehensive_signal(self, symbol: str = "BTCUSDT") -> MysticSignal:
        """Generate comprehensive mystic trading signal"""
        try:
            current_time = datetime.now()

            # Get all mystic data
            schumann_data = await mystic_integration_service.get_schumann_resonance()
            fractal_data = await mystic_integration_service.get_fractal_time_data()
            planetary_data = await mystic_integration_service.get_planetary_alignment()
            mystic_signal_data = await mystic_integration_service.get_mystic_signal_strength()

            # Calculate individual engine signals
            tesla_signal = self.tesla_engine.calculate_tesla_signal(current_time)
            faerie_signal = self.faerie_engine.calculate_faerie_signal(current_time)
            lagos_signal = self.lagos_engine.calculate_lagos_signal(current_time)

            # Combine all signals with weighted importance
            signal_components = self._combine_signals(
                schumann_data,
                fractal_data,
                planetary_data,
                mystic_signal_data,
                tesla_signal,
                faerie_signal,
                lagos_signal,
            )

            # Determine final signal type and strength
            final_signal = self._determine_final_signal(signal_components)

            # Generate reasoning
            reasoning = self._generate_reasoning(signal_components)

            return MysticSignal(
                signal_type=final_signal["type"],
                confidence=final_signal["confidence"],
                strength=final_signal["strength"],
                factors=signal_components,
                timestamp=current_time,
                reasoning=reasoning,
            )

        except Exception as e:
            logger.error(f"Error generating mystic signal: {e}")
            return MysticSignal(
                signal_type=SignalType.HOLD,
                confidence=0.5,
                strength=0.3,
                factors={},
                timestamp=datetime.now(),
                reasoning=["Error in mystic signal generation"],
            )

    def _combine_signals(
        self,
        schumann: Any,
        fractal: Any,
        planetary: Any,
        mystic_signal: Any,
        tesla: Any,
        faerie: Any,
        lagos: Any,
    ) -> dict[str, Any]:
        """Combine all mystic signals with weighted importance"""

        # Weight definitions (total = 1.0)
        weights = {
            "schumann": 0.15,
            "fractal": 0.12,
            "planetary": 0.12,
            "mystic_signal": 0.15,
            "tesla": 0.18,
            "faerie": 0.14,
            "lagos": 0.14,
        }

        # Convert signals to numerical values
        signal_values = {
            "schumann": self._schumann_to_signal(schumann),
            "fractal": self._fractal_to_signal(fractal),
            "planetary": self._planetary_to_signal(planetary),
            "mystic_signal": self._mystic_signal_to_signal(mystic_signal),
            "tesla": self._tesla_to_signal(tesla),
            "faerie": self._faerie_to_signal(faerie),
            "lagos": self._lagos_to_signal(lagos),
        }

        # Calculate weighted average
        total_signal = 0.0
        total_weight = 0.0

        for component, weight in weights.items():
            signal_val = signal_values[component]["value"]
            total_signal += signal_val * weight
            total_weight += weight

        if total_weight > 0:
            final_signal = total_signal / total_weight
        else:
            final_signal = 0.0

        return {
            "final_signal": final_signal,
            "components": signal_values,
            "weights": weights,
            "raw_data": {
                "schumann": schumann,
                "fractal": fractal,
                "planetary": planetary,
                "mystic_signal": mystic_signal,
                "tesla": tesla,
                "faerie": faerie,
                "lagos": lagos,
            },
        }

    def _schumann_to_signal(self, schumann_data: Any) -> dict[str, Any]:
        """Convert Schumann data to signal value"""
        deviation = schumann_data.deviation

        # Higher deviation = stronger signal
        if schumann_data.alert_level == "HIGH":
            signal_value = 0.8 if deviation > 0 else -0.8
        elif schumann_data.alert_level == "MEDIUM":
            signal_value = 0.6 if deviation > 0 else -0.6
        else:
            signal_value = 0.2 if deviation > 0 else -0.2

        return {
            "value": signal_value,
            "direction": ("BUY" if signal_value > 0 else "SELL" if signal_value < 0 else "HOLD"),
            "strength": abs(signal_value),
        }

    def _fractal_to_signal(self, fractal_data: Any) -> dict[str, Any]:
        """Convert fractal data to signal value"""
        # Higher fractal dimension and resonance = stronger signal
        fractal_factor = (fractal_data.fractal_dimension - 2.0) / 0.5  # Normalize around 2.0
        resonance_factor = fractal_data.resonance_peak
        time_factor = (fractal_data.time_compression - 1.0) / 0.2  # Normalize around 1.0

        signal_value = (fractal_factor + resonance_factor + time_factor) / 3

        return {
            "value": max(-1.0, min(1.0, signal_value)),
            "direction": (
                "BUY" if signal_value > 0.1 else "SELL" if signal_value < -0.1 else "HOLD"
            ),
            "strength": abs(signal_value),
        }

    def _planetary_to_signal(self, planetary_data: Any) -> dict[str, Any]:
        """Convert planetary data to signal value"""
        alignment_factor = planetary_data.alignment_strength
        influence_factor = planetary_data.influence_score

        # Stronger alignment and influence = stronger signal
        signal_value = (alignment_factor + influence_factor) / 2

        return {
            "value": signal_value,
            "direction": (
                "BUY" if signal_value > 0.6 else "SELL" if signal_value < 0.4 else "HOLD"
            ),
            "strength": abs(signal_value - 0.5) * 2,
        }

    def _mystic_signal_to_signal(self, mystic_signal_data: dict[str, Any]) -> dict[str, Any]:
        """Convert mystic signal data to signal value"""
        signal_strength = mystic_signal_data["signal_strength"]

        # Convert 0-1 range to -1 to 1 range
        signal_value = (signal_strength - 0.5) * 2

        return {
            "value": signal_value,
            "direction": (
                "BUY" if signal_value > 0.1 else "SELL" if signal_value < -0.1 else "HOLD"
            ),
            "strength": abs(signal_value),
        }

    def _tesla_to_signal(self, tesla_signal: dict[str, Any]) -> dict[str, Any]:
        """Convert Tesla signal to signal value"""
        direction_map = {"BUY": 1.0, "SELL": -1.0, "HOLD": 0.0}
        signal_value = direction_map[tesla_signal["direction"]] * tesla_signal["strength"]

        return {
            "value": signal_value,
            "direction": tesla_signal["direction"],
            "strength": tesla_signal["strength"],
        }

    def _faerie_to_signal(self, faerie_signal: dict[str, Any]) -> dict[str, Any]:
        """Convert Faerie signal to signal value"""
        direction_map = {"BUY": 1.0, "SELL": -1.0, "HOLD": 0.0}
        signal_value = direction_map[faerie_signal["direction"]] * faerie_signal["strength"]

        return {
            "value": signal_value,
            "direction": faerie_signal["direction"],
            "strength": faerie_signal["strength"],
        }

    def _lagos_to_signal(self, lagos_signal: dict[str, Any]) -> dict[str, Any]:
        """Convert Lagos signal to signal value"""
        direction_map = {"BUY": 1.0, "SELL": -1.0, "HOLD": 0.0}
        signal_value = direction_map[lagos_signal["direction"]] * lagos_signal["strength"]

        return {
            "value": signal_value,
            "direction": lagos_signal["direction"],
            "strength": lagos_signal["strength"],
        }

    def _determine_final_signal(self, signal_components: dict[str, Any]) -> dict[str, Any]:
        """Determine final signal type and strength"""
        final_signal = signal_components["final_signal"]

        # Determine signal type based on strength and direction
        if final_signal > 0.7:
            signal_type = SignalType.STRONG_BUY
        elif final_signal > 0.3:
            signal_type = SignalType.BUY
        elif final_signal < -0.7:
            signal_type = SignalType.STRONG_SELL
        elif final_signal < -0.3:
            signal_type = SignalType.SELL
        else:
            signal_type = SignalType.HOLD

        # Calculate confidence based on component agreement
        components = signal_components["components"]
        buy_signals = sum(1 for c in components.values() if c["direction"] == "BUY")
        sell_signals = sum(1 for c in components.values() if c["direction"] == "SELL")
        total_signals = len(components)

        if buy_signals > sell_signals:
            agreement = buy_signals / total_signals
        elif sell_signals > buy_signals:
            agreement = sell_signals / total_signals
        else:
            agreement = 0.5

        confidence = min(0.95, 0.5 + agreement * 0.45)

        return {
            "type": signal_type,
            "confidence": confidence,
            "strength": abs(final_signal),
        }

    def _generate_reasoning(self, signal_components: dict[str, Any]) -> list[str]:
        """Generate human-readable reasoning for the signal"""
        reasoning = []
        components = signal_components["components"]

        # Add reasoning for each component
        if components["tesla"]["strength"] > 0.6:
            reasoning.append(
                f"Tesla 369 resonance: {components['tesla']['direction']} signal (strength: {components['tesla']['strength']:.2f})"
            )

        if components["faerie"]["strength"] > 0.6:
            reasoning.append(
                f"Faerie Star alignment: {components['faerie']['direction']} signal (strength: {components['faerie']['strength']:.2f})"
            )

        if components["lagos"]["strength"] > 0.6:
            reasoning.append(
                f"Lagos cosmic alignment: {components['lagos']['direction']} signal (strength: {components['lagos']['strength']:.2f})"
            )

        if components["schumann"]["strength"] > 0.5:
            reasoning.append(
                f"Schumann resonance: {components['schumann']['direction']} signal (strength: {components['schumann']['strength']:.2f})"
            )

        if components["planetary"]["strength"] > 0.5:
            reasoning.append(
                f"Planetary alignment: {components['planetary']['direction']} signal (strength: {components['planetary']['strength']:.2f})"
            )

        if not reasoning:
            reasoning.append("All mystic factors indicate neutral conditions")

        return reasoning


# Global instance
mystic_signal_engine = MysticSignalEngine()


