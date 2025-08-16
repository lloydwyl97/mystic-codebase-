"""
Mystic Configuration Module
Handles environment variables for mystic integrations, Schumann resonance,
fractal time, and other esoteric trading features.
"""

import os
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class SchumannConfig:
    """Schumann resonance configuration"""

    api_url: str
    base_frequency: float
    alert_threshold: float


@dataclass
class MLConfig:
    """Machine learning configuration"""

    model_url: str
    prediction_threshold: float


@dataclass
class MysticIntegrationConfig:
    """Mystic integration services configuration"""

    tesla_frequency_url: str
    faerie_star_url: str
    lagos_alignment_url: str
    planetary_alignment_url: str
    moon_phase_url: str
    jupiter_venus_url: str
    fractal_time_url: str
    human_resonance_url: str


@dataclass
class RitualEngineConfig:
    """Ritual engine configuration"""

    enabled: bool
    socket_refresh_interval: int


class ExchangeConfig:
    """Exchange configuration for Binance US"""

    def __init__(self):
        self.binance_us_api_key = os.getenv("BINANCE_US_API_KEY", "")
        self.binance_us_secret_key = os.getenv("BINANCE_US_SECRET_KEY", "")


class MysticConfig:
    """Main mystic configuration class"""

    def __init__(self):
        # Exchange Configuration
        self.exchange = ExchangeConfig()

        # Schumann / Fractal / Resonance Services
        self.schumann = SchumannConfig(
            api_url=os.getenv("SCHUMANN_API_URL", "https://api.schumann-resonance.org/data"),
            base_frequency=float(os.getenv("SCHUMANN_BASE_FREQUENCY", "7.83")),
            alert_threshold=float(os.getenv("SCHUMANN_ALERT_THRESHOLD", "15.0")),
        )

        # Machine Learning Model & Prediction
        self.ml = MLConfig(
            model_url=os.getenv("ML_MODEL_URL", "https://ml.mystictrading.com/model"),
            prediction_threshold=float(os.getenv("PREDICTION_THRESHOLD", "0.8")),
        )

        # Realtime Data
        self.realtime_data_url = os.getenv(
            "REALTIME_DATA_URL", "wss://datafeed.mystictrading.com/realtime"
        )

        # Ritual Engine
        self.ritual_engine = RitualEngineConfig(
            enabled=os.getenv("RITUAL_ENGINE_ENABLED", "true").lower() == "true",
            socket_refresh_interval=int(os.getenv("SOCKET_REFRESH_INTERVAL", "5")),
        )

        # Sentry DSN
        self.sentry_dsn = os.getenv("SENTRY_DSN")

        # Debug mode
        self.debug = os.getenv("DEBUG", "False").lower() == "true"

        # Mystic Integration Services
        self.mystic_integrations = MysticIntegrationConfig(
            tesla_frequency_url=os.getenv(
                "TESLA_FREQUENCY_ENGINE_URL",
                "https://api.teslafrequency.org/v1",
            ),
            faerie_star_url=os.getenv("FAERIE_STAR_ENGINE_URL", "https://api.faeriestar.org/v1"),
            lagos_alignment_url=os.getenv(
                "LAGOS_ALIGNMENT_URL", "https://api.lagosalignment.org/v1"
            ),
            planetary_alignment_url=os.getenv(
                "PLANETARY_ALIGNMENT_URL",
                "https://api.planetaryalignments.org/v1",
            ),
            moon_phase_url=os.getenv("MOON_PHASE_URL", "https://api.moonphase.org/v1"),
            jupiter_venus_url=os.getenv(
                "JUPITER_VENUS_URL",
                "https://api.astrodata.io/v1/jupiter-venus",
            ),
            fractal_time_url=os.getenv("FRACTAL_TIME_URL", "https://api.fractaltime.org/v1"),
            human_resonance_url=os.getenv(
                "HUMAN_RESONANCE_URL", "https://api.humanresonance.org/v1"
            ),
        )

    def is_schumann_alert(self, frequency: float) -> bool:
        """Check if Schumann frequency is above alert threshold"""
        return bool(frequency > self.schumann.alert_threshold)

    def get_schumann_deviation(self, frequency: float) -> float:
        """Calculate deviation from base Schumann frequency"""
        return float(abs(frequency - self.schumann.base_frequency))

    def is_prediction_significant(self, confidence: float) -> bool:
        """Check if ML prediction confidence is above threshold"""
        return bool(confidence >= self.ml.prediction_threshold)

    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of all mystic configuration"""
        return {
            "schumann": {
                "api_url": self.schumann.api_url,
                "base_frequency": self.schumann.base_frequency,
                "alert_threshold": self.schumann.alert_threshold,
            },
            "ml": {
                "model_url": self.ml.model_url,
                "prediction_threshold": self.ml.prediction_threshold,
            },
            "realtime_data_url": self.realtime_data_url,
            "ritual_engine": {
                "enabled": self.ritual_engine.enabled,
                "socket_refresh_interval": (self.ritual_engine.socket_refresh_interval),
            },
            "sentry_dsn": self.sentry_dsn is not None,
            "debug": self.debug,
            "mystic_integrations": {
                "tesla_frequency": (self.mystic_integrations.tesla_frequency_url),
                "faerie_star": self.mystic_integrations.faerie_star_url,
                "lagos_alignment": (self.mystic_integrations.lagos_alignment_url),
                "planetary_alignment": (self.mystic_integrations.planetary_alignment_url),
                "moon_phase": self.mystic_integrations.moon_phase_url,
                "jupiter_venus": self.mystic_integrations.jupiter_venus_url,
                "fractal_time": self.mystic_integrations.fractal_time_url,
                "human_resonance": (self.mystic_integrations.human_resonance_url),
            },
        }


# Global instance
mystic_config = MysticConfig()


