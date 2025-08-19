"""
AI Trading System

Advanced AI-powered trading system with multiple analysis modules
"""

from .ai_brains import coin_gecko_meta, trend_analysis
from .ai_breakouts import breakout_detector
from .ai_mystic import mystic_oracle
from .ai_signals import (
    get_trade_summary,
    get_trading_status,
    market_strength_signals,
    risk_adjusted_signals,
    signal_scorer,
    technical_signals,
)
from .ai_signals import (
    mystic_oracle as signals_mystic_oracle,
)
from .ai_signals import (
    trend_analysis as signals_trend_analysis,
)
from .ai_volume import pump_detector
from .poller import cache, get_cache

__all__ = [
    "cache",
    "get_cache",
    "trend_analysis",
    "coin_gecko_meta",
    "breakout_detector",
    "signal_scorer",
    "risk_adjusted_signals",
    "technical_signals",
    "market_strength_signals",
    "signals_trend_analysis",
    "signals_mystic_oracle",
    "get_trading_status",
    "get_trade_summary",
    "pump_detector",
    "mystic_oracle",
]
