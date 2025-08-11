"""
AI Package for Mystic Trading Platform

This package contains AI-related modules and utilities.
"""

from .ai.ai_signals import (
    signal_scorer,
    risk_adjusted_signals,
    technical_signals,
    market_strength_signals,
    trend_analysis,
    mystic_oracle,
    get_trading_status,
    get_trade_summary,
)

__version__ = "1.0.0"
__author__ = "Mystic Trading Team"

__all__ = [
    "signal_scorer",
    "risk_adjusted_signals",
    "technical_signals",
    "market_strength_signals",
    "trend_analysis",
    "mystic_oracle",
    "get_trading_status",
    "get_trade_summary",
]
