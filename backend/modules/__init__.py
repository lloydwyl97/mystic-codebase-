"""
Mystic Trading Platform - Modular Backend Structure

This package contains modularized components of the Mystic Trading Platform backend.
Each module is organized by functionality to improve maintainability and reduce code duplication.
"""

# Import specific modules instead of wildcard imports
from . import ai
from . import api
from . import data
from . import metrics
from . import notifications
from . import signals
from . import strategy
from . import trading

__all__ = [
    "api",
    "trading",
    "data",
    "ai",
    "strategy",
    "notifications",
    "metrics",
    "signals",
]

__version__ = "1.0.0"
__author__ = "Mystic Trading Team"
