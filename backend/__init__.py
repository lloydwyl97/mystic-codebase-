"""
Mystic Trading Platform - Backend Package

Advanced cryptocurrency trading platform with AI-powered strategies,
real-time market data, automated trading, and comprehensive analytics.

This package contains the complete backend implementation including:
- AI trading strategies and signal generation
- Real-time market data processing
- Automated trading execution
- Portfolio management and analytics
- WebSocket communication
- Database management and optimization
- Security and authentication
- Performance monitoring and logging
"""

__version__ = "2.0.0"
__author__ = "Mystic Trading Team"
__description__ = "Advanced AI-powered cryptocurrency trading platform"
__license__ = "MIT"

# Core imports for easy access
try:
    from .main import app
    from .config import Config
    from .database import get_db_context
    from .utils.exceptions import MysticException, handle_exception
except ImportError:
    # Allow partial imports during development
    pass

__all__ = [
    "app",
    "Config",
    "get_db_context",
    "MysticException",
    "handle_exception",
    "__version__",
    "__author__",
    "__description__",
]
