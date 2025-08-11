"""
Utils Module for Mystic Trading Platform

Contains utility functions and classes for the application.
"""

from .exceptions import (
    AIException,
    APIException,
    AuthenticationException,
    DatabaseException,
    ErrorCode,
    MarketDataException,
    MysticException,
    RateLimitException,
    TradingException,
    create_http_exception_handler,
    handle_async_exception,
    handle_exception,
    safe_async_execute,
    safe_execute,
)

__all__ = [
    "MysticException",
    "DatabaseException",
    "APIException",
    "TradingException",
    "MarketDataException",
    "AIException",
    "AuthenticationException",
    "RateLimitException",
    "ErrorCode",
    "handle_exception",
    "handle_async_exception",
    "create_http_exception_handler",
    "safe_execute",
    "safe_async_execute",
]
