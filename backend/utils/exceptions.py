"""
Standardized Exception Handling for Mystic Trading Platform

Provides consistent exception handling across the entire application.
"""

import logging
import traceback
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, Optional, Type, Union

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class ErrorCode(Enum):
    """Standardized error codes for the application"""

    # General errors (1000-1999)
    UNKNOWN_ERROR = 1000
    VALIDATION_ERROR = 1001
    CONFIGURATION_ERROR = 1002
    TIMEOUT_ERROR = 1003

    # Database errors (2000-2999)
    DATABASE_CONNECTION_ERROR = 2000
    DATABASE_QUERY_ERROR = 2001
    DATABASE_TRANSACTION_ERROR = 2002

    # External API errors (3000-3999)
    API_CONNECTION_ERROR = 3000
    API_RATE_LIMIT_ERROR = 3001
    API_AUTHENTICATION_ERROR = 3002
    API_TIMEOUT_ERROR = 3003
    API_RESPONSE_ERROR = 3004

    # Trading errors (4000-4999)
    TRADING_ORDER_ERROR = 4000
    TRADING_BALANCE_ERROR = 4001
    TRADING_SYMBOL_ERROR = 4002
    TRADING_EXCHANGE_ERROR = 4003

    # Market data errors (5000-5999)
    MARKET_DATA_ERROR = 5000
    MARKET_DATA_FETCH_ERROR = 5001
    MARKET_DATA_PARSING_ERROR = 5002

    # AI/ML errors (6000-6999)
    AI_MODEL_ERROR = 6000
    AI_PREDICTION_ERROR = 6001
    AI_TRAINING_ERROR = 6002

    # Authentication/Authorization errors (7000-7999)
    AUTHENTICATION_ERROR = 7000
    AUTHORIZATION_ERROR = 7001
    TOKEN_ERROR = 7002

    # Rate limiting errors (8000-8999)
    RATE_LIMIT_ERROR = 8000
    RATE_LIMIT_EXCEEDED = 8001


class MysticException(Exception):
    """Base exception class for Mystic Trading Platform"""

    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.original_exception = original_exception
        self.timestamp = datetime.now(timezone.timezone.utc)

        # Log the exception
        self._log_exception()

        super().__init__(self.message)

    def _log_exception(self):
        """Log the exception with structured information"""
        log_data = {
            "error_code": self.error_code.value,
            "error_type": self.error_code.name,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }

        if self.original_exception:
            log_data["original_exception"] = {
                "type": type(self.original_exception).__name__,
                "message": str(self.original_exception),
            }

        logger.error(f"MysticException: {log_data}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses"""
        return {
            "error": True,
            "error_code": self.error_code.value,
            "error_type": self.error_code.name,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }


class DatabaseException(MysticException):
    """Database-related exceptions"""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
    ):
        super().__init__(
            message,
            ErrorCode.DATABASE_QUERY_ERROR,
            details,
            original_exception,
        )


class DatabaseConnectionException(MysticException):
    """Database connection-related exceptions"""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
    ):
        super().__init__(
            message,
            ErrorCode.DATABASE_CONNECTION_ERROR,
            details,
            original_exception,
        )


class APIException(MysticException):
    """External API-related exceptions"""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
    ):
        super().__init__(
            message,
            ErrorCode.API_CONNECTION_ERROR,
            details,
            original_exception,
        )


class TradingException(MysticException):
    """Trading-related exceptions"""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
    ):
        super().__init__(message, ErrorCode.TRADING_ORDER_ERROR, details, original_exception)


class MarketDataException(MysticException):
    """Market data-related exceptions"""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
    ):
        super().__init__(message, ErrorCode.MARKET_DATA_ERROR, details, original_exception)


class AIException(MysticException):
    """AI/ML-related exceptions"""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
    ):
        super().__init__(message, ErrorCode.AI_MODEL_ERROR, details, original_exception)


class AnalyticsException(MysticException):
    """Analytics-related exceptions"""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
    ):
        super().__init__(message, ErrorCode.AI_MODEL_ERROR, details, original_exception)


class MetricsException(MysticException):
    """Metrics-related exceptions"""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
    ):
        super().__init__(message, ErrorCode.AI_MODEL_ERROR, details, original_exception)


class AuthenticationException(MysticException):
    """Authentication-related exceptions"""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
    ):
        super().__init__(
            message,
            ErrorCode.AUTHENTICATION_ERROR,
            details,
            original_exception,
        )


class RateLimitException(MysticException):
    """Rate limiting exceptions"""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
    ):
        super().__init__(message, ErrorCode.RATE_LIMIT_ERROR, details, original_exception)


class ModelException(MysticException):
    """Model-related exceptions"""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
    ):
        super().__init__(message, ErrorCode.AI_MODEL_ERROR, details, original_exception)


class NotificationException(MysticException):
    """Notification-related exceptions"""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
    ):
        super().__init__(message, ErrorCode.UNKNOWN_ERROR, details, original_exception)


class StrategyException(MysticException):
    """Strategy-related exceptions"""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
    ):
        super().__init__(message, ErrorCode.TRADING_ORDER_ERROR, details, original_exception)





def handle_exception(
    error_message: str,
    exception_class: Type[MysticException] = MysticException,
    error_code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
    reraise: bool = True,
    default_return: Any = None,
):
    """
    Decorator for standardized exception handling

    Args:
        error_message: Base error message
        exception_class: Exception class to raise
        error_code: Error code to use
        reraise: Whether to reraise the exception
        default_return: Default return value if exception occurs and reraise=False
    """

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except MysticException:
                # Re-raise MysticExceptions as-is
                raise
            except Exception as e:
                # Wrap other exceptions
                if exception_class == MysticException:
                    # Base class accepts error_code
                    mystic_exception = exception_class(
                        message=error_message,
                        error_code=error_code,
                        details={"function": func.__name__},
                        original_exception=e,
                    )
                else:
                    # Specialized classes don't accept error_code
                    mystic_exception = exception_class(
                        message=error_message,
                        details={"function": func.__name__},
                        original_exception=e,
                    )

                if reraise:
                    raise mystic_exception
                else:
                    logger.error(f"Exception in {func.__name__}: {mystic_exception}")
                    return default_return

        return wrapper

    return decorator


def handle_async_exception(
    error_message: str,
    exception_class: Type[MysticException] = MysticException,
    error_code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
    reraise: bool = True,
    default_return: Any = None,
):
    """
    Decorator for standardized async exception handling
    """

    def decorator(func: Callable) -> Callable:
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except MysticException:
                # Re-raise MysticExceptions as-is
                raise
            except Exception as e:
                # Wrap other exceptions
                if exception_class == MysticException:
                    # Base class accepts error_code
                    mystic_exception = exception_class(
                        message=error_message,
                        error_code=error_code,
                        details={"function": func.__name__},
                        original_exception=e,
                    )
                else:
                    # Specialized classes don't accept error_code
                    mystic_exception = exception_class(
                        message=error_message,
                        details={"function": func.__name__},
                        original_exception=e,
                    )

                if reraise:
                    raise mystic_exception
                else:
                    logger.error(f"Exception in {func.__name__}: {mystic_exception}")
                    return default_return

        return wrapper

    return decorator


def create_http_exception_handler():
    """Create standardized HTTP exception handler for FastAPI"""

    async def http_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Handle exceptions and return standardized HTTP responses"""

        # Handle MysticExceptions
        if isinstance(exc, MysticException):
            status_code = _get_status_code_for_error_code(exc.error_code)
            return JSONResponse(status_code=status_code, content=exc.to_dict())

        # Handle FastAPI HTTPExceptions
        if isinstance(exc, HTTPException):
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "error": True,
                    "error_code": ErrorCode.UNKNOWN_ERROR.value,
                    "error_type": "HTTPException",
                    "message": exc.detail,
                    "timestamp": (datetime.now(timezone.timezone.utc).isoformat()),
                },
            )

        # Handle other exceptions
        logger.error(f"Unhandled exception: {exc}\n{traceback.format_exc()}")

        return JSONResponse(
            status_code=500,
            content={
                "error": True,
                "error_code": ErrorCode.UNKNOWN_ERROR.value,
                "error_type": "InternalServerError",
                "message": "An unexpected error occurred",
                "timestamp": datetime.now(timezone.timezone.utc).isoformat(),
            },
        )

    return http_exception_handler


def _get_status_code_for_error_code(error_code: ErrorCode) -> int:
    """Map error codes to HTTP status codes"""
    status_code_map = {
        # General errors
        ErrorCode.UNKNOWN_ERROR: 500,
        ErrorCode.VALIDATION_ERROR: 400,
        ErrorCode.CONFIGURATION_ERROR: 500,
        ErrorCode.TIMEOUT_ERROR: 408,
        # Database errors
        ErrorCode.DATABASE_CONNECTION_ERROR: 503,
        ErrorCode.DATABASE_QUERY_ERROR: 500,
        ErrorCode.DATABASE_TRANSACTION_ERROR: 500,
        # External API errors
        ErrorCode.API_CONNECTION_ERROR: 503,
        ErrorCode.API_RATE_LIMIT_ERROR: 429,
        ErrorCode.API_AUTHENTICATION_ERROR: 401,
        ErrorCode.API_TIMEOUT_ERROR: 408,
        ErrorCode.API_RESPONSE_ERROR: 502,
        # Trading errors
        ErrorCode.TRADING_ORDER_ERROR: 500,
        ErrorCode.TRADING_BALANCE_ERROR: 400,
        ErrorCode.TRADING_SYMBOL_ERROR: 400,
        ErrorCode.TRADING_EXCHANGE_ERROR: 503,
        # Market data errors
        ErrorCode.MARKET_DATA_ERROR: 503,
        ErrorCode.MARKET_DATA_FETCH_ERROR: 503,
        ErrorCode.MARKET_DATA_PARSING_ERROR: 500,
        # AI/ML errors
        ErrorCode.AI_MODEL_ERROR: 500,
        ErrorCode.AI_PREDICTION_ERROR: 500,
        ErrorCode.AI_TRAINING_ERROR: 500,
        # Authentication/Authorization errors
        ErrorCode.AUTHENTICATION_ERROR: 401,
        ErrorCode.AUTHORIZATION_ERROR: 403,
        ErrorCode.TOKEN_ERROR: 401,
        # Rate limiting errors
        ErrorCode.RATE_LIMIT_ERROR: 429,
        ErrorCode.RATE_LIMIT_EXCEEDED: 429,
    }

    return status_code_map.get(error_code, 500)


def safe_execute(func: Callable, *args, **kwargs) -> Union[Any, MysticException]:
    """
    Safely execute a function and return result or exception

    Args:
        func: Function to execute
        *args: Function arguments
        **kwargs: Function keyword arguments

    Returns:
        Function result or MysticException if error occurs
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        return MysticException(
            message=f"Error executing {func.__name__}",
            error_code=ErrorCode.UNKNOWN_ERROR,
            details={"function": func.__name__},
            original_exception=e,
        )


async def safe_async_execute(func: Callable, *args, **kwargs) -> Union[Any, MysticException]:
    """
    Safely execute an async function and return result or exception
    """
    try:
        return await func(*args, **kwargs)
    except Exception as e:
        return MysticException(
            message=f"Error executing {func.__name__}",
            error_code=ErrorCode.UNKNOWN_ERROR,
            details={"function": func.__name__},
            original_exception=e,
        )


