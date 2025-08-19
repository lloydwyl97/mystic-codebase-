import logging
from collections.abc import Awaitable, Callable
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ErrorCode(Enum):
    UNKNOWN_ERROR = 1000
    AI_MODEL_ERROR = 6000


class MysticException(Exception):
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
        details: dict[str, Any] | None = None,
        original_exception: Exception | None = None,
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.original_exception = original_exception
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": True,
            "error_code": self.error_code.value,
            "error_type": self.error_code.name,
            "message": self.message,
            "details": self.details,
        }


class ModelException(Exception):
    pass


class NotificationException(Exception):
    pass


class StrategyException(Exception):
    pass


class TradingException(Exception):
    pass


class AIException(Exception):
    pass


class MarketDataException(Exception):
    pass


class DatabaseConnectionException(Exception):
    pass


class DatabaseException(Exception):
    """General database-related exceptions"""

    pass


class AnalyticsException(MysticException):
    """Analytics-related exceptions"""

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
        original_exception: Exception | None = None,
    ):
        super().__init__(message, ErrorCode.AI_MODEL_ERROR, details, original_exception)


class MetricsException(MysticException):
    """Metrics-related exceptions"""

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
        original_exception: Exception | None = None,
    ):
        super().__init__(message, ErrorCode.AI_MODEL_ERROR, details, original_exception)


class RateLimitException(Exception):
    """Exception raised when API rate limit is exceeded."""

    pass


def handle_async_exception(
    error_message: str,
    exception_class: type[Exception] = Exception,
    reraise: bool = True,
    default_return: Any = None,
) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
    """Decorator to handle exceptions in async functions."""

    def decorator(
        func: Callable[..., Awaitable[Any]],
    ) -> Callable[..., Awaitable[Any]]:
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except exception_class as e:
                logger.error(f"{error_message}: {str(e)}")
                if reraise:
                    raise
                return default_return

        return wrapper

    return decorator


def handle_exception(
    error_message: str,
    exception_class: type[Exception] = Exception,
    reraise: bool = True,
    default_return: Any = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to handle exceptions in synchronous functions."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except exception_class as e:
                logger.error(f"{error_message}: {str(e)}")
                if reraise:
                    raise
                return default_return

        return wrapper

    return decorator


def _get_status_code_for_error_code(error_code: int | ErrorCode) -> int:
    """Map error codes to HTTP status codes for error handling."""
    # Default mapping, can be extended as needed
    code_map = {
        1000: 500,  # UNKNOWN_ERROR -> Internal Server Error
        6000: 500,  # AI_MODEL_ERROR -> Internal Server Error
        400: 400,  # Bad Request
        401: 401,  # Unauthorized
        403: 403,  # Forbidden
        404: 404,  # Not Found
        409: 409,  # Conflict
        422: 422,  # Unprocessable Entity
        429: 429,  # Too Many Requests
        500: 500,  # Internal Server Error
    }
    if isinstance(error_code, ErrorCode):
        return code_map.get(error_code.value, 500)
    return code_map.get(int(error_code), 500)
