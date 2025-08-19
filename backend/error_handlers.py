"""
Error Handlers for Mystic Trading

Centralized error handling for the application using standardized exceptions.
"""

import logging
import os
import traceback
from typing import Any, cast

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded

from backend.utils.exceptions import (
    ErrorCode,
    MysticException,
    _get_status_code_for_error_code,
)

# Get logger
logger = logging.getLogger(__name__)


async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    """Handle rate limit exceeded errors from SlowAPI."""
    # Get retry_after from exception detail if available
    default_retry_after = 60  # Default value
    retry_after: int = default_retry_after

    # Check if exc has detail attribute and if it's a dict with retry_after
    if hasattr(exc, "detail"):
        # Handle the case where detail might be a custom object
        if isinstance(exc.detail, dict):
            # Try to get retry_after using get() method which is safer
            # Provide a default value of None to help with type inference
            detail_dict: dict[str, Any] = cast(dict[str, Any], exc.detail)
            retry_after_value: Any = detail_dict.get("retry_after")
            if retry_after_value is not None:
                retry_after = retry_after_value
        elif hasattr(exc.detail, "retry_after"):
            # If detail is an object with retry_after attribute
            retry_after_value: Any = getattr(exc.detail, "retry_after", None)
            if retry_after_value is not None:
                retry_after = retry_after_value

    # Check if exc itself has retry_after attribute
    if hasattr(exc, "retry_after"):
        retry_after_value: Any = getattr(exc, "retry_after", None)
        if retry_after_value is not None:
            retry_after = retry_after_value

    # Create standardized response
    response_data = {
        "error": True,
        "error_code": ErrorCode.RATE_LIMIT_EXCEEDED.value,
        "error_type": "RateLimitExceeded",
        "message": "Rate limit exceeded",
        "details": {"retry_after": retry_after, "path": str(request.url)},
        "timestamp": traceback.format_exc(),
    }

    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content=response_data,
        headers={"Retry-After": str(retry_after)},
    )


async def custom_rate_limit_handler(request: Request, exc: HTTPException):
    """Handle custom rate limit exceptions (429 errors)."""
    if exc.status_code == 429:
        # Extract retry_after from exception detail if available
        retry_after = 60  # Default value

        if hasattr(exc, "detail") and isinstance(exc.detail, dict):
            retry_after = exc.detail.get("retry_after", retry_after)

        response_data = {
            "error": True,
            "error_code": ErrorCode.RATE_LIMIT_EXCEEDED.value,
            "error_type": "RateLimitExceeded",
            "message": "Rate limit exceeded",
            "details": {"retry_after": retry_after, "path": str(request.url)},
            "timestamp": traceback.format_exc(),
        }

        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content=response_data,
            headers={"Retry-After": str(retry_after)},
        )

    # For other HTTPExceptions, use the standard handler
    return await generic_exception_handler(request, exc)


async def generic_exception_handler(request: Request, exc: Exception):
    """Handle generic exceptions using standardized format."""
    # Handle MysticExceptions
    if isinstance(exc, MysticException):
        status_code = _get_status_code_for_error_code(exc.error_code)
        return JSONResponse(status_code=status_code, content=exc.to_dict())

    # Handle HTTPExceptions
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": True,
                "error_code": ErrorCode.UNKNOWN_ERROR.value,
                "error_type": "HTTPException",
                "message": exc.detail,
                "details": {"path": str(request.url)},
                "timestamp": traceback.format_exc(),
            },
        )

    # Handle other exceptions
    logger.error(f"Unhandled exception: {str(exc)}\n{traceback.format_exc()}")

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": True,
            "error_code": ErrorCode.UNKNOWN_ERROR.value,
            "error_type": "InternalServerError",
            "message": (
                str(exc) if os.getenv("DEBUG") == "true" else "An unexpected error occurred"
            ),
            "details": {"path": str(request.url)},
            "timestamp": traceback.format_exc(),
        },
    )


def register_error_handlers(app: FastAPI):
    """Register all error handlers with the FastAPI application."""
    # Register SlowAPI rate limit handler
    app.exception_handler(RateLimitExceeded)(rate_limit_handler)

    # Register custom rate limit handler for 429 errors from our middleware
    app.exception_handler(HTTPException)(custom_rate_limit_handler)

    # Register generic exception handler
    app.exception_handler(Exception)(generic_exception_handler)


