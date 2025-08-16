"""
Response Sanitizer Middleware

FastAPI middleware for response sanitization.
"""

import json
import logging
from typing import Any, Awaitable, Callable, Dict, Union

from fastapi import Request, Response
from fastapi.responses import JSONResponse

from enhanced_logging import log_operation_performance

logger = logging.getLogger(__name__)

# Sensitive fields to remove from responses
SENSITIVE_FIELDS = ["password", "token", "secret", "key", "api_key"]


@log_operation_performance("response_sanitizer")
async def response_sanitizer_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Union[Response, JSONResponse]:
    """
    Middleware to sanitize response data.
    """
    # Process request
    response = await call_next(request)

    # Only sanitize JSON responses
    if response.headers.get("content-type", "").startswith("application/json"):
        try:
            # Get response body
            if hasattr(response, "body") and response.body:
                if isinstance(response.body, bytes):
                    data_str = response.body.decode("utf-8")
                else:
                    data_str = str(response.body)

                # Parse JSON
                data = json.loads(data_str)

                # Sanitize sensitive data
                sanitized_data = _sanitize_data(data)

                # Create new response
                return JSONResponse(
                    content=sanitized_data,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                )
        except Exception as e:
            logger.error(f"Response sanitizer error: {str(e)}")
            # Return original response if sanitization fails
            return response

    return response


def _sanitize_data(data: Any) -> Any:
    """Recursively sanitize data by removing sensitive fields."""
    if isinstance(data, dict):
        sanitized: Dict[str, Any] = {}
        for key, value in data.items():
            key_str = str(key)
            if isinstance(key, str) and key_str.lower() in SENSITIVE_FIELDS:
                sanitized[key_str] = "***REDACTED***"
            else:
                sanitized[key_str] = _sanitize_data(value)
        return sanitized
    elif isinstance(data, list):
        return [_sanitize_data(item) for item in data]
    else:
        return data


