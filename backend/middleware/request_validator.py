"""
Request Validator Middleware

FastAPI middleware for request validation.
"""

import logging
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse

from enhanced_logging import log_operation_performance

logger = logging.getLogger(__name__)

# Validation rules
REQUIRED_HEADERS = ["user-agent"]
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB
ALLOWED_METHODS = ["GET", "POST", "PUT", "DELETE", "PATCH"]


@log_operation_performance("request_validator")
async def request_validator_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response | JSONResponse:
    """
    Middleware to validate incoming requests.
    """
    # Validate HTTP method
    if request.method not in ALLOWED_METHODS:
        logger.warning(f"Invalid HTTP method: {request.method}")
        return JSONResponse(
            status_code=405,
            content={"detail": f"Method {request.method} not allowed"},
        )

    # Validate required headers
    for header in REQUIRED_HEADERS:
        if header not in request.headers:
            logger.warning(f"Missing required header: {header}")
            return JSONResponse(
                status_code=400,
                content={"detail": f"Missing required header: {header}"},
            )

    # Validate content length
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_CONTENT_LENGTH:
        logger.warning(f"Content too large: {content_length} bytes")
        return JSONResponse(status_code=413, content={"detail": "Request entity too large"})

    # Validate URL length
    if len(str(request.url)) > 2048:
        logger.warning("URL too long")
        return JSONResponse(status_code=414, content={"detail": "Request URI too long"})

    # Process request
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        logger.error(f"Request validator error: {str(e)}")
        return await call_next(request)


