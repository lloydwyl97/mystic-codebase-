"""
Request Logger Middleware

FastAPI middleware for logging requests.
"""

import logging
import time
from typing import Awaitable, Callable, Union

from fastapi import Request, Response
from fastapi.responses import JSONResponse

from enhanced_logging import log_operation_performance

logger = logging.getLogger(__name__)


@log_operation_performance("request_logger")
async def request_logger_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Union[Response, JSONResponse]:
    """
    Middleware to log incoming requests and responses.
    """
    start_time = time.time()

    # Log request
    client_ip = request.client.host if request.client else "unknown"
    logger.info(f"Request: {request.method} {request.url.path} from {client_ip}")

    # Process request
    try:
        response = await call_next(request)

        # Calculate duration
        duration = time.time() - start_time

        # Log response
        logger.info(
            f"Response: {response.status_code} for {request.method} {request.url.path} "
            f"({duration:.3f}s)"
        )

        return response

    except Exception as e:
        # Calculate duration
        duration = time.time() - start_time

        # Log error with specific exception details
        logger.error(
            f"Request logger caught exception for {request.method} {request.url.path}: {str(e)} "
            f"({duration:.3f}s)"
        )

        # Return error response
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})


