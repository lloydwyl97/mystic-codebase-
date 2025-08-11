import logging
import uuid
from typing import Awaitable, Callable

from fastapi import Request
from fastapi.responses import JSONResponse, Response

from enhanced_logging import log_operation_performance

logger = logging.getLogger(__name__)


@log_operation_performance("http_request_handling")
async def health_monitor_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """
    Middleware to monitor request health and performance.
    """
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

    try:
        response = await call_next(request)

        # Add request ID header
        response.headers["X-Request-ID"] = request_id

        return response

    except Exception as e:
        logger.error(f"Request failed: {str(e)} [ID: {request_id}]")
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error",
                "request_id": request_id,
            },
        )
