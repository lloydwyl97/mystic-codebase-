import logging
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse

from enhanced_logging import log_operation_performance

logger = logging.getLogger(__name__)

# Circuit breaker state
circuit_states: dict[str, dict[str, int | float | bool]] = defaultdict(
    lambda: {"failures": 0, "last_failure": 0.0, "is_open": False}
)
FAILURE_THRESHOLD = 5
RESET_TIMEOUT = 60  # seconds


@log_operation_performance("circuit_breaker")
async def circuit_breaker_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response | JSONResponse:
    """
    Middleware to implement circuit breaker pattern.
    """
    endpoint = request.url.path
    current_time = time.time()

    # Check if circuit is open
    if circuit_states[endpoint]["is_open"]:
        # Check if reset timeout has passed
        if current_time - circuit_states[endpoint]["last_failure"] > RESET_TIMEOUT:
            circuit_states[endpoint]["is_open"] = False
            circuit_states[endpoint]["failures"] = 0
        else:
            logger.warning(f"Circuit breaker open for endpoint: {endpoint}")
            return JSONResponse(
                status_code=503,
                content={"detail": "Service temporarily unavailable"},
            )

    try:
        response = await call_next(request)

        # Reset failure count on success
        if response.status_code < 500:
            circuit_states[endpoint]["failures"] = 0

        return response

    except Exception as e:
        # Log the specific exception for debugging
        logger.error(f"Circuit breaker caught exception for endpoint {endpoint}: {str(e)}")
        
        # Increment failure count
        circuit_states[endpoint]["failures"] += 1
        circuit_states[endpoint]["last_failure"] = current_time

        # Check if threshold exceeded
        if circuit_states[endpoint]["failures"] >= FAILURE_THRESHOLD:
            circuit_states[endpoint]["is_open"] = True
            logger.error(f"Circuit breaker opened for endpoint: {endpoint}")

        return JSONResponse(status_code=500, content={"detail": "Internal server error"})


