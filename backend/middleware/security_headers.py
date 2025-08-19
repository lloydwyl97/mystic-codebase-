"""
Security Headers Middleware

FastAPI middleware for adding security headers.
"""

import logging
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse

from enhanced_logging import log_operation_performance

logger = logging.getLogger(__name__)


@log_operation_performance("security_headers")
async def security_headers_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response | JSONResponse:
    """
    Middleware to add security headers to responses.
    """
    # Process request
    response = await call_next(request)

    # Skip CSP for docs endpoints to allow Swagger UI
    is_docs_endpoint = request.url.path in ["/docs", "/redoc", "/openapi.json"]

    # Add security headers (but preserve CORS headers)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

    # Content Security Policy - Skip for docs endpoints
    if not is_docs_endpoint:
        csp_policy = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "img-src 'self' data: https: https://fastapi.tiangolo.com; "
            "font-src 'self' data:; "
            "connect-src 'self' https:; "
            "frame-ancestors 'none'; "
            "form-action 'self'"
        )
        response.headers["Content-Security-Policy"] = csp_policy

    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"

    # Ensure CORS headers are present (add if missing)
    if "access-control-allow-origin" not in response.headers:
        response.headers["Access-Control-Allow-Origin"] = "*"
    if "access-control-allow-credentials" not in response.headers:
        response.headers["Access-Control-Allow-Credentials"] = "true"
    if "access-control-allow-methods" not in response.headers:
        response.headers["Access-Control-Allow-Methods"] = "*"
    if "access-control-allow-headers" not in response.headers:
        response.headers["Access-Control-Allow-Headers"] = "*"

    return response


