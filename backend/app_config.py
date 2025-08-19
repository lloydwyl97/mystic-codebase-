"""
Application Configuration for Mystic Trading

Centralizes application configuration and setup.
"""

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from starlette.middleware.base import BaseHTTPMiddleware

# Import error handlers
from .error_handlers import register_error_handlers

# Import custom aioredis-based middleware
from .middleware.rate_limiter import rate_limit_middleware

# Get logger
logger = logging.getLogger(__name__)


def configure_app(app: FastAPI):
    """Configure the FastAPI application with middleware and settings."""

    # Add CORS middleware with specific origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )

    # Add custom aioredis-based rate limiting middleware
    app.add_middleware(BaseHTTPMiddleware, dispatch=rate_limit_middleware)

    # Add Prometheus metrics
    Instrumentator().instrument(app).expose(app)

    # Register error handlers
    register_error_handlers(app)

    logger.info("Application configured successfully")

    return app


def get_app_settings():
    """Get application settings from environment variables."""
    return {
        "debug": os.getenv("DEBUG", "false").lower() == "true",
        "environment": os.getenv("ENVIRONMENT", "development"),
        "auto_trading_available": (os.getenv("AUTO_TRADING_AVAILABLE", "true").lower() == "true"),
        "api_host": os.getenv("API_HOST", "localhost"),
        "api_port": int(os.getenv("PORT", "8000")),
    }


