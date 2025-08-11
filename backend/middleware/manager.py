"""
Middleware Manager for Mystic Trading

Provides centralized middleware registration and configuration.
"""

import logging
from typing import Any, Dict, Optional

from fastapi import FastAPI

from .cache import cache_middleware_handler
from .circuit_breaker import circuit_breaker_middleware
from .rate_limiter import rate_limit_middleware
from .request_logger import request_logger_middleware
from .request_validator import request_validator_middleware
from .response_sanitizer import response_sanitizer_middleware
from .security_headers import security_headers_middleware

logger = logging.getLogger(__name__)


class MiddlewareManager:
    """Manages middleware registration and configuration"""

    def __init__(self):
        self.middleware_configs: Dict[str, Dict[str, Any]] = {
            "rate_limiter": {"enabled": True, "config": {}},
            "request_logger": {"enabled": True, "config": {}},
            "security_headers": {"enabled": True, "config": {}},
            "cache": {"enabled": True, "config": {}},
            "circuit_breaker": {"enabled": True, "config": {}},
            "request_validator": {"enabled": True, "config": {}},
            "response_sanitizer": {"enabled": True, "config": {}},
        }

    def configure(
        self,
        middleware_name: str,
        enabled: bool = True,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Configure a specific middleware"""
        if middleware_name in self.middleware_configs:
            self.middleware_configs[middleware_name]["enabled"] = enabled
            if config:
                self.middleware_configs[middleware_name]["config"] = config
            logger.info(f"Configured {middleware_name} middleware: enabled={enabled}")
        else:
            logger.warning(f"Unknown middleware: {middleware_name}")

    def register_all(self, app: FastAPI) -> None:
        """Register all enabled middleware with the FastAPI app"""
        logger.info("Registering middleware...")

        # 1. Request logger (should be first to log all requests)
        if self.middleware_configs["request_logger"]["enabled"]:
            app.middleware("http")(request_logger_middleware)
            logger.info("Request logger middleware registered")

        # 2. Security headers
        if self.middleware_configs["security_headers"]["enabled"]:
            app.middleware("http")(security_headers_middleware)
            logger.info("Security headers middleware registered")

        # 3. Rate limiter
        if self.middleware_configs["rate_limiter"]["enabled"]:
            app.middleware("http")(rate_limit_middleware)
            logger.info("Rate limiter middleware registered")

        # 4. Circuit breaker
        if self.middleware_configs["circuit_breaker"]["enabled"]:
            app.middleware("http")(circuit_breaker_middleware)
            logger.info("Circuit breaker middleware registered")

        # 5. Request validator
        if self.middleware_configs["request_validator"]["enabled"]:
            app.middleware("http")(request_validator_middleware)
            logger.info("Request validator middleware registered")

        # 6. Cache middleware
        if self.middleware_configs["cache"]["enabled"]:
            app.middleware("http")(cache_middleware_handler)
            logger.info("Cache middleware registered")

        # 7. Response sanitizer (should be last to sanitize all responses)
        if self.middleware_configs["response_sanitizer"]["enabled"]:
            app.middleware("http")(response_sanitizer_middleware)
            logger.info("Response sanitizer middleware registered")

        logger.info("All middleware registered successfully")


def get_middleware_manager() -> MiddlewareManager:
    """Get the global middleware manager instance"""
    return MiddlewareManager()
