"""
Cache Middleware Handler for Mystic Trading Platform

Provides a simple async cache middleware for FastAPI/Starlette apps.
Uses aioredis if available, else acts as a no-op cache.
"""

import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
import asyncio

try:
    import aioredis
except ImportError:
    aioredis = None

logger = logging.getLogger(__name__)


class CacheMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, redis_url="redis://localhost:6379/0"):
        super().__init__(app)
        self.redis_url = redis_url
        self.redis = None
        if aioredis:
            asyncio.create_task(self._init_redis())

    async def _init_redis(self):
        try:
            self.redis = await aioredis.from_url(self.redis_url)
            logger.info("CacheMiddleware: Connected to Redis")
        except Exception as e:
            logger.error(f"CacheMiddleware: Redis connection failed: {e}")
            self.redis = None

    async def dispatch(self, request: Request, call_next):
        # Example: No-op cache, just pass through
        response = await call_next(request)
        return response


# Exported handler for use in app_config and __init__.py
cache_middleware_handler = CacheMiddleware
