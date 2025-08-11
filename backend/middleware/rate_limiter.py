"""
Custom rate limiting implementation for FastAPI using Redis 5.x and aioredis 2.0.1
Replaces fastapi-limiter which doesn't support Redis 5.x
"""

import time
from typing import Optional, Callable
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
import aioredis


class RateLimiter:
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis: Optional[aioredis.Redis] = None

    async def connect(self):
        """Connect to Redis"""
        if not self.redis:
            self.redis = aioredis.from_url(self.redis_url, decode_responses=True)

    async def close(self):
        """Close Redis connection"""
        if self.redis:
            await self.redis.close()

    async def is_allowed(
        self, key: str, max_requests: int, window_seconds: int
    ) -> tuple[bool, dict]:
        """
        Check if request is allowed based on rate limit

        Args:
            key: Unique identifier for the rate limit (e.g., IP address, user ID)
            max_requests: Maximum number of requests allowed in the window
            window_seconds: Time window in seconds

        Returns:
            tuple: (is_allowed, rate_limit_info)
        """
        await self.connect()

        current_time = int(time.time())
        window_start = current_time - window_seconds

        # Use Redis sorted set to track requests
        pipe = self.redis.pipeline()

        # Remove old entries outside the window
        pipe.zremrangebyscore(key, 0, window_start)

        # Count current requests in window
        pipe.zcard(key)

        # Add current request
        pipe.zadd(key, {str(current_time): current_time})

        # Set expiry on the key
        pipe.expire(key, window_seconds)

        # Execute pipeline
        results = await pipe.execute()
        current_requests = results[1]  # zcard result

        # Check if limit exceeded
        is_allowed = current_requests < max_requests

        # Calculate remaining requests and reset time
        remaining = max(0, max_requests - current_requests)
        reset_time = current_time + window_seconds

        rate_limit_info = {
            "limit": max_requests,
            "remaining": remaining,
            "reset": reset_time,
            "window": window_seconds,
        }

        return is_allowed, rate_limit_info


# Global rate limiter instance
rate_limiter = RateLimiter()


def get_client_ip(request: Request) -> str:
    """Extract client IP from request"""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


async def rate_limit_middleware(
    request: Request,
    max_requests: int = 100,
    window_seconds: int = 60,
    key_func: Optional[Callable[[Request], str]] = None,
):
    """
    Rate limiting middleware for FastAPI

    Args:
        request: FastAPI request object
        max_requests: Maximum requests per window
        window_seconds: Time window in seconds
        key_func: Optional function to generate rate limit key
    """
    # Generate rate limit key
    if key_func:
        key = f"rate_limit:{key_func(request)}"
    else:
        key = f"rate_limit:{get_client_ip(request)}"

    # Check rate limit
    is_allowed, rate_info = await rate_limiter.is_allowed(key, max_requests, window_seconds)

    if not is_allowed:
        raise HTTPException(
            status_code=429,
            detail={"error": "Rate limit exceeded", "rate_limit": rate_info},
        )

    # Add rate limit headers to response
    request.state.rate_limit_info = rate_info


def rate_limit(
    max_requests: int = 100,
    window_seconds: int = 60,
    key_func: Optional[Callable[[Request], str]] = None,
):
    """
    Decorator for rate limiting endpoints

    Usage:
        @app.get("/api/data")
        @rate_limit(max_requests=10, window_seconds=60)
        async def get_data(request: Request):
            return {"data": "example"}
    """

    def decorator(func):
        async def wrapper(*args, request: Request, **kwargs):
            await rate_limit_middleware(request, max_requests, window_seconds, key_func)
            return await func(*args, request=request, **kwargs)

        return wrapper

    return decorator


async def add_rate_limit_headers(request: Request, response: JSONResponse):
    """Add rate limit headers to response"""
    if hasattr(request.state, "rate_limit_info"):
        rate_info = request.state.rate_limit_info
        response.headers["X-RateLimit-Limit"] = str(rate_info["limit"])
        response.headers["X-RateLimit-Remaining"] = str(rate_info["remaining"])
        response.headers["X-RateLimit-Reset"] = str(rate_info["reset"])
        response.headers["X-RateLimit-Window"] = str(rate_info["window"])

    return response
