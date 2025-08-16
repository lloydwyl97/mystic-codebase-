"""
Rate Limiter Manager

Handles rate limiting logic and request tracking.
"""

import logging
import time
from collections import defaultdict
from typing import Dict, List, Tuple

from fastapi import HTTPException, Request

logger = logging.getLogger(__name__)


class RateLimiter:
    """Rate limiting implementation"""

    def __init__(self):
        # Store request counts per IP
        self.requests: Dict[str, List[float]] = defaultdict(list)

        # Rate limit configurations
        self.rate_limits = {
            "default": {
                "requests": 100,
                "window": 60,
            },  # 100 requests per minute
            "auth": {"requests": 5, "window": 60},  # 5 requests per minute
            "api": {
                "requests": 1000,
                "window": 3600,
            },  # 1000 requests per hour
            "websocket": {
                "requests": 100,
                "window": 60,
            },  # 100 requests per minute
        }

        # Endpoint-specific rate limits
        self.endpoint_limits = {
            "/api/auth/login": "auth",
            "/api/auth/register": "auth",
            "/api/coinbase": "api",
            "/ws": "websocket",
        }

        # Last cleanup time
        self.last_full_cleanup = time.time()
        # Full cleanup interval (every hour)
        self.full_cleanup_interval = 3600

    def get_rate_limit(self, path: str) -> Tuple[int, int]:
        """Get rate limit configuration for a path"""
        # Check for endpoint-specific limit
        for endpoint, limit_type in self.endpoint_limits.items():
            if path.startswith(endpoint):
                limit = self.rate_limits[limit_type]
                return limit["requests"], limit["window"]

        # Return default limit
        limit = self.rate_limits["default"]
        return limit["requests"], limit["window"]

    def cleanup_old_requests(self, ip: str, window: int) -> None:
        """Remove requests older than the window for a specific IP"""
        current_time = time.time()
        self.requests[ip] = [
            req_time for req_time in self.requests[ip] if current_time - req_time < window
        ]

        # If the IP has no requests, remove it from the dictionary
        if not self.requests[ip]:
            del self.requests[ip]

        # Periodically clean up all expired requests to prevent memory leaks
        if current_time - self.last_full_cleanup > self.full_cleanup_interval:
            self.full_cleanup()
            self.last_full_cleanup = current_time

    def full_cleanup(self) -> None:
        """Clean up all expired requests across all IPs"""
        current_time = time.time()
        # Find the maximum window size from all rate limits
        max_window = max(limit["window"] for limit in self.rate_limits.values())

        # Clean up each IP's requests
        ips_to_remove: List[str] = []
        for ip, requests in self.requests.items():
            self.requests[ip] = [
                req_time for req_time in requests if current_time - req_time < max_window
            ]
            if not self.requests[ip]:
                ips_to_remove.append(ip)

        # Remove IPs with no requests
        for ip in ips_to_remove:
            del self.requests[ip]

        logger.debug(f"Full cleanup completed. Active IPs: {len(self.requests)}")

    async def check_rate_limit(self, request: Request) -> None:
        """Check if request is within rate limits"""
        try:
            # Get client IP
            client_ip: str = request.client.host if request.client else "unknown"
            path = request.url.path

            # Skip rate limiting for unknown IPs if configured to do so
            if client_ip == "unknown":
                logger.warning("Unknown client IP detected, skipping rate limiting")
                return

            # Get rate limit configuration
            max_requests, window = self.get_rate_limit(path)

            # Cleanup old requests
            self.cleanup_old_requests(client_ip, window)

            # Check if rate limit exceeded
            if len(self.requests[client_ip]) >= max_requests:
                logger.warning(f"Rate limit exceeded for IP {client_ip} on path {path}")
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": "Rate limit exceeded",
                        "retry_after": window,
                        "limit": max_requests,
                        "window": window,
                    },
                )

            # Add current request
            self.requests[client_ip].append(time.time())

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error in rate limiter: {str(e)}")
            # Don't block the request if rate limiter fails
            pass


