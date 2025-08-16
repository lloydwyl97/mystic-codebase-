"""
Advanced Rate Limiter for Mystic Trading Platform

Provides comprehensive rate limiting with:
- Multiple rate limiting strategies
- IP-based and user-based limiting
- Sliding window and token bucket algorithms
- Security monitoring and alerting
- Configurable limits per endpoint
"""

import hashlib
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict
import threading

try:
    import redis
    from redis import Redis
except ImportError:
    redis = None
    Redis = None

from trading_config import trading_config

logger = logging.getLogger(__name__)

# Rate limiting configuration
DEFAULT_RATE_LIMIT = 100  # requests per minute
BURST_RATE_LIMIT = 10     # requests per second
SLIDING_WINDOW_SIZE = 60   # seconds
TOKEN_BUCKET_CAPACITY = 100
TOKEN_BUCKET_RATE = 1.67   # tokens per second (100 per minute)

# Security thresholds
MAX_FAILED_ATTEMPTS = 5
BLOCK_DURATION = 300  # 5 minutes
SUSPICIOUS_ACTIVITY_THRESHOLD = 50  # requests per minute


@dataclass
class RateLimitConfig:
    """Rate limit configuration for endpoints"""
    endpoint: str
    requests_per_minute: int
    requests_per_second: int
    burst_limit: int
    window_size: int
    strategy: str  # 'sliding_window', 'token_bucket', 'fixed_window'


class RateLimitViolation:
    """Rate limit violation record"""
    client_id: str
    endpoint: str
    timestamp: float
    violation_type: str  # 'rate_limit', 'burst_limit', 'suspicious_activity'
    request_count: int
    limit: int


class SecurityMonitor:
    """Monitors for suspicious activity and security threats"""

    def __init__(self):
        self.violations: deque = deque(maxlen=1000)
        self.blocked_clients: Dict[str, float] = {}
        self.suspicious_patterns: Dict[str, int] = defaultdict(int)
        self.security_alerts: deque = deque(maxlen=100)
        self.lock = threading.Lock()

    def record_violation(self, client_id: str, endpoint: str, violation_type: str,
                        request_count: int, limit: int):
        """Record a rate limit violation"""
        violation = RateLimitViolation(
            client_id=client_id,
            endpoint=endpoint,
            timestamp=time.time(),
            violation_type=violation_type,
            request_count=request_count,
            limit=limit
        )

        with self.lock:
            self.violations.append(violation)

            # Check for suspicious patterns
            pattern_key = f"{client_id}:{endpoint}"
            self.suspicious_patterns[pattern_key] += 1

            # Block client if too many violations
            if self.suspicious_patterns[pattern_key] >= MAX_FAILED_ATTEMPTS:
                self.blocked_clients[client_id] = time.time() + BLOCK_DURATION
                self._create_security_alert(client_id, "Client blocked due to repeated violations")

    def is_client_blocked(self, client_id: str) -> bool:
        """Check if client is currently blocked"""
        with self.lock:
            if client_id in self.blocked_clients:
                if time.time() < self.blocked_clients[client_id]:
                    return True
                else:
                    # Remove expired block
                    del self.blocked_clients[client_id]
            return False

    def _create_security_alert(self, client_id: str, reason: str):
        """Create a security alert"""
        alert = {
            'client_id': client_id,
            'reason': reason,
            'timestamp': time.time(),
            'severity': 'high'
        }
        self.security_alerts.append(alert)
        logger.warning(f"Security alert: {reason} for client {client_id}")

    def get_security_stats(self) -> Dict[str, Any]:
        """Get security monitoring statistics"""
        with self.lock:
            return {
                'total_violations': len(self.violations),
                'blocked_clients': len(self.blocked_clients),
                'security_alerts': len(self.security_alerts),
                'suspicious_patterns': dict(self.suspicious_patterns)
            }


class SlidingWindowRateLimiter:
    """Sliding window rate limiter implementation"""

    def __init__(self, window_size: int = SLIDING_WINDOW_SIZE):
        self.window_size = window_size
        self.requests: Dict[str, deque] = defaultdict(lambda: deque())
        self.lock = threading.Lock()

    def is_allowed(self, client_id: str, limit: int) -> bool:
        """Check if request is allowed within sliding window"""
        current_time = time.time()

        with self.lock:
            # Remove old requests outside the window
            while (self.requests[client_id] and
                   current_time - self.requests[client_id][0] > self.window_size):
                self.requests[client_id].popleft()

            # Check if under limit
            if len(self.requests[client_id]) < limit:
                self.requests[client_id].append(current_time)
                return True

            return False

    def get_remaining_requests(self, client_id: str, limit: int) -> int:
        """Get remaining requests for client"""
        current_time = time.time()

        with self.lock:
            # Remove old requests
            while (self.requests[client_id] and
                   current_time - self.requests[client_id][0] > self.window_size):
                self.requests[client_id].popleft()

            return max(0, limit - len(self.requests[client_id]))


class TokenBucketRateLimiter:
    """Token bucket rate limiter implementation"""

    def __init__(self, capacity: int = TOKEN_BUCKET_CAPACITY, rate: float = TOKEN_BUCKET_RATE):
        self.capacity = capacity
        self.rate = rate
        self.tokens: Dict[str, float] = defaultdict(lambda: capacity)
        self.last_update: Dict[str, float] = defaultdict(lambda: time.time())
        self.lock = threading.Lock()

    def is_allowed(self, client_id: str, tokens_required: int = 1) -> bool:
        """Check if request is allowed based on available tokens"""
        current_time = time.time()

        with self.lock:
            # Refill tokens based on time passed
            time_passed = current_time - self.last_update[client_id]
            tokens_to_add = time_passed * self.rate
            self.tokens[client_id] = min(self.capacity, self.tokens[client_id] + tokens_to_add)
            self.last_update[client_id] = current_time

            # Check if enough tokens available
            if self.tokens[client_id] >= tokens_required:
                self.tokens[client_id] -= tokens_required
                return True

            return False

    def get_remaining_tokens(self, client_id: str) -> float:
        """Get remaining tokens for client"""
        current_time = time.time()

        with self.lock:
            # Refill tokens
            time_passed = current_time - self.last_update[client_id]
            tokens_to_add = time_passed * self.rate
            self.tokens[client_id] = min(self.capacity, self.tokens[client_id] + tokens_to_add)
            self.last_update[client_id] = current_time

            return self.tokens[client_id]


class AdvancedRateLimiter:
    """Advanced rate limiter with multiple strategies and security monitoring"""

    def __init__(self):
        self.sliding_window_limiter = SlidingWindowRateLimiter()
        self.token_bucket_limiter = TokenBucketRateLimiter()
        self.security_monitor = SecurityMonitor()
        self.endpoint_configs: Dict[str, RateLimitConfig] = {}
        self.request_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.lock = threading.Lock()
        self.redis_client = None

        # Initialize Redis connection
        self._initialize_redis()

        # Setup default configurations
        self._setup_default_configs()

    def _initialize_redis(self):
        """Initialize Redis connection for distributed rate limiting"""
        if redis is None or Redis is None:
            logger.warning("Redis not available, using local rate limiting only")
            return

        try:
            self.redis_client = Redis(
                host=trading_config.DEFAULT_REDIS_HOST,
                port=trading_config.DEFAULT_REDIS_PORT,
                db=trading_config.DEFAULT_REDIS_DB,
                decode_responses=True
            )

            # Test connection
            self.redis_client.ping()
            logger.info("Redis connection established for rate limiting")

        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            self.redis_client = None

    def _setup_default_configs(self):
        """Setup default rate limit configurations for common endpoints"""
        default_configs = [
            RateLimitConfig("/api/auth/login", 5, 1, 2, 60, "sliding_window"),
            RateLimitConfig("/api/auth/register", 3, 1, 1, 300, "sliding_window"),
            RateLimitConfig("/api/trading/order", 100, 10, 5, 60, "token_bucket"),
            RateLimitConfig("/api/portfolio/balance", 50, 5, 3, 60, "sliding_window"),
            RateLimitConfig("/api/market/data", 200, 20, 10, 60, "token_bucket")
        ]

        for config in default_configs:
            self.endpoint_configs[config.endpoint] = config

    def get_client_id(self, request) -> str:
        """Extract client identifier from request"""
        # Try to get from headers first
        client_id = request.headers.get('X-Client-ID')
        if client_id:
            return client_id

        # Try to get from user authentication
        if hasattr(request, 'user') and request.user:
            return f"user_{request.user.id}"

        # Fallback to IP address
        client_ip = request.client.host if hasattr(request, 'client') else 'unknown'
        return f"ip_{client_ip}"

    def is_rate_limited(self, request, endpoint: str) -> tuple[bool, Dict[str, Any]]:
        """Check if request should be rate limited"""
        client_id = self.get_client_id(request)

        # Check if client is blocked
        if self.security_monitor.is_client_blocked(client_id):
            return True, {
                'blocked': True,
                'reason': 'Client is blocked due to previous violations',
                'retry_after': BLOCK_DURATION
            }

        # Get endpoint configuration
        config = self.endpoint_configs.get(endpoint)
        if not config:
            config = RateLimitConfig(endpoint, DEFAULT_RATE_LIMIT, BURST_RATE_LIMIT,
                                   BURST_RATE_LIMIT, SLIDING_WINDOW_SIZE, "sliding_window")

        # Check rate limiting based on strategy
        if config.strategy == "sliding_window":
            allowed = self.sliding_window_limiter.is_allowed(client_id, config.requests_per_minute)
            remaining = self.sliding_window_limiter.get_remaining_requests(client_id, config.requests_per_minute)
        else:
            allowed = self.token_bucket_limiter.is_allowed(client_id, 1)
            remaining = int(self.token_bucket_limiter.get_remaining_tokens(client_id))

        # Check burst limiting
        burst_allowed = self.sliding_window_limiter.is_allowed(client_id, config.burst_limit)

        # Track request count
        with self.lock:
            self.request_counts[client_id][endpoint] += 1

        # Check for suspicious activity
        if self._is_suspicious_activity(client_id, endpoint):
            self.security_monitor.record_violation(client_id, endpoint, "suspicious_activity",
                                                 self.request_counts[client_id][endpoint], 0)
            return True, {
                'blocked': False,
                'reason': 'Suspicious activity detected',
                'retry_after': 60
            }

        if not allowed or not burst_allowed:
            violation_type = "rate_limit" if not allowed else "burst_limit"
            self.security_monitor.record_violation(client_id, endpoint, violation_type,
                                                 self.request_counts[client_id][endpoint], config.requests_per_minute)

            return True, {
                'blocked': False,
                'reason': f'Rate limit exceeded for {endpoint}',
                'retry_after': 60,
                'remaining_requests': remaining
            }

        return False, {
            'remaining_requests': remaining,
            'reset_time': time.time() + config.window_size
        }

    def _is_suspicious_activity(self, client_id: str, endpoint: str) -> bool:
        """Check for suspicious activity patterns"""
        with self.lock:
            # Check if client is making too many requests
            total_requests = sum(self.request_counts[client_id].values())
            if total_requests > SUSPICIOUS_ACTIVITY_THRESHOLD:
                return True

            # Check for rapid requests to the same endpoint
            endpoint_requests = self.request_counts[client_id][endpoint]
            if endpoint_requests > 20:  # More than 20 requests to same endpoint
                return True

            return False

    def get_rate_limit_info(self, client_id: str, endpoint: str) -> Dict[str, Any]:
        """Get rate limit information for client and endpoint"""
        config = self.endpoint_configs.get(endpoint)
        if not config:
            config = RateLimitConfig(endpoint, DEFAULT_RATE_LIMIT, BURST_RATE_LIMIT,
                                   BURST_RATE_LIMIT, SLIDING_WINDOW_SIZE, "sliding_window")

        if config.strategy == "sliding_window":
            remaining = self.sliding_window_limiter.get_remaining_requests(client_id, config.requests_per_minute)
        else:
            remaining = int(self.token_bucket_limiter.get_remaining_tokens(client_id))

        return {
            'endpoint': endpoint,
            'strategy': config.strategy,
            'limit_per_minute': config.requests_per_minute,
            'limit_per_second': config.requests_per_second,
            'burst_limit': config.burst_limit,
            'remaining_requests': remaining,
            'window_size': config.window_size,
            'is_blocked': self.security_monitor.is_client_blocked(client_id)
        }

    def add_endpoint_config(self, config: RateLimitConfig):
        """Add or update endpoint rate limit configuration"""
        with self.lock:
            self.endpoint_configs[config.endpoint] = config
            logger.info(f"Added rate limit config for {config.endpoint}: {config.requests_per_minute} req/min")

    def get_rate_limit_stats(self) -> Dict[str, Any]:
        """Get comprehensive rate limiting statistics"""
        with self.lock:
            stats = self.security_monitor.get_security_stats()
            stats.update({
                'endpoint_configs': len(self.endpoint_configs),
                'active_clients': len(self.request_counts),
                'redis_available': self.redis_client is not None,
                'total_requests': sum(sum(counts.values()) for counts in self.request_counts.values())
            })
            return stats

    def clear_old_data(self, max_age_hours: int = 24):
        """Clear old rate limiting data"""
        time.time() - (max_age_hours * 3600)

        with self.lock:
            # Clear old request counts
            for client_id in list(self.request_counts.keys()):
                if not any(self.request_counts[client_id].values()):
                    del self.request_counts[client_id]

        logger.info("Cleared old rate limiting data")


# Global rate limiter instance
rate_limiter = AdvancedRateLimiter()


