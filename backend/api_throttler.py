"""
API Throttling System for Mystic Trading Platform

Provides intelligent API call throttling with:
- Rate limiting per endpoint
- Request queuing
- Adaptive throttling
- Performance monitoring
- Graceful degradation
"""

import asyncio
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional

from utils.exceptions import RateLimitException, handle_exception
from trading_config import trading_config

logger = logging.getLogger(__name__)


class ThrottleLevel(Enum):
    """Throttling levels"""

    CONSERVATIVE = "conservative"  # Start with low rates
    MODERATE = "moderate"  # Medium rates
    AGGRESSIVE = "aggressive"  # High rates
    UNLIMITED = "unlimited"  # No throttling


@dataclass
class ThrottleConfig:
    """Configuration for API throttling"""

    requests_per_second: int = 10
    burst_limit: int = 20
    queue_size: int = 100
    timeout: float = 30.0
    retry_attempts: int = 3
    backoff_factor: float = 2.0


@dataclass
class RequestMetrics:
    """Request performance metrics"""

    endpoint: str
    method: str
    timestamp: float
    response_time: float
    status_code: int
    success: bool
    throttled: bool = False


class AdaptiveThrottler:
    """Adaptive API throttling system"""

    def __init__(self):
        self.throttle_level = ThrottleLevel.CONSERVATIVE
        self.configs = {
            ThrottleLevel.CONSERVATIVE: ThrottleConfig(
                requests_per_second=5,
                burst_limit=10,
                queue_size=50,
                timeout=30.0,
            ),
            ThrottleLevel.MODERATE: ThrottleConfig(
                requests_per_second=20,
                burst_limit=40,
                queue_size=100,
                timeout=20.0,
            ),
            ThrottleLevel.AGGRESSIVE: ThrottleConfig(
                requests_per_second=50,
                burst_limit=100,
                queue_size=200,
                timeout=10.0,
            ),
            ThrottleLevel.UNLIMITED: ThrottleConfig(
                requests_per_second=trading_config.DEFAULT_REQUEST_TIMEOUT * 200,
                burst_limit=trading_config.DEFAULT_REQUEST_TIMEOUT * 400,
                queue_size=500,
                timeout=trading_config.DEFAULT_REQUEST_TIMEOUT,
            ),
        }

        # Per-endpoint throttling
        self.endpoint_limits: Dict[str, ThrottleConfig] = defaultdict(
            lambda: self.configs[self.throttle_level]
        )

        # Request tracking
        self.request_history: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=trading_config.DEFAULT_REQUEST_TIMEOUT * 200))
        self.request_metrics: List[RequestMetrics] = []
        self.metrics_lock = threading.Lock()

        # Performance monitoring
        self.performance_stats = {
            "total_requests": 0,
            "throttled_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0,
            "success_rate": 0.0,
        }

        # Adaptive throttling
        self.adaptation_interval = 60  # seconds
        self.last_adaptation = time.time()
        self.adaptation_thread = threading.Thread(target=self._adaptation_loop, daemon=True)
        self.adaptation_thread.start()

        logger.info(f"✅ AdaptiveThrottler initialized with {self.throttle_level.value} level")

    def set_throttle_level(self, level: ThrottleLevel):
        """Set throttling level"""
        self.throttle_level = level
        logger.info(f"✅ Throttle level set to {level.value}")

    def get_current_config(self) -> ThrottleConfig:
        """Get current throttling configuration"""
        return self.configs[self.throttle_level]

    def set_endpoint_limit(self, endpoint: str, config: ThrottleConfig):
        """Set custom limits for specific endpoint"""
        self.endpoint_limits[endpoint] = config
        logger.info(f"✅ Custom limits set for {endpoint}: {config.requests_per_second} req/s")

    def _can_make_request(self, endpoint: str) -> bool:
        """Check if request can be made based on rate limits"""
        config = self.endpoint_limits[endpoint]
        now = time.time()

        # Clean old requests
        while self.request_history[endpoint] and now - self.request_history[endpoint][0] >= 1.0:
            self.request_history[endpoint].popleft()

        # Check rate limit
        current_requests = len(self.request_history[endpoint])
        return current_requests < config.requests_per_second

    def _record_request(self, endpoint: str):
        """Record a request for rate limiting"""
        self.request_history[endpoint].append(time.time())

    @handle_exception("Request throttling failed", RateLimitException)
    async def throttle_request(
        self,
        endpoint: str,
        method: str = "GET",
        func: Optional[Callable[..., Any]] = None,
        *args,
        **kwargs,
    ) -> Any:
        """Throttle and execute API request"""
        config = self.endpoint_limits[endpoint]
        start_time = time.time()

        # Check if request can be made
        if not self._can_make_request(endpoint):
            self.performance_stats["throttled_requests"] += 1
            raise RateLimitException(
                f"Rate limit exceeded for {endpoint}",
                details={
                    "endpoint": endpoint,
                    "limit": config.requests_per_second,
                    "retry_after": 1.0,
                },
            )

        # Record request
        self._record_request(endpoint)

        # Execute request with retries
        last_exception = None
        for attempt in range(config.retry_attempts):
            try:
                if func:
                    if asyncio.iscoroutinefunction(func):
                        result = await func(*args, **kwargs)
                    else:
                        result = func(*args, **kwargs)
                else:
                    result = None

                # Record successful request
                response_time = time.time() - start_time
                self._record_metrics(endpoint, method, response_time, 200, True, False)

                return result

            except Exception as e:
                last_exception = e
                response_time = time.time() - start_time

                # Record failed request
                self._record_metrics(endpoint, method, response_time, 500, False, False)

                # Exponential backoff
                if attempt < config.retry_attempts - 1:
                    wait_time = config.backoff_factor**attempt
                    logger.warning(f"Request failed, retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)

        # All retries failed
        raise last_exception or Exception("Request failed after all retries")

    def _record_metrics(
        self,
        endpoint: str,
        method: str,
        response_time: float,
        status_code: int,
        success: bool,
        throttled: bool,
    ):
        """Record request metrics"""
        with self.metrics_lock:
            metric = RequestMetrics(
                endpoint=endpoint,
                method=method,
                timestamp=time.time(),
                response_time=response_time,
                status_code=status_code,
                success=success,
                throttled=throttled,
            )

            self.request_metrics.append(metric)

            # Update performance stats
            self.performance_stats["total_requests"] += 1
            if not success:
                self.performance_stats["failed_requests"] += 1
            if throttled:
                self.performance_stats["throttled_requests"] += 1

            # Keep only last metrics based on config
            max_metrics = trading_config.DEFAULT_REQUEST_TIMEOUT * 200
            if len(self.request_metrics) > max_metrics:
                self.request_metrics = self.request_metrics[-max_metrics:]

    def _adaptation_loop(self):
        """Background loop for adaptive throttling"""
        while True:
            try:
                time.sleep(self.adaptation_interval)
                self._adapt_throttling()
            except Exception as e:
                logger.error(f"Adaptation loop error: {e}")

    def _adapt_throttling(self):
        """Adapt throttling based on performance metrics"""
        with self.metrics_lock:
            if not self.request_metrics:
                return

            # Calculate performance metrics
            recent_metrics = [
                m
                for m in self.request_metrics
                if time.time() - m.timestamp < self.adaptation_interval
            ]

            if not recent_metrics:
                return

            success_rate = sum(1 for m in recent_metrics if m.success) / len(recent_metrics)
            avg_response_time = sum(m.response_time for m in recent_metrics) / len(recent_metrics)
            throttle_rate = sum(1 for m in recent_metrics if m.throttled) / len(recent_metrics)

            # Update performance stats
            self.performance_stats["success_rate"] = success_rate
            self.performance_stats["average_response_time"] = avg_response_time

            # Adaptive logic
            current_level = self.throttle_level

            if success_rate > 0.95 and avg_response_time < trading_config.DEFAULT_REQUEST_TIMEOUT and throttle_rate < 0.1:
                # Performance is good, can increase throttling
                if current_level == ThrottleLevel.CONSERVATIVE:
                    self.set_throttle_level(ThrottleLevel.MODERATE)
                elif current_level == ThrottleLevel.MODERATE:
                    self.set_throttle_level(ThrottleLevel.AGGRESSIVE)
                elif current_level == ThrottleLevel.AGGRESSIVE:
                    self.set_throttle_level(ThrottleLevel.UNLIMITED)

            elif success_rate < 0.8 or avg_response_time > trading_config.DEFAULT_REQUEST_TIMEOUT or throttle_rate > 0.3:
                # Performance is poor, decrease throttling
                if current_level == ThrottleLevel.UNLIMITED:
                    self.set_throttle_level(ThrottleLevel.AGGRESSIVE)
                elif current_level == ThrottleLevel.AGGRESSIVE:
                    self.set_throttle_level(ThrottleLevel.MODERATE)
                elif current_level == ThrottleLevel.MODERATE:
                    self.set_throttle_level(ThrottleLevel.CONSERVATIVE)

            logger.info(
                f"Adaptation: success_rate={success_rate:.2f}, "
                f"avg_response_time={avg_response_time:.2f}s, "
                f"throttle_rate={throttle_rate:.2f}"
            )

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        with self.metrics_lock:
            stats = self.performance_stats.copy()

            # Add endpoint-specific stats
            endpoint_stats = defaultdict(
                lambda: {
                    "total_requests": 0,
                    "successful_requests": 0,
                    "failed_requests": 0,
                    "throttled_requests": 0,
                    "average_response_time": 0.0,
                }
            )

            for metric in self.request_metrics:
                endpoint = metric.endpoint
                endpoint_stats[endpoint]["total_requests"] += 1

                if metric.success:
                    endpoint_stats[endpoint]["successful_requests"] += 1
                else:
                    endpoint_stats[endpoint]["failed_requests"] += 1

                if metric.throttled:
                    endpoint_stats[endpoint]["throttled_requests"] += 1

            # Calculate averages for each endpoint
            for endpoint, data in endpoint_stats.items():
                if data["total_requests"] > 0:
                    endpoint_metrics = [m for m in self.request_metrics if m.endpoint == endpoint]
                    if endpoint_metrics:
                        data["average_response_time"] = sum(
                            m.response_time for m in endpoint_metrics
                        ) / len(endpoint_metrics)

            stats["endpoint_stats"] = dict(endpoint_stats)
            stats["current_throttle_level"] = self.throttle_level.value
            stats["current_config"] = {
                "requests_per_second": (self.get_current_config().requests_per_second),
                "burst_limit": self.get_current_config().burst_limit,
                "timeout": self.get_current_config().timeout,
            }

            return stats

    def increase_throttling(self):
        """Manually increase throttling level"""
        levels = list(ThrottleLevel)
        current_index = levels.index(self.throttle_level)
        if current_index < len(levels) - 1:
            self.set_throttle_level(levels[current_index + 1])
            logger.info(f"✅ Throttling increased to {self.throttle_level.value}")
        else:
            logger.info("✅ Already at maximum throttling level")

    def decrease_throttling(self):
        """Manually decrease throttling level"""
        levels = list(ThrottleLevel)
        current_index = levels.index(self.throttle_level)
        if current_index > 0:
            self.set_throttle_level(levels[current_index - 1])
            logger.info(f"✅ Throttling decreased to {self.throttle_level.value}")
        else:
            logger.info("✅ Already at minimum throttling level")


# Global throttler instance
api_throttler = AdaptiveThrottler()
