"""
Connection Manager for Mystic Trading

Manages all external service connections with proper error handling and fallbacks.
"""

import logging
import os
import platform
import subprocess
import time
from typing import Any, Dict, Optional

import pika
import redis

from auto_trading_manager import get_auto_trading_manager
from metrics_collector import get_metrics_collector
from notification_service import NotificationService, get_notification_service
from signal_manager import get_signal_manager

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages all external service connections with proper error handling and fallbacks."""

    def __init__(self):
        # Initialize connection variables
        self.redis_client: Optional[Any] = None
        self.rabbitmq_conn = None
        self.notification_service = None
        self.signal_manager = None
        self.auto_trading_manager = None
        self.metrics_collector = None
        self.minio_client = None
        self.influx_client = None
        self.consul_client = None
        self.vault_client = None

    async def initialize(self) -> None:
        """Initialize all connections - wrapper for initialize_connections"""
        await self.initialize_connections()

    async def initialize_connections(self) -> None:
        """Initialize all connections with proper error handling and fallbacks."""
        # Initialize Redis
        self.redis_client = self._initialize_redis()

        # Initialize RabbitMQ
        self.rabbitmq_conn = self._initialize_rabbitmq()

        # Initialize services
        self._initialize_services()

    def _initialize_redis(self) -> Any:
        """Initialize Redis connection - ALWAYS TRY FOR REAL REDIS"""
        # Get Redis URL from environment variables or use defaults
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")

        # Parse host and port from URL for logging
        try:
            from urllib.parse import urlparse

            parsed = urlparse(redis_url)
            redis_host = parsed.hostname or "redis"
            redis_port = parsed.port or 6379
        except Exception:
            redis_host = "redis"
            redis_port = 6379

        logger.info(
            f"Attempting to connect to Redis at {redis_host}:{redis_port} (URL: {redis_url})"
        )

        # Try multiple connection attempts with different strategies
        connection_attempts = [
            # Attempt 1: Direct connection with Redis URL
            lambda: self._try_redis_url_connection(redis_url),
            # Attempt 2: Direct connection with host/port
            lambda: self._try_redis_connection(redis_host, redis_port, socket_timeout=3),
            # Attempt 3: Try with connection pool
            lambda: self._try_redis_connection_pool(redis_host, redis_port),
            # Attempt 4: Try to start Redis service (Windows)
            lambda: self._try_start_redis_service(redis_host, redis_port),
            # Attempt 5: Try with longer timeout
            lambda: self._try_redis_connection(redis_host, redis_port, socket_timeout=10),
        ]

        for i, attempt in enumerate(connection_attempts, 1):
            try:
                logger.info(f"Redis connection attempt {i}...")
                redis_client = attempt()
                if redis_client:
                    logger.info(
                        f"✅ Redis connection established successfully at {redis_host}:{redis_port}"
                    )
                    return redis_client
            except Exception as e:
                logger.warning(f"Redis connection attempt {i} failed: {str(e)}")
                continue

        # All attempts failed - this is critical for live data
        error_msg = "❌ CRITICAL: All Redis connection attempts failed. Live data requires Redis to be running."
        logger.error(error_msg)
        logger.error("Please start Redis server or install Redis for Windows.")
        logger.error("The application cannot provide live data without Redis.")

        # Return None instead of MockRedis - this will cause the app to fail fast
        # rather than silently using mock data
        return None

    def _try_redis_url_connection(self, redis_url: str) -> Optional[Any]:
        """Try to establish a Redis connection using Redis URL."""
        try:
            logger.debug(f"Attempting Redis URL connection to {redis_url}")

            # Create Redis client from URL
            redis_client = redis.from_url(
                redis_url,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30,
            )

            # Use ping() to test connection
            ping_result = redis_client.ping()
            logger.debug(f"Redis ping result: {ping_result}")

            return redis_client
        except redis.ConnectionError as e:
            logger.debug(f"Redis URL connection error: {str(e)}")
            raise e
        except redis.TimeoutError as e:
            logger.debug(f"Redis URL timeout error: {str(e)}")
            raise e
        except Exception as e:
            logger.debug(f"Redis URL connection failed with unexpected error: {str(e)}")
            raise e

    def _try_redis_connection(self, host: str, port: int, socket_timeout: int = 5) -> Optional[Any]:
        """Try to establish a Redis connection with given parameters."""
        try:
            # Log the connection attempt details
            logger.debug(
                f"Attempting Redis connection to {host}:{port} with timeout {socket_timeout}s"
            )

            # Create Redis client with more resilient settings
            redis_client = redis.Redis(
                host=host,
                port=port,
                db=0,
                decode_responses=True,
                socket_timeout=socket_timeout,
                socket_connect_timeout=socket_timeout,
                retry_on_timeout=True,  # Enable retry for better resilience
                health_check_interval=30,  # Enable health check with reasonable interval
            )

            # Use ping() to test connection - ignore type checking for this method
            ping_result = redis_client.ping()
            logger.debug(f"Redis ping result: {ping_result}")

            return redis_client
        except redis.ConnectionError as e:
            logger.debug(f"Redis connection error: {str(e)}")
            raise e
        except redis.TimeoutError as e:
            logger.debug(f"Redis timeout error: {str(e)}")
            raise e
        except Exception as e:
            logger.debug(f"Redis connection failed with unexpected error: {str(e)}")
            raise e

    def _try_redis_connection_pool(self, host: str, port: int) -> Optional[Any]:
        """Try to establish a Redis connection using connection pool."""
        try:
            pool = redis.ConnectionPool(
                host=host,
                port=port,
                db=0,
                decode_responses=True,
                socket_timeout=3,
                socket_connect_timeout=3,
                retry_on_timeout=False,
                max_connections=5,
            )
            redis_client = redis.Redis(connection_pool=pool)
            # Use ping() to test connection - ignore type checking for this method
            redis_client.ping()
            return redis_client
        except Exception as e:
            logger.debug(f"Redis connection pool failed: {str(e)}")
            raise e

    def _try_start_redis_service(self, host: str, port: int) -> Optional[Any]:
        """Try to start Redis service on Windows and then connect."""
        if host != "redis":
            return None  # Only try to start local Redis service

        try:
            # First check if Redis is already running
            try:
                # Try a quick connection first - it might already be running
                quick_client = redis.Redis(
                    host=host,
                    port=port,
                    db=0,
                    decode_responses=True,
                    socket_timeout=1,
                    socket_connect_timeout=1,
                )
                quick_client.ping()
                logger.info("Redis is already running")
                return quick_client
            except Exception:
                # Redis is not running, continue with startup attempts
                pass

            if platform.system() == "Windows":
                # Try to start Redis service
                try:
                    subprocess.run(
                        ["sc", "start", "Redis"],
                        capture_output=True,
                        timeout=10,
                        check=True,
                    )
                    logger.info("Redis service started successfully")
                    # Wait a moment for service to fully start
                    time.sleep(3)
                except subprocess.CalledProcessError:
                    logger.warning("Failed to start Redis service via sc command")
                except subprocess.TimeoutExpired:
                    logger.warning("Timeout starting Redis service")

                # Try to start Redis server directly if service failed
                try:
                    # Check if Redis process is already running
                    redis_running = False
                    try:
                        result = subprocess.run(
                            [
                                "tasklist",
                                "/FI",
                                "IMAGENAME eq redis-server.exe",
                            ],
                            capture_output=True,
                            text=True,
                        )
                        if "redis-server.exe" in result.stdout:
                            logger.info("Redis server is already running as a process")
                            redis_running = True
                    except Exception:
                        pass

                    if not redis_running:
                        redis_server_path = os.path.join(
                            os.getcwd(), "redis-server", "redis-server.exe"
                        )
                        if os.path.exists(redis_server_path):
                            subprocess.Popen(
                                [redis_server_path],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                            )
                            logger.info("Redis server started directly")
                            time.sleep(2)
                except Exception as e:
                    logger.warning(f"Failed to start Redis server directly: {e}")

            # Try connection after attempting to start with increased timeout
            return self._try_redis_connection(host, port, socket_timeout=15)

        except Exception as e:
            raise e

    def check_redis_health(self) -> Dict[str, Any]:
        """Check Redis connection health and provide diagnostics."""
        health_info: Dict[str, Any] = {
            "connected": False,
            "client_type": "none",
            "error": None,
            "diagnostics": {},
        }

        if not self.redis_client:
            health_info["error"] = "No Redis client initialized"
            return health_info

        try:
            # Check client type
            if hasattr(self.redis_client, "__class__"):
                health_info["client_type"] = self.redis_client.__class__.__name__

            # Try to ping Redis
            if hasattr(self.redis_client, "ping"):
                try:
                    self.redis_client.ping()
                    health_info["connected"] = True
                    health_info["diagnostics"]["ping_successful"] = True
                except Exception as ping_error:
                    health_info["error"] = f"Ping failed: {str(ping_error)}"
                    health_info["diagnostics"]["ping_successful"] = False
            else:
                health_info["error"] = "Redis client does not have ping method"
                health_info["diagnostics"]["ping_method_available"] = False

            # Additional diagnostics for real Redis clients
            if health_info["client_type"] == "Redis":
                try:
                    info = self.redis_client.info()
                    health_info["diagnostics"]["redis_version"] = info.get(
                        "redis_version", "unknown"
                    )
                    health_info["diagnostics"]["connected_clients"] = info.get(
                        "connected_clients", 0
                    )
                    health_info["diagnostics"]["used_memory_human"] = info.get(
                        "used_memory_human", "unknown"
                    )
                except Exception as info_error:
                    health_info["diagnostics"]["info_error"] = str(info_error)

        except Exception as e:
            health_info["error"] = f"Health check failed: {str(e)}"

        return health_info

    def _initialize_rabbitmq(self) -> Any:
        """Initialize RabbitMQ connection with fallback."""
        # Get RabbitMQ connection parameters from environment variables or use defaults
        rabbitmq_host = os.getenv("RABBITMQ_HOST", "localhost")
        rabbitmq_port = int(os.getenv("RABBITMQ_PORT", "5672"))

        try:
            rabbitmq_conn = pika.BlockingConnection(
                pika.ConnectionParameters(
                    host=rabbitmq_host,
                    port=rabbitmq_port,
                    socket_timeout=5,
                    heartbeat=60,
                    blocked_connection_timeout=10,
                )
            )
            logger.info(
                f"RabbitMQ connection established successfully at {rabbitmq_host}:{rabbitmq_port}"
            )
            return rabbitmq_conn
        except Exception as e:
            logger.warning(f"RabbitMQ connection failed: {str(e)}")
            logger.info("RabbitMQ is optional for basic functionality")
            return None

    def _initialize_services(self) -> None:
        """Initialize all services that depend on connections."""
        # Initialize notification service
        try:
            self.notification_service = get_notification_service(self.redis_client)
            logger.info("Notification service initialized successfully")
        except Exception as e:
            logger.warning(f"Notification service initialization failed: {str(e)}")
            self.notification_service = NotificationService(self.redis_client)

        # Initialize signal manager
        try:
            self.signal_manager = get_signal_manager(self.redis_client)
            logger.info("Signal manager initialized successfully")
        except Exception as e:
            logger.warning(f"Signal manager initialization failed: {str(e)}")
            self.signal_manager = None

        # Initialize auto trading manager
        try:
            self.auto_trading_manager = get_auto_trading_manager(self.redis_client)
            logger.info("Auto trading manager initialized successfully")
        except Exception as e:
            logger.warning(f"Auto trading manager initialization failed: {str(e)}")
            self.auto_trading_manager = None

        # Initialize metrics collector
        try:
            # Ensure redis_client is a proper Redis instance before passing to metrics collector
            if self.redis_client and hasattr(self.redis_client, "ping"):
                self.metrics_collector = get_metrics_collector(self.redis_client)
                logger.info("Metrics collector initialized successfully")
            else:
                logger.warning(
                    "Redis client not available, skipping metrics collector initialization"
                )
                self.metrics_collector = None
        except Exception as e:
            logger.warning(f"Metrics collector initialization failed: {str(e)}")
            self.metrics_collector = None

    async def close_connections(self) -> None:
        """Close all connections with proper error handling."""
        # Close RabbitMQ connection
        if self.rabbitmq_conn:
            try:
                self.rabbitmq_conn.close()
                logger.info("RabbitMQ connection closed")
            except Exception as e:
                logger.error(f"Error closing RabbitMQ connection: {str(e)}")

    async def close(self) -> None:
        """Close all connections - wrapper for close_connections"""
        await self.close_connections()


# Global connection manager instance
connection_manager = ConnectionManager()


def get_connection_manager():
    """Get the global connection manager instance"""
    return connection_manager
