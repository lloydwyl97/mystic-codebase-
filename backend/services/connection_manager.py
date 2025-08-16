"""
Connection Manager Service
Handles database and external service connections
"""

import logging
from typing import Dict, Any
from datetime import datetime, timezone
import redis
import pika

logger = logging.getLogger(__name__)


class ConnectionManager:
    def __init__(self):
        self.redis_client = None
        self.rabbitmq_connection = None
        self.connection_status = {
            "redis": "disconnected",
            "rabbitmq": "disconnected",
            "database": "disconnected",
        }
        logger.info("âœ… ConnectionManager initialized")

    async def get_connection_status(self) -> Dict[str, Any]:
        """Get status of all connections"""
        return {
            "connections": self.connection_status,
            "timestamp": datetime.now(timezone.timezone.utc).isoformat(),
        }

    async def connect_redis(self, host: str = "localhost", port: int = 6379) -> Dict[str, Any]:
        """Connect to Redis"""
        try:
            self.redis_client = redis.Redis(host=host, port=port, decode_responses=True)
            # Test connection
            self.redis_client.ping()
            self.connection_status["redis"] = "connected"
            logger.info("âœ… Redis connected successfully")
            return {"success": True, "status": "connected"}
        except Exception as e:
            self.connection_status["redis"] = "failed"
            logger.error(f"âŒ Redis connection failed: {e}")
            return {"success": False, "error": str(e)}

    async def connect_rabbitmq(self, host: str = "localhost", port: int = 5672) -> Dict[str, Any]:
        """Connect to RabbitMQ"""
        try:
            self.rabbitmq_connection = pika.BlockingConnection(
                pika.ConnectionParameters(host=host, port=port)
            )
            self.connection_status["rabbitmq"] = "connected"
            logger.info("âœ… RabbitMQ connected successfully")
            return {"success": True, "status": "connected"}
        except Exception as e:
            self.connection_status["rabbitmq"] = "failed"
            logger.error(f"âŒ RabbitMQ connection failed: {e}")
            return {"success": False, "error": str(e)}

    async def test_redis_connection(self) -> Dict[str, Any]:
        """Test Redis connection"""
        try:
            if self.redis_client:
                self.redis_client.ping()
                return {"success": True, "status": "connected"}
            return {"success": False, "error": "Redis client not initialized"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def test_rabbitmq_connection(self) -> Dict[str, Any]:
        """Test RabbitMQ connection"""
        try:
            if self.rabbitmq_connection and not self.rabbitmq_connection.is_closed:
                return {"success": True, "status": "connected"}
            return {
                "success": False,
                "error": "RabbitMQ connection not available",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def get_redis_info(self) -> Dict[str, Any]:
        """Get Redis server information"""
        try:
            if self.redis_client:
                info = self.redis_client.info()
                return {
                    "version": info.get("redis_version", "unknown"),
                    "connected_clients": info.get("connected_clients", 0),
                    "used_memory": info.get("used_memory_human", "unknown"),
                    "uptime": info.get("uptime_in_seconds", 0),
                }
            return {"error": "Redis client not initialized"}
        except Exception as e:
            return {"error": str(e)}

    async def close_connections(self) -> Dict[str, Any]:
        """Close all connections"""
        try:
            if self.redis_client:
                self.redis_client.close()
                self.connection_status["redis"] = "disconnected"

            if self.rabbitmq_connection and not self.rabbitmq_connection.is_closed:
                self.rabbitmq_connection.close()
                self.connection_status["rabbitmq"] = "disconnected"

            logger.info("âœ… All connections closed")
            return {"success": True}
        except Exception as e:
            logger.error(f"âŒ Error closing connections: {e}")
            return {"success": False, "error": str(e)}


# Global instance
connection_manager = ConnectionManager()


