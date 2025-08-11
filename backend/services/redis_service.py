"""
Redis Service
Handles Redis operations and caching with Docker connection
"""

import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import redis.asyncio as redis
import os

logger = logging.getLogger(__name__)


class RedisService:
    def __init__(self):
        self.redis_client = None
        self.connection_status = "disconnected"
        self.stats = {
            "total_operations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "keys_stored": 0,
        }

        # Redis connection configuration
        self.redis_host = os.getenv("REDIS_HOST", "localhost")
        self.redis_port = int(os.getenv("REDIS_PORT", "6379"))
        self.redis_db = int(os.getenv("REDIS_DB", "0"))
        self.redis_password = os.getenv("REDIS_PASSWORD", None)

        logger.info(
            f"✅ RedisService initialized - connecting to {self.redis_host}:{self.redis_port}"
        )
        self._connect()

    def _connect(self):
        """Connect to Redis"""
        try:
            self.redis_client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                db=self.redis_db,
                password=self.redis_password,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30,
            )
            self.connection_status = "connected"
            logger.info("✅ Connected to Redis successfully")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            self.connection_status = "error"
            self.redis_client = None

    async def _ensure_connection(self):
        """Ensure Redis connection is available"""
        if self.redis_client is None:
            self._connect()
        if self.redis_client is None:
            raise Exception("Redis connection not available")

    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from Redis"""
        try:
            await self._ensure_connection()
            self.stats["total_operations"] += 1

            value = await self.redis_client.get(key)
            if value is not None:
                self.stats["cache_hits"] += 1
                return json.loads(value)
            else:
                self.stats["cache_misses"] += 1
                return default
        except Exception as e:
            logger.error(f"❌ Error getting key {key}: {e}")
            return default

    async def set(self, key: str, value: Any, ex: Optional[int] = None) -> bool:
        """Set value in Redis with optional expiry"""
        try:
            await self._ensure_connection()
            self.stats["total_operations"] += 1

            serialized_value = json.dumps(value)
            result = await self.redis_client.set(key, serialized_value, ex=ex)

            # Update stats
            if result:
                self.stats["keys_stored"] = await self.redis_client.dbsize()

            return bool(result)
        except Exception as e:
            logger.error(f"❌ Error setting key {key}: {e}")
            return False

    async def delete(self, key: str) -> int:
        """Delete key from Redis"""
        try:
            await self._ensure_connection()
            self.stats["total_operations"] += 1

            result = await self.redis_client.delete(key)
            self.stats["keys_stored"] = await self.redis_client.dbsize()
            return result
        except Exception as e:
            logger.error(f"❌ Error deleting key {key}: {e}")
            return 0

    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis"""
        try:
            await self._ensure_connection()
            self.stats["total_operations"] += 1

            result = await self.redis_client.exists(key)
            return bool(result)
        except Exception as e:
            logger.error(f"❌ Error checking existence of key {key}: {e}")
            return False

    async def hget(self, name: str, key: str) -> Optional[str]:
        """Get hash field value"""
        try:
            await self._ensure_connection()
            return await self.redis_client.hget(name, key)
        except Exception as e:
            logger.error(f"❌ Error getting hash field {name}:{key}: {e}")
            return None

    async def hset(self, name: str, key: str, value: Any) -> int:
        """Set hash field value"""
        try:
            await self._ensure_connection()
            serialized_value = json.dumps(value) if not isinstance(value, str) else value
            return await self.redis_client.hset(name, key, serialized_value)
        except Exception as e:
            logger.error(f"❌ Error setting hash field {name}:{key}: {e}")
            return 0

    async def hgetall(self, name: str) -> Dict[str, Any]:
        """Get all hash fields"""
        try:
            await self._ensure_connection()
            result = await self.redis_client.hgetall(name)
            # Try to deserialize values
            deserialized = {}
            for key, value in result.items():
                try:
                    deserialized[key] = json.loads(value)
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Could not deserialize value for key {key}: {e}")
                    deserialized[key] = value
            return deserialized
        except Exception as e:
            logger.error(f"❌ Error getting all hash fields for {name}: {e}")
            return {}

    async def lpush(self, name: str, *values: Any) -> int:
        """Push values to list"""
        try:
            await self._ensure_connection()
            serialized_values = [json.dumps(v) if not isinstance(v, str) else v for v in values]
            return await self.redis_client.lpush(name, *serialized_values)
        except Exception as e:
            logger.error(f"❌ Error pushing to list {name}: {e}")
            return 0

    async def lrange(self, name: str, start: int, end: int) -> List[Any]:
        """Get list range"""
        try:
            await self._ensure_connection()
            result = await self.redis_client.lrange(name, start, end)
            # Try to deserialize values
            deserialized = []
            for value in result:
                try:
                    deserialized.append(json.loads(value))
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Could not deserialize list value: {e}")
                    deserialized.append(value)
            return deserialized
        except Exception as e:
            logger.error(f"❌ Error getting list range for {name}: {e}")
            return []

    async def ping(self) -> bool:
        """Ping Redis server"""
        try:
            await self._ensure_connection()
            result = await self.redis_client.ping()
            self.connection_status = "connected" if result else "error"
            return bool(result)
        except Exception as e:
            logger.error(f"❌ Redis ping failed: {e}")
            self.connection_status = "error"
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """Get Redis statistics"""
        try:
            await self._ensure_connection()
            info = await self.redis_client.info()
            dbsize = await self.redis_client.dbsize()

            return {
                "connection_status": self.connection_status,
                "redis_version": info.get("redis_version", "unknown"),
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_human": info.get("used_memory_human", "unknown"),
                "total_operations": self.stats["total_operations"],
                "cache_hits": self.stats["cache_hits"],
                "cache_misses": self.stats["cache_misses"],
                "keys_stored": dbsize,
                "hit_rate": (self.stats["cache_hits"] / max(self.stats["total_operations"], 1)),
                "timestamp": datetime.now(timezone.timezone.utc).isoformat(),
            }
        except Exception as e:
            logger.error(f"❌ Error getting Redis stats: {e}")
            return {
                "connection_status": self.connection_status,
                "error": str(e),
                "timestamp": datetime.now(timezone.timezone.utc).isoformat(),
            }

    async def flushdb(self) -> bool:
        """Clear all data"""
        try:
            await self._ensure_connection()
            await self.redis_client.flushdb()
            self.stats["keys_stored"] = 0
            logger.info("✅ Redis database flushed")
            return True
        except Exception as e:
            logger.error(f"❌ Error flushing database: {e}")
            return False

    async def close(self):
        """Close Redis connection"""
        try:
            if self.redis_client:
                await self.redis_client.close()
            self.connection_status = "disconnected"
            logger.info("Redis connection closed")
        except Exception as e:
            logger.error(f"❌ Error closing Redis connection: {e}")


# Global instance
redis_service = RedisService()


def get_redis_service() -> RedisService:
    """Get the Redis service instance"""
    return redis_service
