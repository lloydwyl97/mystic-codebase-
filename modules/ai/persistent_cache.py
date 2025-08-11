"""
Persistent Cache Module for AI Services
Provides persistent caching functionality for AI services and data.
"""

class PersistentCache:
    _cache = {}

    @classmethod
    def get(cls, key, default=None):
        """Get value from cache"""
        return cls._cache.get(key, default)

    @classmethod
    def set(cls, key, value):
        """Set value in cache"""
        cls._cache[key] = value

    @classmethod
    def delete(cls, key):
        """Delete value from cache"""
        if key in cls._cache:
            del cls._cache[key]

    @classmethod
    def clear(cls):
        """Clear all cache"""
        cls._cache.clear()

    @classmethod
    def exists(cls, key):
        """Check if key exists in cache"""
        return key in cls._cache

    @classmethod
    def get_all(cls):
        """Get all cached data"""
        return cls._cache.copy()


# Global cache instance
persistent_cache = PersistentCache()


def get_persistent_cache():
    """Get the global persistent cache instance"""
    return persistent_cache 