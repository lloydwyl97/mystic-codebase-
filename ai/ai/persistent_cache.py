"""
Persistent Cache System

Provides a persistent cache that can be shared across processes
and survive server restarts using file-based storage.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict

logger = None  # Will be set when logging is available


class PersistentCache:
    def __init__(self, cache_file: str = "ai_cache.json"):
        self.cache_file = cache_file
        self.cache_data = {
            "binance": {},
            "coinbase": {},
            "coingecko": {},
            "last_update": {},
            "created_at": datetime.now().isoformat(),
        }
        self.load_cache()

    def load_cache(self):
        """Load cache from file"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, "r") as f:
                    data = json.load(f)
                    self.cache_data.update(data)
                if logger:
                    logger.info(f"✅ Loaded cache from {self.cache_file}")
        except Exception as e:
            if logger:
                logger.error(f"❌ Error loading cache: {e}")

    def save_cache(self):
        """Save cache to file"""
        try:
            self.cache_data["last_saved"] = datetime.now().isoformat()
            with open(self.cache_file, "w") as f:
                json.dump(self.cache_data, f, indent=2)
            if logger:
                logger.info(f"✅ Saved cache to {self.cache_file}")
        except Exception as e:
            if logger:
                logger.error(f"❌ Error saving cache: {e}")

    def update_binance(self, data: Dict[str, Any]):
        """Update Binance data"""
        self.cache_data["binance"] = data
        self.cache_data["last_update"]["binance"] = datetime.now().isoformat()
        self.save_cache()

    def update_coinbase(self, data: Dict[str, Any]):
        """Update Coinbase data"""
        self.cache_data["coinbase"] = data
        self.cache_data["last_update"]["coinbase"] = datetime.now().isoformat()
        self.save_cache()

    def update_coingecko(self, data: Dict[str, Any]):
        """Update CoinGecko data"""
        self.cache_data["coingecko"] = data
        self.cache_data["last_update"]["coingecko"] = datetime.now().isoformat()
        self.save_cache()

    def get_binance(self) -> Dict[str, Any]:
        """Get Binance data"""
        return self.cache_data.get("binance", {})

    def get_coinbase(self) -> Dict[str, Any]:
        """Get Coinbase data"""
        return self.cache_data.get("coinbase", {})

    def get_coingecko(self) -> Dict[str, Any]:
        """Get CoinGecko data"""
        return self.cache_data.get("coingecko", {})

    def get_last_update(self) -> Dict[str, str]:
        """Get last update timestamps"""
        return self.cache_data.get("last_update", {})

    def is_fresh(self, max_age_seconds: int = 300) -> bool:
        """Check if cache is fresh (updated within max_age_seconds)"""
        last_update = self.cache_data.get("last_update", {})
        if not last_update:
            return False

        # Check if any data source was updated recently
        for source, timestamp in last_update.items():
            try:
                update_time = datetime.fromisoformat(timestamp)
                age = (datetime.now() - update_time).total_seconds()
                if age < max_age_seconds:
                    return True
            except (ValueError, TypeError, AttributeError) as e:
                if logger:
                    logger.debug(f"Failed to parse timestamp for {source}: {e}")
                continue
        return False

    def clear(self):
        """Clear all cache data"""
        self.cache_data = {
            "binance": {},
            "coinbase": {},
            "coingecko": {},
            "last_update": {},
            "created_at": datetime.now().isoformat(),
        }
        self.save_cache()


# Global persistent cache instance
persistent_cache = PersistentCache()


def get_persistent_cache() -> PersistentCache:
    """Get the global persistent cache instance"""
    return persistent_cache
