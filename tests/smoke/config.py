from __future__ import annotations

import os
from typing import Optional


def require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return val


def get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    return os.getenv(name, default)


# Base URL for backend API
BASE_URL: str = os.getenv("BASE_URL", "http://127.0.0.1:9000")


# Exchange credentials (read-only here; tests will only verify presence and use public-safe endpoints)
COINBASE_API_KEY = os.getenv("COINBASE_API_KEY")
BINANCEUS_API_KEY = os.getenv("BINANCEUS_API_KEY")
KRAKEN_API_KEY = os.getenv("KRAKEN_API_KEY")
COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY")

# Optional Redis URL for environments that wire it
REDIS_URL = os.getenv("REDIS_URL")


