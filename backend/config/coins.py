"""Stable re-export of feature flags for dashboard.

Allows imports such as `from backend.config.coins import FEATURED_EXCHANGE, FEATURED_SYMBOLS`
while keeping the single source of truth in the repo-level `config/coins.py`.
"""

try:
    # Primary source of truth
    from config.coins import FEATURED_EXCHANGE, FEATURED_SYMBOLS  # type: ignore
except Exception as e:  # pragma: no cover - safety fallback
    # Conservative defaults if root config fails to import
    FEATURED_EXCHANGE = "binanceus"
    FEATURED_SYMBOLS = ["BTCUSDT", "ETHUSDT"]


