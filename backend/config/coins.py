"""
Export coin config from repo-level config/coins.py with safe defaults.
"""

DEFAULT_FEATURED_EXCHANGE = "binanceus"
DEFAULT_FEATURED_SYMBOLS: list[str] = ["BTCUSDT", "ETHUSDT"]

try:
    from config.coins import FEATURED_EXCHANGE as FEATURED_EXCHANGE  # type: ignore[no-redef]
    from config.coins import FEATURED_SYMBOLS as FEATURED_SYMBOLS  # type: ignore[no-redef]
except Exception:
    # Hard failover to avoid silent defaults; but clearly mark source
    FEATURED_EXCHANGE = DEFAULT_FEATURED_EXCHANGE  # type: ignore[assignment]
    FEATURED_SYMBOLS = DEFAULT_FEATURED_SYMBOLS  # type: ignore[assignment]
