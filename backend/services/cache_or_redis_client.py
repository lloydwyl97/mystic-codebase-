from __future__ import annotations

from typing import Any, Dict

_singleton_cache: Any | None = None


def _build_fallback_cache() -> Any:
    class _DataCache:
        def __init__(self) -> None:
            self.binance: Dict[str, Any] = {}
            self.coinbase: Dict[str, Any] = {}
            self.coingecko: Dict[str, Any] = {}
            self.last_update: Dict[str, Any] = {}

    return _DataCache()


def get_cache() -> Any:  # noqa: ANN401
    global _singleton_cache
    if _singleton_cache is not None:
        return _singleton_cache
    try:
        # Prefer the AI poller's cache if available
        from ai.ai.poller import get_cache as _ai_get_cache  # type: ignore[import-not-found]

        _singleton_cache = _ai_get_cache()
    except Exception:
        _singleton_cache = _build_fallback_cache()
    return _singleton_cache


