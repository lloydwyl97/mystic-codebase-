from __future__ import annotations

from typing import Any

try:
    from backend.exchanges.coinbase_adapter import CoinbaseAdapter as _Adapter  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - graceful fallback if adapter missing
    _Adapter = None  # type: ignore[assignment]


class CoinbaseDataFetcher:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if _Adapter is not None:
            try:
                self._client = _Adapter()
            except Exception:
                self._client = None
        else:
            self._client = None

    def is_ready(self) -> bool:
        return self._client is not None


# Legacy alias expected by old imports
CoinbaseData = CoinbaseDataFetcher
__all__ = ["CoinbaseData", "CoinbaseDataFetcher"]




