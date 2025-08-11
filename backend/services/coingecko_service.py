from __future__ import annotations
from typing import Any, Optional, Dict

# Prefer delegating to the existing client if present
try:
    from backend.exchanges.coingecko_client import CoinGeckoClient as _CGClient  # type: ignore[import-not-found]
except Exception:
    try:
        from exchanges.coingecko_client import CoinGeckoClient as _CGClient  # type: ignore[import-not-found]
    except Exception:
        _CGClient = None  # type: ignore[assignment]


class CoingeckoService:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._client = _CGClient() if _CGClient is not None else None

    def is_ready(self) -> bool:
        return self._client is not None

    # Optional convenience used by legacy code paths; safe no-op default
    def get_simple_price(self, symbol: str, vs: str = "usd") -> Optional[float]:
        if self._client is None:
            return None
        try:
            # Our client typically returns a dict mapping id -> ticker or price info
            data = self._client.get_simple_price([symbol])  # type: ignore[arg-type]
            if isinstance(data, dict):
                sym = symbol.lower()
                curr = vs.lower()
                inner: Dict[str, Any] = data.get(sym, {})  # type: ignore[assignment]
                value = inner.get(curr)
                return float(value) if value is not None else None
            return None
        except Exception:
            return None

    # Some endpoints call this; provide a harmless default
    async def get_global_data(self) -> Dict[str, Any]:
        if self._client is None:
            return {}
        # No direct mapping in client; return empty to be safe
        return {}

# Alias to satisfy import name used by endpoints
CoinGeckoService = CoingeckoService

__all__ = ["CoingeckoService", "CoinGeckoService"]


