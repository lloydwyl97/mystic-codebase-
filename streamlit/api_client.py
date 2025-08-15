import os, requests, functools
from typing import Dict, Any, Optional, Sequence

API = os.environ.get("MYSTIC_BACKEND", "http://127.0.0.1:9000")

@functools.lru_cache(maxsize=1)
def _discover_prefix() -> str:
    """
    Ask the backend for openapi.json and infer the base path.
    Falls back through common prefixes.
    """
    candidates = ["", "/api", "/api/v1", "/v1"]
    for base in candidates:
        try:
            r = requests.get(f"{API}{base}/openapi.json", timeout=2)
            if r.ok and isinstance(r.json(), dict):
                return base
        except Exception:
            pass
    return ""  # best effort


def _get(url: str, params: Dict[str, Any]) -> Dict[str, Optional[Any]]:
    try:
        r = requests.get(url, params=params, timeout=5)
        if r.ok:
            return {"data": r.json()}
    except Exception:
        pass
    return {"data": None}


def dc_get_ohlcv(exchange: str, symbol: str, timeframe: str = "1m", limit: int = 300):
    base = _discover_prefix()
    # conservative set of shapes; *prefix* is the important part
    paths = [
        f"{API}{base}/market/ohlcv",
        f"{API}{base}/markets/ohlcv",
        f"{API}{base}/ohlcv",
        f"{API}{base}/charts/ohlcv",
        f"{API}{base}/live/market/historical",  # legacy
    ]
    # try the three common param names each backend uses
    param_variants = [
        {"symbol": symbol, "timeframe": timeframe, "limit": limit, "exchange": exchange},
        {"ticker": symbol, "interval": timeframe, "limit": limit, "exchange": exchange},
        {"symbol": symbol, "tf": timeframe, "limit": limit, "exchange": exchange},
    ]
    for p in paths:
        for pv in param_variants:
            res = _get(p, pv)
            if res["data"]:          # got something non-empty
                return res
    return {"data": None}


def dc_get_trades(exchange: str, symbol: str, limit: int = 50):
    base = _discover_prefix()
    paths = [
        f"{API}{base}/market/trades",
        f"{API}{base}/markets/trades",
        f"{API}{base}/trades",
        f"{API}{base}/live/trading/trades",     # legacy
    ]
    param_variants = [
        {"symbol": symbol, "limit": limit, "exchange": exchange},
        {"ticker": symbol, "limit": limit, "exchange": exchange},
        {"pair": symbol,   "limit": limit, "exchange": exchange},
    ]
    for p in paths:
        for pv in param_variants:
            res = _get(p, pv)
            if res["data"]:
                return res
    return {"data": None}


def dc_get_prices(symbols: Sequence[str], exchange: Optional[str] = None) -> Dict[str, Optional[Any]]:
    base = _discover_prefix()
    syms = [str(s).upper() for s in symbols]
    if not syms:
        return {"data": None}
    params = {"symbols": ",".join(syms)}
    if exchange:
        params["exchange"] = exchange
    paths = [
        f"{API}{base}/market/prices",
        f"{API}{base}/markets/prices",
        f"{API}{base}/prices",
    ]
    for p in paths:
        res = _get(p, params)
        if res["data"]:
            return res
    return {"data": None}


def dc_health() -> Dict[str, Optional[Any]]:
    base = _discover_prefix()
    paths = [
        f"{API}{base}/system/health-check",
        f"{API}{base}/system/health",
        f"{API}{base}/health",
        f"{API}{base}/health-check",
    ]
    for p in paths:
        res = _get(p, {})
        if res["data"]:
            return res
    return {"data": None}


def dc_autobuy_status() -> Dict[str, Optional[Any]]:
    base = _discover_prefix()
    paths = [
        f"{API}{base}/autobuy/status",
        f"{API}{base}/ai/autobuy/status",
        f"{API}{base}/autobuy/health",
    ]
    for p in paths:
        res = _get(p, {})
        if res["data"]:
            return res
    return {"data": None}


def dc_ai_heartbeat() -> Dict[str, Optional[Any]]:
    base = _discover_prefix()
    paths = [
        f"{API}{base}/ai/heartbeat",
        f"{API}{base}/api/ai/heartbeat",
    ]
    for p in paths:
        res = _get(p, {})
        if res["data"]:
            return res
    return {"data": None}


def dc_get_autobuy_signals(limit: int = 50) -> Dict[str, Optional[Any]]:
    base = _discover_prefix()
    # Prefer explicit autobuy signals; include common prefixed variants
    paths = [
        f"{API}{base}/autobuy/signals",
        f"{API}{base}/api/autobuy/signals",
        f"{API}{base}/live/trading/signals",  # fallback: generic live signals
        f"{API}{base}/api/live/trading/signals",
    ]
    params = {"limit": limit}
    for p in paths:
        res = _get(p, params)
        if res["data"]:
            return res
    return {"data": None}


def dc_get_orders() -> Dict[str, Optional[Any]]:
    base = _discover_prefix()
    paths = [
        f"{API}{base}/orders",
        f"{API}{base}/api/orders",
        f"{API}{base}/trading/orders",
        f"{API}{base}/live/trading/orders",
    ]
    for p in paths:
        res = _get(p, {})
        if res["data"]:
            return res
    return {"data": None}


def dc_get_portfolio_overview() -> Dict[str, Optional[Any]]:
    base = _discover_prefix()
    paths = [
        f"{API}{base}/portfolio/overview",
        f"{API}{base}/api/portfolio/overview",
        f"{API}{base}/portfolio/live",
        f"{API}{base}/api/portfolio/live",
    ]
    for p in paths:
        res = _get(p, {})
        if res["data"]:
            return res
    return {"data": None}


def dc_get_portfolio_positions() -> Dict[str, Optional[Any]]:
    base = _discover_prefix()
    paths = [
        f"{API}{base}/portfolio/positions",
        f"{API}{base}/api/portfolio/positions",
    ]
    for p in paths:
        res = _get(p, {})
        if res["data"]:
            return res
    return {"data": None}
