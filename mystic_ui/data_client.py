import os
from collections.abc import Sequence
from typing import Any

import requests

API = os.getenv("MYSTIC_BACKEND", "http://127.0.0.1:8000").rstrip("/")

# ----------------- helpers -----------------
def _to_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default

# Note: direct requests.get is used in _first_ok to carry metadata; _req kept for compatibility if needed
# Note: direct requests.get is used in _first_ok to carry metadata

def _first_ok(candidates: Sequence[tuple[str, dict[str, Any] | None]]) -> Any | None:
    """
    Try each (path, params) until one returns HTTP 200.
    - On success: return parsed JSON payload
    - On failure: return minimal metadata dict: {"__meta__": {"route": url, "status": code|None, "error": str|None}}
    """
    last_status: int | None = None
    last_route: str | None = None
    last_error: str | None = None
    for path, params in candidates:
        url = f"{API}/{path.lstrip('/')}"
        try:
            r = requests.get(url, params=params, timeout=6)
            last_status = int(r.status_code)
            last_route = url
            if r.status_code == 200:
                try:
                    return r.json()
                except Exception:
                    return {"__meta__": {"route": url, "status": last_status, "error": "invalid_json"}}
        except requests.RequestException as e:
            last_route = url
            last_error = str(e)
            continue
    # Nothing succeeded: return metadata instead of None
    return {"__meta__": {"route": last_route, "status": last_status, "error": last_error or (f"http_{last_status}" if last_status else None)}}

# ----------------- symbol normalization -----------------
def _norm_symbol(sym: str, ex: str) -> str:
    s = str(sym).upper().replace("-", "")           # BTC-USD -> BTCUSD
    if ex == "binanceus" and s.endswith("USD") and not s.endswith("USDT"):
        s = s[:-3] + "USDT"                          # BTCUSD -> BTCUSDT
    return s

def _norm_symbols(symbols: Sequence[str] | None, ex: str) -> list[str]:
    return [] if not symbols else [_norm_symbol(str(x), ex) for x in symbols]

# ----------------- argument parsers -----------------
def _parse_exchange(default: str = "binanceus") -> str:
    return (os.getenv("DISPLAY_EXCHANGE") or default).lower()

def _parse_ohlcv_args(*args: Any, **kwargs: Any) -> tuple[str, str, str, int]:
    """
    Accepts:
      (symbol)
      (symbol, timeframe)
      (symbol, timeframe, limit)
      (exchange, symbol, timeframe)
      (exchange, symbol, timeframe, limit)
      or keyword args: exchange=, symbol=, timeframe=, limit=
    """
    ex: Any = kwargs.get("exchange", _parse_exchange())
    symbol: Any = kwargs.get("symbol")
    timeframe: Any = kwargs.get("timeframe")
    limit: Any = kwargs.get("limit")

    if symbol is None:
        if len(args) == 1:
            symbol = args[0]
        elif len(args) == 2:
            symbol, timeframe = args
        elif len(args) >= 3:
            ex, symbol, timeframe = args[:3]
            if len(args) >= 4:
                limit = args[3]

    if timeframe is None:
        timeframe = "1m"
    limit = _to_int(limit, 300)

    ex_str = str(ex).lower()
    symbol_str = _norm_symbol(str(symbol), ex_str)
    return ex_str, symbol_str, str(timeframe), int(limit)

def _parse_trades_args(*args: Any, **kwargs: Any) -> tuple[str, str, int]:
    """
    Accepts:
      (symbol)
      (symbol, limit)
      (exchange, symbol)
      (exchange, symbol, limit)
      or keyword args: exchange=, symbol=, limit=
    """
    ex: Any = kwargs.get("exchange", _parse_exchange())
    symbol: Any = kwargs.get("symbol")
    limit: Any = kwargs.get("limit")

    if symbol is None:
        if len(args) == 1:
            symbol = args[0]
        elif len(args) >= 2:
            # could be (symbol, limit) or (exchange, symbol)
            a0, a1 = args[0], args[1]
            # if second looks numeric, it's limit
            if isinstance(a1, (int, float)) or (isinstance(a1, str) and a1.isdigit()):
                symbol, limit = a0, a1
            else:
                ex, symbol = a0, a1
            if len(args) >= 3:
                limit = args[2]

    limit = _to_int(limit, 50)
    ex_str = str(ex).lower()
    symbol_str = _norm_symbol(str(symbol), ex_str)
    return ex_str, symbol_str, int(limit)

# ----------------- API wrappers with fallbacks -----------------
def get_health_check() -> Any | None:
    return _first_ok([
        ("/system/health-check", None),
        ("/system/health", None),
        ("/health", None),
        ("/health-check", None),
    ])

def get_autobuy_status() -> Any | None:
    return _first_ok([
        ("/autobuy/status", None),
        ("/ai/autobuy/status", None),
        ("/autobuy/health", None),
    ])

def get_prices(symbols: Sequence[str], exchange: str | None = None) -> Any | None:
    ex = (exchange or _parse_exchange()).lower()
    syms = _norm_symbols(symbols, ex)
    if not syms:
        return None
    p = {"symbols": ",".join(syms), "exchange": ex}
    return _first_ok([
        ("/market/prices", p),
        ("/markets/prices", p),
        ("/prices", p),
    ])

def get_ohlcv(*args: Any, **kwargs: Any) -> Any | None:
    ex, symbol, timeframe, limit = _parse_ohlcv_args(*args, **kwargs)
    # common param aliases
    p1 = {"symbol": symbol, "timeframe": timeframe, "limit": limit, "exchange": ex}
    p2 = {"symbol": symbol, "tf": timeframe, "limit": limit, "exchange": ex}
    p3 = {"ticker": symbol, "interval": timeframe, "limit": limit, "exchange": ex}

    return _first_ok([
        # path-param styles
        (f"/market/ohlcv/{symbol}", {"timeframe": timeframe, "limit": limit, "exchange": ex}),
        (f"/markets/ohlcv/{symbol}", {"timeframe": timeframe, "limit": limit, "exchange": ex}),
        (f"/ohlcv/{symbol}", {"timeframe": timeframe, "limit": limit, "exchange": ex}),
        (f"/live/market/historical/{symbol}", {"timeframe": timeframe, "limit": limit}),
        (f"/charts/ohlcv/{symbol}", {"timeframe": timeframe, "limit": limit, "exchange": ex}),
        # query-param styles
        ("/market/ohlcv", p1), ("/market/ohlcv", p2), ("/market/ohlcv", p3),
        ("/markets/ohlcv", p1), ("/ohlcv", p1), ("/charts/ohlcv", p1),
        ("/live/market/historical", p1),
    ])

def get_trades(*args: Any, **kwargs: Any) -> Any | None:
    ex, symbol, limit = _parse_trades_args(*args, **kwargs)
    p = {"symbol": symbol, "limit": limit, "exchange": ex}

    return _first_ok([
        # path-param styles
        (f"/trades/{symbol}", {"limit": limit, "exchange": ex}),
        (f"/trading/trades/{symbol}", {"limit": limit, "exchange": ex}),
        (f"/live/trading/trades/{symbol}", {"limit": limit}),
        (f"/market/trades/{symbol}", {"limit": limit, "exchange": ex}),
        (f"/recent_trades/{symbol}", {"limit": limit, "exchange": ex}),
        # query-param styles
        ("/trades", p), ("/trading/trades", p),
        ("/live/trading/trades", p),
        ("/market/trades", p), ("/recent_trades", p),
    ])


# ----------------- alerts and ai heartbeat -----------------
def get_ai_heartbeat() -> Any | None:
    return _first_ok([
        ("/ai/heartbeat", None),
        ("/api/ai/heartbeat", None),
    ])


def get_alerts(limit: int = 100) -> Any | None:
    """
    Fetch recent alerts/notifications from available backends.
    Tries multiple canonical paths and shapes.
    """
    p = {"limit": int(limit)}
    return _first_ok([
        # Prefer canonical recent alerts
        ("/api/alerts/recent", p),
        # Generic alerts collections (legacy)
        ("/api/alerts", None),
        ("/alerts", p),
    ])


# ----------------- features and system health -----------------
def get_features() -> Any | None:
    return _first_ok([
        ("/features", None),
        ("/api/features", None),
    ])


def get_system_health_basic() -> Any | None:
    """
    Lightweight server health ping suitable for liveness checks.
    """
    return _first_ok([
        ("/api/health", None),
        ("/health", None),
    ])


def get_system_health_detailed() -> Any | None:
    """
    Detailed system health including services and resources.
    """
    return _first_ok([
        ("/system/health", None),
        ("/health/comprehensive", None),
    ])

