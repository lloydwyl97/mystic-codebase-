import os
from typing import Any, Dict, List, Optional, Mapping, Sequence, cast

import requests


# ----------------- public API -----------------
def to_dash_symbol(symbol: str) -> str:
    """
    Convert common exchange symbols to dashboard format.
    Examples:
      "BTCUSDT" -> "BTC-USD"
      "ETHUSD"  -> "ETH-USD"
      "ada-usdt" -> "ADA-USD"
    """
    s = str(symbol or "").upper().replace(" ", "").replace("_", "").replace("-", "")
    if not s:
        return ""

    # Explicit top mappings (extendable)
    explicit: Dict[str, str] = {
        "BTCUSDT": "BTC-USD",
        "ETHUSDT": "ETH-USD",
        "BNBUSDT": "BNB-USD",
        "ADAUSDT": "ADA-USD",
        "SOLUSDT": "SOL-USD",
        "XRPUSDT": "XRP-USD",
        "DOGEUSDT": "DOGE-USD",
        "DOTUSDT": "DOT-USD",
        "MATICUSDT": "MATIC-USD",
        "LTCUSDT": "LTC-USD",
        "SHIBUSDT": "SHIB-USD",
        # USD-quoted direct
        "BTCUSD": "BTC-USD",
        "ETHUSD": "ETH-USD",
        "BNBUSD": "BNB-USD",
        "ADAUSD": "ADA-USD",
        "SOLUSD": "SOL-USD",
        "XRPUSD": "XRP-USD",
        "DOGEUSD": "DOGE-USD",
        "DOTUSD": "DOT-USD",
        "MATICUSD": "MATIC-USD",
        "LTCUSD": "LTC-USD",
        "SHIBUSD": "SHIB-USD",
    }
    if s in explicit:
        return explicit[s]

    # Generic rule: map XXXUSDT or XXXUSD -> XXX-USD; otherwise, try to split base/quote
    if s.endswith("USDT"):
        base = s[:-4]
        return f"{base}-USD"
    if s.endswith("USD"):
        base = s[:-3]
        return f"{base}-USD"

    # If no common quote, best-effort: insert dash before last 3 or 4 if that looks like a fiat/known suffix
    for q in ("USD", "USDT", "USDC", "EUR", "GBP"):
        if s.endswith(q):
            base = s[: -len(q)]
            # Normalize all to USD for dashboard unless explicitly non-USD
            mapped_quote = "USD" if q in ("USD", "USDT", "USDC") else q
            return f"{base}-{mapped_quote}"

    # Fallback: if already contained a dash originally, return upper-dashed form
    if "-" in str(symbol):
        parts = str(symbol).replace(" ", "").replace("_", "-").upper().split("-")
        if len(parts) >= 2:
            return f"{parts[0]}-USD" if parts[1] in ("USDT", "USDC") else f"{parts[0]}-{parts[1]}"

    # Last resort: return as base-USD with entire input as base
    return f"{s}-USD"


def fetch_candles(exchange: str = "binanceus", symbol: str = "BTCUSDT", interval: str = "1h") -> Dict[str, Any]:
    """
    Fetch candles from the backend and normalize into a stable shape.

    Returns dict:
      {
        "symbol": "BTC-USD",
        "interval": "1h",
        "candles": [ {"ts": int, "o": float, "h": float, "l": float, "c": float, "v": float}, ... ],
        "raw": <full JSON or error info>
      }
    Never raises on missing keys; uses empty list on errors.
    """
    api_base = os.getenv("MYSTIC_BACKEND", "http://127.0.0.1:9000").rstrip("/")
    pair = to_dash_symbol(symbol)
    params = {"exchange": str(exchange or "").lower(), "symbol": pair, "interval": str(interval or "1h")}

    # Some deployments mount FastAPI at /api and also define routes with /api/... producing /api/api/...
    # Try that first as requested, then fall back to single /api/market/candles.
    candidates = ["/api/api/market/candles", "/api/market/candles"]

    last_status: Optional[int] = None
    last_json: Any = None
    last_url: Optional[str] = None
    try:
        for path in candidates:
            try:
                last_url = f"{api_base}{path}"
                r = requests.get(last_url, params=params, timeout=6)
                last_status = r.status_code
                # Accept any 2xx code
                if 200 <= r.status_code < 300:
                    data = _safe_json(r)
                    last_json = data
                    candles = _normalize_candles(data)
                    return {
                        "symbol": pair,
                        "interval": params["interval"],
                        "candles": candles,
                        "raw": data,
                    }
            except requests.RequestException as _:
                # try next candidate
                continue

        # No successful response
        return {
            "symbol": pair,
            "interval": params["interval"],
            "candles": [],
            "raw": {"error": "request_failed", "status": last_status, "response": last_json, "route": last_url},
        }

    except Exception as e:
        return {
            "symbol": pair,
            "interval": params["interval"],
            "candles": [],
            "raw": {"error": str(e), "status": last_status, "response": last_json, "route": last_url},
        }


def safe_number_format(value: Any, decimals: int = 2) -> str:
    """
    Format numbers with thousands separators and fixed decimals.
    Returns a string; handles None or non-numeric inputs gracefully.
    """
    try:
        if value is None or value == "":
            return f"{0:.{decimals}f}"
        # Permit strings like "123.45"
        num = float(value)
        return f"{num:,.{decimals}f}"
    except Exception:
        return f"{0:.{decimals}f}"


# ----------------- internals -----------------
def _safe_json(resp: requests.Response) -> Any:
    try:
        return resp.json()
    except Exception:
        return None


def _to_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def _extract_ts(item: Mapping[str, Any]) -> Optional[int]:
    for k in ("ts", "timestamp", "time", "t", "open_time", "openTime"):
        v = item.get(k)
        if isinstance(v, (int, float)):
            return int(v)
        if isinstance(v, str):
            try:
                return int(float(v))
            except Exception:
                continue
    return None


def _normalize_candles(data: Any) -> List[Dict[str, Any]]:
    """Best-effort normalization across several response shapes."""
    if not data:
        return []

    # Shape A: top-level dict with key "candles"
    if isinstance(data, dict):
        data_dict: Dict[str, Any] = cast(Dict[str, Any], data)
        candles_obj: Any = data_dict.get("candles")
        if isinstance(candles_obj, list):
            return _normalize_candle_list(cast(Sequence[Any], candles_obj))

    # Shape B: data itself is a list
    if isinstance(data, list):
        return _normalize_candle_list(cast(Sequence[Any], data))

    # Unknown shape
    return []


def _normalize_candle_list(items: Sequence[Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for it in items or []:
        # Dict-like candles
        if isinstance(it, dict):
            d = cast(Dict[str, Any], it)
            ts = _extract_ts(d)
            o = d.get("o", d.get("open"))
            h = d.get("h", d.get("high"))
            l = d.get("l", d.get("low"))
            c = d.get("c", d.get("close"))
            v = d.get("v", d.get("volume"))
            if ts is None:
                # some feeds provide close time only
                for alt in ("close_time", "closeTime"):
                    try:
                        ts_val = d.get(alt)
                        ts = int(float(cast(Any, ts_val)))
                        break
                    except Exception:
                        ts = None
                if ts is None:
                    continue
            out.append({
                "ts": int(ts),
                "o": _to_float(o),
                "h": _to_float(h),
                "l": _to_float(l),
                "c": _to_float(c),
                "v": _to_float(v),
            })
            continue

        # List/tuple-like candles: try common positions
        if isinstance(it, (list, tuple)):
            seq = list(cast(Sequence[Any], it))
            if len(seq) < 6:
                continue
            # Try binance Kline [openTime, open, high, low, close, volume, closeTime, ...]
            try:
                ts = int(float(seq[0]))
                o = _to_float(seq[1])
                h = _to_float(seq[2])
                l = _to_float(seq[3])
                c = _to_float(seq[4])
                v = _to_float(seq[5])
                out.append({"ts": ts, "o": o, "h": h, "l": l, "c": c, "v": v})
                continue
            except Exception:
                pass

            # Try shape [ts, o, h, l, c, v] directly
            try:
                ts = int(float(seq[0]))
                o, h, l, c, v = map(_to_float, seq[1:6])
                out.append({"ts": ts, "o": o, "h": h, "l": l, "c": c, "v": v})
                continue
            except Exception:
                pass

    return out


