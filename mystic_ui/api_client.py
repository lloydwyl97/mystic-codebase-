import os, json, time
from typing import Any, Dict, List, Optional
import requests

BACKEND = os.environ.get("MYSTIC_BACKEND", "http://127.0.0.1:9000").rstrip("/")
SESSION = requests.Session()

def _get(url: str, params: Dict[str, Any], timeout: int = 10) -> Dict[str, Any]:
    try:
        r = SESSION.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"ok": False, "error": str(e), "url": url, "params": params, "timestamp": int(time.time())}

def get_candles(
    symbol: str,
    timeframe: str = "1h",
    limit: int = 300,
    exchange: str = "binanceus",
) -> Dict[str, Any]:
    """
    Calls /api/api/market/candles (this is the one you confirmed returns data)
    NOTE: symbol MUST be hyphen-format on your backend (e.g., BTC-USD, ETH-USD)
    """
    url = f"{BACKEND}/api/api/market/candles"
    params = {
        "exchange": exchange,
        "symbol": symbol,       # REQUIRED (ticker/interval won't satisfy validator)
        "timeframe": timeframe, # '1m','5m','1h','4h','1d' (backend may return 1h regardless; we still pass it)
        "limit": limit,
    }
    data = _get(url, params)
    # Normalize shape -> always return dict with 'candles' list
    candles = data.get("candles") if isinstance(data, dict) else None
    if not isinstance(candles, list):
        candles = []
    return {
        "ok": True if candles else False,
        "symbol": data.get("symbol", symbol),
        "interval": data.get("interval", timeframe),
        "candles": candles,
        "live_data": data.get("live_data", False),
        "raw": data,
        "url": url,
        "params": params,
    }
