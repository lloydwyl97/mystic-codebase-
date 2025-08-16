import time
from typing import List, Tuple
from mystic_ui.api_client import get_candles

# Curated BinanceUS pairs (hyphen format)
SEED = [
    "BTC-USD","ETH-USD","BNB-USD","SOL-USD","ADA-USD",
    "DOGE-USD","AVAX-USD","MATIC-USD","LTC-USD","ATOM-USD",
    "XRP-USD","TRX-USD","LINK-USD","BCH-USD","DOT-USD",
    "NEAR-USD","ETC-USD","XLM-USD","FIL-USD","ALGO-USD"
]

_cache = {"ts": 0, "syms": []}

def _score_volume_24h(candles: List[dict]) -> float:
    # Use whatever interval backend returns (often 1h). Sum last 24 entries.
    window = candles[-24:] if len(candles) >= 24 else candles
    return float(sum((c.get("volume") or 0) for c in window))

def resolve_top10(timeframe: str = "1h", limit: int = 300) -> List[str]:
    global _cache
    now = time.time()
    if _cache["syms"] and (now - _cache["ts"] < 300):  # 5 min cache
        return _cache["syms"]

    scored: List[Tuple[str, float]] = []
    for sym in SEED:
        res = get_candles(symbol=sym, timeframe=timeframe, limit=limit, exchange="binanceus")
        candles = res.get("candles", [])
        if candles:
            scored.append((sym, _score_volume_24h(candles)))

    scored.sort(key=lambda x: x[1], reverse=True)
    top = [s for s, _ in scored[:10]]
    if top:
        _cache = {"ts": now, "syms": top}
    return top or SEED[:10]
