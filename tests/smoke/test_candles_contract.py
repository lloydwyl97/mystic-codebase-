import math
import os

import httpx

BASE = os.getenv("MYSTIC_BACKEND", "http://127.0.0.1:9000").rstrip("/")


def test_candles_contract():
    with httpx.Client(timeout=10.0) as c:
        r = c.get(f"{BASE}/api/market/candles", params={"symbol":"BTCUSDT","interval":"1h","limit":50})
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list)
        if not data:
            return  # allow empty but correct
        row = data[0]
        for k in ("timestamp","open","high","low","close","volume"):
            assert k in row
        # types
        assert isinstance(row["timestamp"], int)
        for k in ("open","high","low","close","volume"):
            assert isinstance(row[k], (int, float)) and not math.isnan(float(row[k]))
        # monotonic timestamps (ascending)
        ts = [r["timestamp"] for r in data if "timestamp" in r]
        assert ts == sorted(ts)
        # milliseconds
        assert ts[0] > 10_000_000_000  # > ~2001 in seconds; so must be ms
        # limit respected
        assert len(data) <= 50
        # cache header present
        cc = r.headers.get("cache-control","" ).lower()
        assert "max-age=" in cc


