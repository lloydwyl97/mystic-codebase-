import os

import pytest
import requests


def _backend() -> str | None:
    return os.getenv("MYSTIC_BACKEND")


@pytest.mark.skipif(_backend() is None, reason="MYSTIC_BACKEND not set")
def test_canonical_candles_200_and_cache_header():
    base = _backend().rstrip("/")  # type: ignore[union-attr]
    url = f"{base}/api/market/candles"
    r = requests.get(url, params={"symbol": "BTCUSDT", "interval": "1h", "limit": 1}, timeout=15)
    assert r.status_code == 200, f"expected 200 from {url}, got {r.status_code}: {r.text}"
    # Cache header present and short TTL
    cc = r.headers.get("Cache-Control", "")
    assert "max-age=15" in cc or "max-age=\n15" in cc, f"Cache-Control max-age=15 missing, got: {cc!r}"
    # Body shape is list (normalized OHLCV rows)
    data = r.json()
    assert isinstance(data, list), f"Expected list body, got {type(data)}"


@pytest.mark.skipif(_backend() is None, reason="MYSTIC_BACKEND not set")
def test_alias_candles_removed_flag_false():
    base = _backend().rstrip("/")  # type: ignore[union-attr]
    url = f"{base}/api/market/candles"
    r = requests.get(url, params={"symbol": "BTCUSDT", "interval": "1h", "limit": 1}, timeout=15)
    assert r.status_code == 200, f"expected 200 from {url}, got {r.status_code}: {r.text}"


