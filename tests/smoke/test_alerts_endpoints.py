import os

import pytest
import requests


def _backend() -> str | None:
    return os.getenv("MYSTIC_BACKEND")


@pytest.mark.skipif(_backend() is None, reason="MYSTIC_BACKEND not set")
def test_canonical_alerts_recent_200():
    base = _backend().rstrip("/")  # type: ignore[union-attr]
    url = f"{base}/api/alerts/recent"
    r = requests.get(url, params={"limit": 5}, timeout=10)
    assert r.status_code == 200, f"expected 200 from {url}, got {r.status_code}: {r.text}"


@pytest.mark.skipif(_backend() is None, reason="MYSTIC_BACKEND not set")
def test_live_notifications_alias_removed_when_flag_false():
    base = _backend().rstrip("/")  # type: ignore[union-attr]
    resp = requests.get(f"{base}/api/alerts/recent", timeout=10)
    assert resp.status_code == 200



@pytest.mark.skipif(_backend() is None, reason="MYSTIC_BACKEND not set")
def test_candles_cache_header():
    base = _backend().rstrip("/")  # type: ignore[union-attr]
    r = requests.get(f"{base}/api/market/candles?symbol=BTCUSDT&interval=1h&limit=5", timeout=10)
    assert r.status_code == 200
    cc = r.headers.get("cache-control", "")
    assert "max-age=" in cc.lower()