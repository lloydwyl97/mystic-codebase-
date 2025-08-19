import os

import httpx

BASE = os.getenv("MYSTIC_BACKEND", "http://127.0.0.1:9000").rstrip("/")


def _normalize_alerts(payload):
    if isinstance(payload, dict) and "alerts" in payload and isinstance(payload["alerts"], list):
        return payload["alerts"]
    if isinstance(payload, list):
        return payload
    return []


def test_alerts_recent_contract():
    with httpx.Client(timeout=10.0) as c:
        r = c.get(f"{BASE}/api/alerts/recent")
        assert r.status_code == 200
        alerts = _normalize_alerts(r.json())
        assert isinstance(alerts, list)
        # shape is loose; if items exist, ensure dict-like
        if alerts:
            assert isinstance(alerts[0], dict)


