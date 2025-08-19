from __future__ import annotations

import typing as t

from fastapi.testclient import TestClient


def _get_client() -> TestClient:
    from backend.app_factory import create_app

    app = create_app()
    return TestClient(app)


def test_portfolio_transactions() -> None:
    client = _get_client()
    r = client.get("/api/portfolio/transactions")
    assert r.status_code == 200
    data: dict[str, t.Any] = r.json()
    assert isinstance(data, dict)
    assert isinstance(data.get("items"), list)
    assert isinstance(data.get("count"), int)


def test_ai_leaderboard() -> None:
    client = _get_client()
    r = client.get("/api/ai/strategies/leaderboard")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)


def test_websocket_status() -> None:
    client = _get_client()
    r = client.get("/api/websocket/status")
    assert r.status_code == 200
    data: dict[str, t.Any] = r.json()
    assert isinstance(data, dict)
    assert "websocket" in data


def test_whale_alerts_get_empty_ok() -> None:
    client = _get_client()
    r = client.get("/api/whale/alerts")
    assert r.status_code == 200
    data: dict[str, t.Any] = r.json()
    assert isinstance(data.get("alerts"), list)


def test_whale_alerts_ingest_and_read() -> None:
    client = _get_client()
    one = {"symbol": "BTCUSDT", "amount": 1_000_000, "tx": "abc123"}
    r_ingest = client.post("/api/whale/alerts/ingest", json=one)
    assert r_ingest.status_code == 200
    assert r_ingest.json().get("ok") is True

    r_get = client.get("/api/whale/alerts", params={"limit": 5})
    assert r_get.status_code == 200
    data = r_get.json()
    assert isinstance(data.get("alerts"), list)
    # Not asserting count strictly due to TTL and prior state, but shape must be valid


def test_whale_alerts_ingest_bulk() -> None:
    client = _get_client()
    bulk = {
        "alerts": [
            {"symbol": "ETHUSDT", "amount": 500_000, "tx": "tx1"},
            {"symbol": "SOLUSDT", "amount": 300_000, "tx": "tx2"},
        ]
    }
    r_ingest = client.post("/api/whale/alerts/ingest_bulk", json=bulk)
    assert r_ingest.status_code == 200
    data = r_ingest.json()
    assert data.get("ok") is True
    assert isinstance(data.get("ingested"), int)


