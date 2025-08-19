from __future__ import annotations

import os
import typing as t

import pytest
from fastapi.testclient import TestClient


def _get_client() -> TestClient:
    from backend.app_factory import create_app

    app = create_app()
    return TestClient(app)


_RUN_TRADING = os.environ.get("RUN_TRADING_ACTION_TESTS", "0").lower() in {"1", "true", "yes"}


@pytest.mark.skipif(not _RUN_TRADING, reason="Live trading action tests disabled by default")
def test_trading_start_stop_alias() -> None:
    client = _get_client()
    r1 = client.post("/trading/start")
    assert r1.status_code in (200, 202)
    assert isinstance(r1.json(), dict)

    r2 = client.post("/trading/stop")
    assert r2.status_code in (200, 202)
    assert isinstance(r2.json(), dict)


@pytest.mark.skipif(not _RUN_TRADING, reason="Live trading action tests disabled by default")
def test_trading_cancel_all_alias() -> None:
    client = _get_client()
    r = client.post("/trading/cancel-all")
    assert r.status_code in (200, 202)
    data: t.Dict[str, t.Any] = r.json()
    assert isinstance(data, dict)
    assert data.get("ok") in (True, False)


