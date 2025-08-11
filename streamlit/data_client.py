from __future__ import annotations

"""Central data client for Streamlit dashboard.

Provides a canonical facade over the legacy `dashboard.data_client` so that all
pages import from `streamlit.data_client` and use a single BASE_URL.
"""

import os

# Single source of truth for backend base URL
BASE_URL = os.getenv("DASHBOARD_BASE_URL", "http://127.0.0.1:9000").rstrip("/")
if not globals().get("_BASE_URL_LOGGED"):
    print(f"[streamlit] BASE_URL={BASE_URL}")
    _BASE_URL_LOGGED = True  # type: ignore

# Wire the underlying client to use BASE_URL by setting env var expected by legacy client
os.environ.setdefault("BACKEND_URL", BASE_URL)

try:
    # Prefer legacy dashboard client already used across pages
    from dashboard import data_client as _core  # type: ignore
except Exception:  # pragma: no cover
    # Fallback to the copy inside pages if needed
    from streamlit.pages.components import data_client as _core  # type: ignore

# Re-export the public API expected by pages
from typing import Any, Dict, List, Optional  # noqa: E402

FetchResult = _core.FetchResult  # type: ignore[attr-defined]

get_prices = _core.get_prices
get_ticker = _core.get_ticker
get_ohlcv = _core.get_ohlcv
get_orderbook = _core.get_orderbook
get_trades = _core.get_trades
get_balances = _core.get_balances
get_ai_signals = _core.get_ai_signals
get_autobuy_heartbeat = _core.get_autobuy_heartbeat
get_ai_heartbeat = _core.get_ai_heartbeat
get_autobuy_status = _core.get_autobuy_status
get_autobuy_signals = _core.get_autobuy_signals
get_autobuy_decision = getattr(_core, "get_autobuy_decision", lambda symbol: _core.get_autobuy_signals(1))
start_autobuy = _core.start_autobuy
stop_autobuy = _core.stop_autobuy
system_health = _core.system_health
get_health_check = _core.get_health_check
advanced_events = _core.advanced_events
advanced_performance = _core.advanced_performance
get_portfolio_overview = _core.get_portfolio_overview
get_trading_orders = _core.get_trading_orders
get_risk_alerts = _core.get_risk_alerts
get_market_liquidity = _core.get_market_liquidity
get_analytics_performance = _core.get_analytics_performance
get_alerts = _core.get_alerts
compute_spread_from_price_entry = _core.compute_spread_from_price_entry
clear_cache = _core.clear_cache

__all__ = [
    "BASE_URL",
    "FetchResult",
    "fetch_api",
    "post_api",
    "get_prices",
    "get_ticker",
    "get_ohlcv",
    "get_orderbook",
    "get_trades",
    "get_balances",
    "get_ai_signals",
    "get_autobuy_heartbeat",
    "get_ai_heartbeat",
    "get_autobuy_status",
    "get_autobuy_signals",
    "get_autobuy_decision",
    "start_autobuy",
    "stop_autobuy",
    "system_health",
    "get_health_check",
    "advanced_events",
    "advanced_performance",
    "get_portfolio_overview",
    "get_trading_orders",
    "get_risk_alerts",
    "get_market_liquidity",
    "get_analytics_performance",
    "get_alerts",
    "compute_spread_from_price_entry",
    "clear_cache",
]

# Generic helpers for hub pages that still need raw endpoints
def fetch_api(endpoint: str):
    from streamlit.api_client import api_client as _api  # local import to avoid cycles
    return _api.fetch_api_data(endpoint)


def post_api(endpoint: str, data: dict):
    from streamlit.api_client import api_client as _api
    return _api.post_api_data(endpoint, data)


