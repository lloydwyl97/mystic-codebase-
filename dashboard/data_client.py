from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Type, cast, List

import streamlit as st

# Resolve API client without hard depending on Streamlit package path
try:  # Prefer Streamlit pages client when available
    from streamlit._pages.components.api_client import api_client as _api_client  # type: ignore[import-not-found]
except Exception:
    try:  # Fallback if running from Streamlit app root
        from _pages.components.api_client import api_client as _api_client  # type: ignore[import-not-found]
    except Exception:
        _api_client = None  # type: ignore[assignment]

if _api_client is None:
    # Minimal inline HTTP client using requests; mirrors the Streamlit client behavior
    import os
    import requests

    class _InlineAPIClient:
        def __init__(self) -> None:
            self.base_url = os.getenv("BACKEND_URL", "http://localhost:9000")
            self.timeout = 10
            self.session = requests.Session()
            self.session.headers.update({
                "Content-Type": "application/json",
                "User-Agent": "Mystic-Dashboard/2.0",
            })
            self.api_prefix = "/api"
            self.triple_api_prefix = self.api_prefix + self.api_prefix

        def _build_url(self, endpoint: str) -> str:
            if endpoint.startswith(self.triple_api_prefix + "/") or endpoint.startswith(self.api_prefix + "/"):
                return f"{self.base_url}{endpoint}"
            return f"{self.base_url}{self.api_prefix}{endpoint if endpoint.startswith('/') else '/' + endpoint}"

        def fetch_api_data(self, endpoint: str):  # noqa: ANN001
            try:
                url = self._build_url(endpoint)
                response = self.session.get(url, timeout=self.timeout)
                if response.status_code != 200:
                    # Try common fallbacks
                    fallback_paths = []
                    if endpoint.startswith("/trading/"):
                        fallback_paths.append("/live" + endpoint)
                    if not endpoint.startswith("/api/"):
                        fallback_paths.append("/api" + endpoint)
                        fallback_paths.append("/api/api" + endpoint)
                    if endpoint == "/analytics/performance":
                        fallback_paths += [
                            "/strategies/performance",
                            "/api/strategies/performance",
                        ]
                    if endpoint == "/risk/alerts":
                        fallback_paths += [
                            "/risk/metrics",
                            "/api/risk/metrics",
                        ]
                    if endpoint == "/market/liquidity":
                        fallback_paths += [
                            "/market/live",
                            "/api/market/live",
                        ]
                    for alt in fallback_paths:
                        r2 = self.session.get(self._build_url(alt), timeout=self.timeout)
                        if r2.status_code == 200:
                            payload2 = r2.json()
                            return payload2 if (isinstance(payload2, dict) and "data" in payload2) else {"data": payload2}
                if response.status_code == 200:
                    payload = response.json()
                    return payload if (isinstance(payload, dict) and "data" in payload) else {"data": payload}
                return None
            except requests.exceptions.RequestException:
                return None

        def post_api_data(self, endpoint: str, data):  # noqa: ANN001, D401
            try:
                url = self._build_url(endpoint)
                response = self.session.post(url, json=data, timeout=self.timeout)
                if response.status_code in (200, 201):
                    payload = response.json()
                    return payload if (isinstance(payload, dict) and "data" in payload) else {"data": payload}
                return None
            except requests.exceptions.RequestException:
                return None

    api_client = _InlineAPIClient()
else:
    api_client = _api_client
from .schemas import Ticker, OHLCV, OrderBook, Trade, Balance, AIHeartbeat, AlertItem, Candle


@dataclass
class FetchResult:
    data: Any
    latency_ms: int
    cache_age_s: Optional[int]
    status: str


def _request(method: str, endpoint: str, payload: Optional[Dict[str, Any]] = None, page: str = "dash", name: str = "call") -> Tuple[Optional[Dict[str, Any]], int]:
    start = time.perf_counter()
    delay = 0.2
    for _ in range(3):
        try:
            if method == "GET":
                resp = api_client.fetch_api_data(endpoint)  # type: ignore[no-untyped-call]
            elif method == "POST":
                resp = api_client.post_api_data(endpoint, payload or {})  # type: ignore[no-untyped-call]
            else:
                resp = None
            if resp is not None:
                latency = int((time.perf_counter() - start) * 1000)
                try:
                    print(f"[{page}] {name} endpoint={endpoint} latency={latency}ms status=ok")
                except Exception:
                    pass
                return cast(Optional[Dict[str, Any]], resp), latency
        except Exception:
            pass
        time.sleep(delay)
        delay = min(1.5, delay * 2)
    latency = int((time.perf_counter() - start) * 1000)
    try:
        print(f"[{page}] {name} endpoint={endpoint} latency={latency}ms status=fail")
    except Exception:
        pass
    return None, latency


def _extract(payload: Optional[Dict[str, Any]]) -> Any:
    if isinstance(payload, dict) and "data" in payload:
        return payload["data"]
    return payload


def _finalize(page: str, name: str, payload: Any, latency_ms: int, cache_age_s: Optional[int]) -> FetchResult:
    status = "ok" if payload is not None else "fail"
    st.write(f"[{page}] {name} latency={latency_ms}ms cache={cache_age_s or 0}s status={status}")
    try:
        print(f"[{page}] {name} latency={latency_ms}ms cache={cache_age_s or 0}s status={status}")
    except Exception:
        pass
    return FetchResult(data=payload, latency_ms=latency_ms, cache_age_s=cache_age_s, status=status)


def _validate(model: Type[Any], data: Any, toast_prefix: str) -> Any:
    try:
        if data is None:
            return None
        if model is Ticker:
            return model(**data)
        if model is OHLCV:
            if isinstance(data, dict) and ("data" in data or "candles" in data):
                return model(**cast(Dict[str, Any], data))
            if isinstance(data, list):
                return model(candles=[Candle(**c) for c in cast(List[Dict[str, Any]], data)])
            return None
        if model is OrderBook:
            return model(**cast(Dict[str, Any], data))
        if model is Trade:
            if isinstance(data, list):
                ll = cast(List[Dict[str, Any]], data)
                return [Trade(**t) for t in ll]
            return Trade(**cast(Dict[str, Any], data))
        if model is Balance:
            if isinstance(data, list):
                lb = cast(List[Dict[str, Any]], data)
                return [Balance(**b) for b in lb]
            return Balance(**cast(Dict[str, Any], data))
        if model is AIHeartbeat:
            return model(**cast(Dict[str, Any], data))
        if model is AlertItem:
            if isinstance(data, list):
                la = cast(List[Dict[str, Any]], data)
                return [AlertItem(**a) for a in la]
            return AlertItem(**cast(Dict[str, Any], data))
        return data
    except Exception as e:
        st.toast(f"Validation error: {toast_prefix}: {e}", icon="❌")
        return None


@st.cache_data(show_spinner=False, ttl=2)
def get_prices(symbols: list[str]) -> FetchResult:
    q = ",".join([s.upper().replace("/", "-") for s in symbols])
    payload, latency = _request("GET", f"/market/prices?symbols={q}", page="markets", name="get_prices")
    data = _extract(payload)
    return _finalize("dash", "get_prices", data, latency, 0)


@st.cache_data(show_spinner=False, ttl=2)
def get_ticker(exchange: str, symbol: str) -> FetchResult:
    qsym = symbol.upper().replace("/", "-")
    payload, latency = _request("GET", f"/market/prices?symbols={qsym}", page="markets", name=f"get_ticker {exchange}:{qsym}")
    data = _extract(payload)
    entry: Optional[Dict[str, Any]] = None
    if isinstance(data, dict):
        dct: Dict[str, Any] = cast(Dict[str, Any], data)
        prices: Dict[str, Any] = cast(Dict[str, Any], dct.get("prices", dct))
        entry = cast(Optional[Dict[str, Any]], prices.get(qsym))
    v = _validate(Ticker, entry, "ticker")
    out = v.model_dump() if hasattr(v, "model_dump") else v
    return _finalize("dash", "get_ticker", out, latency, 0)


@st.cache_data(show_spinner=False, ttl=2)
def get_ohlcv(exchange: str, symbol: str, timeframe: str) -> FetchResult:
    sym = symbol.upper().replace("/", "-")
    payload, latency = _request("GET", f"/live/market/historical/{sym}?timeframe={timeframe}&limit=300", page="markets", name=f"get_ohlcv {exchange}:{sym}:{timeframe}")
    data = _extract(payload)
    v = _validate(OHLCV, data, "ohlcv")
    out = v.model_dump() if hasattr(v, "model_dump") else v
    return _finalize("dash", "get_ohlcv", out, latency, 0)


@st.cache_data(show_spinner=False, ttl=2)
def get_orderbook(exchange: str, symbol: str, depth: int = 50) -> FetchResult:
    # If specific orderbook endpoint exists in backend, wire here; otherwise return empty
    payload, latency = _request("GET", f"/live/market/overview", page="markets", name=f"get_orderbook {exchange}:{symbol}:{depth}")
    data = _extract(payload)
    return _finalize("dash", "get_orderbook", data, latency, 0)


@st.cache_data(show_spinner=False, ttl=2)
def get_trades(exchange: str, symbol: str, limit: int = 100) -> FetchResult:
    sym = symbol.upper().replace("/", "-")
    payload, latency = _request("GET", f"/live/trading/trades?symbol={sym}&limit={limit}", page="markets", name=f"get_trades {exchange}:{sym}:{limit}")
    data = _extract(payload)
    v = _validate(Trade, data, "trades")
    if isinstance(v, list):
        out_list: List[Any] = []
        items: List[Any] = cast(List[Any], v)
        for t in items:
            try:
                out_list.append(t.model_dump() if hasattr(t, "model_dump") else t)  # type: ignore[attr-defined]
            except Exception:
                out_list.append(t)
        out = out_list
    else:
        out = v.model_dump() if hasattr(v, "model_dump") else v
    return _finalize("dash", "get_trades", out, latency, 0)


@st.cache_data(show_spinner=False, ttl=2)
def get_balances(exchange: str) -> FetchResult:
    payload, latency = _request("GET", "/live/trading/balance", page="portfolio", name=f"get_balances {exchange}")
    data = _extract(payload)
    v = _validate(Balance, data, "balances")
    if isinstance(v, list):
        out_list: List[Any] = []
        items: List[Any] = cast(List[Any], v)
        for b in items:
            try:
                out_list.append(b.model_dump() if hasattr(b, "model_dump") else b)  # type: ignore[attr-defined]
            except Exception:
                out_list.append(b)
        out = out_list
    else:
        out = v.model_dump() if hasattr(v, "model_dump") else v
    return _finalize("dash", "get_balances", out, latency, 0)


@st.cache_data(show_spinner=False, ttl=2)
def get_ai_signals(symbol: str, timeframe: str) -> FetchResult:
    payload, latency = _request("GET", "/ai/signals", page="signals", name=f"get_ai_signals {symbol}:{timeframe}")
    data = _extract(payload)
    return _finalize("dash", "get_ai_signals", data, latency, 0)


@st.cache_data(show_spinner=False, ttl=2)
def get_autobuy_heartbeat() -> FetchResult:
    payload, latency = _request("GET", "/autobuy/status", page="signals", name="get_autobuy_heartbeat")
    data = _extract(payload)
    return _finalize("dash", "get_autobuy_heartbeat", data, latency, 0)


@st.cache_data(show_spinner=False, ttl=2)
def get_ai_heartbeat() -> FetchResult:
    payload, latency = _request("GET", "/ai/heartbeat", page="signals", name="get_ai_heartbeat")
    data = _extract(payload)
    v = _validate(AIHeartbeat, data, "ai_heartbeat")
    out = v.model_dump() if hasattr(v, "model_dump") else v
    return _finalize("dash", "get_ai_heartbeat", out, latency, 0)


@st.cache_data(show_spinner=False, ttl=2)
def get_autobuy_status() -> FetchResult:
    payload, latency = _request("GET", "/autobuy/status", page="signals", name="get_autobuy_status")
    data = _extract(payload)
    return _finalize("dash", "get_autobuy_status", data, latency, 0)


@st.cache_data(show_spinner=False, ttl=2)
def get_autobuy_signals(limit: int = 50) -> FetchResult:
    payload, latency = _request("GET", f"/autobuy/signals?limit={limit}", page="signals", name=f"get_autobuy_signals {limit}")
    data = _extract(payload)
    return _finalize("dash", "get_autobuy_signals", data, latency, 0)


@st.cache_data(show_spinner=False, ttl=2)
def get_autobuy_decision(symbol: str) -> FetchResult:
    sym = symbol.upper().replace("/", "-")
    payload, latency = _request("GET", f"/autobuy/decision?symbol={sym}", page="signals", name=f"get_autobuy_decision {sym}")
    data = _extract(payload)
    return _finalize("dash", "get_autobuy_decision", data, latency, 0)


def start_autobuy() -> bool:
    payload, _ = _request("POST", "/autobuy/control/start", {}, page="signals", name="start_autobuy")
    ok = bool(_extract(payload))
    if ok:
        st.toast("Autobuy started", icon="✅")
    else:
        st.toast("Failed to start autobuy", icon="⚠️")
    return ok


def stop_autobuy() -> bool:
    payload, _ = _request("POST", "/autobuy/control/stop", {}, page="signals", name="stop_autobuy")
    ok = bool(_extract(payload))
    if ok:
        st.toast("Autobuy stopped", icon="✅")
    else:
        st.toast("Failed to stop autobuy", icon="⚠️")
    return ok


@st.cache_data(show_spinner=False, ttl=2)
def system_health() -> FetchResult:
    payload, latency = _request("GET", "/system/health-check", page="system", name="system_health")
    data = _extract(payload)
    return _finalize("dash", "system_health", data, latency, 0)


@st.cache_data(show_spinner=False, ttl=2)
def get_health_check() -> FetchResult:
    payload, latency = _request("GET", "/system/health-check", page="system", name="get_health_check")
    data = _extract(payload)
    return _finalize("dash", "get_health_check", data, latency, 0)


@st.cache_data(show_spinner=False, ttl=2)
def advanced_events() -> FetchResult:
    payload, latency = _request("GET", "/system/events", page="system", name="advanced_events")
    data = _extract(payload)
    return _finalize("dash", "advanced_events", data, latency, 0)


@st.cache_data(show_spinner=False, ttl=2)
def advanced_performance() -> FetchResult:
    payload, latency = _request("GET", "/system/performance", page="system", name="advanced_performance")
    data = _extract(payload)
    return _finalize("dash", "advanced_performance", data, latency, 0)


@st.cache_data(show_spinner=False, ttl=2)
def get_portfolio_overview() -> FetchResult:
    payload, latency = _request("GET", "/portfolio/overview", page="portfolio", name="get_portfolio_overview")
    data = _extract(payload)
    return _finalize("dash", "get_portfolio_overview", data, latency, 0)


@st.cache_data(show_spinner=False, ttl=2)
def get_trading_orders() -> FetchResult:
    payload, latency = _request("GET", "/trading/orders", page="trading", name="get_trading_orders")
    data = _extract(payload)
    return _finalize("dash", "get_trading_orders", data, latency, 0)


@st.cache_data(show_spinner=False, ttl=2)
def get_risk_alerts() -> FetchResult:
    payload, latency = _request("GET", "/risk/alerts", page="risk", name="get_risk_alerts")
    data = _extract(payload)
    if not data:
        payload2, latency2 = _request("GET", "/system/events", page="risk", name="get_system_events")
        data = _extract(payload2)
        latency = latency2
    return _finalize("dash", "get_risk_alerts", data, latency, 0)


@st.cache_data(show_spinner=False, ttl=2)
def get_market_liquidity() -> FetchResult:
    payload, latency = _request("GET", "/market/liquidity", page="market", name="get_market_liquidity")
    data = _extract(payload)
    if not data:
        payload2, latency2 = _request("GET", "/market/live", page="market", name="get_market_live")
        data = _extract(payload2)
        latency = latency2
    return _finalize("dash", "get_market_liquidity", data, latency, 0)


@st.cache_data(show_spinner=False, ttl=2)
def get_analytics_performance() -> FetchResult:
    payload, latency = _request("GET", "/analytics/performance", page="analytics", name="get_analytics_performance")
    data = _extract(payload)
    return _finalize("dash", "get_analytics_performance", data, latency, 0)


def compute_spread_from_price_entry(entry: Dict[str, Any]) -> Optional[float]:
    try:
        bid = float(entry.get("bid", 0) or 0)
        ask = float(entry.get("ask", 0) or 0)
        if bid > 0 and ask > 0 and ask >= bid:
            return float(ask - bid)
        return None
    except Exception:
        return None


@st.cache_data(show_spinner=False, ttl=3)
def get_alerts(limit: int = 100) -> FetchResult:
    payload, latency = _request("GET", "/api/alerts/recent")
    data = _extract(payload)
    return _finalize("dash", "get_alerts", data, latency, 0)


# Note: get_prices is already defined above with same functionality


def clear_cache() -> bool:
    """Clear dashboard cache - connects to live backend"""
    try:
        # Clear Streamlit cache
        st.cache_data.clear()
        
        # Clear backend cache if endpoint available
        payload, _ = _request("POST", "/system/clear-cache", {}, page="system", name="clear_cache")
        return bool(_extract(payload))
    except Exception:
        return False


