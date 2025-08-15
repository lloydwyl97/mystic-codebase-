from __future__ import annotations

from typing import Any, Dict, List, MutableMapping, Tuple, cast

import streamlit as st
import plotly.graph_objects as go  # type: ignore[import-not-found]

from streamlit.ui.data_adapter import fetch_candles, safe_number_format
from streamlit.ui.symbols import render_symbol_strip, ensure_state_defaults


_st = cast(Any, st)


@_st.cache_data(show_spinner=False, ttl=60)
def _get_candles_cached(exchange: str, symbol: str, interval: str) -> Dict[str, Any]:
    return fetch_candles(exchange=exchange, symbol=symbol, interval=interval)


def _to_price_series(candles: List[Dict[str, Any]]) -> Tuple[List[int], List[float], List[float], List[float], List[float]]:
    ts: List[int] = []
    o: List[float] = []
    h: List[float] = []
    l: List[float] = []
    c: List[float] = []
    for item in candles:
        ts.append(int(item.get("ts", 0)))
        o.append(float(item.get("o", 0)))
        h.append(float(item.get("h", 0)))
        l.append(float(item.get("l", 0)))
        c.append(float(item.get("c", 0)))
    return ts, o, h, l, c


def _sma(values: List[float], length: int) -> List[float]:
    if length <= 1:
        return list(values)
    out: List[float] = []
    rolling_sum = 0.0
    for i, v in enumerate(values):
        rolling_sum += v
        if i >= length:
            rolling_sum -= values[i - length]
        n = min(i + 1, length)
        out.append(rolling_sum / n)
    return out


def render_market_section() -> None:
    s = cast(MutableMapping[str, Any], _st.session_state)
    exchange = str(s.get("exchange", "binanceus"))
    symbol = str(s.get("symbol", "BTCUSDT"))
    interval = str(s.get("interval", "1h"))

    with _st.spinner("Loading market data..."):
        res = _get_candles_cached(exchange, symbol, interval)
    candles: List[Dict[str, Any]] = cast(List[Dict[str, Any]], res.get("candles", []))

    if not candles:
        _st.warning(f"No candles yet for {symbol} on BinanceUS (interval: {interval}).")
        return

    ts, o, h, l, closes = _to_price_series(candles)

    left, right = _st.columns([3, 2])
    with left:
        _st.subheader("OHLC")
        fig_any: Any = go.Figure(data=[
            go.Candlestick(x=ts, open=o, high=h, low=l, close=closes)
        ])
        fig_any.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10))
        _st.plotly_chart(fig_any, use_container_width=True)
        with _st.expander("Raw JSON: Candles", expanded=False):
            _st.json(res.get("raw"))

    with right:
        _st.subheader("Snapshot")
        last = closes[-1] if closes else 0.0
        chg = (last - closes[-2]) / closes[-2] * 100.0 if len(closes) > 1 and closes[-2] != 0 else 0.0
        _st.metric(
            label=f"{symbol} @ {interval}",
            value=safe_number_format(last, 2),
            delta=f"{safe_number_format(chg, 2)}%",
        )

        _st.caption(f"Candles: {len(candles)}  •  Exchange: {exchange}")

    with _st.expander("Details", expanded=False):
        _st.dataframe(candles[-200:])


def render_signals_autobuy_section() -> None:
    s = cast(MutableMapping[str, Any], _st.session_state)
    exchange = str(s.get("exchange", "binanceus"))
    symbol = str(s.get("symbol", "BTCUSDT"))
    interval = str(s.get("interval", "1h"))

    with _st.spinner("Loading signal inputs..."):
        res = _get_candles_cached(exchange, symbol, interval)
    candles: List[Dict[str, Any]] = cast(List[Dict[str, Any]], res.get("candles", []))

    if not candles:
        _st.warning("No candle data for signals.")
        return

    _, _o, _h, _l, closes = _to_price_series(candles)
    sma_fast = _sma(closes, 10)
    sma_slow = _sma(closes, 30)

    _st.subheader("Signals — SMA Crossover (derived from live candles)")
    _st.line_chart({"close": closes, "SMA10": sma_fast, "SMA30": sma_slow})

    if len(closes) >= 2:
        last_fast = sma_fast[-1]
        last_slow = sma_slow[-1]
        signal = "Neutral"
        if last_fast > last_slow:
            signal = "Bullish Bias"
        elif last_fast < last_slow:
            signal = "Bearish Bias"
        _st.metric("Current Bias", signal)

    _st.caption("Autobuy logic can subscribe to the above bias without direct HTTP calls.")


def render_portfolio_orders_section() -> None:
    s = cast(MutableMapping[str, Any], _st.session_state)
    exchange = str(s.get("exchange", "binanceus"))
    symbol = str(s.get("symbol", "BTCUSDT"))
    interval = str(s.get("interval", "1h"))

    with _st.spinner("Loading candles for portfolio signals..."):
        res = _get_candles_cached(exchange, symbol, interval)
    candles: List[Dict[str, Any]] = cast(List[Dict[str, Any]], res.get("candles", []))

    if not candles:
        _st.warning("No candle data available.")
        return

    _ts, _o, _h, _l, closes = _to_price_series(candles)
    sma_fast = _sma(closes, 10)
    sma_slow = _sma(closes, 30)

    # Detect simple crossover events as hypothetical order signals
    events: List[Dict[str, Any]] = []
    for i in range(1, len(closes)):
        prev_fast = sma_fast[i - 1]
        prev_slow = sma_slow[i - 1]
        cur_fast = sma_fast[i]
        cur_slow = sma_slow[i]
        crossed_up = prev_fast <= prev_slow and cur_fast > cur_slow
        crossed_down = prev_fast >= prev_slow and cur_fast < cur_slow
        if crossed_up or crossed_down:
            action = "BUY" if crossed_up else "SELL"
            events.append({
                "idx": i,
                "action": action,
                "price": closes[i],
            })

    left, right = _st.columns([2, 3])
    with left:
        _st.subheader("Hypothetical Signals (SMA Cross)")
        if events:
            _st.dataframe(events[-20:])
        else:
            _st.caption("No recent cross events detected.")

    with right:
        _st.subheader("Equity Curve (Toy) — 1x on BUY, flat on SELL")
        equity: List[float] = []
        position = 0  # 0 flat, 1 long
        cash = 1.0
        last_price = closes[0]
        for i, px in enumerate(closes):
            # Apply events
            ev = next((e for e in events if e["idx"] == i), None)
            if ev:
                if ev["action"] == "BUY":
                    position = 1
                elif ev["action"] == "SELL":
                    position = 0
            # Update equity as cash plus unrealized PnL of a 1x position
            pnl = (px - last_price) / last_price if position == 1 and last_price else 0.0
            cash *= (1.0 + pnl)
            equity.append(cash)
            last_price = px

        _st.line_chart({"equity": equity})


def render_alerts_section() -> None:
    s = cast(MutableMapping[str, Any], _st.session_state)
    exchange = str(s.get("exchange", "binanceus"))
    symbol = str(s.get("symbol", "BTCUSDT"))
    interval = str(s.get("interval", "1h"))

    with _st.spinner("Scanning candles for alert conditions..."):
        res = _get_candles_cached(exchange, symbol, interval)
    candles: List[Dict[str, Any]] = cast(List[Dict[str, Any]], res.get("candles", []))

    if not candles:
        _st.warning("No candle data available.")
        return

    _ts, _o, _h, _l, closes = _to_price_series(candles)

    alerts: List[Dict[str, Any]] = []
    for i in range(1, len(closes)):
        prev = closes[i - 1]
        cur = closes[i]
        if prev == 0:
            continue
        pct = (cur - prev) / prev * 100.0
        if abs(pct) >= 2.0:
            alerts.append({
                "idx": i,
                "type": "Volatility Spike",
                "delta%": round(pct, 2),
                "price": cur,
            })

    _st.subheader("Derived Alerts (from candles)")
    if alerts:
        _st.dataframe(alerts[-50:])
    else:
        _st.caption("No recent volatility spikes detected (|Δ| >= 2%).")


def render_system_advanced_section() -> None:
    _st.subheader("System Advanced")
    s = cast(MutableMapping[str, Any], _st.session_state)
    _st.write({
        "exchange": s.get("exchange"),
        "symbol": s.get("symbol"),
        "interval": s.get("interval"),
    })
    _st.caption("Advanced diagnostics can be added via shared helpers without direct HTTP.")


def main() -> None:
    ensure_state_defaults()
    _st.set_page_config(page_title="Mystic — Super Dashboard", layout="wide")
    _st.title("Mystic Super Dashboard")

    # Top symbol strip (no direct HTTP; uses data adapter under the hood)
    render_symbol_strip()

    # Status line
    s = cast(MutableMapping[str, Any], _st.session_state)
    _st.caption(
        f"Exchange: {s.get('exchange')}  •  Symbol: {s.get('symbol')}  •  Interval: {s.get('interval')}"
    )

    tabs = _st.tabs([
        "Market",
        "Signals & Autobuy",
        "Portfolio Orders",
        "Alerts",
        "System Advanced",
    ])

    with tabs[0]:
        render_market_section()
    with tabs[1]:
        render_signals_autobuy_section()
    with tabs[2]:
        render_portfolio_orders_section()
    with tabs[3]:
        render_alerts_section()
    with tabs[4]:
        render_system_advanced_section()


if __name__ == "__main__":
    main()
