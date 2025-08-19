import os
from typing import Any, cast

import pandas as pd
import plotly.graph_objects as go  # type: ignore[import-not-found]
import requests
import streamlit as st

from backend.config.coins import FEATURED_EXCHANGE
from mystic_ui import api_client
from mystic_ui import ui_common as ui
from mystic_ui.data_client import get_trades
from mystic_ui.display import render_kpis, render_table
from mystic_ui.top10_resolver import resolve_top10_binanceus

API = os.getenv("MYSTIC_BACKEND", "http://127.0.0.1:9000")
_st = st


def render() -> None:
	st.title(" Market")

	global_data: dict[str, Any] = cast(dict[str, Any], api_client.request_json("GET", "/market/global") or {})
	trends_data: dict[str, Any] = cast(dict[str, Any], api_client.request_json("GET", "/market/live") or {})

	gd = global_data.get("global_data") or global_data
	render_kpis([
		("Total Market Cap (USD)", gd.get("total_market_cap", {}).get("usd") if isinstance(gd.get("total_market_cap"), dict) else gd.get("total_market_cap"), None),
		("Total Volume (USD)", gd.get("total_volume", {}).get("usd") if isinstance(gd.get("total_volume"), dict) else gd.get("total_volume"), None),
		("24h Market Cap %", gd.get("market_cap_change_percentage_24h", gd.get("market_cap_change_24h")), None),
	])

	st.subheader("Prices Snapshot")
	render_table(trends_data)

	tf = ui.timeframe_select("1h")
	base_api = os.getenv("MYSTIC_BACKEND", "http://127.0.0.1:9000")
	sym = st.selectbox("Symbol", resolve_top10_binanceus(base_api, limit=10), index=0)

	@st.cache_data(ttl=15, show_spinner=False)
	def _cached_candles(symbol: str, interval: str, limit: int):
		return api_client.get_candles(symbol, exchange="binance", interval=interval, limit=limit)

	columns = _cached_candles(sym, tf, 300)
	data = api_client.coerce_candles_to_columns(columns)
	df = ui.to_df(data)

	st.subheader(f"{sym} — KPIs")
	render_symbol_kpis(sym, tf)

	if df.empty:
		st.warning("No candles.") 
	else:
		_render_candles(df)

	st.subheader("AI Explain")
	render_ai_explain(sym)

	st.subheader(f"{sym} — Recent Trades")
	render_recent_trades(sym, limit=100)



def _render_candles(df: pd.DataFrame) -> None:
	fig = go.Figure(
		data=[go.Candlestick(
			x=df["ts"],
			open=df["open"],
			high=df["high"],
			low=df["low"],
			close=df["close"],
			showlegend=False,
		)]
	)
	fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=500)
	st.plotly_chart(fig, use_container_width=True)


def render_symbol_kpis(symbol: str, timeframe: str):
	@st.cache_data(ttl=15, show_spinner=False)
	def _cached_candles(symbol: str, interval: str, limit: int):
		return api_client.get_candles(symbol, exchange="binance", interval=interval, limit=limit)

	columns = _cached_candles(symbol, timeframe, 300)
	resp = api_client.coerce_candles_to_columns(columns)
	df = ui.to_df(resp)

	if not df.empty:
		last = float(df["close"].iloc[-1])
		pct = ((last - float(df["close"].iloc[0])) / float(df["close"].iloc[0])) * 100.0 if df["close"].iloc[0] else 0.0
		hi = float(df["high"].max())
		lo = float(df["low"].min())
		vol = float(df["volume"].tail(24).sum())

		ui.metric_row([
			("Last", f"{last:,.2f}", None),
			("24h Δ", f"{last - float(df['close'].iloc[-25]) if len(df)>25 else 0:,.2f} (+{pct:,.2f}%)", "Δ vs first candle in window"),
			("High 24h", f"{hi:,.2f}", None),
			("Low 24h", f"{lo:,.2f}", None),
			("Volume 24h", f"{vol:,.0f}", None),
		])
	else:
		st.warning("No candles returned (try 1h/1d).")


def render_ai_explain(symbol: str):
	_st.subheader(f"AI — What it used for {symbol}")
	try:
		r = requests.get(f"{API}/api/ai/explain/attribution", params={"symbol": symbol}, timeout=6)
		data: Any = r.json()
	except Exception as e:
		_st.error(f"Explain error: {e}")
		return
	if not (isinstance(data, dict) and cast(dict[str, Any], data).get("ok")):
		_st.info("No AI attribution yet.")
		return
	used_any: Any = cast(dict[str, Any], data).get("used") or {}
	used: dict[str, Any] = cast(dict[str, Any], used_any if isinstance(used_any, dict) else {})
	_st.json(
		{
			"mode": os.environ.get("AI_TRADE_MODE", "off"),
			"inputs": used.get("inputs"),
			"weights": used.get("weights"),
			"reason": used.get("reason"),
			"ts": cast(dict[str, Any], data).get("ts"),
		}
	)


def render_recent_trades(symbol: str, limit: int = 100):
	res_tr = get_trades(FEATURED_EXCHANGE, symbol, limit)
	trades: list[Any] = []
	if isinstance(res_tr, dict) and "__meta__" in res_tr:
		meta_any: Any = res_tr["__meta__"] if "__meta__" in res_tr else {}
		meta: dict[str, Any] = cast(dict[str, Any], meta_any if isinstance(meta_any, dict) else {})
		route = meta.get("route")
		status = meta.get("status")
		err = meta.get("error")
		msg = "Trades unavailable"
		if route or status is not None:
			msg += f" — {route or ''} {f'({status})' if status is not None else ''}"
		if err:
			msg += f" • {err}"
		_st.info(msg.strip())
	else:
		payload_tr: Any = res_tr
		if isinstance(payload_tr, dict):
			p2: dict[str, Any] = cast(dict[str, Any], payload_tr)
			t_any = p2.get("trades") or p2.get("data")
			if isinstance(t_any, list):
				trades = t_any
		elif isinstance(payload_tr, list):
			trades = payload_tr
	_st.caption(f"{symbol} — recent trades ({len(trades)})")
	if trades:
		_st.dataframe(trades, height=460, use_container_width=True)
	else:
		_st.info("No trades from backend")

