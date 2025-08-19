import streamlit as st
from mystic_ui import api_client
from mystic_ui.top10_resolver import resolve_top10
from mystic_ui import ui_common as ui

# set_page_config is centralized in mystic_ui/app.py
st.title("Mystic Super Dashboard BinanceUS")

tf = ui.timeframe_select("1h")
symbols = resolve_top10(tf, limit=300)
sym = st.selectbox("Symbol (Top 10 by 24h volume)", symbols, index=0)
if st.button("Refresh"):
	st.experimental_rerun()

@st.cache_data(ttl=15, show_spinner=False)
def _cached_candles(symbol: str, interval: str, limit: int):
	return api_client.get_candles(symbol, exchange="binance", interval=interval, limit=limit)

rows = _cached_candles(sym, tf, 300)

def _valid_candle_row(r: dict) -> bool:
	return all(k in r for k in ("timestamp","open","high","low","close","volume"))

if isinstance(rows, list) and rows and not _valid_candle_row(rows[0]):
	print("WARN: Candle row shape unexpected; keys=", list(rows[0].keys()))

resp = api_client._coerce_candles_to_columns(rows)
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

	st.subheader("Price (Candles)")
	st.line_chart(df.set_index("ts")["close"])
else:
	st.warning("No candles returned (try 1h/1d).")

ui.safe_json("Raw JSON (candles)", resp)
ui.safe_json("Request info", {"exchange":"binanceus","symbol":sym,"timeframe":tf,"limit":300})


