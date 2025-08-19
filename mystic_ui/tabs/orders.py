import streamlit as st

from mystic_ui.api_client import request_json
from mystic_ui.display import render_table


def render() -> None:
	st.title(" Orders")

	col1, col2 = st.columns(2)
	with col1:
		st.subheader("Place Market Order")
		sym = st.text_input("Symbol (e.g. BTC-USD)", "BTC-USD")
		side = st.selectbox("Side", ["buy","sell"])
		qty = st.number_input("Qty", min_value=0.0001, value=0.001, step=0.001, format="%.6f")
		if st.button("Submit Market"):
			payload = {
				"symbol": sym,
				"side": side,
				"type": "market",
				"quantity": qty,
			}
			st.write(request_json("POST", "/live/trading/order", json=payload))

	with col2:
		st.subheader("Cancel")
		csym = st.text_input("Cancel Symbol", "BTC-USD", key="csym")
		if st.button("Cancel Symbol"):
			st.write(request_json("POST", f"/trading/cancel/{csym}", json={}))
		if st.button("Cancel All"):
			st.write(request_json("POST", "/trading/cancel-all", json={}))

	st.subheader("Open Orders")
	try:
		render_table(request_json("GET", "/live/trading/orders"))
	except Exception:
		render_table(request_json("GET", "/binance/orders"))


