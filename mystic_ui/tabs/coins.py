import streamlit as st

from mystic_ui.api_client import request_json


def render() -> None:
	st.title(" Coins Explorer")

	coins = request_json("GET", "/coins") or []
	st.dataframe(coins if isinstance(coins, list) else [coins])

	sym = st.text_input("Lookup Symbol (e.g. BTC-USD)", "BTC-USD")
	if st.button("Fetch /api/coins/{symbol}"):
		st.json(request_json("GET", f"/coins/{sym}"))


