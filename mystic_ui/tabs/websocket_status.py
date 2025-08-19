import streamlit as st
from mystic_ui.api_client import get_ws_status


def render() -> None:
	st.title(" WebSocket")

	st.json(get_ws_status())


