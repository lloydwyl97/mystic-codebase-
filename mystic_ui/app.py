import streamlit as st

st.set_page_config(page_title="Super Dashboard", layout="wide")
st.markdown("## Super Dashboard")

# Import tab renderers
from mystic_ui.tabs.ai_strategies import render as render_ai_strategies
from mystic_ui.tabs.analytics import render as render_analytics
from mystic_ui.tabs.autobuy import render as render_autobuy
from mystic_ui.tabs.coins import render as render_coins
from mystic_ui.tabs.experimental import render as render_experimental
from mystic_ui.tabs.health import render as render_health
from mystic_ui.tabs.live_trading import render as render_live_trading
from mystic_ui.tabs.market import render as render_market
from mystic_ui.tabs.orders import render as render_orders
from mystic_ui.tabs.portfolio import render as render_portfolio
from mystic_ui.tabs.settings import render as render_settings
from mystic_ui.tabs.signals import render as render_signals
from mystic_ui.tabs.websocket_status import render as render_websocket

tabs = st.tabs([
    "Market", "Portfolio", "Orders", "Live Trading", "AutoBuy",
    "AI Strategies", "Analytics", "Signals", "Health",
    "Settings", "Experimental", "WebSocket", "Coins"
])

(
    tab_market,
    tab_portfolio,
    tab_orders,
    tab_live_trading,
    tab_autobuy,
    tab_ai_strategies,
    tab_analytics,
    tab_signals,
    tab_health,
    tab_settings,
    tab_experimental,
    tab_websocket,
    tab_coins,
) = tabs

with tab_market:
    render_market()

with tab_portfolio:
    render_portfolio()

with tab_orders:
    render_orders()

with tab_live_trading:
    render_live_trading()

with tab_autobuy:
    render_autobuy()

with tab_ai_strategies:
    render_ai_strategies()

with tab_analytics:
    render_analytics()

with tab_signals:
    render_signals()

with tab_health:
    render_health()

with tab_settings:
    render_settings()

with tab_experimental:
    render_experimental()

with tab_websocket:
    render_websocket()

with tab_coins:
    render_coins()

