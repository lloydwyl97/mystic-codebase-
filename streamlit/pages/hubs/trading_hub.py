"""
📊 TRADING HUB - Portfolio, orders, market analysis, risk management
Consolidated trading operations with smart navigation
"""

import streamlit as st

# Import our modular components
from streamlit.pages.components.common_utils import (
    render_page_header,
)

# Import the real, full-featured page functions
from streamlit.pages.trading.portfolio import render_portfolio_page
from streamlit.pages.trading.orders import render_orders_page
from streamlit.pages.trading.signals import render_signals_page
from streamlit.pages.trading.market_overview import render_market_overview_page
from streamlit.pages.trading.whale_alerts import render_whale_alerts_page
from streamlit.pages.trading.risk_management import render_risk_management_page
from streamlit.pages.trading.live_trading import render_live_trading_page
from streamlit.pages.trading.performance_report import render_performance_report_page

# Backup: create a .bak_cursor copy once
try:
    import shutil
    from pathlib import Path
    _p = Path(__file__)
    _bak = _p.with_suffix(_p.suffix + ".bak_cursor")
    if not _bak.exists():
        shutil.copyfile(_p, _bak)
except Exception:
    pass


def render_trading_hub():
    """Display trading hub with consolidated trading operations"""

    # Page header
    render_page_header(
        "📊 Trading Hub",
        "Portfolio, orders, market analysis, and risk management",
    )

    # Trading section navigation
    st.subheader("🎯 Trading Operations")

    trading_sections = {
        "💼 Portfolio": "portfolio",
        "📋 Orders": "orders",
        "📡 Signals": "signals",
        "📊 Market Overview": "market_overview",
        "🐋 Whale Alerts": "whale_alerts",
        "🛡️ Risk Management": "risk_management",
        "📈 Live Trading": "live_trading",
        "📊 Performance Report": "performance_report",
    }

    # Create tabs for different trading sections
    selected_section = st.selectbox(
        "Select Trading Section:", list(trading_sections.keys()), index=0, key="trading_hub_section"
    )

    # Render selected section using the real, full-featured page functions
    if selected_section == "💼 Portfolio":
        render_portfolio_page()
    elif selected_section == "📋 Orders":
        render_orders_page()
    elif selected_section == "📡 Signals":
        render_signals_page()
    elif selected_section == "📊 Market Overview":
        render_market_overview_page()
    elif selected_section == "🐋 Whale Alerts":
        render_whale_alerts_page()
    elif selected_section == "🛡️ Risk Management":
        render_risk_management_page()
    elif selected_section == "📈 Live Trading":
        render_live_trading_page()
    elif selected_section == "📊 Performance Report":
        render_performance_report_page()
