"""
🚀 AUTOBUY HUB - Complete Autobuy System Integration
Real-time autobuy monitoring, control, and management with enhanced trading functionality
"""

import streamlit as st
from streamlit.ui_guard import display_guard
from streamlit.data_client import fetch_api as _fetch_api, post_api as _post_api  # central client
import plotly.express as px

# Import our modular components
from streamlit.data_client import BASE_URL  # expose base URL if needed
from ..components.common_utils import (
    render_page_header,
    render_metrics_grid,
    render_dataframe_safely,
    format_currency,
    format_percentage,
    get_fallback_data,
)
from ..components.responsive_layout import (
    responsive_columns,
)

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

# Import the real, full-featured legacy trading page functions
from ..pages.trading.portfolio import render_portfolio_page
from ..pages.trading.orders import render_orders_page
from ..pages.trading.signals import render_signals_page
from ..pages.trading.market_overview import render_market_overview_page
from ..pages.trading.whale_alerts import render_whale_alerts_page
from ..pages.trading.risk_management import render_risk_management_page
from ..pages.trading.live_trading import render_live_trading_page
from ..pages.trading.performance_report import render_performance_report_page


def render_autobuy_hub():
    """Render the complete Autobuy Hub with all functionality and enhanced trading integration"""

    # Page header
    render_page_header(
        "🚀 Autobuy System",
        "Real-time automated trading monitoring and control",
    )

    # Fetch autobuy data
    autobuy_status = _fetch_api("/api/autobuy/status")
    autobuy_stats = _fetch_api("/api/autobuy/stats")
    autobuy_trades = _fetch_api("/api/autobuy/trades")
    autobuy_config = _fetch_api("/api/autobuy/config")
    autobuy_signals = _fetch_api("/api/autobuy/signals")

    # Check data availability
    has_data = any([autobuy_status, autobuy_stats, autobuy_trades, autobuy_config])
    data_source = "api" if has_data else "fallback"

    # Status indicators
    col_status1, col_status2 = responsive_columns(2)
    with col_status1:
        if data_source != "fallback":
            st.success(f"🟢 Autobuy System Connected - Source: {data_source.upper()}")
        else:
            st.warning("⚠️ Using fallback data - Autobuy APIs unavailable")

    with col_status2:
        if autobuy_status and autobuy_status.get("data"):
            status_data = autobuy_status.get("data", {})
            system_status = (
                "🟢 Running" if status_data.get("is_running", False) else "🔴 Stopped"
            )
            st.info(f"System Status: {system_status}")
        else:
            st.info("System Status: Unknown")

    # Enhanced Navigation Tabs
    st.subheader("🎯 Autobuy & Trading Operations")

    # Create comprehensive tabs for all autobuy and trading functionality
    (
        tab_autobuy,
        tab_portfolio,
        tab_orders,
        tab_signals,
        tab_market,
        tab_whales,
        tab_risk,
        tab_live,
        tab_performance,
    ) = st.tabs(
        [
            "🚀 Autobuy Control",
            "💼 Portfolio",
            "📋 Orders",
            "📡 Signals",
            "📊 Market",
            "🐋 Whales",
            "🛡️ Risk",
            "📈 Live Trading",
            "📊 Performance",
        ]
    )

    # Tab 1: Autobuy Control
    with tab_autobuy:
        render_autobuy_control_section(
            autobuy_status,
            autobuy_stats,
            autobuy_trades,
            autobuy_config,
            autobuy_signals,
        )

    # Tab 2: Portfolio
    with tab_portfolio:
        render_portfolio_page()

    # Tab 3: Orders
    with tab_orders:
        render_orders_page()

    # Tab 4: Signals
    with tab_signals:
        render_signals_page()

    # Tab 5: Market Overview
    with tab_market:
        render_market_overview_page()

    # Tab 6: Whale Alerts
    with tab_whales:
        render_whale_alerts_page()

    # Tab 7: Risk Management
    with tab_risk:
        render_risk_management_page()

    # Tab 8: Live Trading
    with tab_live:
        render_live_trading_page()

    # Tab 9: Performance Report
    with tab_performance:
        render_performance_report_page()


def render_autobuy_control_section(
    autobuy_status,
    autobuy_stats,
    autobuy_trades,
    autobuy_config,
    autobuy_signals,
):
    """Render the autobuy control section with all existing functionality"""

    # System Controls
    st.subheader("🎛️ System Controls")
    col_control1, col_control2, col_control3, col_control4 = responsive_columns(4)

    with col_control1:
        if st.button("▶️ Start Autobuy", type="primary", use_container_width=True):
            # Start autobuy system
            start_result = _post_api("/api/autobuy/control/start", {})
            if start_result:
                st.success("Autobuy system started!")
            else:
                st.error("Failed to start autobuy system")

    with col_control2:
        if st.button("⏹️ Stop Autobuy", type="secondary", use_container_width=True):
            # Stop autobuy system
            stop_result = _post_api("/api/autobuy/control/stop", {})
            if stop_result:
                st.success("Autobuy system stopped!")
            else:
                st.error("Failed to stop autobuy system")

    with col_control3:
        if st.button("🚨 Emergency Stop", type="secondary", use_container_width=True):
            # Emergency stop
            emergency_result = _post_api("/api/autobuy/control/emergency-stop", {})
            if emergency_result:
                st.error("🚨 Emergency stop activated!")
            else:
                st.error("Failed to activate emergency stop")

    with col_control4:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()

    # Autobuy Statistics
    st.subheader("📊 Autobuy Statistics")

    # Get autobuy stats
    stats_data = (
        autobuy_stats.get("data", {})
        if autobuy_stats
        else get_fallback_data("autobuy_stats")
    )

    autobuy_metrics = [
        {
            "label": "Total Trades",
            "value": str(stats_data.get("total_trades", 0)),
            "delta": None,
        },
        {
            "label": "Success Rate",
            "value": format_percentage(stats_data.get("success_rate", 0)),
            "delta": None,
        },
        {
            "label": "Total Volume",
            "value": format_currency(stats_data.get("total_volume", 0)),
            "delta": None,
        },
        {
            "label": "Active Trades",
            "value": str(stats_data.get("active_trades", 0)),
            "delta": None,
        },
        {
            "label": "Total Profit",
            "value": format_currency(stats_data.get("total_profit", 0)),
            "delta": format_percentage(stats_data.get("profit_percentage", 0)),
        },
        {
            "label": "System Uptime",
            "value": f"{stats_data.get('system_uptime_hours', 0):.1f}h",
            "delta": None,
        },
    ]

    render_metrics_grid(autobuy_metrics, cols=3)

    # Trading Pairs Status
    st.subheader("📈 Trading Pairs Status")

    # Get trading pairs status
    pairs_data = (
        autobuy_config.get("data", {}).get("trading_pairs", {})
        if autobuy_config
        else {}
    )

    if pairs_data:
        # Create pairs overview
        pairs_overview = []
        for symbol, config in pairs_data.items():
            if isinstance(config, dict):
                pairs_overview.append(
                    {
                        "Symbol": symbol,
                        "Status": (
                            "🟢 Active"
                            if config.get("enabled", False)
                            else "🔴 Inactive"
                        ),
                        "Min Amount": format_currency(
                            config.get("min_trade_amount", 0)
                        ),
                        "Max Amount": format_currency(
                            config.get("max_trade_amount", 0)
                        ),
                        "Target Frequency": (
                            f"{config.get('target_frequency', 0)}/day"
                        ),
                    }
                )

        if pairs_overview:
            render_dataframe_safely(pairs_overview, "Trading Pairs Configuration")

    # Recent Trades
    st.subheader("📋 Recent Trades")

    trades_data = autobuy_trades.get("data", []) if autobuy_trades else []

    if trades_data:
        # Format trades for display
        formatted_trades = []
        for trade in trades_data[-10:]:  # Last 10 trades
            formatted_trades.append(
                {
                    "Time": trade.get("timestamp", "Unknown"),
                    "Symbol": trade.get("symbol", "Unknown"),
                    "Side": trade.get("side", "Unknown"),
                    "Amount": format_currency(trade.get("amount", 0)),
                    "Price": format_currency(trade.get("price", 0)),
                    "Status": trade.get("status", "Unknown"),
                    "Confidence": f"{trade.get('confidence', 0):.1f}%",
                }
            )

        render_dataframe_safely(formatted_trades, "Recent Autobuy Trades")
    else:
        st.info("No recent trades found")

    # Performance Charts
    st.subheader("📊 Performance Analytics")

    col_chart1, col_chart2 = responsive_columns(2)

    with col_chart1:
        # Profit/Loss over time chart
        if trades_data:
            # Create P&L chart
            dates = [trade.get("timestamp", "") for trade in trades_data[-20:]]
            profits = [trade.get("profit", 0) for trade in trades_data[-20:]]

            fig_pl = px.line(
                x=dates,
                y=profits,
                title="Profit/Loss Over Time",
                labels={"x": "Time", "y": "P&L (USD)"},
            )
            fig_pl.update_layout(height=400)
            st.plotly_chart(fig_pl, use_container_width=True)

    with col_chart2:
        # Success rate chart
        if trades_data:
            # Calculate success rate over time
            success_rates = []
            time_periods = []

            # Group trades by time periods and calculate success rates
            for i in range(0, len(trades_data), 5):  # Every 5 trades
                period_trades = trades_data[i : i + 5]
                if period_trades:
                    successful = len(
                        [
                            t
                            for t in period_trades
                            if t.get("status") == "completed" and t.get("profit", 0) > 0
                        ]
                    )
                    success_rate = (successful / len(period_trades)) * 100
                    success_rates.append(success_rate)
                    time_periods.append(f"Period {len(time_periods) + 1}")

            if success_rates:
                fig_success = px.bar(
                    x=time_periods,
                    y=success_rates,
                    title="Success Rate Over Time",
                    labels={"x": "Time Period", "y": "Success Rate (%)"},
                )
                fig_success.update_layout(height=400)
                st.plotly_chart(fig_success, use_container_width=True)

    # Advanced Configuration
    st.subheader("⚙️ Advanced Configuration")

    # Get current configuration
    config_data = autobuy_config.get("data", {}) if autobuy_config else {}

    col_config1, col_config2 = responsive_columns(2)

    with col_config1:
        st.subheader("Risk Parameters")

        # Risk management settings
        max_daily_trades = st.number_input(
            "Max Daily Trades",
            min_value=1,
            max_value=100,
            value=config_data.get("max_daily_trades", 48),
            help="Maximum number of trades per day",
        )

        max_daily_volume = st.number_input(
            "Max Daily Volume (USD)",
            min_value=100,
            max_value=10000,
            value=config_data.get("max_daily_volume", 2000),
            help="Maximum trading volume per day",
        )

        stop_loss_percent = st.slider(
            "Stop Loss (%)",
            min_value=1.0,
            max_value=20.0,
            value=config_data.get("stop_loss_percent", 5.0),
            step=0.5,
            help="Stop loss percentage",
        )

        take_profit_percent = st.slider(
            "Take Profit (%)",
            min_value=1.0,
            max_value=50.0,
            value=config_data.get("take_profit_percent", 10.0),
            step=0.5,
            help="Take profit percentage",
        )

    with col_config2:
        st.subheader("Signal Parameters")

        # Signal filtering settings
        min_confidence = st.slider(
            "Minimum Confidence (%)",
            min_value=50,
            max_value=95,
            value=config_data.get("min_confidence", 75),
            help="Minimum confidence level for trade execution",
        )

        min_volume_multiplier = st.slider(
            "Minimum Volume Multiplier",
            min_value=1.0,
            max_value=5.0,
            value=config_data.get("min_volume_multiplier", 1.5),
            step=0.1,
            help="Minimum volume above average for signal validation",
        )

        signal_confirmation_count = st.number_input(
            "Signal Confirmations Required",
            min_value=1,
            max_value=10,
            value=config_data.get("signal_confirmation_count", 3),
            help="Number of confirming signals required",
        )

        mystic_weight = st.slider(
            "Mystic Signal Weight (%)",
            min_value=0,
            max_value=100,
            value=config_data.get("mystic_weight", 40),
            help="Weight given to mystic signals in decision making",
        )

    # Save configuration button
    if st.button("💾 Save Configuration", type="primary", use_container_width=True):
        # Prepare configuration data
        new_config = {
            "max_daily_trades": max_daily_trades,
            "max_daily_volume": max_daily_volume,
            "stop_loss_percent": stop_loss_percent,
            "take_profit_percent": take_profit_percent,
            "min_confidence": min_confidence,
            "min_volume_multiplier": min_volume_multiplier,
            "signal_confirmation_count": signal_confirmation_count,
            "mystic_weight": mystic_weight,
        }

        # Save configuration
        config_result = _post_api("/api/autobuy/config", new_config)
        if config_result:
            st.success("Configuration saved successfully!")
        else:
            st.error("Failed to save configuration")

    # Real-time Signals
    st.subheader("📡 Real-time Signals")

    signals_data = autobuy_signals.get("data", []) if autobuy_signals else []

    if signals_data:
        # Display recent signals
        for signal in signals_data[-5:]:  # Last 5 signals
            signal_type = signal.get("type", "unknown")
            symbol = signal.get("symbol", "Unknown")
            confidence = signal.get("confidence", 0)
            strength = signal.get("strength", 0)
            timestamp = signal.get("timestamp", "Unknown")

            # Color coding based on signal type
            if signal_type == "buy":
                st.success(
                    f"🟢 BUY {symbol} - Confidence: {confidence:.1f}% - Strength: {strength:.2f} - {timestamp}"
                )
            elif signal_type == "sell":
                st.error(
                    f"🔴 SELL {symbol} - Confidence: {confidence:.1f}% - Strength: {strength:.2f} - {timestamp}"
                )
            else:
                st.info(
                    f"🟡 {signal_type.upper()} {symbol} - Confidence: {confidence:.1f}% - Strength: {strength:.2f} - {timestamp}"
                )
    else:
        st.info("No recent signals available")

    # System Health
    st.subheader("🏥 System Health")

    # Get system health data
    health_data = _fetch_api("/api/autobuy/health")

    if health_data and health_data.get("data"):
        health_info = health_data.get("data", {})

        health_metrics = [
            {
                "label": "API Status",
                "value": (
                    "🟢 Healthy"
                    if health_info.get("api_status", False)
                    else "🔴 Unhealthy"
                ),
                "delta": None,
            },
            {
                "label": "Database Status",
                "value": (
                    "🟢 Connected"
                    if health_info.get("db_status", False)
                    else "🔴 Disconnected"
                ),
                "delta": None,
            },
            {
                "label": "Cache Status",
                "value": (
                    "🟢 Active"
                    if health_info.get("cache_status", False)
                    else "🔴 Inactive"
                ),
                "delta": None,
            },
            {
                "label": "Last Update",
                "value": health_info.get("last_update", "Unknown"),
                "delta": None,
            },
        ]

        render_metrics_grid(health_metrics, cols=4)
    else:
        st.warning("System health data unavailable")
