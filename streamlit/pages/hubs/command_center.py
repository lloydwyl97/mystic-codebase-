"""
🏠 COMMAND CENTER HUB - Real-time overview and quick actions
Consolidated main dashboard with smart widgets and progressive loading
Enhanced with advanced monitoring and quantum features
"""

import streamlit as st
from datetime import datetime

# Import our modular components using canonical streamlit package
from streamlit.data_client import fetch_api as _fetch_api, BASE_URL  # central fetch helpers
from streamlit.pages.components.common_utils import (
    render_page_header,
    render_metrics_grid,
    render_dataframe_safely,
    format_currency,
    format_percentage,
    get_fallback_data,
)
from streamlit.pages.components.responsive_layout import (
    responsive_columns,
)

# Import advanced monitoring components via canonical root
from streamlit.pages.advanced_tech.phase5_monitoring import render_phase5_monitoring_page
from streamlit.pages.advanced_tech.quantum_computing import render_quantum_computing_page
from streamlit.pages.advanced_tech.blockchain import render_blockchain_page
from streamlit.pages.advanced_tech.mining_operations import render_mining_operations_page
from streamlit.pages.advanced_tech.experimental_services import (
    render_experimental_services_page,
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


def render_command_center():
    """Display command center with consolidated overview and smart widgets"""

    # Page header
    render_page_header(
        "🏠 Command Center", "Real-time system overview and quick actions"
    )

    # Fetch live data using centralized API client
    portfolio_data = _fetch_api("/api/portfolio/overview")
    autobuy_data = _fetch_api("/api/autobuy/status")
    ai_data = _fetch_api("/api/ai/strategies")
    live_market_data = _fetch_api("/api/market/live")
    live_portfolio = _fetch_api("/api/portfolio/live")
    system_status = _fetch_api("/api/system/status")
    trading_signals = _fetch_api("/api/trading/signals")

    # Advanced monitoring data
    _fetch_api("/api/phase5/metrics")
    _fetch_api("/api/quantum/systems")
    _fetch_api("/api/blockchain/status")
    _fetch_api("/api/mining/status")
    _fetch_api("/api/experimental/status")

    # Check if we have any data and determine data source
    has_data = any(
        [
            portfolio_data,
            autobuy_data,
            ai_data,
            live_market_data,
            live_portfolio,
            system_status,
            trading_signals,
        ]
    )
    data_source = "api" if has_data else "fallback"

    # Test backend connectivity via health endpoint
    _health = _fetch_api("/health") or _fetch_api("/api/health")
    backend_status = "Connected" if _health else "Disconnected"

    # Status indicators
    col_status1, col_status2 = responsive_columns(2)
    with col_status1:
        if data_source != "fallback":
            st.success(f"🟢 Live Data Connected - Source: {data_source.upper()}")
        else:
            st.warning("⚠️ Using fallback data - Live APIs unavailable")

    with col_status2:
        st.info(f"Backend Status: {backend_status}")
        st.caption(f"Backend URL: {BASE_URL}")

    # Enhanced Navigation Tabs
    st.subheader("🎯 System Overview & Advanced Monitoring")

    # Create comprehensive tabs for all monitoring functionality
    (
        tab_overview,
        tab_phase5,
        tab_quantum,
        tab_blockchain,
        tab_mining,
        tab_experimental,
    ) = st.tabs(
        [
            "📊 Overview",
            "🔬 Phase 5",
            "⚛️ Quantum",
            "🔗 Blockchain",
            "⛏️ Mining",
            "🧪 Experimental",
        ]
    )

    # Tab 1: Overview
    with tab_overview:
        render_overview_section(
            portfolio_data,
            autobuy_data,
            ai_data,
            live_market_data,
            live_portfolio,
            system_status,
            trading_signals,
        )

    # Tab 2: Phase 5 Monitoring
    with tab_phase5:
        render_phase5_monitoring_page()

    # Tab 3: Quantum Computing
    with tab_quantum:
        render_quantum_computing_page()

    # Tab 4: Blockchain
    with tab_blockchain:
        render_blockchain_page()

    # Tab 5: Mining Operations
    with tab_mining:
        render_mining_operations_page()

    # Tab 6: Experimental Services
    with tab_experimental:
        render_experimental_services_page()


def render_overview_section(
    portfolio_data,
    autobuy_data,
    ai_data,
    live_market_data,
    live_portfolio,
    system_status,
    trading_signals,
):
    """Render the main overview section with all existing functionality plus enhanced monitoring"""

    # Key metrics with live data
    portfolio_actual = (
        portfolio_data.get("data", {})
        if portfolio_data
        else get_fallback_data("portfolio")
    )
    autobuy_actual = autobuy_data.get("data", {}) if autobuy_data else {}
    ai_actual = (
        ai_data.get("data", {}) if ai_data else get_fallback_data("ai_strategies")
    )

    # Create metrics for responsive grid
    metrics = [
        {
            "label": "Portfolio Value",
            "value": format_currency(portfolio_actual.get("total_value", 0)),
            "delta": format_percentage(portfolio_actual.get("daily_change", 0)),
        },
        {
            "label": "Autobuy Status",
            "value": (
                "🟢 Active"
                if autobuy_actual.get("is_running", False)
                else "🔴 Inactive"
            ),
            "delta": None,
        },
        {
            "label": "Active AI Strategies",
            "value": str(
                len(
                    [
                        s
                        for s in ai_actual.get("ai_strategies", [])
                        if s.get("status") == "ACTIVE"
                    ]
                )
            ),
            "delta": None,
        },
        {
            "label": "Active Positions",
            "value": str(portfolio_actual.get("positions", 0)),
            "delta": None,
        },
    ]

    # Render metrics in responsive grid
    render_metrics_grid(metrics, cols=4)

    # Enhanced System Status with Advanced Monitoring
    st.subheader("🔧 Enhanced System Status")

    # Get advanced system data
    phase5_data = _fetch_api("/api/phase5/metrics")
    quantum_data = _fetch_api("/api/quantum/systems")
    blockchain_data = _fetch_api("/api/blockchain/status")
    mining_data = _fetch_api("/api/mining/status")
    experimental_data = _fetch_api("/api/experimental/status")

    # Advanced system metrics
    col_adv1, col_adv2, col_adv3, col_adv4 = responsive_columns(4)

    with col_adv1:
        # Phase 5 Status
        if phase5_data and phase5_data.get("data"):
            phase5_metrics = phase5_data.get("data", {})
            neuro_sync = phase5_metrics.get("neuro_sync", "0%")
            st.metric("🧠 Neuro-Sync", neuro_sync)
        else:
            st.metric("🧠 Neuro-Sync", "N/A")

    with col_adv2:
        # Quantum Status
        if quantum_data and quantum_data.get("data"):
            quantum_systems = quantum_data.get("data", [])
            online_quantum = len(
                [q for q in quantum_systems if q.get("status") == "Online"]
            )
            total_quantum = len(quantum_systems)
            st.metric("⚛️ Quantum Systems", f"{online_quantum}/{total_quantum}")
        else:
            st.metric("⚛️ Quantum Systems", "N/A")

    with col_adv3:
        # Blockchain Status
        if blockchain_data and blockchain_data.get("data"):
            blockchain_status = blockchain_data.get("data", {})
            blockchain_health = blockchain_status.get("health", "Unknown")
            st.metric("🔗 Blockchain", blockchain_health)
        else:
            st.metric("🔗 Blockchain", "N/A")

    with col_adv4:
        # Mining Status
        if mining_data and mining_data.get("data"):
            mining_status = mining_data.get("data", {})
            mining_active = mining_status.get("active_miners", 0)
            st.metric("⛏️ Active Miners", str(mining_active))
        else:
            st.metric("⛏️ Active Miners", "N/A")

    # Quick Actions Panel
    st.subheader("⚡ Quick Actions")

    col_actions1, col_actions2, col_actions3, col_actions4 = responsive_columns(4)

    with col_actions1:
        if st.button("📊 View Portfolio", use_container_width=True):
            st.info("Navigate to Trading Hub → Portfolio")

    with col_actions2:
        if st.button("🚀 Autobuy Control", use_container_width=True):
            st.info("Navigate to Autobuy Hub")

    with col_actions3:
        if st.button("🤖 AI Strategies", use_container_width=True):
            st.info("Navigate to AI Intelligence Hub")

    with col_actions4:
        if st.button("⚙️ System Admin", use_container_width=True):
            st.info("Navigate to System Control Hub")

    # Advanced Quick Actions
    col_adv_actions1, col_adv_actions2, col_adv_actions3, col_adv_actions4 = (
        responsive_columns(4)
    )

    with col_adv_actions1:
        if st.button("🔬 Phase 5 Monitor", use_container_width=True):
            st.info("View Phase 5 Monitoring")

    with col_adv_actions2:
        if st.button("⚛️ Quantum Status", use_container_width=True):
            st.info("View Quantum Computing Status")

    with col_adv_actions3:
        if st.button("🔗 Blockchain", use_container_width=True):
            st.info("View Blockchain Operations")

    with col_adv_actions4:
        if st.button("⛏️ Mining Ops", use_container_width=True):
            st.info("View Mining Operations")

    # Live market overview
    if live_market_data and live_market_data.get("data"):
        st.subheader("📊 Live Market Overview")
        live_market_actual = live_market_data.get("data", {})
        market_data = live_market_actual.get("markets", [])
        render_dataframe_safely(market_data, "Market Data")

    # Enhanced System Status Summary
    st.subheader("🔧 Enhanced System Status")
    system_actual = system_status.get("data", {}) if system_status else {}

    system_metrics = [
        {
            "label": "Market Data Status",
            "value": system_actual.get("market_data_status", "Unknown"),
            "delta": None,
        },
        {
            "label": "Exchange Connections",
            "value": str(system_actual.get("exchange_connections", 0)),
            "delta": None,
        },
        {
            "label": "Uptime",
            "value": system_actual.get("uptime", "Unknown"),
            "delta": None,
        },
        {
            "label": "Experimental Services",
            "value": (
                "🟢 Active"
                if experimental_data
                and experimental_data.get("data", {}).get("status") == "active"
                else "🔴 Inactive"
            ),
            "delta": None,
        },
    ]

    render_metrics_grid(system_metrics, cols=4)

    # Recent trading signals
    if trading_signals and trading_signals.get("data"):
        st.subheader("📡 Recent Trading Signals")
        signals_actual = trading_signals.get("data", {})
        signals_data = signals_actual.get("signals", [])
        render_dataframe_safely(signals_data, "Trading Signals")

    # Performance summary
    st.subheader("📈 Performance Summary")

    # Get performance data
    performance_data = _fetch_api("/api/dashboard/performance")
    if performance_data:
        perf_actual = performance_data.get("data", {})
        portfolio_perf = perf_actual.get("portfolio_performance", {})

        perf_metrics = [
            {
                "label": "24h Return",
                "value": format_percentage(portfolio_perf.get("total_return_24h", 0)),
                "delta": None,
            },
            {
                "label": "7d Return",
                "value": format_percentage(portfolio_perf.get("total_return_7d", 0)),
                "delta": None,
            },
            {
                "label": "Sharpe Ratio",
                "value": f"{portfolio_perf.get('sharpe_ratio', 0):.2f}",
                "delta": None,
            },
            {
                "label": "Win Rate",
                "value": f"{portfolio_perf.get('win_rate', 0):.1f}%",
                "delta": None,
            },
        ]

        render_metrics_grid(perf_metrics, cols=4)

    # Enhanced Smart Alerts Panel
    st.subheader("🔔 Enhanced Smart Alerts")

    # Get alerts data
    alerts_data = _fetch_api("/api/alerts/recent")

    if alerts_data and alerts_data.get("data"):
        alerts = alerts_data.get("data", [])
        for alert in alerts[:5]:  # Show last 5 alerts
            alert_type = alert.get("type", "info")
            message = alert.get("message", "No message")
            timestamp = alert.get("timestamp", "Unknown")

            if alert_type == "error":
                st.error(f"🚨 {message} ({timestamp})")
            elif alert_type == "warning":
                st.warning(f"⚠️ {message} ({timestamp})")
            else:
                st.info(f"ℹ️ {message} ({timestamp})")

    # Advanced System Health Dashboard
    st.subheader("🏥 Advanced System Health Dashboard")

    # Get comprehensive health data
    health_data = _fetch_api("/api/system/health")
    phase5_health = _fetch_api("/api/phase5/health")
    quantum_health = _fetch_api("/api/quantum/health")
    blockchain_health = _fetch_api("/api/blockchain/health")
    _fetch_api("/api/mining/health")

    # Health metrics grid
    col_health1, col_health2, col_health3, col_health4 = responsive_columns(4)

    with col_health1:
        # Main System Health
        if health_data and health_data.get("data"):
            main_health = health_data.get("data", {})
            cpu_usage = main_health.get("cpu_usage", 0)
            memory_usage = main_health.get("memory_usage", 0)
            st.metric("💻 CPU Usage", f"{cpu_usage:.1f}%")
            st.metric("🧠 Memory Usage", f"{memory_usage:.1f}%")

    with col_health2:
        # Phase 5 Health
        if phase5_health and phase5_health.get("data"):
            phase5_health_data = phase5_health.get("data", {})
            phase5_status = phase5_health_data.get("status", "Unknown")
            st.metric("🔬 Phase 5", phase5_status)

    with col_health3:
        # Quantum Health
        if quantum_health and quantum_health.get("data"):
            quantum_health_data = quantum_health.get("data", {})
            quantum_status = quantum_health_data.get("status", "Unknown")
            st.metric("⚛️ Quantum", quantum_status)

    with col_health4:
        # Blockchain Health
        if blockchain_health and blockchain_health.get("data"):
            blockchain_health_data = blockchain_health.get("data", {})
            blockchain_status = blockchain_health_data.get("status", "Unknown")
            st.metric("🔗 Blockchain", blockchain_status)

    # Real-time System Events
    st.subheader("📡 Real-time System Events")

    # Get system events
    events_data = _fetch_api("/api/system/events")

    if events_data and events_data.get("data"):
        events = events_data.get("data", [])
        for event in events[:10]:  # Show last 10 events
            event_type = event.get("type", "info")
            event_message = event.get("message", "No message")
            event_timestamp = event.get("timestamp", "Unknown")
            event_source = event.get("source", "System")

            if event_type == "critical":
                st.error(f"🚨 [{event_source}] {event_message} ({event_timestamp})")
            elif event_type == "warning":
                st.warning(f"⚠️ [{event_source}] {event_message} ({event_timestamp})")
            elif event_type == "success":
                st.success(f"✅ [{event_source}] {event_message} ({event_timestamp})")
            else:
                st.info(f"ℹ️ [{event_source}] {event_message} ({event_timestamp})")
    else:
        st.info("No recent system events")

    # Footer with enhanced system info
    st.markdown("---")
    col_footer1, col_footer2, col_footer3 = responsive_columns(3)

    with col_footer1:
        st.caption("Ultimate Dashboard v2.0")

    with col_footer2:
        st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")

    with col_footer3:
        st.caption("Advanced Monitoring: Active")
