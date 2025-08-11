"""
⚙️ SYSTEM CONTROL HUB - Admin, tools, notifications
Consolidated system administration with smart navigation
Focused on core system management (advanced tech moved to Advanced Tech Hub)
"""

import streamlit as st
from streamlit.ui_guard import display_guard
from streamlit.data_client import fetch_api as _fetch_api, post_api as _post_api  # type: ignore[attr-defined]
from datetime import datetime
import plotly.express as px

# Import our modular components
from streamlit.api_client import api_client  # centralized api client
from streamlit.pages.components.common_utils import (
    render_page_header,
    render_metrics_grid,
)
from streamlit.pages.components.responsive_layout import (
    responsive_columns,
)

# Import the real, full-featured system admin page functions
from streamlit.pages.system_admin.system_admin import render_system_admin_page
from streamlit.pages.system_admin.notifications import render_notifications_page
from streamlit.pages.system_admin.tools import render_tools_page

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


def render_system_control_hub():
    """Display system control hub with consolidated admin operations"""

    # Page header
    render_page_header("⚙️ System Control Hub", "Admin, tools, and notifications")

    # System section navigation
    st.subheader("🔧 System Operations")

    system_sections = {
        "📊 System Admin": "system_admin",
        "🔔 Notifications": "notifications",
        "🛠️ Tools": "tools",
    }

    # Create tabs for different system sections
    selected_section = st.selectbox(
        "Select System Section:", list(system_sections.keys()), index=0, key="system_control_section"
    )

    # Render selected section using the real, full-featured page functions
    if selected_section == "📊 System Admin":
        render_system_admin_page()
    elif selected_section == "🔔 Notifications":
        render_notifications_page()
    elif selected_section == "🛠️ Tools":
        render_tools_page()

    # System Health Overview
    st.subheader("🏥 System Health Overview")

    # Get system health data
    with display_guard("System Health Fetch"):
        health_data = _fetch_api("/api/system/health")
        _fetch_api("/api/system/status")

    if health_data and health_data.get("data"):
        health_info = health_data.get("data", {})

        health_metrics = [
            {
                "label": "CPU Usage",
                "value": f"{health_info.get('cpu_usage', 0):.1f}%",
                "delta": None,
            },
            {
                "label": "Memory Usage",
                "value": f"{health_info.get('memory_usage', 0):.1f}%",
                "delta": None,
            },
            {
                "label": "Disk Usage",
                "value": f"{health_info.get('disk_usage', 0):.1f}%",
                "delta": None,
            },
            {
                "label": "Network Status",
                "value": (
                    "🟢 Online"
                    if health_info.get("network_status", False)
                    else "🔴 Offline"
                ),
                "delta": None,
            },
        ]

        render_metrics_grid(health_metrics, cols=4)

    # Quick System Actions
    st.subheader("⚡ Quick System Actions")

    col_actions1, col_actions2, col_actions3, col_actions4 = responsive_columns(4)

    with col_actions1:
        if st.button("🔄 Restart System", type="secondary", use_container_width=True):
            restart_result = _post_api("/api/system/restart", {})
            if restart_result:
                st.success("System restart initiated!")
            else:
                st.error("Failed to restart system")

    with col_actions2:
        if st.button("🧹 Clear Cache", type="secondary", use_container_width=True):
            cache_result = _post_api("/api/system/clear-cache", {})
            if cache_result:
                st.success("Cache cleared successfully!")
            else:
                st.error("Failed to clear cache")

    with col_actions3:
        if st.button("📊 Generate Report", type="secondary", use_container_width=True):
            report_result = _post_api("/api/system/generate-report", {})
            if report_result:
                st.success("System report generated!")
            else:
                st.error("Failed to generate report")

    with col_actions4:
        if st.button("🔍 System Check", type="secondary", use_container_width=True):
            check_result = _post_api("/api/system/health-check", {})
            if check_result:
                st.success("System health check completed!")
            else:
                st.error("Failed to complete health check")

    # System Configuration
    st.subheader("⚙️ System Configuration")

    # Get current configuration
    config_data = _fetch_api("/api/system/config")

    if config_data and config_data.get("data"):
        current_config = config_data.get("data", {})

        col_config1, col_config2 = responsive_columns(2)

        with col_config1:
            st.subheader("Performance Settings")

            # Performance configuration
            max_cpu_usage = st.slider(
                "Max CPU Usage (%)",
                min_value=50,
                max_value=100,
                value=current_config.get("max_cpu_usage", 80),
                help="Maximum allowed CPU usage",
            )

            max_memory_usage = st.slider(
                "Max Memory Usage (%)",
                min_value=50,
                max_value=100,
                value=current_config.get("max_memory_usage", 85),
                help="Maximum allowed memory usage",
            )

            log_level = st.selectbox(
                "Log Level",
                ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                index=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"].index(
                    current_config.get("log_level", "INFO")
                ),
                help="System logging level",
                key="system_log_level"
            )

        with col_config2:
            st.subheader("Security Settings")

            # Security configuration
            enable_2fa = st.checkbox(
                "Enable 2FA",
                value=current_config.get("enable_2fa", True),
                help="Enable two-factor authentication",
            )

            session_timeout = st.number_input(
                "Session Timeout (minutes)",
                min_value=5,
                max_value=1440,
                value=current_config.get("session_timeout", 30),
                help="Session timeout in minutes",
            )

            max_login_attempts = st.number_input(
                "Max Login Attempts",
                min_value=3,
                max_value=10,
                value=current_config.get("max_login_attempts", 5),
                help="Maximum login attempts before lockout",
            )

        # Save configuration button
        if st.button("💾 Save Configuration", type="primary", use_container_width=True):
            # Prepare configuration data
            new_config = {
                "max_cpu_usage": max_cpu_usage,
                "max_memory_usage": max_memory_usage,
                "log_level": log_level,
                "enable_2fa": enable_2fa,
                "session_timeout": session_timeout,
                "max_login_attempts": max_login_attempts,
            }

            # Save configuration
            config_result = _post_api("/api/system/config", new_config)
            if config_result:
                st.success("Configuration saved successfully!")
            else:
                st.error("Failed to save configuration")

    # Recent System Events
    st.subheader("📡 Recent System Events")

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

    # System Performance Analytics
    st.subheader("📊 System Performance Analytics")

    # Get performance data
    performance_data = _fetch_api("/api/system/performance")

    if performance_data and performance_data.get("data"):
        perf_data = performance_data.get("data", {})

        col_perf1, col_perf2 = responsive_columns(2)

        with col_perf1:
            # CPU Usage Over Time
            if "cpu_usage_history" in perf_data:
                cpu_history = perf_data["cpu_usage_history"]
                fig_cpu = px.line(
                    x=cpu_history.get("timestamps", []),
                    y=cpu_history.get("values", []),
                    title="CPU Usage Over Time",
                    labels={"x": "Time", "y": "CPU Usage (%)"},
                )
                fig_cpu.update_layout(height=300)
                st.plotly_chart(fig_cpu, use_container_width=True)

        with col_perf2:
            # Memory Usage Over Time
            if "memory_usage_history" in perf_data:
                memory_history = perf_data["memory_usage_history"]
                fig_memory = px.line(
                    x=memory_history.get("timestamps", []),
                    y=memory_history.get("values", []),
                    title="Memory Usage Over Time",
                    labels={"x": "Time", "y": "Memory Usage (%)"},
                )
                fig_memory.update_layout(height=300)
                st.plotly_chart(fig_memory, use_container_width=True)
    else:
        st.info("No performance analytics available")

    # Footer with system info
    st.markdown("---")
    col_footer1, col_footer2, col_footer3 = responsive_columns(3)

    with col_footer1:
        st.caption("System Control Hub v2.0")

    with col_footer2:
        st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")

    with col_footer3:
        st.caption("Advanced Tech: See Advanced Tech Hub")
