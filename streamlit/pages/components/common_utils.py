"""
🛠️ COMMON UTILITIES - Shared functions for the dashboard
Centralized utility functions for rendering, formatting, and data handling
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, List
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px


def render_page_header(title: str, subtitle: str = ""):
    """Render a consistent page header with title and subtitle"""
    st.title(title)
    if subtitle:
        st.markdown(f"*{subtitle}*")
    st.markdown("---")


def render_metrics_grid(metrics: List[Dict[str, Any]], cols: int = 4):
    """Render metrics in a responsive grid layout"""
    if not metrics:
        return

    # Create columns for metrics
    columns = st.columns(cols)

    for i, metric in enumerate(metrics):
        col_idx = i % cols
        with columns[col_idx]:
            st.metric(
                label=metric.get("label", ""),
                value=metric.get("value", ""),
                delta=metric.get("delta"),
            )


def render_dataframe_safely(data: List[Dict[str, Any]], title: str = ""):
    """Safely render a dataframe with error handling"""
    if not data:
        st.info("No data available")
        return

    try:
        df = pd.DataFrame(data)
        if title:
            st.subheader(title)
        st.dataframe(df, use_container_width=True)
    except Exception as e:
        st.error(f"Error rendering data: {str(e)}")


def format_currency(value: float) -> str:
    """Format a number as currency"""
    try:
        if value >= 1_000_000_000:
            return f"${value/1_000_000_000:.2f}B"
        elif value >= 1_000_000:
            return f"${value/1_000_000:.2f}M"
        elif value >= 1_000:
            return f"${value/1_000:.2f}K"
        else:
            return f"${value:.2f}"
    except (TypeError, ValueError):
        return "$0.00"


def format_percentage(value: float) -> str:
    """Format a number as percentage"""
    try:
        return f"{value:.2f}%"
    except (TypeError, ValueError):
        return "0.00%"


def get_fallback_data(data_type: str) -> Dict[str, Any]:
    """Get fallback data when APIs are unavailable"""
    fallback_data = {
        "portfolio": {
            "total_value": 100000.00,
            "daily_change": 2.5,
            "total_pnl": 5000.00,
            "pnl_percentage": 5.0,
            "positions": 5,
            "win_rate": 75.0,
            "positions_list": [
                {
                    "symbol": "BTCUSDT",
                    "amount": 0.5,
                    "value": 25000.00,
                    "pnl": 1250.00,
                },
                {
                    "symbol": "ETHUSDT",
                    "amount": 2.0,
                    "value": 4000.00,
                    "pnl": 200.00,
                },
                {
                    "symbol": "SOLUSDT",
                    "amount": 100.0,
                    "value": 15000.00,
                    "pnl": 750.00,
                },
            ],
        },
        "ai_strategies": {
            "ai_strategies": [
                {
                    "name": "Neural Network",
                    "status": "ACTIVE",
                    "performance": 12.5,
                    "last_updated": "2024-01-15",
                    "confidence": 85.0,
                },
                {
                    "name": "LSTM Model",
                    "status": "ACTIVE",
                    "performance": 8.2,
                    "last_updated": "2024-01-15",
                    "confidence": 78.0,
                },
                {
                    "name": "Random Forest",
                    "status": "ACTIVE",
                    "performance": 15.1,
                    "last_updated": "2024-01-15",
                    "confidence": 92.0,
                },
            ]
        },
        "market_data": {
            "markets": [
                {
                    "symbol": "BTCUSDT",
                    "price": 50000.00,
                    "change_24h": 2.5,
                    "volume": 1000000.00,
                    "market_cap": 1000000000.00,
                },
                {
                    "symbol": "ETHUSDT",
                    "price": 3000.00,
                    "change_24h": 1.8,
                    "volume": 500000.00,
                    "market_cap": 300000000.00,
                },
                {
                    "symbol": "SOLUSDT",
                    "price": 150.00,
                    "change_24h": 5.2,
                    "volume": 200000.00,
                    "market_cap": 50000000.00,
                },
            ]
        },
    }

    return fallback_data.get(data_type, {})


def get_color_by_value(value: Any, threshold: float = 0) -> str:
    """Get color based on value (positive/negative)"""
    # Handle string values by converting to float if possible
    if isinstance(value, str):
        try:
            # Remove any non-numeric characters except decimal point and minus
            clean_value = ''.join(c for c in value if c.isdigit() or c in '.-')
            if clean_value:
                value = float(clean_value)
            else:
                return "gray"  # Default color for non-numeric strings
        except (ValueError, TypeError):
            return "gray"  # Default color for unparseable strings
    
    # Now handle numeric values
    try:
        numeric_value = float(value)
        if numeric_value > threshold:
            return "green"
        elif numeric_value < threshold:
            return "red"
        else:
            return "gray"
    except (ValueError, TypeError):
        return "gray"  # Default color for non-numeric values


def render_phase5_overlay(metrics: Dict[str, str]) -> Dict[str, str]:
    """Render Phase5 overlay metrics and return default values"""
    st.subheader("🌟 Phase5 Overlay Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("🧠 Neuro Sync", metrics.get("neuro_sync", "N/A"))

    with col2:
        st.metric("🌌 Cosmic Signal", metrics.get("cosmic_signal", "N/A"))

    with col3:
        st.metric("✨ Aura Alignment", metrics.get("aura_alignment", "N/A"))

    with col4:
        st.metric("🌀 Interdim Activity", metrics.get("interdim_activity", "N/A"))
    
    # Return default values for fallback
    return {
        "neuro_sync": metrics.get("neuro_sync", "0%"),
        "cosmic_signal": metrics.get("cosmic_signal", "0%"),
        "aura_alignment": metrics.get("aura_alignment", "0%"),
        "interdim_activity": metrics.get("interdim_activity", "0%")
    }


def render_quantum_indicators(indicators: Dict[str, Any]):
    """Render quantum indicators"""
    st.subheader("⚛️ Quantum Indicators")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("🔮 Quantum Signal", indicators.get("quantum_signal", "N/A"))

    with col2:
        st.metric("🎯 Trade Probability", indicators.get("trade_probability", "N/A"))

    with col3:
        st.metric("📊 Entropy Index", indicators.get("entropy_index", "N/A"))


def render_quantum_waveform_chart(waveform_data: Dict[str, Any]):
    """Render quantum waveform chart"""
    st.subheader("🌊 Quantum Waveform")

    try:
        y_data = waveform_data.get("y", [])
        if y_data:
            x_data = list(range(len(y_data)))

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=x_data,
                    y=y_data,
                    mode="lines",
                    name="Quantum Waveform",
                    line=dict(color="cyan", width=2),
                )
            )

            fig.update_layout(
                title="Quantum Waveform Analysis",
                xaxis_title="Time",
                yaxis_title="Amplitude",
                height=400,
                showlegend=False,
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No quantum waveform data available")
    except Exception as e:
        st.error(f"Error rendering quantum waveform: {str(e)}")


def create_performance_chart(
    data: List[Dict[str, Any]], x_key: str, y_key: str, title: str
):
    """Create a performance chart using plotly"""
    try:
        if not data:
            st.info("No data available for chart")
            return

        x_values = [item.get(x_key, 0) for item in data]
        y_values = [item.get(y_key, 0) for item in data]

        fig = px.line(
            x=x_values,
            y=y_values,
            title=title,
            labels={"x": x_key.title(), "y": y_key.title()},
        )

        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")


def render_status_indicator(
    status: bool, true_text: str = "Active", false_text: str = "Inactive"
):
    """Render a status indicator with color coding"""
    if status:
        st.success(f"🟢 {true_text}")
    else:
        st.error(f"🔴 {false_text}")


def format_timestamp(timestamp: str) -> str:
    """Format timestamp for display"""
    try:
        # Try to parse and format the timestamp
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError, AttributeError) as e:
        # Log the specific error for debugging
        import logging
        logging.debug(f"Timestamp formatting failed: {e}")
        return str(timestamp)


def render_loading_spinner(text: str = "Loading..."):
    """Render a loading spinner with custom text"""
    with st.spinner(text):
        st.empty()


def render_error_message(error: str, title: str = "Error"):
    """Render a formatted error message"""
    st.error(f"**{title}:** {error}")


def render_success_message(message: str, title: str = "Success"):
    """Render a formatted success message"""
    st.success(f"**{title}:** {message}")


def render_warning_message(message: str, title: str = "Warning"):
    """Render a formatted warning message"""
    st.warning(f"**{title}:** {message}")


def render_info_message(message: str, title: str = "Info"):
    """Render a formatted info message"""
    st.info(f"**{title}:** {message}")
