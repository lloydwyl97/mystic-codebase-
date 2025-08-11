"""
📱 RESPONSIVE LAYOUT - Mobile-friendly design components
Responsive layout utilities for better mobile and desktop experience
"""

import streamlit as st
from typing import Tuple


def responsive_columns(num_cols: int) -> Tuple:
    """Create responsive columns that adapt to screen size"""
    if num_cols <= 0:
        return (st.container(),)

    # For mobile devices, use fewer columns
    if st.session_state.get("is_mobile", False):
        # On mobile, use max 2 columns
        actual_cols = min(num_cols, 2)
    else:
        actual_cols = num_cols

    return st.columns(actual_cols)


def mobile_friendly_metric(label: str, value: str, delta: str = None):
    """Render a mobile-friendly metric"""
    st.metric(label=label, value=value, delta=delta)


def mobile_optimized_sidebar():
    """Apply mobile optimizations to the sidebar"""
    # Add mobile detection CSS
    st.markdown(
        """
        <style>
        /* Mobile Responsive CSS */
        @media (max-width: 768px) {
            section[data-testid="stSidebar"] {
                width: 100% !important;
                min-width: 100% !important;
                max-width: 100% !important;
                position: relative !important;
            }
            
            .main .block-container {
                padding: 0.5rem !important;
                margin-left: 0 !important;
            }
            
            .stButton > button {
                width: 100% !important;
                margin: 0.25rem 0 !important;
            }
            
            .stSelectbox > div > div {
                width: 100% !important;
            }
            
            .stMetric {
                margin: 0.5rem 0 !important;
            }
        }
        
        /* Tablet Responsive CSS */
        @media (min-width: 769px) and (max-width: 1024px) {
            section[data-testid="stSidebar"] {
                width: 300px !important;
                min-width: 300px !important;
                max-width: 300px !important;
            }
        }
        
        /* Desktop Responsive CSS */
        @media (min-width: 1025px) {
            section[data-testid="stSidebar"] {
                width: 400px !important;
                min-width: 350px !important;
                max-width: 500px !important;
            }
        }
        </style>
    """,
        unsafe_allow_html=True,
    )


def responsive_dataframe(data, title: str = ""):
    """Render a responsive dataframe with mobile optimization"""
    if title:
        st.subheader(title)

    # Use container width for responsive behavior
    st.dataframe(data, use_container_width=True)


def responsive_chart(fig, title: str = ""):
    """Render a responsive chart with mobile optimization"""
    if title:
        st.subheader(title)

    # Use container width for responsive behavior
    st.plotly_chart(fig, use_container_width=True)


def mobile_friendly_button(text: str, key: str = None, type: str = "primary"):
    """Create a mobile-friendly button"""
    return st.button(text, key=key, type=type, use_container_width=True)


def responsive_tabs(tab_names: list):
    """Create responsive tabs that work well on mobile"""
    return st.tabs(tab_names)


def mobile_optimized_expander(title: str, expanded: bool = False):
    """Create a mobile-optimized expander"""
    return st.expander(title, expanded=expanded)


def responsive_metric_grid(metrics: list, cols: int = 4):
    """Create a responsive metric grid"""
    # Determine actual number of columns based on screen size
    if st.session_state.get("is_mobile", False):
        actual_cols = min(cols, 2)
    else:
        actual_cols = cols

    # Create columns
    columns = st.columns(actual_cols)

    # Distribute metrics across columns
    for i, metric in enumerate(metrics):
        col_idx = i % actual_cols
        with columns[col_idx]:
            st.metric(
                label=metric.get("label", ""),
                value=metric.get("value", ""),
                delta=metric.get("delta"),
            )


def mobile_friendly_selectbox(label: str, options: list, index: int = 0):
    """Create a mobile-friendly selectbox"""
    return st.selectbox(label, options, index=index, help=f"Select {label.lower()}")


def responsive_slider(label: str, min_value: float, max_value: float, value: float):
    """Create a responsive slider"""
    return st.slider(label, min_value=min_value, max_value=max_value, value=value)


def mobile_optimized_number_input(
    label: str, min_value: float, max_value: float, value: float
):
    """Create a mobile-optimized number input"""
    return st.number_input(label, min_value=min_value, max_value=max_value, value=value)


def responsive_checkbox(label: str, value: bool = False):
    """Create a responsive checkbox"""
    return st.checkbox(label, value=value)


def mobile_friendly_text_input(label: str, value: str = ""):
    """Create a mobile-friendly text input"""
    return st.text_input(label, value=value)


def responsive_text_area(label: str, value: str = ""):
    """Create a responsive text area"""
    return st.text_area(label, value=value)


def mobile_optimized_file_uploader(label: str, type: str = None):
    """Create a mobile-optimized file uploader"""
    return st.file_uploader(label, type=type)


def responsive_download_button(label: str, data, file_name: str, mime: str = None):
    """Create a responsive download button"""
    return st.download_button(
        label=label,
        data=data,
        file_name=file_name,
        mime=mime,
        use_container_width=True,
    )


def mobile_friendly_radio(label: str, options: list, index: int = 0):
    """Create a mobile-friendly radio button group"""
    return st.radio(label, options, index=index, horizontal=True)


def responsive_multiselect(label: str, options: list, default: list = None):
    """Create a responsive multiselect"""
    return st.multiselect(label, options, default=default)


def mobile_optimized_date_input(label: str, value=None):
    """Create a mobile-optimized date input"""
    return st.date_input(label, value=value)


def responsive_time_input(label: str, value=None):
    """Create a responsive time input"""
    return st.time_input(label, value=value)


def mobile_friendly_color_picker(label: str, value: str = "#000000"):
    """Create a mobile-friendly color picker"""
    return st.color_picker(label, value=value)


def responsive_progress_bar(value: float, text: str = None):
    """Create a responsive progress bar"""
    return st.progress(value, text=text)


def mobile_optimized_spinner(text: str = "Loading..."):
    """Create a mobile-optimized spinner"""
    return st.spinner(text)


def responsive_success(message: str):
    """Display a responsive success message"""
    st.success(message)


def responsive_error(message: str):
    """Display a responsive error message"""
    st.error(message)


def responsive_warning(message: str):
    """Display a responsive warning message"""
    st.warning(message)


def responsive_info(message: str):
    """Display a responsive info message"""
    st.info(message)


def mobile_friendly_caption(text: str):
    """Display a mobile-friendly caption"""
    st.caption(text)


def responsive_markdown(text: str):
    """Display responsive markdown"""
    st.markdown(text)


def mobile_optimized_code(code: str, language: str = "python"):
    """Display mobile-optimized code"""
    st.code(code, language=language)


def responsive_json(data):
    """Display responsive JSON"""
    st.json(data)


def mobile_friendly_balloons():
    """Display mobile-friendly balloons"""
    st.balloons()


def responsive_snow():
    """Display responsive snow effect"""
    st.snow()


def create_metric_card(label: str, value: str, color: str = "blue"):
    """Create a styled metric card with color coding"""
    # Color mapping for different metric types
    color_map = {
        "green": "#00ff00",
        "red": "#ff0000",
        "orange": "#ffa500",
        "blue": "#0066cc",
        "purple": "#800080",
        "yellow": "#ffff00",
    }

    # Get the color value
    color_value = color_map.get(color.lower(), color_map["blue"])

    # Create styled metric card
    st.markdown(
        f"""
    <div style="
        background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%);
        border: 2px solid {color_value};
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    ">
        <div style="color: {color_value}; font-size: 24px; font-weight: bold; margin-bottom: 5px;">
            {value}
        </div>
        <div style="color: #ffffff; font-size: 14px;">
            {label}
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def create_chart_container(fig, title: str = ""):
    """Create a styled chart container with title and responsive design"""
    # Create container with styling
    st.markdown(
        """
    <div style="
        background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%);
        border: 2px solid #00e6e6;
        border-radius: 10px;
        padding: 15px;
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    ">
    """,
        unsafe_allow_html=True,
    )

    # Display title if provided
    if title:
        st.markdown(
            f"<h4 style='color: #00e6e6; text-align: center; margin-bottom: 15px;'>{title}</h4>",
            unsafe_allow_html=True,
        )

    # Display the chart
    st.plotly_chart(fig, use_container_width=True)

    # Close container
    st.markdown("</div>", unsafe_allow_html=True)
