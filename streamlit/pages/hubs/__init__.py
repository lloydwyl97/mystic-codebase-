"""Hubs package for Streamlit dashboard."""

"""
Hubs Module - Centralized hub management for the Mystic Super Dashboard
"""

# Import all hub functions
from .command_center import render_command_center
from .trading_hub import render_trading_hub
from .ai_intelligence_hub import render_ai_intelligence_hub
from .autobuy_hub import render_autobuy_hub
from .system_control_hub import render_system_control_hub
from .advanced_tech_hub import render_advanced_tech_hub

# Hub registry for dynamic loading
HUBS_REGISTRY = {
    "command_center": render_command_center,
    "trading_hub": render_trading_hub,
    "ai_intelligence_hub": render_ai_intelligence_hub,
    "autobuy_hub": render_autobuy_hub,
    "system_control_hub": render_system_control_hub,
    "advanced_tech_hub": render_advanced_tech_hub,
}

# Export all hub functions
__all__ = [
    "render_command_center",
    "render_trading_hub",
    "render_ai_intelligence_hub",
    "render_autobuy_hub",
    "render_system_control_hub",
    "render_advanced_tech_hub",
    "HUBS_REGISTRY",
]
