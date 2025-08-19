"""Icon helpers with safe fallbacks for Streamlit dashboard.

Re-exports from legacy `pages/components/icons.py` as canonical `streamlit.icons`.
"""

from __future__ import annotations

# No runtime typing imports needed here
from mystic_ui.icon_loader import get_coin_icon, render_text_badge

__all__ = ["get_coin_icon", "render_text_badge"]



