"""Icon helpers with safe fallbacks for Streamlit dashboard.

Re-exports from legacy `pages/components/icons.py` as canonical `streamlit.icons`.
"""

from __future__ import annotations

# No runtime typing imports needed here

try:
    from streamlit._pages.components.icons import get_coin_icon, render_text_badge  # type: ignore
except Exception:  # pragma: no cover
    from ._pages.components.icons import get_coin_icon, render_text_badge  # type: ignore

__all__ = ["get_coin_icon", "render_text_badge"]


