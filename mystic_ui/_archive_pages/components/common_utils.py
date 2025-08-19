from __future__ import annotations

import typing as t


def safe_number_format(value: t.Any, digits: int = 2) -> str:
    try:
        return f"{float(value):,.{digits}f}"
    except Exception:
        return "-"


def ensure_state_defaults(state: dict[str, t.Any] | None | t.MutableMapping[str, t.Any], defaults: dict[str, t.Any]) -> dict[str, t.Any]:
    s = dict(state) if state is not None else {}
    for k, v in defaults.items():
        s.setdefault(k, v)
    return s


def inject_global_theme() -> None:
    # no-op placeholder for archived pages; keep public surface
    return None


def get_app_state() -> dict[str, t.Any]:
    # lightweight state impl for archived pages
    return {}


def render_sidebar_controls() -> None:
    # no-op wrapper used by archived pages
    return None


class display_guard:
    """Context manager wrapper for archived pages (no-op)."""

    def __init__(self, _label: str | None = None) -> None:
        self._label = _label

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


