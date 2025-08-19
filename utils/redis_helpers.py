"""
Lightweight helpers to normalize Redis return types across sync/async clients
and different configurations (bytes vs str), so downstream JSON parsing and
string operations are safe.

Usage:
- Use to_str(...) for single-value reads like lpop/get
- Use to_str_list([...]) for list reads like lrange
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def to_str(value: Any) -> str | None:
    """Return value as str, decoding bytes with UTF-8; None stays None.

    This avoids bytes passed into json.loads or string operations.
    """
    if value is None:
        return None
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except Exception:
            # Fallback with errors ignored to avoid hard failures on rare bad bytes
            return value.decode("utf-8", errors="ignore")
    if isinstance(value, str):
        return value
    # For non-bytes, non-str (rare for Redis), coerce to str explicitly
    return str(value)


def to_str_list(values: Any) -> list[str]:
    """Normalize a Redis list result (possibly list[bytes|str]) to list[str]."""
    if values is None:
        return []
    if not isinstance(values, Iterable) or isinstance(values, str | bytes):
        # Defensive: unexpected scalar; coerce to single-element list
        s = to_str(values)
        return [s] if s is not None else []
    result: list[str] = []
    for item in values:
        s = to_str(item)
        if s is not None:
            result.append(s)
    return result


