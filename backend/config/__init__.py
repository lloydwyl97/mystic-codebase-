"""Backend config package.

This package provides configuration modules for the backend. It also re-exports
selected settings from the root `config` package to offer a stable import path
(`backend.config.*`) for Streamlit and backend code.
"""

# Intentionally minimal; concrete modules live alongside this file.


# --- settings re-export shim (safe, idempotent) ---
try:
    from .settings import settings as settings  # preferred re-export
except Exception:  # ultra-safe env fallback
    import os
    from typing import Optional
    class _Settings:
        def __getattr__(self, name: str) -> Optional[str]: return os.getenv(name)
        def __getitem__(self, k: str) -> Optional[str]:    return os.getenv(k)
    settings = _Settings()
# --------------------------------------------------

__all__ = ["settings"]
