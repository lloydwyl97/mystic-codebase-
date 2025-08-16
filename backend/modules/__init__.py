"""Lightweight aggregator for backend.modules (tolerates optional submodules)."""
from importlib import import_module

__all__ = []
# Add likely subpackages here; missing ones are skipped cleanly.
for _name in [
    "ai", "data", "trading", "signals", "strategy", "metrics",
    "notifications", "api", "market_data", "analytics"
]:
    try:
        globals()[_name] = import_module(f"{__name__}.{_name}")
        __all__.append(_name)
    except Exception:
        # Module is optional or has its own dependencies not loaded yet.
        pass
