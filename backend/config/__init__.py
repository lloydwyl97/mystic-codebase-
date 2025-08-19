from .settings import settings

__all__ = ["settings"]

# shim: provide AUTO_BUY_CONFIG if missing
try:
    from .autobuy import AUTO_BUY_CONFIG  # noqa
except Exception:
    AUTO_BY_CONFIG = {}
    AUTO_BUY_CONFIG = {}
