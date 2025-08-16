import importlib, sys
_pkg = importlib.import_module("backend.modules.ai")
for _name in getattr(_pkg, "__all__", []):
    globals()[_name] = getattr(_pkg, _name)
# Make sure importing backend.ai.persistent_cache works
sys.modules[f"{__name__}.persistent_cache"] = importlib.import_module("backend.modules.ai.persistent_cache")
# --- Compatibility shims for legacy callers (update_* methods) ---
# We already import import_module above in this file.
try:
    _pcm = import_module("backend.modules.ai.persistent_cache")
    PC = getattr(_pcm, "PersistentCache", None)

    def _bind_update(ns_name):
        def _update(self, data=None, **kwargs):
            payload = data if data is not None else kwargs
            key = f"market:{ns_name}"
            try:
                # Prefer the real setter if present
                return getattr(self, "set")(key, payload)
            except Exception:
                # Best-effort in-memory fallback
                self._store = getattr(self, "_store", {})
                self._store[key] = payload
                return True
        return _update

    if PC and not hasattr(PC, "update_binance"):
        setattr(PC, "update_binance", _bind_update("binanceus"))
    if PC and not hasattr(PC, "update_coinbase"):
        setattr(PC, "update_coinbase", _bind_update("coinbase"))
    if PC and not hasattr(PC, "update_coingecko"):
        setattr(PC, "update_coingecko", _bind_update("coingecko"))
except Exception:
    # Dont break import on environments that dont need this.
    pass
