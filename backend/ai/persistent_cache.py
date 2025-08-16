from importlib import import_module as _im
_mod = _im("backend.modules.ai.persistent_cache")
# re-export everything public
for _k in dir(_mod):
    if not _k.startswith("_"):
        globals()[_k] = getattr(_mod, _k)
