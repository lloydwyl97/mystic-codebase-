from importlib import import_module
from pathlib import Path

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware

BASE_PKG = "backend"
PKG_DIR = Path(__file__).resolve().parent

def create_app() -> FastAPI:
    app = FastAPI(title="Mystic Backend", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 1) Prefer a consolidated router if present (no double-includes)
    used_consolidated = False
    try:
        mod = import_module(f"{BASE_PKG}.endpoints.consolidated_router")
        router = getattr(mod, "router", None)
        if isinstance(router, APIRouter):
            app.include_router(router)
            used_consolidated = True
    except Exception:
        pass  # fall back to discovery

    # 2) Otherwise, auto-discover any *_endpoints.py exposing `router`
    if not used_consolidated:
        endpoints_root = PKG_DIR / "endpoints"
        if endpoints_root.is_dir():
            for py in endpoints_root.rglob("*_endpoints.py"):
                mod_name = f"{BASE_PKG}." + str(py.relative_to(PKG_DIR)).replace("\\", "/").replace("/", ".")[:-3]
                try:
                    mod = import_module(mod_name)
                    router = getattr(mod, "router", None)
                    if isinstance(router, APIRouter):
                        app.include_router(router)
                except Exception:
                    # don't fail whole app if one module is bad
                    continue

    @app.get("/healthz")
    def _healthz():
        return {"ok": True}

    return app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(create_app(), host="127.0.0.1", port=9000)
