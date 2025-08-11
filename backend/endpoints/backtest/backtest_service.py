from fastapi import APIRouter

router = APIRouter(prefix="/api/backtest")


@router.get("/ping")
def ping():
    return {"status": "ok", "module": "backtest_service_stub"}

class BacktestService:
    """Minimal stub to satisfy legacy imports without affecting runtime behavior."""

    def __init__(self) -> None:
        pass

    def health(self) -> dict:
        return {"status": "ok"}

__all__ = ["router", "ping", "BacktestService"]


