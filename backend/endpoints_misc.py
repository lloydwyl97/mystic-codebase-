import logging
import time
from typing import Any

from .app_factory import app
from .services.service_manager import service_manager

logger = logging.getLogger("main")


@app.get("/health")
async def health_check() -> dict[str, str | float]:
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "message": "Mystic Trading Platform is running",
    }


@app.get("/api/health/comprehensive")
async def comprehensive_health_check() -> dict[str, str | float | dict[str, Any]]:
    try:
        health_status: dict[str, str | float | dict[str, Any]] = {
            "status": "healthy",
            "timestamp": time.time(),
            "version": "1.0.0",
            "services": {},
        }
        if service_manager:
            health_status["services"] = service_manager.get_health_status()
        if service_manager and service_manager.redis_client:
            try:
                service_manager.redis_client.ping()
                services_dict = health_status["services"]
                if isinstance(services_dict, dict):
                    services_dict["redis"] = "healthy"
            except Exception:
                services_dict = health_status["services"]
                if isinstance(services_dict, dict):
                    services_dict["redis"] = "unhealthy"
        return health_status
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": time.time(),
            "error": str(e),
        }


@app.get("/api/version")
async def get_version() -> dict[str, str | list[str]]:
    return {
        "version": "1.0.0",
        "build_date": "2024-06-22",
        "environment": "production",
        "features": [
            "real-time trading",
            "AI-powered analytics",
            "social trading",
            "mobile PWA support",
            "advanced order types",
            "risk management",
            "auto-trading bots",
        ],
    }


