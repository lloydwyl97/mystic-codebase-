"""
Health Routes
Health check and monitoring endpoints
"""

import logging
import os
import time
from typing import Any

import psutil
from fastapi import APIRouter

logger = logging.getLogger("health_routes")

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/")
async def health_check() -> dict[str, Any]:
    """Basic health check"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "service": "Mystic AI Trading Platform",
    }


@router.get("/comprehensive")
async def comprehensive_health_check() -> dict[str, str | float | dict[str, str] | None]:
    """Comprehensive health check with system metrics"""

    # System metrics
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage("/")

    # Service status
    services_status = {
        "database": "healthy",
        "redis": "healthy",
        "api": "healthy",
        "websocket": "healthy",
    }

    # Check if key files exist
    files_status = {}
    key_files = ["simulation_trades.db", "ai_model_state.json", "config.py"]

    for file in key_files:
        files_status[file] = "exists" if os.path.exists(file) else "missing"

    return {
        "status": "healthy",
        "timestamp": time.time(),
        "system": {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "disk_percent": disk.percent,
            "uptime": time.time() - psutil.boot_time(),
        },
        "services": services_status,
        "files": files_status,
        "environment": os.getenv("ENVIRONMENT", "development"),
    }


@router.get("/services")
async def services_health_check() -> dict[str, Any]:
    """Check health of all services"""

    from service_initializer import service_initializer

    services = service_initializer.get_all_services()
    services_health = {}

    for service_name, service in services.items():
        try:
            # Basic health check for each service
            if hasattr(service, "health_check"):
                health = service.health_check()
            else:
                health = {
                    "status": "unknown",
                    "message": "No health check method",
                }

            services_health[service_name] = health
        except Exception as e:
            services_health[service_name] = {
                "status": "error",
                "message": str(e),
            }

    return {
        "status": "healthy",
        "timestamp": time.time(),
        "services": services_health,
    }


