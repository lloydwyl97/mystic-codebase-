# status_router.py
"""
Status Router - Dashboard API for AI Module Monitoring
Provides real-time status of all AI trading modules.
Built for Windows 11 Home + PowerShell + Docker.
"""

from fastapi import APIRouter, HTTPException
import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/status", tags=["System Status"])

# AI Module configurations
AI_MODULES = {
    "ai_strategy_generator": {
        "name": "AI Strategy Generator",
        "description": "Generates new trading strategies using GPT-4",
        "ping_file": "./logs/ai_strategy_generator.ping",
        "container": "mystic_ai_strategy_generator",
    },
    "sentiment_monitor": {
        "name": "Sentiment Monitor",
        "description": "Analyzes crypto news sentiment",
        "ping_file": "./logs/sentiment_monitor.ping",
        "container": "mystic_sentiment_monitor",
    },
    "anomaly_guardian": {
        "name": "Anomaly Guardian",
        "description": "Detects price anomalies using ML",
        "ping_file": "./logs/anomaly_guardian.ping",
        "container": "mystic_anomaly_guardian",
    },
    "risk_optimizer": {
        "name": "Risk Optimizer",
        "description": "Dynamically adjusts risk parameters",
        "ping_file": "./logs/risk_optimizer.ping",
        "container": "mystic_risk_optimizer",
    },
    "trade_explainer": {
        "name": "Trade Explainer",
        "description": "AI-powered trade analysis",
        "ping_file": "./logs/trade_explainer.ping",
        "container": "mystic_trade_explainer",
    },
    "strategy_reaper": {
        "name": "Strategy Reaper",
        "description": "Removes underperforming strategies",
        "ping_file": "./logs/strategy_reaper.ping",
        "container": "mystic_strategy_reaper",
    },
    "portfolio_ai_balance": {
        "name": "Portfolio AI Balance",
        "description": "Portfolio rebalancing recommendations",
        "ping_file": "./logs/portfolio_ai_balance.ping",
        "container": "mystic_portfolio_ai_balance",
    },
    "train_fast_mutations": {
        "name": "Mutation Trainer",
        "description": "Trains and evolves strategies",
        "ping_file": "./logs/train_fast_mutations.ping",
        "container": "mystic_train_fast_mutations",
    },
    "trade_logger": {
        "name": "Trade Logger",
        "description": "Logs all trading activities",
        "ping_file": "./logs/trade_logger.ping",
        "container": "mystic_trade_logger",
    },
    "strategy_mutator": {
        "name": "Strategy Mutator",
        "description": "Mutates and evolves strategies",
        "ping_file": "./logs/strategy_mutator.ping",
        "container": "mystic_strategy_mutator",
    },
    "hyper_optimizer": {
        "name": "Hyper Optimizer",
        "description": "Optimizes strategy parameters",
        "ping_file": "./logs/hyper_optimizer.ping",
        "container": "mystic_optimizer",
    },
}


def read_ping_file(ping_file: str) -> Dict[str, Any]:
    """Read ping file and return status data"""
    try:
        if os.path.exists(ping_file):
            with open(ping_file, "r") as f:
                data = json.load(f)
                return data
        return {"status": "offline", "last_update": None}
    except Exception as e:
        return {"status": "error", "error": str(e), "last_update": None}


def calculate_health_percentage(last_update: str) -> int:
    """Calculate health percentage based on last update time"""
    if not last_update:
        return 0

    try:
        last_time = datetime.fromisoformat(last_update.replace("Z", "+00:00"))
        now = datetime.timezone.utcnow()
        time_diff = (now - last_time).total_seconds()

        # Consider healthy if updated within last 5 minutes
        if time_diff < 300:
            return 100
        elif time_diff < 600:
            return 75
        elif time_diff < 1800:
            return 50
        elif time_diff < 3600:
            return 25
        else:
            return 0
    except Exception as e:
        logger.error(f"Error calculating health percentage: {e}")
        return 0


@router.get("/")
async def get_system_status() -> Dict[str, Any]:
    """Get overall system status"""
    try:
        total_modules = len(AI_MODULES)
        online_modules = 0
        total_health = 0

        module_statuses = []

        for module_id, config in AI_MODULES.items():
            ping_data = read_ping_file(config["ping_file"])
            health = calculate_health_percentage(ping_data.get("last_update"))

            if health > 0:
                online_modules += 1
            total_health += health

            module_status = {
                "id": module_id,
                "name": config["name"],
                "description": config["description"],
                "container": config["container"],
                "status": ping_data.get("status", "offline"),
                "health_percentage": health,
                "last_update": ping_data.get("last_update"),
                "data": {k: v for k, v in ping_data.items() if k not in ["status", "last_update"]},
            }
            module_statuses.append(module_status)

        overall_health = total_health // total_modules if total_modules > 0 else 0
        system_status = (
            "healthy" if overall_health > 75 else "warning" if overall_health > 50 else "critical"
        )

        return {
            "timestamp": datetime.timezone.utcnow().isoformat(),
            "system_status": system_status,
            "overall_health": overall_health,
            "total_modules": total_modules,
            "online_modules": online_modules,
            "offline_modules": total_modules - online_modules,
            "modules": module_statuses,
        }

    except Exception as e:
        from backend.utils.exceptions import APIException, ErrorCode
        raise APIException(
            message="Failed to get system status",
            error_code=ErrorCode.API_RESPONSE_ERROR,
            details={"original_error": str(e)},
            original_exception=e
        )


@router.get("/modules")
async def get_modules_status() -> List[Dict[str, Any]]:
    """Get status of all AI modules"""
    try:
        module_statuses = []

        for module_id, config in AI_MODULES.items():
            ping_data = read_ping_file(config["ping_file"])
            health = calculate_health_percentage(ping_data.get("last_update"))

            module_status = {
                "id": module_id,
                "name": config["name"],
                "description": config["description"],
                "container": config["container"],
                "status": ping_data.get("status", "offline"),
                "health_percentage": health,
                "last_update": ping_data.get("last_update"),
                "data": {k: v for k, v in ping_data.items() if k not in ["status", "last_update"]},
            }
            module_statuses.append(module_status)

        return module_statuses

    except Exception as e:
        from backend.utils.exceptions import APIException, ErrorCode
        raise APIException(
            message="Failed to get modules status",
            error_code=ErrorCode.API_RESPONSE_ERROR,
            details={"original_error": str(e)},
            original_exception=e
        )


@router.get("/modules/{module_id}")
async def get_module_status(module_id: str) -> Dict[str, Any]:
    """Get status of a specific AI module"""
    try:
        if module_id not in AI_MODULES:
            raise HTTPException(status_code=404, detail="Module not found")

        config = AI_MODULES[module_id]
        ping_data = read_ping_file(config["ping_file"])
        health = calculate_health_percentage(ping_data.get("last_update"))

        return {
            "id": module_id,
            "name": config["name"],
            "description": config["description"],
            "container": config["container"],
            "status": ping_data.get("status", "offline"),
            "health_percentage": health,
            "last_update": ping_data.get("last_update"),
            "data": {k: v for k, v in ping_data.items() if k not in ["status", "last_update"]},
        }

    except HTTPException:
        raise
    except Exception as e:
        from backend.utils.exceptions import APIException, ErrorCode
        raise APIException(
            message="Failed to get module status",
            error_code=ErrorCode.API_RESPONSE_ERROR,
            details={"module_id": module_id, "original_error": str(e)},
            original_exception=e
        )


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Simple health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.timezone.utcnow().isoformat(),
        "service": "AI Trading System Status Monitor",
    }


@router.get("/logs/{module_id}")
async def get_module_logs(module_id: str, lines: int = 50) -> Dict[str, Any]:
    """Get recent logs for a specific module."""
    try:
        if module_id not in AI_MODULES:
            raise HTTPException(status_code=404, detail="Module not found")

        # This would normally read from actual log files
        # For now, return simulated logs
        logs = [
            f"[{datetime.timezone.utcnow().strftime('%Y-%m-%d %H:%M:%S')}] {module_id}: Module is running",
            f"[{datetime.timezone.utcnow().strftime('%Y-%m-%d %H:%M:%S')}] {module_id}: Processing data...",
            f"[{datetime.timezone.utcnow().strftime('%Y-%m-%d %H:%M:%S')}] {module_id}: Task completed successfully",
        ]

        return {
            "module": module_id,
            "logs": logs[-lines:],  # Return last N lines
            "total_lines": len(logs),
            "timestamp": datetime.timezone.utcnow().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        from backend.utils.exceptions import APIException, ErrorCode
        raise APIException(
            message="Failed to get module logs",
            error_code=ErrorCode.API_RESPONSE_ERROR,
            details={"module_id": module_id, "original_error": str(e)},
            original_exception=e
        )
