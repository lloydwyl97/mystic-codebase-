"""
Advanced Signal Endpoints

Handles all advanced signal related API endpoints including live signals, health monitoring, and testing.
"""

import logging
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from enhanced_logging import log_event, log_operation_performance

logger = logging.getLogger(__name__)


def get_signal_manager():
    """Get signal manager instance"""
    try:
        from signal_manager import SignalManager

        return SignalManager()
    except ImportError as e:
        logger.error(f"SignalManager not available: {str(e)}")
        raise HTTPException(status_code=500, detail="Signal service unavailable")


def get_health_monitor():
    """Get health monitor instance"""
    try:
        from health_monitor import HealthMonitor

        return HealthMonitor()
    except ImportError as e:
        logger.error(f"HealthMonitor not available: {str(e)}")
        raise HTTPException(status_code=500, detail="Health monitoring service unavailable")


def get_redis_client():
    """Get Redis client instance"""
    try:
        from database import get_redis_client as get_db_redis_client

        return get_db_redis_client()
    except Exception as e:
        logger.error(f"Failed to get Redis client: {str(e)}")
        raise HTTPException(status_code=500, detail="Redis service unavailable")


router = APIRouter()


@router.get("/live/{symbol}")
@log_operation_performance("live_signal_generation")
async def get_live_signals(
    symbol: str, signal_manager: Any = Depends(lambda: get_signal_manager())
):
    """Get live trading signals for a specific symbol"""
    try:
        signals = await signal_manager.get_live_signals(symbol)
        return {"symbol": symbol, "signals": signals, "timestamp": time.time()}
    except Exception as e:
        logger.error(f"Error getting live signals for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting live signals: {str(e)}")


@router.get("/status")
async def get_signal_status(
    signal_manager: Any = Depends(lambda: get_signal_manager()),
):
    """Get signal system status"""
    try:
        status = await signal_manager.get_status()
        return status
    except Exception as e:
        logger.error(f"Error getting signal status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting signal status: {str(e)}")


@router.post("/activate")
@log_operation_performance("signal_activation")
async def activate_signals(
    signal_manager: Any = Depends(lambda: get_signal_manager()),
):
    """Activate signal generation"""
    try:
        result = await signal_manager.activate_signals()
        log_event("signals_activated", "Signal generation activated")
        return result
    except Exception as e:
        logger.error(f"Error activating signals: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error activating signals: {str(e)}")


@router.post("/self-heal")
@log_operation_performance("self_healing")
async def self_heal_signals(
    health_monitor: Any = Depends(lambda: get_health_monitor()),
):
    """Trigger self-healing for signal system"""
    try:
        result = await health_monitor.self_heal()
        log_event("self_healing_triggered", "Signal system self-healing initiated")
        return result
    except Exception as e:
        logger.error(f"Error triggering self-healing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error triggering self-healing: {str(e)}")


@router.get("/health")
async def get_signal_health(
    health_monitor: Any = Depends(lambda: get_health_monitor()),
):
    """Get signal system health"""
    try:
        health = await health_monitor.get_signal_health()
        return health
    except Exception as e:
        logger.error(f"Error getting signal health: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting signal health: {str(e)}")


@router.post("/strategy/{strategy_name}/toggle")
async def toggle_strategy(
    strategy_name: str,
    signal_manager: Any = Depends(lambda: get_signal_manager()),
    redis_client: Any = Depends(lambda: get_redis_client()),
):
    """Toggle a specific signal strategy"""
    try:
        result = await signal_manager.toggle_strategy(strategy_name)
        return result
    except Exception as e:
        logger.error(f"Error toggling strategy {strategy_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error toggling strategy: {str(e)}")


@router.get("/history/{symbol}")
async def get_signal_history(
    symbol: str, limit: int = 50, signal_manager: Any = Depends(lambda: get_signal_manager())
):
    """Get signal history for a symbol"""
    try:
        signals = await signal_manager.get_signal_history(symbol, limit)
        return {
            "symbol": symbol,
            "signals": signals,
            "count": len(signals) if signals else 0,
        }
    except Exception as e:
        logger.error(f"Error getting signal history: {str(e)}")
        raise HTTPException(status_code=500, detail="Error getting signal history")


@router.get("/strategies")
async def get_signal_strategies(signal_manager: Any = Depends(lambda: get_signal_manager())):
    """Get available signal strategies"""
    try:
        strategies = await signal_manager.get_strategies()
        return {"strategies": strategies}
    except Exception as e:
        logger.error(f"Error getting signal strategies: {str(e)}")
        raise HTTPException(status_code=500, detail="Error getting signal strategies")


@router.post("/tests/run")
async def run_signal_tests(signal_manager: Any = Depends(lambda: get_signal_manager())):
    """Run signal system tests"""
    try:
        test_results = await signal_manager.run_tests()
        return test_results
    except Exception as e:
        logger.error(f"Error running signal tests: {str(e)}")
        raise HTTPException(status_code=500, detail="Error running signal tests")


@router.get("/tests/statistics")
async def get_signal_test_statistics(signal_manager: Any = Depends(lambda: get_signal_manager())):
    """Get signal test statistics"""
    try:
        statistics = await signal_manager.get_test_statistics()
        return statistics
    except Exception as e:
        logger.error(f"Error getting test statistics: {str(e)}")
        raise HTTPException(status_code=500, detail="Error getting test statistics")


@router.get("/tests/recent")
async def get_recent_signal_tests(
    limit: int = 10, signal_manager: Any = Depends(lambda: get_signal_manager())
):
    """Get recent signal tests"""
    try:
        recent_tests = await signal_manager.get_recent_tests(limit)
        return {"recent_tests": recent_tests}
    except Exception as e:
        logger.error(f"Error getting recent tests: {str(e)}")
        raise HTTPException(status_code=500, detail="Error getting recent tests")


@router.get("/tests/run/{run_id}")
async def get_signal_test_run(
    run_id: str, signal_manager: Any = Depends(lambda: get_signal_manager())
):
    """Get specific signal test run details"""
    try:
        test_run = await signal_manager.get_test_run(run_id)
        return test_run
    except Exception as e:
        logger.error(f"Error getting test run: {str(e)}")
        raise HTTPException(status_code=500, detail="Error getting test run")


@router.get("/metrics")
async def get_signal_metrics(signal_manager: Any = Depends(lambda: get_signal_manager())):
    """Get signal system metrics"""
    try:
        metrics = await signal_manager.get_metrics()
        return {
            "metrics": metrics,
            "timestamp": time.time(),
        }
    except Exception as e:
        logger.error(f"Error getting signal metrics: {str(e)}")
        raise HTTPException(status_code=500, detail="Error getting signal metrics")


@router.get("/system-health")
async def get_signal_system_health(health_monitor: Any = Depends(lambda: get_health_monitor())):
    """Get signal system health status"""
    try:
        system_health = await health_monitor.get_system_health()
        return system_health
    except Exception as e:
        logger.error(f"Error getting system health: {str(e)}")
        raise HTTPException(status_code=500, detail="Error getting system health")
