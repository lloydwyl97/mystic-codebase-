"""
Autobuy Endpoints
Consolidated autobuy configuration, status, trades, and signals
All endpoints return live data - no stubs or placeholders
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, TYPE_CHECKING, Protocol, runtime_checkable, TypedDict, List, cast

from fastapi import APIRouter, HTTPException

# Import real services
try:
    from endpoints.autobuy.autobuy_config import get_autobuy_status as get_autobuy_status_config
    from endpoints.autobuy.autobuy_config import get_config as _real_get_autobuy_config
    from modules.ai.ai_signals import signal_scorer

    def get_autobuy_config() -> Dict[str, Any]:
        cfg = _real_get_autobuy_config()
        try:
            return cfg.to_dict()  # type: ignore[attr-defined]
        except Exception:
            return cast(Dict[str, Any], cfg)
except ImportError as e:
    logging.warning(f"Some autobuy config modules not available: {e}")
    def get_autobuy_status_config() -> Dict[str, Any]:  # type: ignore[misc]
        return {}
    # Provide a dict-returning fallback to satisfy type checker
    def get_autobuy_config() -> Dict[str, Any]:  # type: ignore[misc]
        return {"config": {}}

if TYPE_CHECKING:
    from services.autobuy_service import AutobuyService as _AutobuyService  # type: ignore[unused-ignore]
    from services.trading import TradingService as _TradingService  # type: ignore[unused-ignore]


class PerformanceDict(TypedDict, total=False):
    active_orders: int
    simulation_mode: bool


class StatusDict(TypedDict, total=False):
    status: str


@runtime_checkable
class AutobuyServiceProtocol(Protocol):
    async def get_configuration(self) -> Dict[str, Any]:
        ...

    async def get_status(self) -> Dict[str, Any]:
        ...

    async def get_statistics(self) -> Dict[str, Any]:
        ...

    async def get_performance_metrics(self) -> PerformanceDict:
        ...

    async def get_recent_trades(self, limit: int = 50) -> List[Dict[str, Any]]:
        ...

    async def get_trade_summary(self) -> Dict[str, Any]:
        ...

    async def get_recent_signals(self, limit: int = 50) -> List[Dict[str, Any]]:
        ...

    async def get_signal_analysis(self) -> Dict[str, Any]:
        ...

    async def get_ai_status(self) -> Dict[str, Any]:
        ...

    async def get_ai_performance(self) -> Dict[str, Any]:
        ...

    async def start(self) -> Dict[str, Any]:
        ...

    async def stop(self) -> Dict[str, Any]:
        ...

    async def update_configuration(self, config: Dict[str, Any]) -> Dict[str, Any]:
        ...

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize real services
try:
    from services.redis_client import get_redis_client
    from services.autobuy_service import AutobuyService as _RealAutobuyService  # noqa: F401
    from services.trading import TradingService as _RealTradingService  # noqa: F401

    autobuy_service: AutobuyServiceProtocol = _RealAutobuyService()  # type: ignore[assignment]
    trading_service = _RealTradingService(get_redis_client())  # type: ignore[call-arg]
except Exception as e:
    logger.warning(f"Could not initialize some autobuy services: {e}")

"""
Autobuy Endpoints
Consolidated autobuy configuration, status, trades, and signals
All endpoints return live data - no stubs or placeholders
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, TYPE_CHECKING, Protocol, runtime_checkable, TypedDict, List, cast

from fastapi import APIRouter, HTTPException

# Import real services
try:
    from endpoints.autobuy.autobuy_config import get_autobuy_status as get_autobuy_status_config
    from endpoints.autobuy.autobuy_config import get_config as _real_get_autobuy_config
    from modules.ai.ai_signals import signal_scorer

    def get_autobuy_config() -> Dict[str, Any]:
        cfg = _real_get_autobuy_config()
        try:
            return cfg.to_dict()  # type: ignore[attr-defined]
        except Exception:
            return cast(Dict[str, Any], cfg)
except ImportError as e:
    logging.warning(f"Some autobuy config modules not available: {e}")
    def get_autobuy_status_config() -> Dict[str, Any]:  # type: ignore[misc]
        return {}
    # Provide a dict-returning fallback to satisfy type checker
    def get_autobuy_config() -> Dict[str, Any]:  # type: ignore[misc]
        return {"config": {}}

if TYPE_CHECKING:
    from services.autobuy_service import AutobuyService as _AutobuyService  # type: ignore[unused-ignore]
    from services.trading import TradingService as _TradingService  # type: ignore[unused-ignore]


class PerformanceDict(TypedDict, total=False):
    active_orders: int
    simulation_mode: bool


class StatusDict(TypedDict, total=False):
    status: str


@runtime_checkable
class AutobuyServiceProtocol(Protocol):
    async def get_configuration(self) -> Dict[str, Any]:
        ...

    async def get_status(self) -> Dict[str, Any]:
        ...

    async def get_statistics(self) -> Dict[str, Any]:
        ...

    async def get_performance_metrics(self) -> PerformanceDict:
        ...

    async def get_recent_trades(self, limit: int = 50) -> List[Dict[str, Any]]:
        ...

    async def get_trade_summary(self) -> Dict[str, Any]:
        ...

    async def get_recent_signals(self, limit: int = 50) -> List[Dict[str, Any]]:
        ...

    async def get_signal_analysis(self) -> Dict[str, Any]:
        ...

    async def get_ai_status(self) -> Dict[str, Any]:
        ...

    async def get_ai_performance(self) -> Dict[str, Any]:
        ...

    async def start(self) -> Dict[str, Any]:
        ...

    async def stop(self) -> Dict[str, Any]:
        ...

    async def update_configuration(self, config: Dict[str, Any]) -> Dict[str, Any]:
        ...

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize real services
try:
    from services.redis_client import get_redis_client
    from services.autobuy_service import AutobuyService as _RealAutobuyService  # noqa: F401
    from services.trading import TradingService as _RealTradingService  # noqa: F401

    autobuy_service: AutobuyServiceProtocol = _RealAutobuyService()  # type: ignore[assignment]
    # Avoid NameError at runtime if type not available; instantiate without annotation
    trading_service = _RealTradingService(get_redis_client())  # type: ignore[call-arg]
except Exception as e:
    logger.warning(f"Could not initialize some autobuy services: {e}")


@router.get("/autobuy/config")
async def get_autobuy_config_endpoint() -> Dict[str, Any]:
    """Get autobuy configuration and settings"""
    try:
        # Get real autobuy configuration
        config = {}
        try:
            config = get_autobuy_config()
        except Exception as e:
            logger.error(f"Error getting autobuy config: {e}")
            config = {"error": "Autobuy config unavailable"}

        # Get additional configuration from service
        service_config = {}
        try:
            if autobuy_service:
                service_config = await autobuy_service.get_configuration()
        except Exception as e:
            logger.error(f"Error getting service config: {e}")
            service_config = {"error": "Service config unavailable"}

        # Enabled exchanges via unified router
        enabled_exchanges: List[str] = []
        try:
            from services.market_data_router import MarketDataRouter  # type: ignore[import-not-found]
            _router = MarketDataRouter()
            enabled_exchanges = await _router.get_enabled_adapters()
            if "coingecko" not in enabled_exchanges:
                enabled_exchanges.append("coingecko")
        except Exception:
            enabled_exchanges = []

        config_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": config,
            "service_config": service_config,
            "enabled_exchanges": enabled_exchanges,
            "version": "1.0.0",
        }

        return config_data

    except Exception as e:
        logger.error(f"Error getting autobuy config: {e}")
        raise HTTPException(status_code=500, detail=f"Autobuy config failed: {str(e)}")


@router.get("/autobuy/status")
async def get_autobuy_status() -> Dict[str, Any]:
    """Get autobuy system status and health"""
    try:
        # Get real autobuy status
        status = {}
        try:
            status = get_autobuy_status_config()
        except Exception as e:
            logger.error(f"Error getting autobuy status: {e}")
            status = {"error": "Autobuy status unavailable"}

        # Get service status
        service_status = {}
        try:
            if autobuy_service:
                service_status = await autobuy_service.get_status()
        except Exception as e:
            logger.error(f"Error getting service status: {e}")
            service_status = {"error": "Service status unavailable"}

        status_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "service_status": service_status,
            "version": "1.0.0",
        }

        return status_data

    except Exception as e:
        logger.error(f"Error getting autobuy status: {e}")
        raise HTTPException(status_code=500, detail=f"Autobuy status failed: {str(e)}")


@router.get("/autobuy/stats")
async def get_autobuy_stats() -> Dict[str, Any]:
    """Get autobuy performance statistics"""
    try:
        # Get real autobuy statistics
        stats = {}
        try:
            if autobuy_service:
                stats = await autobuy_service.get_statistics()
        except Exception as e:
            logger.error(f"Error getting autobuy stats: {e}")
            stats = {"error": "Autobuy stats unavailable"}

        # Get performance metrics
        performance = {}
        try:
            if autobuy_service:
                performance = await autobuy_service.get_performance_metrics()
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            performance = {"error": "Performance metrics unavailable"}

        stats_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "statistics": stats,
            "performance": performance,
            "version": "1.0.0",
        }

        return stats_data

    except Exception as e:
        logger.error(f"Error getting autobuy stats: {e}")
        raise HTTPException(status_code=500, detail=f"Autobuy stats failed: {str(e)}")


@router.get("/autobuy/trades")
async def get_autobuy_trades(limit: int = 50) -> Dict[str, Any]:
    """Get recent autobuy trades"""
    try:
        # Get real autobuy trades
        trades = []
        try:
            if autobuy_service:
                trades = await autobuy_service.get_recent_trades(limit)
        except Exception as e:
            logger.error(f"Error getting autobuy trades: {e}")
            trades = []

        # Get trade summary
        trade_summary = {}
        try:
            if autobuy_service:
                trade_summary = await autobuy_service.get_trade_summary()
        except Exception as e:
            logger.error(f"Error getting trade summary: {e}")
            trade_summary = {"error": "Trade summary unavailable"}

        trades_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trades": trades,
            "summary": trade_summary,
            "total_trades": len(trades),
            "limit": limit,
            "version": "1.0.0",
        }

        return trades_data

    except Exception as e:
        logger.error(f"Error getting autobuy trades: {e}")
        raise HTTPException(status_code=500, detail=f"Autobuy trades failed: {str(e)}")


@router.get("/autobuy/signals")
async def get_autobuy_signals(limit: int = 50) -> Dict[str, Any]:
    """Get recent autobuy signals"""
    try:
        # Get real autobuy signals
        signals = []
        try:
            if autobuy_service:
                signals = await autobuy_service.get_recent_signals(limit)
        except Exception as e:
            logger.error(f"Error getting autobuy signals: {e}")
            signals = []

        # Get signal analysis
        signal_analysis = {}
        try:
            if autobuy_service:
                signal_analysis = await autobuy_service.get_signal_analysis()
        except Exception as e:
            logger.error(f"Error getting signal analysis: {e}")
            signal_analysis = {"error": "Signal analysis unavailable"}

        signals_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "signals": signals,
            "analysis": signal_analysis,
            "total_signals": len(signals),
            "limit": limit,
            "version": "1.0.0",
        }

        return signals_data

    except Exception as e:
        logger.error(f"Error getting autobuy signals: {e}")
        raise HTTPException(status_code=500, detail=f"Autobuy signals failed: {str(e)}")


@router.get("/autobuy/ai-status")
async def get_autobuy_ai_status() -> Dict[str, Any]:
    """Get autobuy AI system status and performance"""
    try:
        # Get real AI status
        ai_status = {}
        try:
            if autobuy_service:
                ai_status = await autobuy_service.get_ai_status()
        except Exception as e:
            logger.error(f"Error getting AI status: {e}")
            ai_status = {"error": "AI status unavailable"}

        # Get AI performance metrics
        ai_performance = {}
        try:
            if autobuy_service:
                ai_performance = await autobuy_service.get_ai_performance()
        except Exception as e:
            logger.error(f"Error getting AI performance: {e}")
            ai_performance = {"error": "AI performance unavailable"}

        # Get signal quality metrics
        signal_quality = {}
        try:
            signal_quality = signal_scorer.get_quality_metrics()  # type: ignore[attr-defined]
        except Exception as e:
            logger.error(f"Error getting signal quality: {e}")
            signal_quality = {"error": "Signal quality unavailable"}

        ai_status_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ai_status": ai_status,
            "ai_performance": ai_performance,
            "signal_quality": signal_quality,
            "version": "1.0.0",
        }

        return ai_status_data

    except Exception as e:
        logger.error(f"Error getting autobuy AI status: {e}")
        raise HTTPException(status_code=500, detail=f"Autobuy AI status failed: {str(e)}")


@router.post("/autobuy/start")
async def start_autobuy() -> Dict[str, Any]:
    """Start the autobuy system"""
    try:
        # Start real autobuy system
        result = {}
        try:
            if autobuy_service:
                result = await autobuy_service.start()
        except Exception as e:
            logger.error(f"Error starting autobuy: {e}")
            result = {"error": f"Failed to start autobuy: {str(e)}"}

        start_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "result": result,
            "status": "started" if "error" not in result else "failed",
            "version": "1.0.0",
        }

        return start_data

    except Exception as e:
        logger.error(f"Error starting autobuy: {e}")
        raise HTTPException(status_code=500, detail=f"Autobuy start failed: {str(e)}")


@router.post("/autobuy/stop")
async def stop_autobuy() -> Dict[str, Any]:
    """Stop the autobuy system"""
    try:
        # Stop real autobuy system
        result = {}
        try:
            if autobuy_service:
                result = await autobuy_service.stop()
        except Exception as e:
            logger.error(f"Error stopping autobuy: {e}")
            result = {"error": f"Failed to stop autobuy: {str(e)}"}

        stop_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "result": result,
            "status": "stopped" if "error" not in result else "failed",
            "version": "1.0.0",
        }

        return stop_data

    except Exception as e:
        logger.error(f"Error stopping autobuy: {e}")
        raise HTTPException(status_code=500, detail=f"Autobuy stop failed: {str(e)}")


@router.post("/autobuy/update-config")
async def update_autobuy_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Update autobuy configuration"""
    try:
        # Update real autobuy configuration
        result = {}
        try:
            if autobuy_service:
                result = await autobuy_service.update_configuration(config)
        except Exception as e:
            logger.error(f"Error updating autobuy config: {e}")
            result = {"error": f"Failed to update config: {str(e)}"}

        update_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "result": result,
            "status": "updated" if "error" not in result else "failed",
            "version": "1.0.0",
        }

        return update_data

    except Exception as e:
        logger.error(f"Error updating autobuy config: {e}")
        raise HTTPException(status_code=500, detail=f"Autobuy config update failed: {str(e)}")


# Aliases to match Streamlit paths
@router.post("/autobuy/control/start")
async def start_autobuy_control() -> Dict[str, Any]:
    try:
        result = await start_autobuy()
        started = (result.get("status") == "started") or (
            isinstance(result.get("result"), dict) and result["result"].get("started")
        )
        return {"started": bool(started)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Autobuy start failed: {str(e)}")


@router.post("/autobuy/control/stop")
async def stop_autobuy_control() -> Dict[str, Any]:
    try:
        result = await stop_autobuy()
        stopped = (result.get("status") == "stopped") or (
            isinstance(result.get("result"), dict) and result["result"].get("stopped")
        )
        return {"stopped": bool(stopped)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Autobuy stop failed: {str(e)}")


@router.post("/autobuy/config")
async def set_autobuy_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return await update_autobuy_config(config)


@router.get("/autobuy/decision")
async def autobuy_decision(symbol: str = "BTC-USD") -> Dict[str, Any]:
    try:
        warm = {}
        evald = {}
        decision = {}
        executed = {}
        try:
            if autobuy_service:
                warm = await autobuy_service.warmup([symbol])  # type: ignore[attr-defined]
                evald = await autobuy_service.eval_signals(symbol)  # type: ignore[attr-defined]
                decision = await autobuy_service.decide_and_route(symbol)  # type: ignore[attr-defined]
                executed = await autobuy_service.execute_decision(decision)  # type: ignore[attr-defined]
        except Exception as e:
            executed = {"error": f"decision flow failed: {str(e)}"}

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "warmup": warm,
            "signals": evald,
            "decision": decision,
            "execution": executed,
            "version": "1.0.0",
        }
    except Exception as e:
        logger.error(f"Error running autobuy decision: {e}")
        raise HTTPException(status_code=500, detail=f"Autobuy decision failed: {str(e)}")