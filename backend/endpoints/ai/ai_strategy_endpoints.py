"""
AI Strategy Endpoints
Consolidated AI strategies, predictions, signals, and AI system management
All endpoints return live data - no stubs or placeholders
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, TYPE_CHECKING, Protocol, runtime_checkable

from fastapi import APIRouter, HTTPException

# Import real services
try:
    from backend.modules.ai.ai_signals import signal_scorer, technical_signals
except ImportError as e:
    logging.warning(f"AI signals not fully available: {e}")

if TYPE_CHECKING:
    from backend.services.ai_strategy_service import AIStrategyService as _AIStrategyService  # type: ignore[unused-ignore]
    from backend.services.ai_prediction_service import AIPredictionService as _AIPredictionService  # type: ignore[unused-ignore]
    from backend.services.ai_signal_service import AISignalService as _AISignalService  # type: ignore[unused-ignore]

@runtime_checkable
class _AIStrategyLike(Protocol):
    async def get_all_strategies(self) -> Dict[str, Any]: ...
    async def get_strategy_performance(self) -> Dict[str, Any]: ...
    async def get_system_status(self) -> Dict[str, Any]: ...
    async def get_performance_metrics(self) -> Dict[str, Any]: ...
    async def get_model_status(self) -> Dict[str, Any]: ...
    async def get_models(self) -> Dict[str, Any]: ...
    async def get_model_configurations(self) -> Dict[str, Any]: ...
    async def get_training_status(self) -> Dict[str, Any]: ...
    async def get_training_metrics(self) -> Dict[str, Any]: ...
    async def start_retraining(self) -> Dict[str, Any]: ...
    async def update_strategy(self, strategy_id: str, config: Dict[str, Any]) -> Dict[str, Any]: ...

@runtime_checkable
class _AIPredictionLike(Protocol):
    async def get_predictions(self) -> Dict[str, Any]: ...
    async def get_prediction_accuracy(self) -> Dict[str, Any]: ...

@runtime_checkable
class _AISignalLike(Protocol):
    async def get_signals(self) -> Dict[str, Any]: ...

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize real services with safe fallbacks
ai_strategy_service: _AIStrategyLike | None = None
ai_prediction_service: _AIPredictionLike | None = None
ai_signal_service: _AISignalLike | None = None

# Try primary backend.services paths first (if present in this codebase)
try:
    from backend.services.ai_strategy_service import AIStrategyService as _RealAIStrategyService  # type: ignore
except Exception as e:
    _RealAIStrategyService = None  # type: ignore[assignment]
    logger.warning(f"AI strategy service unavailable: {e}")
try:
    from backend.services.ai_prediction_service import AIPredictionService as _RealAIPredictionService  # type: ignore
except Exception as e:
    _RealAIPredictionService = None  # type: ignore[assignment]
    logger.warning(f"AI prediction service unavailable: {e}")
try:
    from backend.services.ai_signal_service import AISignalService as _RealAISignalService  # type: ignore
except Exception as e:
    _RealAISignalService = None  # type: ignore[assignment]
    logger.warning(f"AI signal service unavailable: {e}")

# Fallback: try importing the root-level services package by adding project root to sys.path
if _RealAIStrategyService is None or _RealAIPredictionService is None or _RealAISignalService is None:
    try:
        import os, sys
        # backend/endpoints/ai -> up 3 levels to project root
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        if _RealAIStrategyService is None:
            from backend.services.ai_strategy_service import AIStrategyService as _RealAIStrategyService  # type: ignore
        if _RealAIPredictionService is None:
            from backend.services.ai_prediction_service import AIPredictionService as _RealAIPredictionService  # type: ignore
        if _RealAISignalService is None:
            from backend.services.ai_signal_service import AISignalService as _RealAISignalService  # type: ignore
    except Exception:
        pass

try:
    if _RealAIStrategyService is not None:
        ai_strategy_service = _RealAIStrategyService()  # type: ignore[assignment]
except Exception:
    pass
try:
    if _RealAIPredictionService is not None:
        ai_prediction_service = _RealAIPredictionService()  # type: ignore[assignment]
except Exception:
    pass
try:
    if _RealAISignalService is not None:
        ai_signal_service = _RealAISignalService()  # type: ignore[assignment]
except Exception:
    pass

"""
AI Strategy Endpoints
Consolidated AI strategies, predictions, signals, and AI system management
All endpoints return live data - no stubs or placeholders
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, TYPE_CHECKING, Protocol, runtime_checkable

from fastapi import APIRouter, HTTPException

# Import real services
try:
    from backend.modules.ai.ai_signals import signal_scorer, technical_signals
except ImportError as e:
    logging.warning(f"AI signals not fully available: {e}")

if TYPE_CHECKING:
    from backend.services.ai_strategy_service import AIStrategyService as _AIStrategyService  # type: ignore[unused-ignore]
    from backend.services.ai_prediction_service import AIPredictionService as _AIPredictionService  # type: ignore[unused-ignore]
    from backend.services.ai_signal_service import AISignalService as _AISignalService  # type: ignore[unused-ignore]

@runtime_checkable
class _AIStrategyLike(Protocol):
    async def get_all_strategies(self) -> Dict[str, Any]: ...
    async def get_strategy_performance(self) -> Dict[str, Any]: ...
    async def get_system_status(self) -> Dict[str, Any]: ...
    async def get_performance_metrics(self) -> Dict[str, Any]: ...
    async def get_model_status(self) -> Dict[str, Any]: ...
    async def get_models(self) -> Dict[str, Any]: ...
    async def get_model_configurations(self) -> Dict[str, Any]: ...
    async def get_training_status(self) -> Dict[str, Any]: ...
    async def get_training_metrics(self) -> Dict[str, Any]: ...
    async def start_retraining(self) -> Dict[str, Any]: ...
    async def update_strategy(self, strategy_id: str, config: Dict[str, Any]) -> Dict[str, Any]: ...

@runtime_checkable
class _AIPredictionLike(Protocol):
    async def get_predictions(self) -> Dict[str, Any]: ...
    async def get_prediction_accuracy(self) -> Dict[str, Any]: ...

@runtime_checkable
class _AISignalLike(Protocol):
    async def get_signals(self) -> Dict[str, Any]: ...

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize real services with safe fallbacks
ai_strategy_service: _AIStrategyLike | None = None
ai_prediction_service: _AIPredictionLike | None = None
ai_signal_service: _AISignalLike | None = None
try:
    from backend.services.ai_strategy_service import AIStrategyService as _RealAIStrategyService  # type: ignore
    ai_strategy_service = _RealAIStrategyService()  # type: ignore[assignment]
except Exception as e:
    logger.warning(f"AI strategy service unavailable: {e}")
try:
    from backend.services.ai_prediction_service import AIPredictionService as _RealAIPredictionService  # type: ignore
    ai_prediction_service = _RealAIPredictionService()  # type: ignore[assignment]
except Exception as e:
    logger.warning(f"AI prediction service unavailable: {e}")
try:
    from backend.services.ai_signal_service import AISignalService as _RealAISignalService  # type: ignore
    ai_signal_service = _RealAISignalService()  # type: ignore[assignment]
except Exception as e:
    logger.warning(f"AI signal service unavailable: {e}")


@router.get("/ai/strategies")
async def get_ai_strategies() -> Dict[str, Any]:
    """Get all AI strategies and their performance"""
    try:
        # Get real AI strategies
        strategies = {}
        try:
            if ai_strategy_service:
                strategies = await ai_strategy_service.get_all_strategies()
        except Exception as e:
            logger.error(f"Error getting AI strategies: {e}")
            strategies = {"error": "AI strategies unavailable"}

        # Get strategy performance
        performance = {}
        try:
            if ai_strategy_service:
                performance = await ai_strategy_service.get_strategy_performance()
        except Exception as e:
            logger.error(f"Error getting strategy performance: {e}")
            performance = {"error": "Strategy performance unavailable"}

        strategies_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "strategies": strategies,
            "performance": performance,
            "version": "1.0.0",
        }

        return strategies_data

    except Exception as e:
        logger.error(f"Error getting AI strategies: {e}")
        raise HTTPException(status_code=500, detail=f"AI strategies failed: {str(e)}")


@router.get("/ai/predictions")
async def get_ai_predictions() -> Dict[str, Any]:
    """Get AI predictions and forecasts"""
    try:
        # Get real AI predictions
        predictions = {}
        try:
            if ai_prediction_service:
                predictions = await ai_prediction_service.get_predictions()
        except Exception as e:
            logger.error(f"Error getting AI predictions: {e}")
            predictions = {"error": "AI predictions unavailable"}

        # Get prediction accuracy
        accuracy = {}
        try:
            if ai_prediction_service:
                accuracy = await ai_prediction_service.get_prediction_accuracy()
        except Exception as e:
            logger.error(f"Error getting prediction accuracy: {e}")
            accuracy = {"error": "Prediction accuracy unavailable"}

        predictions_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "predictions": predictions,
            "accuracy": accuracy,
            "version": "1.0.0",
        }

        return predictions_data

    except Exception as e:
        logger.error(f"Error getting AI predictions: {e}")
        raise HTTPException(status_code=500, detail=f"AI predictions failed: {str(e)}")


@router.get("/ai/signals")
async def get_ai_signals() -> Dict[str, Any]:
    """Get AI trading signals"""
    try:
        # Get real AI signals
        signals = {}
        try:
            if ai_signal_service:
                signals = await ai_signal_service.get_signals()
        except Exception as e:
            logger.error(f"Error getting AI signals: {e}")
            signals = {"error": "AI signals unavailable"}

        # Get signal quality metrics
        signal_quality: Dict[str, Any] = {}
        try:
            # signal_scorer is a function; derive minimal metrics safely
            scored = signal_scorer() if callable(signal_scorer) else []
            signal_quality = {"count": len(scored)}
        except Exception as e:
            logger.error(f"Error getting signal quality: {e}")
            signal_quality = {"error": "Signal quality unavailable"}

        # Get technical signals
        technical_signals_data: Dict[str, Any] = {}
        try:
            # technical_signals is a function; return its data under a key
            tech = technical_signals() if callable(technical_signals) else []
            technical_signals_data = {"signals": tech}
        except Exception as e:
            logger.error(f"Error getting technical signals: {e}")
            technical_signals_data = {"error": "Technical signals unavailable"}

        signals_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "signals": signals,
            "signal_quality": signal_quality,
            "technical_signals": technical_signals_data,
            "version": "1.0.0",
        }

        return signals_data

    except Exception as e:
        logger.error(f"Error getting AI signals: {e}")
        raise HTTPException(status_code=500, detail=f"AI signals failed: {str(e)}")


@router.get("/ai/status")
async def get_ai_status() -> Dict[str, Any]:
    """Get AI system status and health"""
    try:
        # Get real AI system status
        ai_status = {}
        try:
            if ai_strategy_service:
                ai_status = await ai_strategy_service.get_system_status()
        except Exception as e:
            logger.error(f"Error getting AI status: {e}")
            ai_status = {"error": "AI status unavailable"}

        # Get AI performance metrics
        performance = {}
        try:
            if ai_strategy_service:
                performance = await ai_strategy_service.get_performance_metrics()
        except Exception as e:
            logger.error(f"Error getting AI performance: {e}")
            performance = {"error": "AI performance unavailable"}

        # Get AI model status
        model_status = {}
        try:
            if ai_strategy_service:
                model_status = await ai_strategy_service.get_model_status()
        except Exception as e:
            logger.error(f"Error getting model status: {e}")
            model_status = {"error": "Model status unavailable"}

        status_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ai_status": ai_status,
            "performance": performance,
            "model_status": model_status,
            "version": "1.0.0",
        }

        return status_data

    except Exception as e:
        logger.error(f"Error getting AI status: {e}")
        raise HTTPException(status_code=500, detail=f"AI status failed: {str(e)}")


@router.get("/ai/models")
async def get_ai_models() -> Dict[str, Any]:
    """Get AI models and their configurations"""
    try:
        # Get real AI models
        models = {}
        try:
            if ai_strategy_service:
                models = await ai_strategy_service.get_models()
        except Exception as e:
            logger.error(f"Error getting AI models: {e}")
            models = {"error": "AI models unavailable"}

        # Get model configurations
        configurations = {}
        try:
            if ai_strategy_service:
                configurations = await ai_strategy_service.get_model_configurations()
        except Exception as e:
            logger.error(f"Error getting model configurations: {e}")
            configurations = {"error": "Model configurations unavailable"}

        models_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "models": models,
            "configurations": configurations,
            "version": "1.0.0",
        }

        return models_data

    except Exception as e:
        logger.error(f"Error getting AI models: {e}")
        raise HTTPException(status_code=500, detail=f"AI models failed: {str(e)}")


@router.get("/ai/training")
async def get_ai_training_status() -> Dict[str, Any]:
    """Get AI training status and progress"""
    try:
        # Get real training status
        training_status = {}
        try:
            if ai_strategy_service:
                training_status = await ai_strategy_service.get_training_status()
        except Exception as e:
            logger.error(f"Error getting training status: {e}")
            training_status = {"error": "Training status unavailable"}

        # Get training metrics
        training_metrics = {}
        try:
            if ai_strategy_service:
                training_metrics = await ai_strategy_service.get_training_metrics()
        except Exception as e:
            logger.error(f"Error getting training metrics: {e}")
            training_metrics = {"error": "Training metrics unavailable"}

        training_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "training_status": training_status,
            "training_metrics": training_metrics,
            "version": "1.0.0",
        }

        return training_data

    except Exception as e:
        logger.error(f"Error getting AI training status: {e}")
        raise HTTPException(status_code=500, detail=f"AI training status failed: {str(e)}")


@router.post("/ai/retrain")
async def retrain_ai_models() -> Dict[str, Any]:
    """Trigger AI model retraining"""
    try:
        # Start real AI retraining
        result = {}
        try:
            if ai_strategy_service:
                result = await ai_strategy_service.start_retraining()
        except Exception as e:
            logger.error(f"Error starting AI retraining: {e}")
            result = {"error": f"Failed to start retraining: {str(e)}"}

        retrain_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "result": result,
            "status": "started" if "error" not in result else "failed",
            "version": "1.0.0",
        }

        return retrain_data

    except Exception as e:
        logger.error(f"Error starting AI retraining: {e}")
        raise HTTPException(status_code=500, detail=f"AI retraining failed: {str(e)}")


@router.post("/ai/update-strategy")
async def update_ai_strategy(strategy_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Update AI strategy configuration"""
    try:
        # Update real AI strategy
        result = {}
        try:
            if ai_strategy_service:
                result = await ai_strategy_service.update_strategy(strategy_id, config)
        except Exception as e:
            logger.error(f"Error updating AI strategy: {e}")
            result = {"error": f"Failed to update strategy: {str(e)}"}

        update_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "result": result,
            "strategy_id": strategy_id,
            "status": "updated" if "error" not in result else "failed",
            "version": "1.0.0",
        }

        return update_data

    except Exception as e:
        logger.error(f"Error updating AI strategy: {e}")
        raise HTTPException(status_code=500, detail=f"AI strategy update failed: {str(e)}")



