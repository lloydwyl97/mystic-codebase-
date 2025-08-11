"""
AI Prediction Service (compat shim)
Provides a minimal AIPredictionService expected by endpoints.
"""

from __future__ import annotations

from typing import Any, Dict


class AIPredictionService:
    async def get_predictions(self) -> Dict[str, Any]:
        return {"predictions": []}

    async def get_prediction_accuracy(self) -> Dict[str, Any]:
        return {"accuracy": {}}


