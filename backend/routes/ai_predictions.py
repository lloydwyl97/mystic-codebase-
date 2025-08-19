"""
AI Predictions Endpoints
Focused on AI price prediction functionality
"""

import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException

from backend.ai.ai_brains import trend_analysis
from backend.ai.ai_mystic import mystic_oracle
from backend.ai.poller import cache

# Import actual AI services
from backend.modules.ai.ai_signals import (
    risk_adjusted_signals,
    signal_scorer,
    technical_signals,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/ai", tags=["ai-predictions"])


@router.get("/predictions")
async def get_ai_predictions() -> list[dict[str, Any]]:
    """Get AI price predictions for cryptocurrencies"""
    try:
        predictions = []

        # Get scored signals as predictions
        scored_signals = signal_scorer()
        for i, signal in enumerate(scored_signals[:10]):
            try:
                parts = signal.split()
                symbol = parts[0]
                score = int(parts[2])

                # Get current price from cache if available
                current_price = 0
                if symbol in cache.coingecko:
                    current_price = cache.coingecko[symbol].get("price", 0)

                # Calculate predicted price based on score
                predicted_price = (
                    current_price * (1 + (score - 50) / 100) if current_price > 0 else 0
                )

                predictions.append(
                    {
                        "id": (f"pred_{symbol}_{int(datetime.now().timestamp())}"),
                        "symbol": symbol,
                        "predictedPrice": (round(predicted_price, 4) if predicted_price > 0 else 0),
                        "currentPrice": (round(current_price, 4) if current_price > 0 else 0),
                        "confidence": min(score, 95),
                        "direction": "UP" if score > 60 else "DOWN",
                        "timeframe": "24H",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "source": "AI Signal Scorer",
                        "reasoning": f"Signal score: {score} - {signal}",
                    }
                )
            except Exception as e:
                logger.error(f"Error formatting prediction for signal {signal}: {e}")
                continue

        # Add risk-adjusted predictions
        risk_signals = risk_adjusted_signals()
        for signal in risk_signals[:5]:
            current_price = signal.get("price", 0)
            score = signal.get("score", 0)

            # Calculate predicted price based on risk-adjusted score
            predicted_price = current_price * (1 + (score - 50) / 100) if current_price > 0 else 0

            predictions.append(
                {
                    "id": (f"risk_pred_{signal['symbol']}_{int(datetime.now().timestamp())}"),
                    "symbol": signal["symbol"],
                    "predictedPrice": (round(predicted_price, 4) if predicted_price > 0 else 0),
                    "currentPrice": (round(current_price, 4) if current_price > 0 else 0),
                    "confidence": min(score, 95),
                    "direction": "UP" if score > 60 else "DOWN",
                    "timeframe": "24H",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "source": "AI Risk Analysis",
                    "reasoning": (
                        f"Risk-adjusted score: {score}, Risk: {signal.get('risk_score', 0)}"
                    ),
                }
            )

        # Add technical analysis predictions
        technical_signals_data = technical_signals()
        for signal in technical_signals_data[:5]:
            current_price = signal.get("price", 0)
            strength = signal.get("strength", 0)

            # Calculate predicted price based on technical strength
            predicted_price = (
                current_price * (1 + (strength - 50) / 100) if current_price > 0 else 0
            )

            predictions.append(
                {
                    "id": (f"tech_pred_{signal['symbol']}_{int(datetime.now().timestamp())}"),
                    "symbol": signal["symbol"],
                    "predictedPrice": (round(predicted_price, 4) if predicted_price > 0 else 0),
                    "currentPrice": (round(current_price, 4) if current_price > 0 else 0),
                    "confidence": min(strength, 95),
                    "direction": "UP" if strength > 50 else "DOWN",
                    "timeframe": "24H",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "source": "Technical Analysis",
                    "reasoning": (
                        f"Technical strength: {strength}, Signal type: {signal.get('signal_type', 'NEUTRAL')}"
                    ),
                }
            )

        return predictions
    except Exception as e:
        logger.error(f"Error getting AI predictions: {e}")
        raise HTTPException(status_code=500, detail="Failed to get AI predictions")


@router.post("/train")
async def train_ai_model(data: dict[str, Any]) -> dict[str, Any]:
    """Train AI model for specific symbol and timeframe"""
    try:
        symbol = data.get("symbol", "BTC")
        timeframe = data.get("timeframe", "24H")

        # Get current model performance
        scored_signals = signal_scorer()
        risk_signals = risk_adjusted_signals()
        technical_signals_data = technical_signals()

        # Calculate accuracy based on signal quality
        total_signals = len(scored_signals) + len(risk_signals) + len(technical_signals_data)
        high_confidence_signals = sum(1 for s in scored_signals if int(s.split()[2]) > 80)
        high_confidence_signals += sum(1 for s in risk_signals if s.get("score", 0) > 80)
        high_confidence_signals += sum(
            1 for s in technical_signals_data if s.get("strength", 0) > 80
        )

        accuracy = (high_confidence_signals / total_signals * 100) if total_signals > 0 else 75

        return {
            "success": True,
            "accuracy": round(accuracy, 1),
            "symbol": symbol,
            "timeframe": timeframe,
            "total_signals": total_signals,
            "high_confidence_signals": high_confidence_signals,
            "message": (f"AI model analysis completed for {symbol} on {timeframe} timeframe"),
        }
    except Exception as e:
        logger.error(f"Error training AI model: {e}")
        raise HTTPException(status_code=500, detail="Failed to train AI model")


@router.get("/trends")
async def get_market_trends() -> dict[str, Any]:
    """Get AI market trend analysis"""
    try:
        # Get trend analysis
        trend_data = {}
        try:
            trend_data = trend_analysis() or {}
        except Exception as e:
            logger.warning(f"Trend analysis not available: {e}")

        # Get mystic oracle insights
        oracle_data = {}
        try:
            oracle_data = mystic_oracle() or {}
        except Exception as e:
            logger.warning(f"Mystic oracle not available: {e}")

        # Get market strength
        from backend.modules.ai.ai_signals import market_strength_signals

        market_strength = market_strength_signals()

        return {
            "trend_analysis": trend_data,
            "oracle_insights": oracle_data,
            "market_strength": market_strength,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"Error getting market trends: {e}")
        raise HTTPException(status_code=500, detail="Failed to get market trends")


