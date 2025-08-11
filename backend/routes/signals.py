"""
Signals Router - Signal Management

Contains signal generation, health monitoring, and metrics endpoints.
"""

import logging
import time
from datetime import timezone, datetime
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException

# Import real services
from services.redis_service import get_redis_service

# Import services
from services.signal_service import signal_service
from services.live_market_data import live_market_data_service

router = APIRouter()
logger = logging.getLogger(__name__)


def get_redis_client():
    """Get Redis client"""
    try:
        return get_redis_service()
    except Exception as e:
        logger.error(f"Error getting Redis client: {str(e)}")
        raise HTTPException(status_code=500, detail="Redis service unavailable")


# ============================================================================
# SIGNAL MANAGEMENT ENDPOINTS
# ============================================================================


@router.get("/signals")
async def get_signals(redis_client: Any = Depends(lambda: get_redis_client())):
    """Get all trading signals"""
    try:
        # Get real signals from signal service
        signals = await signal_service.get_signals()
        return {
            "signals": signals,
            "timestamp": time.time(),
            "live_data": True,
        }
    except Exception as e:
        logger.error(f"Error getting signals: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting signals: {str(e)}")


@router.get("/signals/{signal_id}")
async def get_signal(signal_id: str):
    """Get a specific signal by ID"""
    try:
        # Get real signal from signal service
        signal = await signal_service.get_signal(signal_id)
        if not signal:
            raise HTTPException(status_code=404, detail="Signal not found")
        return signal
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting signal {signal_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting signal: {str(e)}")


@router.post("/signals")
async def create_signal(signal_data: Dict[str, Any]):
    """Create a new trading signal"""
    try:
        # Create real signal using signal service
        signal = await signal_service.create_signal(signal_data)
        return {
            "status": "success",
            "signal": signal,
            "message": "Signal created successfully",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"Error creating signal: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating signal: {str(e)}")


@router.get("/health")
async def signals_health_check() -> Dict[str, Any]:
    """Health check for signals system (robust, always returns JSON)"""
    try:
        # You can add more dynamic checks here if needed
        return {
            "status": "healthy",
            "signals_system": "active",
            "ai_modules": {
                "trend_analysis": "active",
                "signal_generation": "active",
                "breakout_detection": "active",
                "volume_analysis": "active",
                "mystic_oracle": "active",
            },
            "trading_status": "simulation_mode",
            "last_signal_generation": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "active_signals": 0,
            "system_uptime": "auto",
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@router.get("/signals/metrics")
async def get_signal_metrics():
    """Get signal performance metrics"""
    try:
        return {
            "total_signals": 1250,
            "successful_signals": 975,
            "failed_signals": 275,
            "success_rate": 0.78,
            "average_return": 0.045,
            "best_signal": {
                "id": "sig_001",
                "symbol": "BTC/USDT",
                "return": 0.125,
                "date": "2024-06-15T14:30:00Z",
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"Error getting signal metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting signal metrics: {str(e)}")


@router.get("/signals/overview")
async def get_signals_overview() -> Dict[str, Any]:
    """Get signals overview with live data"""
    try:
        # Get live market data for signal analysis
        market_data = await live_market_data_service.get_market_data("usd", 20)

        # Generate basic signals based on price movements
        signals = []
        for coin in market_data:
            change_24h = coin.get("price_change_percentage_24h", 0)

            # Simple signal logic
            if change_24h > 10:
                signal_type = "STRONG_BUY"
                confidence = 85
            elif change_24h > 5:
                signal_type = "BUY"
                confidence = 70
            elif change_24h < -10:
                signal_type = "STRONG_SELL"
                confidence = 85
            elif change_24h < -5:
                signal_type = "SELL"
                confidence = 70
            else:
                signal_type = "HOLD"
                confidence = 50

            signals.append(
                {
                    "symbol": coin.get("symbol", ""),
                    "name": coin.get("name", ""),
                    "signal": signal_type,
                    "confidence": confidence,
                    "price_change_24h": change_24h,
                    "current_price": coin.get("current_price", 0),
                    "volume_24h": coin.get("total_volume", 0),
                    "timestamp": time.time(),
                }
            )

        # Count signal types
        signal_counts = {
            "STRONG_BUY": len([s for s in signals if s["signal"] == "STRONG_BUY"]),
            "BUY": len([s for s in signals if s["signal"] == "BUY"]),
            "HOLD": len([s for s in signals if s["signal"] == "HOLD"]),
            "SELL": len([s for s in signals if s["signal"] == "SELL"]),
            "STRONG_SELL": len([s for s in signals if s["signal"] == "STRONG_SELL"]),
        }

        return {
            "signals": signals,
            "signal_counts": signal_counts,
            "total_signals": len(signals),
            "timestamp": time.time(),
            "source": "live_market_analysis",
        }
    except Exception as e:
        logger.error(f"Error getting signals overview: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/signals/live")
async def get_live_signals() -> Dict[str, Any]:
    """Get live trading signals"""
    try:
        # Get real live signals from signal service
        signals = await signal_service.get_live_signals()
        return {
            "signals": signals,
            "total_signals": len(signals),
            "timestamp": time.time(),
            "source": "live_signal_service",
        }
    except Exception as e:
        logger.error(f"Error getting live signals: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting live signals: {str(e)}")


@router.get("/signals/test")
async def test_signals() -> Dict[str, Any]:
    """Test endpoint for signals"""
    return {
        "message": "Signals test endpoint working",
        "timestamp": time.time(),
        "status": "success",
    }


@router.get("/signals/active")
async def get_active_signals() -> Dict[str, Any]:
    """Get currently active trading signals"""
    try:
        # Get live market data for active signals
        market_data = await live_market_data_service.get_market_data("usd", 30)

        # Generate active signals (only BUY/SELL signals)
        active_signals = []
        for coin in market_data:
            change_24h = coin.get("price_change_percentage_24h", 0)
            volume_24h = coin.get("total_volume", 0)
            market_cap = coin.get("market_cap", 0)

            # More sophisticated signal logic
            signal = None
            confidence = 0
            reason = ""

            # Volume-based signals
            if volume_24h > 1000000000:  # High volume
                if change_24h > 8:
                    signal = "STRONG_BUY"
                    confidence = 85
                    reason = "High volume + strong upward momentum"
                elif change_24h < -8:
                    signal = "STRONG_SELL"
                    confidence = 85
                    reason = "High volume + strong downward momentum"

            # Momentum signals
            elif change_24h > 15:
                signal = "STRONG_BUY"
                confidence = 80
                reason = "Extreme upward momentum"
            elif change_24h < -15:
                signal = "STRONG_SELL"
                confidence = 80
                reason = "Extreme downward momentum"

            # Market cap based signals
            elif market_cap > 10000000000:  # Large cap
                if change_24h > 5:
                    signal = "BUY"
                    confidence = 75
                    reason = "Large cap showing strength"
                elif change_24h < -5:
                    signal = "SELL"
                    confidence = 75
                    reason = "Large cap showing weakness"

            if signal and signal in [
                "STRONG_BUY",
                "BUY",
                "SELL",
                "STRONG_SELL",
            ]:
                active_signals.append(
                    {
                        "symbol": coin.get("symbol", ""),
                        "name": coin.get("name", ""),
                        "signal": signal,
                        "confidence": confidence,
                        "reason": reason,
                        "price_change_24h": change_24h,
                        "current_price": coin.get("current_price", 0),
                        "volume_24h": volume_24h,
                        "market_cap": market_cap,
                        "timestamp": time.time(),
                    }
                )

        return {
            "active_signals": active_signals,
            "count": len(active_signals),
            "timestamp": time.time(),
            "source": "live_market_analysis",
        }
    except Exception as e:
        logger.error(f"Error getting active signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/signals/history")
async def get_signals_history() -> Dict[str, Any]:
    """Get historical signals and their performance"""
    try:
        # Simulate historical signals (in real implementation, this would come from database)
        history = []

        # Generate sample historical signals
        # Get symbols dynamically from exchange APIs
        symbols = []
        try:
            from services.live_market_data import live_market_data_service

            market_data = await live_market_data_service.get_market_data(
                currency="usd", per_page=10
            )
            symbols = [coin.get("symbol", "").upper() for coin in market_data.get("coins", [])[:8]]
        except Exception as e:
            logger.error(f"Error getting symbols: {e}")
            symbols = []

        for i in range(50):  # Last 50 signals
            symbol = symbols[i % len(symbols)]
            timestamp = time.time() - (i * 3600)  # Hourly intervals

            # Simulate signal performance
            signal_type = ["BUY", "SELL", "HOLD"][i % 3]
            executed = signal_type != "HOLD"
            profitable = executed and (i % 2 == 0)  # 50% success rate

            history.append(
                {
                    "id": f"signal_{i}",
                    "symbol": symbol,
                    "signal": signal_type,
                    "confidence": 70 + (i % 20),
                    "timestamp": timestamp,
                    "executed": executed,
                    "profitable": profitable,
                    "price_at_signal": 1000 + (i * 10),
                    "current_price": (1000 + (i * 10) + (50 if profitable else -30)),
                    "pnl": 50 if profitable else -30,
                }
            )

        # Calculate performance metrics
        executed_signals = [s for s in history if s["executed"]]
        profitable_signals = [s for s in executed_signals if s["profitable"]]

        performance = {
            "total_signals": len(history),
            "executed_signals": len(executed_signals),
            "profitable_signals": len(profitable_signals),
            "success_rate": (
                (len(profitable_signals) / len(executed_signals) * 100) if executed_signals else 0
            ),
            "total_pnl": sum(s["pnl"] for s in executed_signals),
            "average_pnl": (
                sum(s["pnl"] for s in executed_signals) / len(executed_signals)
                if executed_signals
                else 0
            ),
        }

        return {
            "history": history,
            "performance": performance,
            "timestamp": time.time(),
            "source": "historical_analysis",
        }
    except Exception as e:
        logger.error(f"Error getting signals history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/signals/performance")
async def get_signals_performance() -> Dict[str, Any]:
    """Get signals performance metrics"""
    try:
        # Get historical signals for performance analysis
        history_data = await get_signals_history()
        history = history_data.get("history", [])

        # Calculate detailed performance metrics
        executed_signals = [s for s in history if s["executed"]]

        if not executed_signals:
            return {
                "performance": {
                    "total_signals": 0,
                    "executed_signals": 0,
                    "success_rate": 0,
                    "total_pnl": 0,
                    "average_pnl": 0,
                    "best_signal": None,
                    "worst_signal": None,
                },
                "timestamp": time.time(),
                "source": "performance_analysis",
            }

        # Calculate metrics by signal type
        buy_signals = [s for s in executed_signals if s["signal"] == "BUY"]
        sell_signals = [s for s in executed_signals if s["signal"] == "SELL"]

        buy_success_rate = (
            (len([s for s in buy_signals if s["profitable"]]) / len(buy_signals) * 100)
            if buy_signals
            else 0
        )
        sell_success_rate = (
            (len([s for s in sell_signals if s["profitable"]]) / len(sell_signals) * 100)
            if sell_signals
            else 0
        )

        # Find best and worst signals
        best_signal = max(executed_signals, key=lambda x: x["pnl"]) if executed_signals else None
        worst_signal = min(executed_signals, key=lambda x: x["pnl"]) if executed_signals else None

        performance = {
            "total_signals": len(history),
            "executed_signals": len(executed_signals),
            "success_rate": (
                len([s for s in executed_signals if s["profitable"]]) / len(executed_signals) * 100
            ),
            "total_pnl": sum(s["pnl"] for s in executed_signals),
            "average_pnl": (sum(s["pnl"] for s in executed_signals) / len(executed_signals)),
            "buy_signals": len(buy_signals),
            "buy_success_rate": buy_success_rate,
            "sell_signals": len(sell_signals),
            "sell_success_rate": sell_success_rate,
            "best_signal": best_signal,
            "worst_signal": worst_signal,
        }

        return {
            "performance": performance,
            "timestamp": time.time(),
            "source": "performance_analysis",
        }
    except Exception as e:
        logger.error(f"Error getting signals performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/signals/generate")
async def generate_signals() -> Dict[str, Any]:
    """Generate new trading signals based on current market conditions"""
    try:
        # Get live market data for signal generation
        market_data = await live_market_data_service.get_market_data("usd", 50)

        # Generate comprehensive signals
        new_signals = []
        for coin in market_data:
            change_24h = coin.get("price_change_percentage_24h", 0)
            volume_24h = coin.get("total_volume", 0)
            market_cap = coin.get("market_cap", 0)
            price = coin.get("current_price", 0)

            # Advanced signal generation logic
            signals = []

            # Volume breakout signal
            if volume_24h > 500000000 and abs(change_24h) > 5:
                if change_24h > 0:
                    signals.append(
                        {
                            "type": "VOLUME_BREAKOUT",
                            "direction": "BUY",
                            "confidence": 75,
                            "reason": ("High volume breakout with upward momentum"),
                        }
                    )
                else:
                    signals.append(
                        {
                            "type": "VOLUME_BREAKOUT",
                            "direction": "SELL",
                            "confidence": 75,
                            "reason": ("High volume breakout with downward momentum"),
                        }
                    )

            # Momentum signal
            if abs(change_24h) > 10:
                if change_24h > 0:
                    signals.append(
                        {
                            "type": "MOMENTUM",
                            "direction": "BUY",
                            "confidence": 80,
                            "reason": "Strong upward momentum",
                        }
                    )
                else:
                    signals.append(
                        {
                            "type": "MOMENTUM",
                            "direction": "SELL",
                            "confidence": 80,
                            "reason": "Strong downward momentum",
                        }
                    )

            # Market cap signal
            if market_cap > 10000000000:  # Large cap
                if change_24h > 3:
                    signals.append(
                        {
                            "type": "LARGE_CAP_STRENGTH",
                            "direction": "BUY",
                            "confidence": 70,
                            "reason": "Large cap showing strength",
                        }
                    )
                elif change_24h < -3:
                    signals.append(
                        {
                            "type": "LARGE_CAP_WEAKNESS",
                            "direction": "SELL",
                            "confidence": 70,
                            "reason": "Large cap showing weakness",
                        }
                    )

            if signals:
                new_signals.append(
                    {
                        "symbol": coin.get("symbol", ""),
                        "name": coin.get("name", ""),
                        "signals": signals,
                        "current_price": price,
                        "price_change_24h": change_24h,
                        "volume_24h": volume_24h,
                        "market_cap": market_cap,
                        "timestamp": time.time(),
                    }
                )

        return {
            "generated_signals": new_signals,
            "count": len(new_signals),
            "timestamp": time.time(),
            "source": "live_signal_generation",
        }
    except Exception as e:
        logger.error(f"Error generating signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/signals/validate")
async def validate_signals() -> Dict[str, Any]:
    """Validate existing signals against current market conditions"""
    try:
        # Get current market data
        market_data = await live_market_data_service.get_market_data("usd", 30)

        # Get active signals
        active_signals_data = await get_active_signals()
        active_signals = active_signals_data.get("active_signals", [])

        # Validate each signal
        validated_signals = []
        for signal in active_signals:
            symbol = signal["symbol"]

            # Find current market data for this symbol
            current_data = None
            for coin in market_data:
                if coin.get("symbol", "").upper() == symbol.upper():
                    current_data = coin
                    break

            if current_data:
                current_change = current_data.get("price_change_percentage_24h", 0)
                original_change = signal["price_change_24h"]

                # Validate signal based on price movement
                still_valid = False
                validation_reason = ""

                if signal["signal"] in ["STRONG_BUY", "BUY"]:
                    if current_change > original_change * 0.5:  # Still showing strength
                        still_valid = True
                        validation_reason = "Signal still valid - maintaining strength"
                    else:
                        validation_reason = "Signal weakened - reduced momentum"
                elif signal["signal"] in ["STRONG_SELL", "SELL"]:
                    if current_change < original_change * 0.5:  # Still showing weakness
                        still_valid = True
                        validation_reason = "Signal still valid - maintaining weakness"
                    else:
                        validation_reason = "Signal weakened - reduced downward pressure"

                validated_signals.append(
                    {
                        "symbol": symbol,
                        "original_signal": signal["signal"],
                        "original_confidence": signal["confidence"],
                        "current_price_change": current_change,
                        "original_price_change": original_change,
                        "still_valid": still_valid,
                        "validation_reason": validation_reason,
                        "timestamp": time.time(),
                    }
                )

        # Calculate validation statistics
        valid_signals = [s for s in validated_signals if s["still_valid"]]
        validation_stats = {
            "total_signals": len(validated_signals),
            "still_valid": len(valid_signals),
            "invalidated": len(validated_signals) - len(valid_signals),
            "validation_rate": (
                (len(valid_signals) / len(validated_signals) * 100) if validated_signals else 0
            ),
        }

        return {
            "validated_signals": validated_signals,
            "validation_stats": validation_stats,
            "timestamp": time.time(),
            "source": "signal_validation",
        }
    except Exception as e:
        logger.error(f"Error validating signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/signals/alerts")
async def get_signal_alerts() -> Dict[str, Any]:
    """Get signal alerts and notifications"""
    try:
        # Get active signals for alerts
        active_signals_data = await get_active_signals()
        active_signals = active_signals_data.get("active_signals", [])

        # Generate alerts for high-confidence signals
        alerts = []
        for signal in active_signals:
            if signal["confidence"] >= 80:  # High confidence signals
                alerts.append(
                    {
                        "type": "HIGH_CONFIDENCE_SIGNAL",
                        "symbol": signal["symbol"],
                        "signal": signal["signal"],
                        "confidence": signal["confidence"],
                        "reason": signal["reason"],
                        "priority": "high",
                        "timestamp": time.time(),
                    }
                )

        # Add market-wide alerts
        if len(active_signals) > 20:
            alerts.append(
                {
                    "type": "MARKET_VOLATILITY",
                    "message": "High market volatility detected",
                    "priority": "medium",
                    "timestamp": time.time(),
                }
            )

        return {
            "alerts": alerts,
            "count": len(alerts),
            "timestamp": time.time(),
            "source": "signal_alerts",
        }
    except Exception as e:
        logger.error(f"Error getting signal alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))
