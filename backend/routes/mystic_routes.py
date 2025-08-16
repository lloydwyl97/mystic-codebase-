"""
Mystic Routes
Handles endpoints for mystic integrations, Schumann resonance,
fractal time, and other esoteric trading features.
"""

import logging
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from mystic_config import mystic_config
from backend.services.mystic_integration_service import mystic_integration_service
from backend.services.mystic_signal_engine import mystic_signal_engine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/mystic", tags=["mystic"])


@router.get("/config")
async def get_mystic_config() -> Dict[str, Any]:
    """Get mystic configuration summary"""
    try:
        return {
            "status": "success",
            "data": mystic_config.get_config_summary(),
            "message": "Mystic configuration retrieved successfully",
        }
    except Exception as e:
        logger.error(f"Error retrieving mystic config: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve mystic configuration")


@router.get("/schumann")
async def get_schumann_resonance() -> Dict[str, Any]:
    """Get current Schumann resonance data"""
    try:
        schumann_data = await mystic_integration_service.get_schumann_resonance()

        return {
            "status": "success",
            "data": {
                "frequency": schumann_data.frequency,
                "amplitude": schumann_data.amplitude,
                "timestamp": schumann_data.timestamp.isoformat(),
                "deviation": schumann_data.deviation,
                "alert_level": schumann_data.alert_level,
                "base_frequency": mystic_config.schumann.base_frequency,
                "alert_threshold": mystic_config.schumann.alert_threshold,
            },
            "message": "Schumann resonance data retrieved successfully",
        }
    except Exception as e:
        logger.error(f"Error retrieving Schumann data: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve Schumann resonance data",
        )


@router.get("/fractal-time")
async def get_fractal_time_data() -> Dict[str, Any]:
    """Get current fractal time data"""
    try:
        fractal_data = await mystic_integration_service.get_fractal_time_data()

        return {
            "status": "success",
            "data": {
                "fractal_dimension": fractal_data.fractal_dimension,
                "time_compression": fractal_data.time_compression,
                "resonance_peak": fractal_data.resonance_peak,
                "timestamp": fractal_data.timestamp.isoformat(),
            },
            "message": "Fractal time data retrieved successfully",
        }
    except Exception as e:
        logger.error(f"Error retrieving fractal time data: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve fractal time data")


@router.get("/planetary-alignment")
async def get_planetary_alignment() -> Dict[str, Any]:
    """Get current planetary alignment data"""
    try:
        planetary_data = await mystic_integration_service.get_planetary_alignment()

        return {
            "status": "success",
            "data": {
                "alignment_strength": planetary_data.alignment_strength,
                "planets_involved": planetary_data.planets_involved,
                "influence_score": planetary_data.influence_score,
                "timestamp": planetary_data.timestamp.isoformat(),
            },
            "message": "Planetary alignment data retrieved successfully",
        }
    except Exception as e:
        logger.error(f"Error retrieving planetary alignment data: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve planetary alignment data",
        )


@router.get("/signal-strength")
async def get_mystic_signal_strength() -> Dict[str, Any]:
    """Get overall mystic signal strength"""
    try:
        signal_data = await mystic_integration_service.get_mystic_signal_strength()

        return {
            "status": "success",
            "data": signal_data,
            "message": "Mystic signal strength calculated successfully",
        }
    except Exception as e:
        logger.error(f"Error calculating mystic signal strength: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to calculate mystic signal strength",
        )


@router.get("/moon-phase")
async def get_moon_phase() -> Dict[str, Any]:
    """Get current moon phase information"""
    try:
        from datetime import datetime

        now = datetime.now()
        days_since_new = (now.day + now.month * 30) % 29.5

        # Calculate moon phase factor
        if days_since_new < 2 or days_since_new > 27:
            moon_factor = 0.8  # New moon
        elif 13 < days_since_new < 16:
            moon_factor = 0.9  # Full moon
        else:
            moon_factor = 0.5  # Other phases

        # Get moon phase name
        if days_since_new < 3.7:
            moon_phase = "New Moon"
        elif days_since_new < 7.4:
            moon_phase = "Waxing Crescent"
        elif days_since_new < 11.1:
            moon_phase = "First Quarter"
        elif days_since_new < 14.8:
            moon_phase = "Waxing Gibbous"
        elif days_since_new < 18.5:
            moon_phase = "Full Moon"
        elif days_since_new < 22.1:
            moon_phase = "Waning Gibbous"
        elif days_since_new < 25.8:
            moon_phase = "Last Quarter"
        else:
            moon_phase = "Waning Crescent"

        return {
            "status": "success",
            "data": {
                "phase": moon_phase,
                "factor": moon_factor,
                "timestamp": now.isoformat(),
            },
            "message": "Moon phase data retrieved successfully",
        }
    except Exception as e:
        logger.error(f"Error retrieving moon phase data: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve moon phase data")


@router.get("/all-data")
async def get_all_mystic_data() -> Dict[str, Any]:
    """Get all mystic data in one request"""
    try:
        schumann = await mystic_integration_service.get_schumann_resonance()
        fractal = await mystic_integration_service.get_fractal_time_data()
        planetary = await mystic_integration_service.get_planetary_alignment()
        signal = await mystic_integration_service.get_mystic_signal_strength()

        return {
            "status": "success",
            "data": {
                "schumann_resonance": {
                    "frequency": schumann.frequency,
                    "amplitude": schumann.amplitude,
                    "deviation": schumann.deviation,
                    "alert_level": schumann.alert_level,
                    "timestamp": schumann.timestamp.isoformat(),
                },
                "fractal_time": {
                    "fractal_dimension": fractal.fractal_dimension,
                    "time_compression": fractal.time_compression,
                    "resonance_peak": fractal.resonance_peak,
                    "timestamp": fractal.timestamp.isoformat(),
                },
                "planetary_alignment": {
                    "alignment_strength": planetary.alignment_strength,
                    "planets_involved": planetary.planets_involved,
                    "influence_score": planetary.influence_score,
                    "timestamp": planetary.timestamp.isoformat(),
                },
                "mystic_signal": signal,
                "config": mystic_config.get_config_summary(),
            },
            "message": "All mystic data retrieved successfully",
        }
    except Exception as e:
        logger.error(f"Error retrieving all mystic data: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve mystic data")


@router.get("/health")
async def mystic_health_check() -> Dict[str, Any]:
    """Health check for mystic services"""
    try:
        # Quick test of core functionality
        schumann = await mystic_integration_service.get_schumann_resonance()

        return {
            "status": "healthy",
            "services": {
                "schumann_resonance": "operational",
                "fractal_time": "operational",
                "planetary_alignment": "operational",
                "mystic_signal": "operational",
            },
            "last_schumann_frequency": schumann.frequency,
            "timestamp": schumann.timestamp.isoformat(),
        }
    except Exception as e:
        logger.error(f"Mystic health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "services": {
                "schumann_resonance": "error",
                "fractal_time": "error",
                "planetary_alignment": "error",
                "mystic_signal": "error",
            },
        }


# New endpoints for the mystic signal engine
@router.get("/signal-engine/comprehensive")
async def get_comprehensive_mystic_signal(
    symbol: str = "BTCUSDT",
) -> Dict[str, Any]:
    """Get comprehensive mystic trading signal with Tesla 369, Faerie Star, and Lagos integration"""
    try:
        mystic_signal = await mystic_signal_engine.generate_comprehensive_signal(symbol)

        return {
            "status": "success",
            "data": {
                "symbol": symbol,
                "signal_type": mystic_signal.signal_type.value,
                "confidence": mystic_signal.confidence,
                "strength": mystic_signal.strength,
                "timestamp": mystic_signal.timestamp.isoformat(),
                "reasoning": mystic_signal.reasoning,
                "factors": mystic_signal.factors,
            },
            "message": "Comprehensive mystic signal generated successfully",
        }
    except Exception as e:
        logger.error(f"Error generating comprehensive mystic signal: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to generate comprehensive mystic signal",
        )


@router.get("/signal-engine/tesla")
async def get_tesla_signal() -> Dict[str, Any]:
    """Get Tesla 369 frequency signal"""
    try:
        from datetime import datetime

        tesla_signal = mystic_signal_engine.tesla_engine.calculate_tesla_signal(datetime.now())

        return {
            "status": "success",
            "data": {
                "direction": tesla_signal["direction"],
                "strength": tesla_signal["strength"],
                "resonance": tesla_signal["resonance"],
                "vortex_strength": tesla_signal["vortex_strength"],
                "frequency": tesla_signal["frequency"],
                "timestamp": datetime.now().isoformat(),
            },
            "message": "Tesla 369 signal calculated successfully",
        }
    except Exception as e:
        logger.error(f"Error calculating Tesla signal: {e}")
        raise HTTPException(status_code=500, detail="Failed to calculate Tesla 369 signal")


@router.get("/signal-engine/faerie")
async def get_faerie_signal() -> Dict[str, Any]:
    """Get Faerie Star alignment signal"""
    try:
        from datetime import datetime

        faerie_signal = mystic_signal_engine.faerie_engine.calculate_faerie_signal(datetime.now())

        return {
            "status": "success",
            "data": {
                "direction": faerie_signal["direction"],
                "strength": faerie_signal["strength"],
                "phase": faerie_signal["phase"],
                "element": faerie_signal["element"],
                "magic_strength": faerie_signal["magic_strength"],
                "lunar_cycle": faerie_signal["lunar_cycle"],
                "solar_cycle": faerie_signal["solar_cycle"],
                "timestamp": datetime.now().isoformat(),
            },
            "message": "Faerie Star signal calculated successfully",
        }
    except Exception as e:
        logger.error(f"Error calculating Faerie signal: {e}")
        raise HTTPException(status_code=500, detail="Failed to calculate Faerie Star signal")


@router.get("/signal-engine/lagos")
async def get_lagos_signal() -> Dict[str, Any]:
    """Get Lagos alignment signal"""
    try:
        from datetime import datetime

        lagos_signal = mystic_signal_engine.lagos_engine.calculate_lagos_signal(datetime.now())

        return {
            "status": "success",
            "data": {
                "direction": lagos_signal["direction"],
                "strength": lagos_signal["strength"],
                "cycle": lagos_signal["cycle"],
                "energy": lagos_signal["energy"],
                "alignment_strength": lagos_signal["alignment_strength"],
                "cosmic_cycle": lagos_signal["cosmic_cycle"],
                "energy_cycle": lagos_signal["energy_cycle"],
                "timestamp": datetime.now().isoformat(),
            },
            "message": "Lagos alignment signal calculated successfully",
        }
    except Exception as e:
        logger.error(f"Error calculating Lagos signal: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to calculate Lagos alignment signal",
        )


@router.get("/signal-engine/all-engines")
async def get_all_engine_signals() -> Dict[str, Any]:
    """Get signals from all mystic engines (Tesla, Faerie, Lagos)"""
    try:
        from datetime import datetime

        current_time = datetime.now()

        tesla_signal = mystic_signal_engine.tesla_engine.calculate_tesla_signal(current_time)
        faerie_signal = mystic_signal_engine.faerie_engine.calculate_faerie_signal(current_time)
        lagos_signal = mystic_signal_engine.lagos_engine.calculate_lagos_signal(current_time)

        return {
            "status": "success",
            "data": {
                "timestamp": current_time.isoformat(),
                "tesla_369": {
                    "direction": tesla_signal["direction"],
                    "strength": tesla_signal["strength"],
                    "resonance": tesla_signal["resonance"],
                    "vortex_strength": tesla_signal["vortex_strength"],
                    "frequency": tesla_signal["frequency"],
                },
                "faerie_star": {
                    "direction": faerie_signal["direction"],
                    "strength": faerie_signal["strength"],
                    "phase": faerie_signal["phase"],
                    "element": faerie_signal["element"],
                    "magic_strength": faerie_signal["magic_strength"],
                },
                "lagos_alignment": {
                    "direction": lagos_signal["direction"],
                    "strength": lagos_signal["strength"],
                    "cycle": lagos_signal["cycle"],
                    "energy": lagos_signal["energy"],
                    "alignment_strength": lagos_signal["alignment_strength"],
                },
            },
            "message": "All engine signals calculated successfully",
        }
    except Exception as e:
        logger.error(f"Error calculating all engine signals: {e}")
        raise HTTPException(status_code=500, detail="Failed to calculate engine signals")


@router.get("/signal-engine/trading-recommendation")
async def get_trading_recommendation(
    symbol: str = "BTCUSDT",
) -> Dict[str, Any]:
    """Get trading recommendation based on mystic signals"""
    try:
        mystic_signal = await mystic_signal_engine.generate_comprehensive_signal(symbol)

        # Determine trading recommendation
        if mystic_signal.signal_type.value in ["STRONG_BUY", "BUY"]:
            recommendation = "BUY"
            urgency = "high" if mystic_signal.signal_type.value == "STRONG_BUY" else "medium"
        elif mystic_signal.signal_type.value in ["STRONG_SELL", "SELL"]:
            recommendation = "SELL"
            urgency = "high" if mystic_signal.signal_type.value == "STRONG_SELL" else "medium"
        else:
            recommendation = "HOLD"
            urgency = "low"

        return {
            "status": "success",
            "data": {
                "symbol": symbol,
                "recommendation": recommendation,
                "urgency": urgency,
                "confidence": mystic_signal.confidence,
                "signal_strength": mystic_signal.strength,
                "signal_type": mystic_signal.signal_type.value,
                "reasoning": mystic_signal.reasoning,
                "timestamp": mystic_signal.timestamp.isoformat(),
                "factors": mystic_signal.factors,
            },
            "message": "Trading recommendation generated successfully",
        }
    except Exception as e:
        logger.error(f"Error generating trading recommendation: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate trading recommendation")


