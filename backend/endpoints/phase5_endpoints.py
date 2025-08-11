"""
Phase5 Overlay endpoints for the Mystic Trading Platform

Contains endpoints for Phase5 overlay metrics and quantum visualization services.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

import numpy as np
import redis.asyncio as redis
from fastapi import APIRouter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/phase5", tags=["phase5"])

# Redis connection
r = redis.Redis(decode_responses=True)


async def fetch_metric(key: str, default: str = "..."):
    """Fetch metric from Redis"""
    try:
        value = await r.get(key)
        return value or default
    except Exception as e:
        logger.error(f"Error fetching metric {key}: {e}")
        return default


@router.get("/metrics")
async def get_phase5_metrics() -> Dict[str, Any]:
    """Get Phase5 metrics"""
    try:
        neuro_sync = await fetch_metric("neuro_sync_index")
        cosmic_signal = await fetch_metric("cosmic_harmonic_status")
        aura_alignment = await fetch_metric("aura_alignment_score")
        interdim_activity = await fetch_metric("interdim_signal_strength")

        return {
            "neuro_sync": neuro_sync,
            "cosmic_signal": cosmic_signal,
            "aura_alignment": aura_alignment,
            "interdim_activity": interdim_activity,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting Phase5 metrics: {e}")
        return {
            "neuro_sync": "...",
            "cosmic_signal": "...",
            "aura_alignment": "...",
            "interdim_activity": "...",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


@router.get("/signal-types")
async def get_phase5_signal_types() -> Dict[str, Any]:
    """Get Phase 5 signal types"""
    return {
        "signal_types": ["Neuro-Sync", "Cosmic Harmonic", "Aura Alignment", "Interdim Signal", "Quantum Coherence"]
    }


@router.get("/monitoring-levels")
async def get_phase5_monitoring_levels() -> Dict[str, Any]:
    """Get Phase 5 monitoring levels"""
    return {
        "levels": ["Low", "Medium", "High", "Critical"]
    }


@router.get("/time-periods")
async def get_phase5_time_periods() -> Dict[str, Any]:
    """Get Phase 5 time periods"""
    return {
        "periods": ["1h", "4h", "1d", "1w", "1m"]
    }


@router.get("/alert-types")
async def get_phase5_alert_types() -> Dict[str, Any]:
    """Get Phase 5 alert types"""
    return {
        "alert_types": ["Signal", "Threshold", "Anomaly", "Trend"]
    }


@router.get("/trends")
async def get_phase5_trends() -> Dict[str, Any]:
    """Get Phase 5 trends"""
    return {
        "trends": ["Rising", "Falling", "Stable", "Volatile"],
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@router.get("/distribution")
async def get_phase5_distribution() -> Dict[str, Any]:
    """Get Phase 5 distribution"""
    return {
        "distribution": {
            "signal_strength": [10, 20, 30, 25, 15],
            "time_periods": ["1h", "4h", "1d", "1w", "1m"]
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@router.get("/harmonization")
async def get_phase5_harmonization() -> Dict[str, Any]:
    """Get Phase 5 harmonization"""
    return {
        "harmonization_score": 0.85,
        "alignment_level": "High",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@router.get("/thresholds")
async def get_phase5_thresholds() -> Dict[str, Any]:
    """Get Phase 5 thresholds"""
    return {
        "thresholds": {
            "signal": 0.7,
            "alert": 0.8,
            "critical": 0.9
        }
    }


@router.get("/alerts")
async def get_phase5_alerts() -> Dict[str, Any]:
    """Get Phase 5 alerts"""
    return {
        "alerts": [
            {"type": "Signal", "message": "High neuro-sync detected", "level": "Medium"},
            {"type": "Threshold", "message": "Cosmic harmonic threshold exceeded", "level": "High"}
        ],
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@router.get("/recent-activity")
async def get_phase5_recent_activity() -> Dict[str, Any]:
    """Get Phase 5 recent activity"""
    return {
        "activities": [
            {"time": "2024-01-15T10:30:00Z", "event": "Signal detected", "strength": 0.8},
            {"time": "2024-01-15T10:25:00Z", "event": "Threshold check", "strength": 0.6}
        ],
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@router.get("/monitoring/settings")
async def get_phase5_monitoring_settings() -> Dict[str, Any]:
    """Get Phase 5 monitoring settings"""
    return {
        "settings": {
            "auto_monitoring": True,
            "alert_enabled": True,
            "threshold_level": "Medium"
        }
    }


@router.get("/overlay-metrics")
async def get_phase5_overlay_metrics() -> Dict[str, Any]:
    """Get Phase5 overlay metrics"""
    try:
        neuro_sync = await fetch_metric("neuro_sync_index")
        cosmic_signal = await fetch_metric("cosmic_harmonic_status")
        aura_alignment = await fetch_metric("aura_alignment_score")
        interdim_activity = await fetch_metric("interdim_signal_strength")

        return {
            "neuro_sync": neuro_sync,
            "cosmic_signal": cosmic_signal,
            "aura_alignment": aura_alignment,
            "interdim_activity": interdim_activity,
        }
    except Exception as e:
        logger.error(f"Error getting Phase5 overlay metrics: {e}")
        return {
            "neuro_sync": "...",
            "cosmic_signal": "...",
            "aura_alignment": "...",
            "interdim_activity": "...",
        }


@router.get("/quantum-indicators")
async def get_quantum_indicators() -> Dict[str, Any]:
    """Get quantum indicators data"""
    try:
        q_signal = await fetch_metric("quantum_signal_level")
        q_prob = await fetch_metric("quantum_trade_probability")
        q_entropy = await fetch_metric("quantum_entropy_index")

        return {
            "quantum_signal": q_signal,
            "trade_probability": q_prob,
            "entropy_index": q_entropy,
        }
    except Exception as e:
        logger.error(f"Error getting quantum indicators: {e}")
        return {
            "quantum_signal": "N/A",
            "trade_probability": "N/A",
            "entropy_index": "N/A",
        }


async def get_waveform_data() -> List[float]:
    """Get quantum waveform data"""
    try:
        raw = await r.lrange("quantum_waveform_data", 0, -1)
        waveform_data = []
        for x in raw[-300:]:
            try:
                value = float(str(x))
                # Validate for NaN and infinite values
                if not (np.isnan(value) or np.isinf(value)):
                    waveform_data.append(value)
                else:
                    # Replace invalid values with safe fallback
                    waveform_data.append(0.0)
            except (ValueError, TypeError):
                # Replace invalid data with safe fallback
                waveform_data.append(0.0)
        return waveform_data
    except Exception as e:
        logger.error(f"Error fetching waveform data: {e}")
        return []


@router.get("/quantum-waveform")
async def get_quantum_waveform() -> Dict[str, Any]:
    """Get quantum waveform data for charting"""
    try:
        y = await get_waveform_data()
        if not y:
            return {"data": [], "message": "No waveform data available yet."}

        # Validate waveform data
        if any(np.isnan(y)) or any(np.isinf(y)):
            logger.warning("Invalid waveform data detected, using fallback")
            y = [0.0] * len(y) if y else [0.0] * 100

        x = list(range(len(y)))

        return {
            "x": x,
            "y": y,
            "message": "Waveform data retrieved successfully",
        }
    except Exception as e:
        logger.error(f"Error getting quantum waveform: {e}")
        return {"x": [], "y": [], "message": "Error retrieving waveform data"}


@router.get("/status")
async def get_phase5_status() -> Dict[str, Any]:
    """Get Phase5 service status"""
    return {
        "status": "healthy",
        "service": "phase5-endpoints",
        "version": "1.0.0",
        "endpoints": [
            "/api/phase5/overlay-metrics",
            "/api/phase5/quantum-indicators",
            "/api/phase5/quantum-waveform",
            "/api/phase5/status",
        ],
    }
