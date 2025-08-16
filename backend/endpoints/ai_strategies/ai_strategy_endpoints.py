"""AI Strategy System Live Endpoints"""

from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from backend.services.ai_strategies import get_ai_strategy_service
from backend.services.portfolio_service import portfolio_service

router = APIRouter()


@router.get("/ai-strategy/status")
async def get_ai_strategy_status():
    """Get AI strategy system status"""
    try:
        ai_service = get_ai_strategy_service()
        status = await ai_service.get_status()
        return status
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get AI strategy status: {str(e)}",
        )


@router.get("/ai-strategy/leaderboard")
async def get_ai_strategy_leaderboard():
    """Get AI strategy leaderboard"""
    try:
        ai_service = get_ai_strategy_service()
        leaderboard = await ai_service.get_leaderboard()
        return {"strategies": leaderboard}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get AI strategy leaderboard: {str(e)}",
        )


@router.get("/ai-strategy/auto-buy/config")
async def get_auto_buy_config():
    """Get auto-buy configuration"""
    try:
        ai_service = get_ai_strategy_service()
        config = await ai_service.get_auto_buy_config()
        return config
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get auto-buy config: {str(e)}")


@router.get("/ai-strategy/auto-buy/history")
async def get_auto_buy_history():
    """Get auto-buy history"""
    try:
        ai_service = get_ai_strategy_service()
        history = await ai_service.get_auto_buy_history()
        return {"history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get auto-buy history: {str(e)}")


@router.get("/ai-strategy/events")
async def get_ai_strategy_events(limit: int = 10):
    """Get recent AI strategy events"""
    try:
        ai_service = get_ai_strategy_service()
        events = await ai_service.get_events(limit=limit)
        return {"events": events}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get AI strategy events: {str(e)}",
        )


@router.get("/ai-strategy/position")
async def get_current_position():
    """Get current trading position"""
    try:
        portfolio = await portfolio_service.get_current_position()
        return portfolio
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get current position: {str(e)}")


@router.get("/ai-strategy/analytics/performance")
async def get_performance_analytics():
    """Get performance analytics"""
    try:
        ai_service = get_ai_strategy_service()
        analytics = await ai_service.get_performance_analytics()
        return analytics
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get performance analytics: {str(e)}",
        )


@router.get("/ai-strategy/mutations")
async def get_mutations():
    """Get strategy mutations"""
    try:
        ai_service = get_ai_strategy_service()
        mutations = await ai_service.get_mutations()
        return {"mutations": mutations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get mutations: {str(e)}")


@router.get("/ai-strategy/logs/recent")
async def get_recent_logs(lines: int = 50):
    """Get recent logs"""
    try:
        ai_service = get_ai_strategy_service()
        logs = await ai_service.get_recent_logs(lines=lines)
        return {"logs": logs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get recent logs: {str(e)}")


@router.get("/ai-strategy/health")
async def get_ai_strategy_health():
    """Health check for AI strategy system"""
    try:
        ai_service = get_ai_strategy_service()
        health = await ai_service.get_health()
        return health
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get AI strategy health: {str(e)}",
        )


# POST endpoints
@router.post("/ai-strategy/leaderboard/add")
async def add_strategy(strategy: Dict[str, Any]):
    """Add new strategy to leaderboard"""
    try:
        ai_service = get_ai_strategy_service()
        result = await ai_service.add_strategy(strategy)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add strategy: {str(e)}")


@router.post("/ai-strategy/mutations/add")
async def add_mutation(mutation: Dict[str, Any]):
    """Add new mutation"""
    try:
        ai_service = get_ai_strategy_service()
        result = await ai_service.add_mutation(mutation)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add mutation: {str(e)}")


@router.post("/ai-strategy/position/update")
async def update_position(position: Dict[str, Any]):
    """Update current position"""
    try:
        portfolio = await portfolio_service.update_position(position)
        return portfolio
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update position: {str(e)}")


@router.delete("/ai-strategy/position/clear")
async def clear_position():
    """Clear current position"""
    try:
        portfolio = await portfolio_service.clear_position()
        return portfolio
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear position: {str(e)}")



