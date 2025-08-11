"""
Advanced Analytics Endpoints

Handles all advanced analytics related API endpoints including performance, trade history, and AI insights.
"""

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException

from services.analytics_service import analytics_service
from services.order_service import order_service

logger = logging.getLogger(__name__)


def get_redis_client():
    """Get Redis client instance"""
    try:
        from database import get_redis_client as get_db_redis_client

        return get_db_redis_client()
    except Exception as e:
        logger.error(f"Failed to get Redis client: {str(e)}")
        raise HTTPException(status_code=500, detail="Redis service unavailable")


router = APIRouter()


@router.get("/performance")
async def get_performance_metrics(
    timeframe: str = "30d",
    redis_client: Any = Depends(lambda: get_redis_client()),
) -> Dict[str, Any]:
    """Get comprehensive performance metrics"""
    try:
        # Get real performance data from analytics service
        performance_data = await analytics_service.get_performance_metrics(timeframe)
        return performance_data
    except Exception as e:
        logger.error(f"Error getting performance metrics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting performance metrics: {str(e)}",
        )


@router.get("/trade-history")
async def get_trade_history(
    limit: int = 100,
    offset: int = 0,
    symbol: Optional[str] = None,
    strategy: Optional[str] = None,
    redis_client: Any = Depends(lambda: get_redis_client()),
) -> Dict[str, Any]:
    """Get detailed trade history"""
    try:
        # Get real trade history from order service
        trade_history = await order_service.get_trade_history(
            limit=limit, offset=offset, symbol=symbol, strategy=strategy
        )
        return trade_history
    except Exception as e:
        logger.error(f"Error getting trade history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting trade history: {str(e)}")


@router.get("/strategies")
async def get_strategy_performance(
    redis_client: Any = Depends(lambda: get_redis_client()),
) -> Dict[str, Any]:
    """Get strategy performance comparison"""
    try:
        # Get real strategy performance from analytics service
        strategies = await analytics_service.get_strategy_performance()
        return {"strategies": strategies}
    except Exception as e:
        logger.error(f"Error getting strategy performance: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting strategy performance: {str(e)}",
        )


@router.get("/ai-insights")
async def get_ai_insights(
    redis_client: Any = Depends(lambda: get_redis_client()),
) -> Dict[str, Any]:
    """Get AI-powered trading insights"""
    try:
        # Get real AI insights from analytics service
        insights = await analytics_service.get_ai_insights()
        return {"insights": insights}
    except Exception as e:
        logger.error(f"Error getting AI insights: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting AI insights: {str(e)}")


@router.get("/portfolio-performance")
async def get_portfolio_performance(timeframe: str = "30d"):
    """Get portfolio performance analytics (live)"""
    try:
        # Get live portfolio performance from analytics_service
        performance = await analytics_service.get_portfolio_performance(timeframe)
        return performance
    except Exception as e:
        logger.error(f"Error getting portfolio performance: {str(e)}")
        raise HTTPException(status_code=500, detail="Error getting portfolio performance")


@router.get("/risk-metrics")
async def get_risk_metrics():
    """Get risk metrics (live)"""
    try:
        # Get live risk metrics from analytics_service
        risk_metrics = await analytics_service.get_risk_metrics()
        return risk_metrics
    except Exception as e:
        logger.error(f"Error getting risk metrics: {str(e)}")
        raise HTTPException(status_code=500, detail="Error getting risk metrics")


@router.get("/market-analysis")
async def get_market_analysis(symbol: str = "BTC/USDT"):
    """Get market analysis (live)"""
    try:
        # Get live market analysis from analytics_service
        analysis = await analytics_service.get_market_analysis(symbol)
        return analysis
    except Exception as e:
        logger.error(f"Error getting market analysis: {str(e)}")
        raise HTTPException(status_code=500, detail="Error getting market analysis")
