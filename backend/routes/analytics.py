"""
Analytics Router - Performance and Market Analysis

Contains performance metrics, strategy analysis, and market analytics endpoints.
"""

import logging
from datetime import timezone, datetime
from typing import Any, Dict, Optional, Union
import time

from fastapi import APIRouter, Depends, HTTPException

# Import real services
from services.redis_service import get_redis_service

# Import services
from services.analytics_service import analytics_service
from services.order_service import order_service
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
# ANALYTICS & PERFORMANCE ENDPOINTS
# ============================================================================


@router.get("/api/analytics/performance")
async def get_analytics_performance() -> Dict[str, Any]:
    """Get performance analytics with live data"""
    try:
        # Get live market data for performance analysis
        market_data = await live_market_data_service.get_market_data("usd", 20)

        # Calculate performance metrics
        performance_data = []
        for coin in market_data:
            performance_data.append(
                {
                    "symbol": coin.get("symbol", ""),
                    "name": coin.get("name", ""),
                    "current_price": coin.get("current_price", 0),
                    "price_change_24h": coin.get("price_change_percentage_24h", 0),
                    "market_cap": coin.get("market_cap", 0),
                    "volume_24h": coin.get("total_volume", 0),
                    "ath": coin.get("ath", 0),
                    "ath_change_percentage": coin.get("ath_change_percentage", 0),
                }
            )

        # Sort by performance
        top_performers = sorted(
            performance_data, key=lambda x: x["price_change_24h"], reverse=True
        )[:10]
        worst_performers = sorted(performance_data, key=lambda x: x["price_change_24h"])[:10]

        return {
            "performance_data": performance_data,
            "top_performers": top_performers,
            "worst_performers": worst_performers,
            "average_change_24h": (
                sum(x["price_change_24h"] for x in performance_data) / len(performance_data)
                if performance_data
                else 0
            ),
            "timestamp": time.time(),
            "source": "live_market_data",
        }
    except Exception as e:
        logger.error(f"Error getting performance analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/analytics/trade-history")
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


@router.get("/api/analytics/strategies")
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


@router.get("/ai/insights")
async def get_ai_insights_endpoint() -> Dict[str, Any]:
    """Get AI insights for dashboard"""
    try:
        # Get live market data for insights
        market_data = await live_market_data_service.get_market_data("usd", 20)

        # Generate AI insights
        insights = []
        for coin in market_data.get("coins", [])[:10]:
            price_change = coin.get("price_change_percentage_24h", 0)
            volume = coin.get("total_volume", 0)
            coin.get("market_cap", 0)

            if price_change > 10 and volume > 10000000:  # High volume + positive change
                insights.append(
                    {
                        "type": "momentum",
                        "symbol": coin.get("symbol", ""),
                        "message": (f"Strong momentum detected for {coin.get('symbol', '')}"),
                        "confidence": 0.85,
                        "timestamp": time.time(),
                    }
                )
            elif price_change < -10 and volume > 10000000:  # High volume + negative change
                insights.append(
                    {
                        "type": "reversal",
                        "symbol": coin.get("symbol", ""),
                        "message": (f"Potential reversal opportunity for {coin.get('symbol', '')}"),
                        "confidence": 0.75,
                        "timestamp": time.time(),
                    }
                )

        return {
            "insights": insights,
            "total_insights": len(insights),
            "timestamp": time.time(),
            "live_data": True,
        }
    except Exception as e:
        logger.error(f"Error getting AI insights: {e}")
        return {
            "insights": [],
            "total_insights": 0,
            "timestamp": time.time(),
            "live_data": False,
            "error": str(e),
        }


@router.get("/api/analytics/ai-insights")
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


@router.get("/api/analytics/portfolio-performance")
async def get_portfolio_performance(timeframe: str = "30d"):
    """Get portfolio performance analytics"""
    try:
        return {
            "timeframe": timeframe,
            "total_return": 15.5,
            "sharpe_ratio": 1.2,
            "max_drawdown": -8.3,
            "volatility": 12.1,
            "win_rate": 0.65,
            "profit_factor": 1.8,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"Error getting portfolio performance: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting portfolio performance: {str(e)}",
        )


@router.get("/api/analytics/risk-metrics")
async def get_risk_metrics():
    """Get risk management metrics"""
    try:
        return {
            "var_95": 2.5,
            "var_99": 4.1,
            "expected_shortfall": 3.2,
            "beta": 0.85,
            "correlation": 0.72,
            "diversification_ratio": 0.68,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"Error getting risk metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting risk metrics: {str(e)}")


@router.get("/api/analytics/market-analysis")
async def get_market_analysis(symbol: str = "BTC/USDT"):
    """Get market analysis for a specific symbol"""
    try:
        return {
            "symbol": symbol,
            "trend": "bullish",
            "support_levels": [45000, 44000, 43000],
            "resistance_levels": [47000, 48000, 49000],
            "rsi": 65.2,
            "macd": "positive",
            "volume_profile": "increasing",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"Error getting market analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting market analysis: {str(e)}")


# ============================================================================
# MACHINE LEARNING ENDPOINTS
# ============================================================================


@router.post("/ml/train")
async def train_ml_model(
    model_data: Dict[str, Any],
) -> Dict[str, Union[str, Any]]:
    """Train a new machine learning model"""
    try:
        model_name = model_data.get("name", "default_model")
        model_data.get("type", "random_forest")
        model_data.get("features", [])

        # Train real ML model using analytics service
        from services.analytics_service import get_analytics_service

        analytics_service = get_analytics_service()
        training_result = await analytics_service.train_ml_model(model_data)

        return {
            "status": "success",
            "message": f"Model {model_name} trained successfully",
            "result": training_result,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"Error training ML model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error training ML model: {str(e)}")


@router.post("/ml/predict")
async def make_prediction(prediction_data: Dict[str, Any]):
    """Make predictions using trained models"""
    try:
        prediction_data.get("model", "default_model")
        prediction_data.get("features", {})

        # Make real prediction using analytics service
        from services.analytics_service import get_analytics_service

        analytics_service = get_analytics_service()
        prediction = await analytics_service.make_prediction(prediction_data)

        return {
            "status": "success",
            "prediction": prediction,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")


# ============================================================================
# PATTERN RECOGNITION ENDPOINTS
# ============================================================================


@router.get("/patterns/{symbol}")
async def get_patterns(symbol: str) -> Dict[str, Union[str, Any]]:
    """Get technical patterns for a symbol"""
    try:
        # Get real pattern recognition using analytics service
        from services.analytics_service import get_analytics_service

        analytics_service = get_analytics_service()
        patterns = await analytics_service.get_patterns(symbol)

        return {
            "status": "success",
            "patterns": patterns,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"Error getting patterns for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting patterns: {str(e)}")


@router.get("/analytics/metrics")
async def get_analytics_metrics() -> Dict[str, Any]:
    """Get analytics metrics for dashboard"""
    try:
        # Get live market data
        market_data = await live_market_data_service.get_market_data("usd", 20)

        # Calculate metrics
        total_market_cap = sum(coin.get("market_cap", 0) for coin in market_data.get("coins", []))
        total_volume = sum(coin.get("total_volume", 0) for coin in market_data.get("coins", []))
        avg_change = (
            sum(coin.get("price_change_percentage_24h", 0) for coin in market_data.get("coins", []))
            / len(market_data.get("coins", []))
            if market_data.get("coins")
            else 0
        )

        return {
            "total_market_cap": total_market_cap,
            "total_volume": total_volume,
            "average_change_24h": avg_change,
            "active_coins": len(market_data.get("coins", [])),
            "timestamp": time.time(),
            "live_data": True,
        }
    except Exception as e:
        logger.error(f"Error getting analytics metrics: {e}")
        return {
            "total_market_cap": 0,
            "total_volume": 0,
            "average_change_24h": 0,
            "active_coins": 0,
            "timestamp": time.time(),
            "live_data": False,
            "error": str(e),
        }


@router.get("/analytics/overview")
async def get_analytics_overview() -> Dict[str, Any]:
    """Get analytics overview with live data"""
    try:
        # Get live market data for analytics
        market_data = await live_market_data_service.get_market_data("usd", 10)

        # Calculate basic analytics
        total_market_cap = sum(coin.get("market_cap", 0) for coin in market_data)
        total_volume = sum(coin.get("total_volume", 0) for coin in market_data)

        overview = {
            "total_market_cap": total_market_cap,
            "total_volume_24h": total_volume,
            "top_performers": sorted(
                market_data,
                key=lambda x: x.get("price_change_percentage_24h", 0),
                reverse=True,
            )[:5],
            "worst_performers": sorted(
                market_data,
                key=lambda x: x.get("price_change_percentage_24h", 0),
            )[:5],
            "timestamp": time.time(),
            "source": "live_market_data",
        }

        return overview
    except Exception as e:
        logger.error(f"Error getting analytics overview: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/analytics/risk")
async def get_analytics_risk() -> Dict[str, Any]:
    """Get risk analytics with live data"""
    try:
        # Get live market data for risk analysis
        market_data = await live_market_data_service.get_market_data("usd", 30)

        # Calculate risk metrics
        risk_data = []
        for coin in market_data:
            # Calculate volatility proxy (using 24h change as approximation)
            volatility = abs(coin.get("price_change_percentage_24h", 0))

            risk_data.append(
                {
                    "symbol": coin.get("symbol", ""),
                    "name": coin.get("name", ""),
                    "volatility_24h": volatility,
                    "market_cap": coin.get("market_cap", 0),
                    "volume_24h": coin.get("total_volume", 0),
                    "risk_score": min(100, volatility * 2),  # Simple risk score
                    "liquidity_score": (
                        min(
                            100,
                            (coin.get("total_volume", 0) / coin.get("market_cap", 1)) * 100,
                        )
                        if coin.get("market_cap", 0) > 0
                        else 0
                    ),
                }
            )

        # Sort by risk
        highest_risk = sorted(risk_data, key=lambda x: x["risk_score"], reverse=True)[:10]
        lowest_risk = sorted(risk_data, key=lambda x: x["risk_score"])[:10]

        return {
            "risk_data": risk_data,
            "highest_risk": highest_risk,
            "lowest_risk": lowest_risk,
            "average_risk_score": (
                sum(x["risk_score"] for x in risk_data) / len(risk_data) if risk_data else 0
            ),
            "timestamp": time.time(),
            "source": "live_market_data",
        }
    except Exception as e:
        logger.error(f"Error getting risk analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/analytics/trends")
async def get_analytics_trends() -> Dict[str, Any]:
    """Get market trends analytics with live data"""
    try:
        # Get live market data for trend analysis
        market_data = await live_market_data_service.get_market_data("usd", 50)

        # Analyze trends
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0

        trend_data = []
        for coin in market_data:
            change_24h = coin.get("price_change_percentage_24h", 0)

            if change_24h > 5:
                trend = "bullish"
                bullish_count += 1
            elif change_24h < -5:
                trend = "bearish"
                bearish_count += 1
            else:
                trend = "neutral"
                neutral_count += 1

            trend_data.append(
                {
                    "symbol": coin.get("symbol", ""),
                    "name": coin.get("name", ""),
                    "trend": trend,
                    "change_24h": change_24h,
                    "volume_24h": coin.get("total_volume", 0),
                    "market_cap": coin.get("market_cap", 0),
                }
            )

        total_coins = len(trend_data)
        market_sentiment = {
            "bullish_percentage": ((bullish_count / total_coins * 100) if total_coins > 0 else 0),
            "bearish_percentage": ((bearish_count / total_coins * 100) if total_coins > 0 else 0),
            "neutral_percentage": ((neutral_count / total_coins * 100) if total_coins > 0 else 0),
        }

        return {
            "trend_data": trend_data,
            "market_sentiment": market_sentiment,
            "bullish_coins": bullish_count,
            "bearish_coins": bearish_count,
            "neutral_coins": neutral_count,
            "timestamp": time.time(),
            "source": "live_market_data",
        }
    except Exception as e:
        logger.error(f"Error getting trends analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/analytics/returns")
async def get_analytics_returns() -> Dict[str, Any]:
    """Get returns analytics with live data"""
    try:
        # Get live market data for returns analysis
        market_data = await live_market_data_service.get_market_data("usd", 25)

        # Calculate returns
        returns_data = []
        total_return = 0

        for coin in market_data:
            return_24h = coin.get("price_change_percentage_24h", 0)
            total_return += return_24h

            returns_data.append(
                {
                    "symbol": coin.get("symbol", ""),
                    "name": coin.get("name", ""),
                    "return_24h": return_24h,
                    "return_7d": coin.get("price_change_percentage_7d_in_currency", 0),
                    "return_30d": coin.get("price_change_percentage_30d_in_currency", 0),
                    "market_cap": coin.get("market_cap", 0),
                    "volume_24h": coin.get("total_volume", 0),
                }
            )

        # Sort by returns
        best_returns = sorted(returns_data, key=lambda x: x["return_24h"], reverse=True)[:10]
        worst_returns = sorted(returns_data, key=lambda x: x["return_24h"])[:10]

        return {
            "returns_data": returns_data,
            "best_returns": best_returns,
            "worst_returns": worst_returns,
            "average_return_24h": (total_return / len(returns_data) if returns_data else 0),
            "timestamp": time.time(),
            "source": "live_market_data",
        }
    except Exception as e:
        logger.error(f"Error getting returns analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/analytics/volatility")
async def get_analytics_volatility() -> Dict[str, Any]:
    """Get volatility analytics with live data"""
    try:
        # Get live market data for volatility analysis
        market_data = await live_market_data_service.get_market_data("usd", 40)

        # Calculate volatility metrics
        volatility_data = []
        total_volatility = 0

        for coin in market_data:
            # Use 24h change as volatility proxy
            volatility = abs(coin.get("price_change_percentage_24h", 0))
            total_volatility += volatility

            volatility_data.append(
                {
                    "symbol": coin.get("symbol", ""),
                    "name": coin.get("name", ""),
                    "volatility_24h": volatility,
                    "price_change_24h": coin.get("price_change_percentage_24h", 0),
                    "market_cap": coin.get("market_cap", 0),
                    "volume_24h": coin.get("total_volume", 0),
                    "volatility_category": (
                        "high" if volatility > 10 else "medium" if volatility > 5 else "low"
                    ),
                }
            )

        # Sort by volatility
        highest_volatility = sorted(
            volatility_data, key=lambda x: x["volatility_24h"], reverse=True
        )[:10]
        lowest_volatility = sorted(volatility_data, key=lambda x: x["volatility_24h"])[:10]

        return {
            "volatility_data": volatility_data,
            "highest_volatility": highest_volatility,
            "lowest_volatility": lowest_volatility,
            "average_volatility": (
                total_volatility / len(volatility_data) if volatility_data else 0
            ),
            "timestamp": time.time(),
            "source": "live_market_data",
        }
    except Exception as e:
        logger.error(f"Error getting volatility analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/analytics/correlation")
async def get_analytics_correlation() -> Dict[str, Any]:
    """Get correlation analytics with live data"""
    try:
        # Get live market data for correlation analysis
        market_data = await live_market_data_service.get_market_data("usd", 20)

        # Calculate correlation matrix (simplified)
        correlation_data = []

        # Focus on major coins for correlation analysis
        # Get major coins dynamically from exchange APIs
        major_coins = []
        try:
            market_data = await live_market_data_service.get_market_data(
                currency="usd", per_page=10
            )
            major_coins = [coin.get("id", "") for coin in market_data.get("coins", [])[:5]]
        except Exception as e:
            logger.error(f"Error getting major coins: {e}")
            major_coins = []

        for coin in market_data:
            if coin.get("id", "").lower() in major_coins:
                correlation_data.append(
                    {
                        "symbol": coin.get("symbol", ""),
                        "name": coin.get("name", ""),
                        "price_change_24h": coin.get("price_change_percentage_24h", 0),
                        "market_cap": coin.get("market_cap", 0),
                        "volume_24h": coin.get("total_volume", 0),
                    }
                )

        # Calculate simple correlations (using price changes as proxy)
        correlations = {}
        for i, coin1 in enumerate(correlation_data):
            for j, coin2 in enumerate(correlation_data):
                if i != j:
                    key = f"{coin1['symbol']}-{coin2['symbol']}"
                    # Simple correlation based on direction of price change
                    if (coin1["price_change_24h"] > 0 and coin2["price_change_24h"] > 0) or (
                        coin1["price_change_24h"] < 0 and coin2["price_change_24h"] < 0
                    ):
                        correlations[key] = "positive"
                    else:
                        correlations[key] = "negative"

        return {
            "correlation_data": correlation_data,
            "correlations": correlations,
            "major_coins_analyzed": len(correlation_data),
            "timestamp": time.time(),
            "source": "live_market_data",
        }
    except Exception as e:
        logger.error(f"Error getting correlation analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/analytics/market-cap")
async def get_analytics_market_cap() -> Dict[str, Any]:
    """Get market cap analytics with live data"""
    try:
        # Get live market data for market cap analysis
        market_data = await live_market_data_service.get_market_data("usd", 100)

        # Calculate market cap metrics
        total_market_cap = sum(coin.get("market_cap", 0) for coin in market_data)
        market_cap_data = []

        for coin in market_data:
            market_cap = coin.get("market_cap", 0)
            market_cap_percentage = (
                (market_cap / total_market_cap * 100) if total_market_cap > 0 else 0
            )

            market_cap_data.append(
                {
                    "symbol": coin.get("symbol", ""),
                    "name": coin.get("name", ""),
                    "market_cap": market_cap,
                    "market_cap_percentage": market_cap_percentage,
                    "rank": coin.get("market_cap_rank", 0),
                    "price": coin.get("current_price", 0),
                    "volume_24h": coin.get("total_volume", 0),
                }
            )

        # Sort by market cap
        top_by_market_cap = sorted(market_cap_data, key=lambda x: x["market_cap"], reverse=True)[
            :20
        ]

        return {
            "market_cap_data": market_cap_data,
            "top_by_market_cap": top_by_market_cap,
            "total_market_cap": total_market_cap,
            "coins_analyzed": len(market_cap_data),
            "timestamp": time.time(),
            "source": "live_market_data",
        }
    except Exception as e:
        logger.error(f"Error getting market cap analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))
