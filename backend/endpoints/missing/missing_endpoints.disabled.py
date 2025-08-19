"""
Missing API Endpoints for Frontend Integration
Provides the missing endpoints that the frontend components require for live data.
"""

import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException

from backend.ai.ai_brains import trend_analysis
from backend.ai.ai_mystic import mystic_oracle
from backend.ai.auto_trade import get_trading_status
from backend.ai.trade_tracker import (
    get_active_trades,
    get_trade_history,
    get_trade_summary,
)

# Import actual AI services
from backend.modules.ai.ai_signals import (
    market_strength_signals,
    risk_adjusted_signals,
    signal_scorer,
    technical_signals,
)
from backend.modules.ai.persistent_cache import get_persistent_cache

logger = logging.getLogger(__name__)
router = APIRouter(tags=["missing-endpoints"])


def get_real_portfolio_data() -> dict[str, Any]:
    """Get real portfolio data from AI services"""
    try:
        cache = get_persistent_cache()
        trade_summary = get_trade_summary()
        active_trades = get_active_trades()

        # Calculate portfolio metrics
        total_value = (
            sum(trade["amount"] * trade["buy_price"] for trade in active_trades.values())
            if active_trades
            else 0
        )
        total_cost = (
            sum(trade["amount"] * trade["buy_price"] for trade in active_trades.values())
            if active_trades
            else 0
        )
        total_profit = trade_summary.get("total_pnl", 0)
        profit_percentage = (total_profit / total_cost * 100) if total_cost > 0 else 0

        # Get assets from active trades
        assets = []
        for symbol, trade in active_trades.items():
            current_price = 0
            coingecko_data = cache.get_coingecko()
            binance_data = cache.get_binance()

            if symbol in coingecko_data:
                current_price = coingecko_data[symbol].get("price", trade["buy_price"])
            elif symbol in binance_data:
                current_price = binance_data[symbol]
            else:
                current_price = trade["buy_price"]

            current_value = trade["amount"] * current_price
            profit = current_value - (trade["amount"] * trade["buy_price"])
            profit_percentage_asset = (
                (profit / (trade["amount"] * trade["buy_price"]) * 100)
                if trade["amount"] * trade["buy_price"] > 0
                else 0
            )
            allocation = (current_value / total_value * 100) if total_value > 0 else 0

            assets.append(
                {
                    "symbol": symbol,
                    "name": symbol,
                    "quantity": trade["amount"],
                    "avgPrice": trade["buy_price"],
                    "currentPrice": current_price,
                    "currentValue": current_value,
                    "profit": profit,
                    "profitPercentage": profit_percentage_asset,
                    "allocation": allocation,
                }
            )

        # Get top performers from cache
        top_performers = []
        coingecko_data = cache.get_coingecko()
        binance_data = cache.get_binance()

        for symbol, data in coingecko_data.items():
            if symbol in binance_data:
                top_performers.append(
                    {
                        "symbol": symbol,
                        "change": data.get("price_change_24h", 0),
                        "value": data.get("price", 0),
                    }
                )

        # Sort by change and take top 3
        top_performers.sort(key=lambda x: x["change"], reverse=True)
        top_performers = top_performers[:3]

        return {
            "totalValue": total_value,
            "totalCost": total_cost,
            "totalProfit": total_profit,
            "profitPercentage": profit_percentage,
            "dailyChange": 0,  # Would need daily tracking
            "weeklyChange": 0,  # Would need weekly tracking
            "monthlyChange": 0,  # Would need monthly tracking
            "activePositions": len(active_trades),
            "assets": assets,
            "topPerformers": top_performers,
        }
    except Exception as e:
        logger.error(f"Error getting real portfolio data: {e}")
        return {
            "totalValue": 0,
            "totalCost": 0,
            "totalProfit": 0,
            "profitPercentage": 0,
            "dailyChange": 0,
            "weeklyChange": 0,
            "monthlyChange": 0,
            "activePositions": 0,
            "assets": [],
            "topPerformers": [],
        }


def get_real_analytics_metrics() -> dict[str, Any]:
    """Get real analytics metrics from AI services"""
    try:
        trade_summary = get_trade_summary()
        trade_history = get_trade_history(limit=100)

        total_trades = trade_summary.get("total_trades", 0)
        profitable_trades = trade_summary.get("profitable_trades", 0)
        losing_trades = total_trades - profitable_trades
        win_rate = trade_summary.get("win_rate", 0)
        total_pnl = trade_summary.get("total_pnl", 0)
        avg_pnl = trade_summary.get("avg_pnl", 0)

        # Calculate additional metrics from trade history
        if trade_history:
            profits = [
                t.get("profit_loss", 0) for t in trade_history if t.get("profit_loss", 0) > 0
            ]
            losses = [t.get("profit_loss", 0) for t in trade_history if t.get("profit_loss", 0) < 0]

            avg_win = sum(profits) / len(profits) if profits else 0
            avg_loss = sum(losses) / len(losses) if losses else 0
            best_trade = max(profits) if profits else 0
            worst_trade = min(losses) if losses else 0

            # Calculate risk metrics
            risk_reward_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
            profit_factor = abs(sum(profits) / sum(losses)) if sum(losses) != 0 else 0
        else:
            avg_win = avg_loss = best_trade = worst_trade = 0
            risk_reward_ratio = profit_factor = 0

        return {
            "totalTrades": total_trades,
            "winningTrades": profitable_trades,
            "losingTrades": losing_trades,
            "winRate": win_rate,
            "totalPnL": total_pnl,
            "averagePnL": avg_pnl,
            "maxDrawdown": None,  # Would need drawdown calculation
            "sharpeRatio": None,  # Would need volatility calculation
            "profitFactor": profit_factor,
            "averageTradeDuration": None,  # Would need duration tracking
            "bestTrade": best_trade,
            "worstTrade": worst_trade,
            "calmarRatio": None,  # Would need calculation
            "sortinoRatio": None,  # Would need calculation
            "maxConsecutiveLosses": None,  # Would need calculation
            "averageWin": avg_win,
            "averageLoss": avg_loss,
            "riskRewardRatio": risk_reward_ratio,
            "dailyPnL": 0,  # Would need daily tracking
            "dailyReturn": 0,  # Would need daily tracking
        }
    except Exception as e:
        logger.error(f"Error getting real analytics metrics: {e}")
        return {
            "totalTrades": 0,
            "winningTrades": 0,
            "losingTrades": 0,
            "winRate": 0,
            "totalPnL": 0,
            "averagePnL": 0,
            "maxDrawdown": None,
            "sharpeRatio": None,
            "profitFactor": 0,
            "averageTradeDuration": None,
            "bestTrade": 0,
            "worstTrade": 0,
            "calmarRatio": None,
            "sortinoRatio": None,
            "maxConsecutiveLosses": None,
            "averageWin": 0,
            "averageLoss": 0,
            "riskRewardRatio": 0,
            "dailyPnL": 0,
            "dailyReturn": 0,
        }


def get_real_trading_history() -> list[dict[str, Any]]:
    """Get real trading history from AI services"""
    try:
        trade_history = get_trade_history(limit=20)

        formatted_history = []
        for trade in trade_history:
            formatted_history.append(
                {
                    "id": trade.get("order_id", "unknown"),
                    "symbol": trade.get("symbol", "unknown"),
                    "type": trade.get("type", "unknown").lower(),
                    "quantity": trade.get("amount", 0),
                    "price": trade.get("price", 0),
                    "pnl": trade.get("profit_loss", 0),
                    "timestamp": trade.get("timestamp", datetime.now().isoformat()),
                    "strategy": "AI Trading",
                    "status": "completed",
                    "entryReason": "AI signal analysis",
                    "exitReason": "Take profit or stop loss",
                    "riskLevel": "medium",
                    "tags": ["AI", "Automated"],
                }
            )

        return formatted_history
    except Exception as e:
        logger.error(f"Error getting real trading history: {e}")
        return []


def get_real_strategies() -> list[dict[str, Any]]:
    """Get real strategies from AI services"""
    try:
        # Get AI signal performance
        scored_signals = signal_scorer()
        risk_signals = risk_adjusted_signals()
        technical_signals_data = technical_signals()

        strategies = [
            {
                "name": "AI Signal Scorer",
                "trades": len(scored_signals),
                "winRate": None,  # Would need tracking
                "totalPnL": 0,  # Would need tracking
                "sharpeRatio": None,  # Would need calculation
                "maxDrawdown": None,  # Would need calculation
                "avgTradeDuration": None,  # Would need tracking
                "profitFactor": None,  # Would need calculation
            },
            {
                "name": "Risk-Adjusted Trading",
                "trades": len(risk_signals),
                "winRate": None,  # Would need tracking
                "totalPnL": 0,  # Would need tracking
                "sharpeRatio": None,  # Would need calculation
                "maxDrawdown": None,  # Would need calculation
                "avgTradeDuration": None,  # Would need tracking
                "profitFactor": None,  # Would need calculation
            },
            {
                "name": "Technical Analysis",
                "trades": len(technical_signals_data),
                "winRate": None,  # Would need tracking
                "totalPnL": 0,  # Would need tracking
                "sharpeRatio": None,  # Would need calculation
                "maxDrawdown": None,  # Would need calculation
                "avgTradeDuration": None,  # Would need tracking
                "profitFactor": None,  # Would need calculation
            },
        ]

        return strategies
    except Exception as e:
        logger.error(f"Error getting real strategies: {e}")
        return []


def get_real_ai_insights() -> list[dict[str, Any]]:
    """Get real AI insights from AI services"""
    try:
        insights = []

        # Get market strength analysis
        market_strength = market_strength_signals()
        if market_strength.get("market_strength", 0) > 30:
            insights.append(
                {
                    "type": "market",
                    "title": "Strong Market Momentum",
                    "description": (
                        f"Market strength at {market_strength.get('market_strength', 0)}% indicates bullish conditions."
                    ),
                    "impact": "positive",
                    "confidence": 85,
                    "recommendations": [
                        "Consider increasing position sizes",
                        "Focus on momentum strategies",
                    ],
                }
            )

        # Get trend analysis
        try:
            trend_data = trend_analysis()
            if trend_data:
                insights.append(
                    {
                        "type": "trend",
                        "title": "Trend Analysis",
                        "description": (
                            f"Trend analysis: {trend_data.get('summary', 'Market trends detected')}"
                        ),
                        "impact": "neutral",
                        "confidence": 78,
                        "recommendations": [
                            "Monitor trend continuation",
                            "Adjust position sizing",
                        ],
                    }
                )
        except Exception as e:
            logger.warning(f"Trend analysis not available: {e}")

        # Get mystic oracle insights
        try:
            oracle_data = mystic_oracle()
            if oracle_data:
                insights.append(
                    {
                        "type": "oracle",
                        "title": "Mystic Oracle Insight",
                        "description": (
                            f"Oracle prediction: {oracle_data.get('prediction', 'Market patterns analyzed')}"
                        ),
                        "impact": "positive",
                        "confidence": 92,
                        "recommendations": [
                            "Consider oracle guidance",
                            "Monitor for pattern completion",
                        ],
                    }
                )
        except Exception as e:
            logger.warning(f"Mystic oracle not available: {e}")

        # Get trading performance insight
        trade_summary = get_trade_summary()
        if trade_summary.get("win_rate", 0) > 70:
            insights.append(
                {
                    "type": "performance",
                    "title": "Excellent Win Rate",
                    "description": (
                        f"Current win rate of {trade_summary.get('win_rate', 0)}% indicates strong performance."
                    ),
                    "impact": "positive",
                    "confidence": 90,
                    "recommendations": [
                        "Maintain current strategy",
                        "Consider increasing position sizes",
                    ],
                }
            )

        return insights
    except Exception as e:
        logger.error(f"Error getting real AI insights: {e}")
        return []


def get_real_candlestick_data(symbol: str, interval: str) -> list[dict[str, Any]]:
    """Get real candlestick data from cache"""
    try:
        # Try to get real candlestick data from market data service
        try:
            from backend.services.market_data_service import MarketDataService

            market_data_service = MarketDataService()
            if market_data_service and hasattr(market_data_service, "get_candlestick_data"):
                data = await market_data_service.get_candlestick_data(symbol, interval)
                return data
        except ImportError:
            pass

        # Try to get from persistent cache if available
        cache = get_persistent_cache()
        if cache and hasattr(cache, "get_candlestick_data"):
            data = cache.get_candlestick_data(symbol, interval)
            return data

        # If no live candlestick data service is available, return empty list
        logger.warning(f"No live candlestick data service available for {symbol}")
        return []
    except Exception as e:
        logger.error(f"Error getting real candlestick data: {e}")
        return []


def get_real_alerts() -> list[dict[str, Any]]:
    """Get real alerts from AI services"""
    try:
        cache = get_persistent_cache()
        alerts = []

        # Check for extreme price movements
        coingecko_data = cache.get_coingecko()
        for symbol, data in coingecko_data.items():
            price_change = data.get("price_change_24h", 0)
            if price_change > 20:
                alerts.append(
                    {
                        "type": "EXTREME_GAIN",
                        "symbol": symbol,
                        "message": f"{symbol} +{price_change:.2f}%",
                        "priority": "high",
                        "timestamp": datetime.now().isoformat(),
                    }
                )
            elif price_change < -20:
                alerts.append(
                    {
                        "type": "EXTREME_LOSS",
                        "symbol": symbol,
                        "message": f"{symbol} {price_change:.2f}%",
                        "priority": "high",
                        "timestamp": datetime.now().isoformat(),
                    }
                )

        # Check market strength
        market_strength = market_strength_signals()
        if market_strength.get("market_strength", 0) > 50:
            alerts.append(
                {
                    "type": "MARKET_BULLISH",
                    "message": (
                        f"Strong market momentum: {market_strength.get('market_strength', 0)}%"
                    ),
                    "priority": "medium",
                    "timestamp": datetime.now().isoformat(),
                }
            )
        elif market_strength.get("market_weakness", 0) > 50:
            alerts.append(
                {
                    "type": "MARKET_BEARISH",
                    "message": (f"Market weakness: {market_strength.get('market_weakness', 0)}%"),
                    "priority": "medium",
                    "timestamp": datetime.now().isoformat(),
                }
            )

        # Check trading status
        trading_status = get_trading_status()
        if not trading_status["trading_enabled"]:
            alerts.append(
                {
                    "type": "TRADING_DISABLED",
                    "message": "Auto trading is currently disabled",
                    "priority": "low",
                    "timestamp": datetime.now().isoformat(),
                }
            )

        return alerts
    except Exception as e:
        logger.error(f"Error getting real alerts: {e}")
        return []


@router.get("/portfolio")
async def get_portfolio() -> dict[str, Any]:
    """Get portfolio data"""
    try:
        return get_real_portfolio_data()
    except Exception as e:
        logger.error(f"Error getting portfolio: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/metrics")
async def get_analytics_metrics() -> dict[str, Any]:
    """Get analytics metrics"""
    try:
        return get_real_analytics_metrics()
    except Exception as e:
        logger.error(f"Error getting analytics metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trading/history")
async def get_trading_history(limit: int = 10) -> list[dict[str, Any]]:
    """Get trading history"""
    try:
        history = get_real_trading_history()
        return history[:limit]
    except Exception as e:
        logger.error(f"Error getting trading history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/strategies")
async def get_analytics_strategies() -> list[dict[str, Any]]:
    """Get strategies data"""
    try:
        return get_real_strategies()
    except Exception as e:
        logger.error(f"Error getting strategies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ai/insights")
async def get_ai_insights() -> list[dict[str, Any]]:
    """Get AI insights"""
    try:
        return get_real_ai_insights()
    except Exception as e:
        logger.error(f"Error getting AI insights: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/market/candles")
async def get_candlestick_data(symbol: str = "BTC", interval: str = "1h") -> list[dict[str, Any]]:
    """Get candlestick data"""
    try:
        return await get_real_candlestick_data(symbol, interval)
    except Exception as e:
        logger.error(f"Error getting candlestick data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/wallet/transactions")
async def get_wallet_transactions(limit: int = 10) -> list[dict[str, Any]]:
    """Get wallet transactions"""
    try:
        # This would need actual wallet integration
        return []
    except Exception as e:
        logger.error(f"Error getting wallet transactions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts")
async def get_alerts() -> list[dict[str, Any]]:
    """Get alerts"""
    try:
        return get_real_alerts()
    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))



