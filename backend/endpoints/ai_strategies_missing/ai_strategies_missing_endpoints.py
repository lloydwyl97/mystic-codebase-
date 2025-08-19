"""
Missing AI Strategies Endpoints

Provides missing AI strategies endpoint that returns live data:
- AI Strategies List
"""

import logging
import time
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from backend.modules.ai.persistent_cache import get_persistent_cache

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/api/ai/strategies")
async def get_ai_strategies() -> Dict[str, Any]:
    """
    Get AI strategies with live data

    Returns comprehensive AI strategy information including:
    - Active strategies
    - Performance metrics
    - Strategy parameters
    - AI model status
    """
    try:
        # Get live data from persistent cache
        cache = get_persistent_cache()
        strategies = []
        strategy_id = 1

        # Assume these services are available for live metrics
        from backend.services.trading_logs_service import TradingLogsService
        from backend.services.ai_model_registry import AIModelRegistry
        from backend.services.analytics_service import AnalyticsService

        trading_logs_service = TradingLogsService()
        ai_model_registry = AIModelRegistry()
        analytics_service = AnalyticsService()

        # Process Binance data for strategies
        for symbol, price_data in cache.get_binance().items():
            if isinstance(price_data, dict) and "price" in price_data:
                base_symbol = symbol.replace("USDT", "")

                # Fetch live trading stats
                trading_stats = trading_logs_service.get_stats(base_symbol)
                # Fetch live AI model info
                model_info = ai_model_registry.get_model_info(base_symbol)
                # Fetch live risk/analytics metrics
                risk_metrics = analytics_service.get_risk_metrics(base_symbol)

                # Build strategy types (breakout, mean_reversion, momentum)
                for strategy_type in ["breakout", "mean_reversion", "momentum"]:
                    strategies.append(
                        {
                            "id": f"strategy_{strategy_id}",
                            "name": f"{base_symbol}_{strategy_type.capitalize()}_AI",
                            "type": strategy_type,
                            "symbol": base_symbol,
                            "description": f"AI {strategy_type.replace('_', ' ')} strategy for {base_symbol}",
                            "status": trading_stats.get("status") if trading_stats else None,
                            "performance": {
                                "daily_return": (
                                    trading_stats.get("daily_return") if trading_stats else None
                                ),
                                "weekly_return": (
                                    trading_stats.get("weekly_return") if trading_stats else None
                                ),
                                "monthly_return": (
                                    trading_stats.get("monthly_return") if trading_stats else None
                                ),
                                "total_return": (
                                    trading_stats.get("total_return") if trading_stats else None
                                ),
                                "sharpe_ratio": (
                                    risk_metrics.get("sharpe_ratio") if risk_metrics else None
                                ),
                            },
                            "parameters": model_info.get("parameters") if model_info else None,
                            "ai_model": (
                                {
                                    "model_type": (
                                        model_info.get("model_type") if model_info else None
                                    ),
                                    "version": model_info.get("version") if model_info else None,
                                    "last_trained": (
                                        model_info.get("last_trained") if model_info else None
                                    ),
                                    "accuracy": model_info.get("accuracy") if model_info else None,
                                    "status": model_info.get("status") if model_info else None,
                                }
                                if model_info
                                else None
                            ),
                            "trading_stats": (
                                {
                                    "total_trades": (
                                        trading_stats.get("total_trades") if trading_stats else None
                                    ),
                                    "winning_trades": (
                                        trading_stats.get("winning_trades")
                                        if trading_stats
                                        else None
                                    ),
                                    "losing_trades": (
                                        trading_stats.get("losing_trades")
                                        if trading_stats
                                        else None
                                    ),
                                    "win_rate": (
                                        trading_stats.get("win_rate") if trading_stats else None
                                    ),
                                    "avg_trade_duration": (
                                        trading_stats.get("avg_trade_duration")
                                        if trading_stats
                                        else None
                                    ),
                                    "total_volume": (
                                        trading_stats.get("total_volume") if trading_stats else None
                                    ),
                                }
                                if trading_stats
                                else None
                            ),
                            "risk_metrics": (
                                {
                                    "max_drawdown": (
                                        risk_metrics.get("max_drawdown") if risk_metrics else None
                                    ),
                                    "volatility": (
                                        risk_metrics.get("volatility") if risk_metrics else None
                                    ),
                                    "var_95": risk_metrics.get("var_95") if risk_metrics else None,
                                    "current_risk": (
                                        risk_metrics.get("current_risk") if risk_metrics else None
                                    ),
                                }
                                if risk_metrics
                                else None
                            ),
                            "live_data": True,
                            "timestamp": time.time(),
                            "source": "persistent_cache",
                        }
                    )
                    strategy_id += 1

        # Process Coinbase data for additional strategies (arbitrage)
        for symbol, price_data in cache.get_coinbase().items():
            if isinstance(price_data, dict) and "price" in price_data:
                base_symbol = symbol.replace("-USD", "")
                trading_stats = trading_logs_service.get_stats(base_symbol)
                model_info = ai_model_registry.get_model_info(base_symbol)
                risk_metrics = analytics_service.get_risk_metrics(base_symbol)
                strategies.append(
                    {
                        "id": f"strategy_{strategy_id}",
                        "name": f"{base_symbol}_Arbitrage_AI",
                        "type": "arbitrage",
                        "symbol": base_symbol,
                        "description": f"AI arbitrage detection between exchanges for {base_symbol}",
                        "status": trading_stats.get("status") if trading_stats else None,
                        "performance": {
                            "daily_return": (
                                trading_stats.get("daily_return") if trading_stats else None
                            ),
                            "weekly_return": (
                                trading_stats.get("weekly_return") if trading_stats else None
                            ),
                            "monthly_return": (
                                trading_stats.get("monthly_return") if trading_stats else None
                            ),
                            "total_return": (
                                trading_stats.get("total_return") if trading_stats else None
                            ),
                            "sharpe_ratio": (
                                risk_metrics.get("sharpe_ratio") if risk_metrics else None
                            ),
                        },
                        "parameters": model_info.get("parameters") if model_info else None,
                        "ai_model": (
                            {
                                "model_type": model_info.get("model_type") if model_info else None,
                                "version": model_info.get("version") if model_info else None,
                                "last_trained": (
                                    model_info.get("last_trained") if model_info else None
                                ),
                                "accuracy": model_info.get("accuracy") if model_info else None,
                                "status": model_info.get("status") if model_info else None,
                            }
                            if model_info
                            else None
                        ),
                        "trading_stats": (
                            {
                                "total_trades": (
                                    trading_stats.get("total_trades") if trading_stats else None
                                ),
                                "winning_trades": (
                                    trading_stats.get("winning_trades") if trading_stats else None
                                ),
                                "losing_trades": (
                                    trading_stats.get("losing_trades") if trading_stats else None
                                ),
                                "win_rate": (
                                    trading_stats.get("win_rate") if trading_stats else None
                                ),
                                "avg_trade_duration": (
                                    trading_stats.get("avg_trade_duration")
                                    if trading_stats
                                    else None
                                ),
                                "total_volume": (
                                    trading_stats.get("total_volume") if trading_stats else None
                                ),
                            }
                            if trading_stats
                            else None
                        ),
                        "risk_metrics": (
                            {
                                "max_drawdown": (
                                    risk_metrics.get("max_drawdown") if risk_metrics else None
                                ),
                                "volatility": (
                                    risk_metrics.get("volatility") if risk_metrics else None
                                ),
                                "var_95": risk_metrics.get("var_95") if risk_metrics else None,
                                "current_risk": (
                                    risk_metrics.get("current_risk") if risk_metrics else None
                                ),
                            }
                            if risk_metrics
                            else None
                        ),
                        "live_data": True,
                        "timestamp": time.time(),
                        "source": "persistent_cache",
                    }
                )
                strategy_id += 1

        # Calculate overall statistics (live only)
        total_strategies = len(strategies)
        active_strategies = len([s for s in strategies if s.get("status") == "active"])
        avg_performance = (
            sum(
                s["performance"]["daily_return"]
                for s in strategies
                if s["performance"] and s["performance"]["daily_return"] is not None
            )
            / total_strategies
            if strategies
            else 0
        )
        avg_confidence = (
            sum(
                s["ai_model"]["accuracy"]
                for s in strategies
                if s["ai_model"] and s["ai_model"]["accuracy"] is not None
            )
            / total_strategies
            if strategies
            else 0
        )

        return {
            "strategies": strategies,
            "summary": {
                "total_strategies": total_strategies,
                "active_strategies": active_strategies,
                "monitoring_strategies": total_strategies - active_strategies,
                "avg_daily_performance": avg_performance,
                "avg_ai_confidence": avg_confidence,
                "total_ai_models": total_strategies,
                "active_ai_models": active_strategies,
            },
            "performance_breakdown": {
                "by_type": {
                    "breakout": len([s for s in strategies if s["type"] == "breakout"]),
                    "mean_reversion": len([s for s in strategies if s["type"] == "mean_reversion"]),
                    "momentum": len([s for s in strategies if s["type"] == "momentum"]),
                    "arbitrage": len([s for s in strategies if s["type"] == "arbitrage"]),
                },
                "by_status": {
                    "active": active_strategies,
                    "monitoring": total_strategies - active_strategies,
                },
            },
            "live_data": True,
            "timestamp": time.time(),
            "source": "persistent_cache",
        }

    except Exception as e:
        logger.error(f"Error getting AI strategies: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting AI strategies: {str(e)}")



