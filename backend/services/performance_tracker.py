"""
Advanced Performance Tracking Service
Tracks detailed performance metrics for all trading activities
"""

import logging
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from backend.services.websocket_manager import websocket_manager

logger = logging.getLogger(__name__)


@dataclass
class TradeResult:
    """Trade execution result with performance metrics"""

    trade_id: str
    symbol: str
    entry_price: float
    exit_price: float
    position_size: float
    profit_loss: float
    profit_percentage: float
    duration_hours: float
    ml_probability: float
    sentiment_score: float
    original_score: int
    enhanced_score: int
    risk_level: str
    success: bool
    timestamp: datetime


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""

    total_trades: int
    successful_trades: int
    win_rate: float
    average_profit: float
    average_loss: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    ml_accuracy: float
    sentiment_accuracy: float
    enhanced_score_accuracy: float
    best_performing_hours: list[int]
    best_performing_symbols: list[str]
    risk_adjusted_return: float
    timestamp: datetime


class AdvancedPerformanceTracker:
    """Advanced performance tracking and analysis"""

    def __init__(self):
        self.trades: list[TradeResult] = []
        self.ml_predictions: list[dict[str, Any]] = []
        self.sentiment_predictions: list[dict[str, Any]] = []
        self.performance_history: list[PerformanceMetrics] = []
        self.min_trades_for_analysis = 10
        self.hourly_performance: dict[int, dict[str, Any]] = {}
        self.symbol_performance: dict[str, dict[str, Any]] = {}
        self.enhanced_scores: list[dict[str, Any]] = []

    async def record_trade_result(self, trade_result: TradeResult):
        """Record a completed trade result"""
        self.trades.append(trade_result)

        # Update performance metrics
        if len(self.trades) >= self.min_trades_for_analysis:
            await self._update_performance_metrics()

        # Broadcast trade result
        await websocket_manager.broadcast_json(
            {
                "type": "trade_result",
                "data": {
                    "trade_id": trade_result.trade_id,
                    "symbol": trade_result.symbol,
                    "profit_loss": trade_result.profit_loss,
                    "profit_percentage": trade_result.profit_percentage,
                    "success": trade_result.success,
                    "timestamp": trade_result.timestamp.isoformat(),
                },
            }
        )

        logger.info(
            f"Trade recorded: {trade_result.symbol} - P&L: {trade_result.profit_percentage:.2f}%"
        )

    async def record_ml_prediction(self, prediction: dict[str, Any]):
        """Record ML prediction for accuracy tracking"""
        self.ml_predictions.append(prediction)

    async def record_sentiment_prediction(self, prediction: dict[str, Any]):
        """Record sentiment prediction for accuracy tracking"""
        self.sentiment_predictions.append(prediction)

    async def _update_performance_metrics(self):
        """Update comprehensive performance metrics"""
        if len(self.trades) < self.min_trades_for_analysis:
            return

        # Basic metrics
        total_trades = len(self.trades)
        successful_trades = len([t for t in self.trades if t.success])
        win_rate = successful_trades / total_trades if total_trades > 0 else 0

        # Profit/Loss analysis
        profits = [t.profit_percentage for t in self.trades if t.profit_percentage > 0]
        losses = [t.profit_percentage for t in self.trades if t.profit_percentage < 0]

        average_profit = statistics.mean(profits) if profits else 0
        average_loss = statistics.mean(losses) if losses else 0

        # Profit factor
        total_profit = sum(profits) if profits else 0
        total_loss = abs(sum(losses)) if losses else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else float("inf")

        # Risk-adjusted metrics
        returns = [t.profit_percentage for t in self.trades]
        if returns:
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            max_drawdown = self._calculate_max_drawdown(returns)
            risk_adjusted_return = self._calculate_risk_adjusted_return(returns)
        else:
            sharpe_ratio = 0
            max_drawdown = 0
            risk_adjusted_return = 0

        # ML accuracy
        ml_accuracy = self._calculate_ml_accuracy()

        # Sentiment accuracy
        sentiment_accuracy = self._calculate_sentiment_accuracy()

        # Enhanced score accuracy
        enhanced_score_accuracy = self._calculate_enhanced_score_accuracy()

        # Time-based analysis
        best_performing_hours = self._find_best_performing_hours()

        # Symbol-based analysis
        best_performing_symbols = self._find_best_performing_symbols()

        # Create performance metrics
        metrics = PerformanceMetrics(
            total_trades=total_trades,
            successful_trades=successful_trades,
            win_rate=win_rate,
            average_profit=average_profit,
            average_loss=average_loss,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            ml_accuracy=ml_accuracy,
            sentiment_accuracy=sentiment_accuracy,
            enhanced_score_accuracy=enhanced_score_accuracy,
            best_performing_hours=best_performing_hours,
            best_performing_symbols=best_performing_symbols,
            risk_adjusted_return=risk_adjusted_return,
            timestamp=datetime.now(),
        )

        self.performance_history.append(metrics)

        # Broadcast performance update
        await websocket_manager.broadcast_json(
            {
                "type": "performance_update",
                "data": {
                    "total_trades": metrics.total_trades,
                    "win_rate": metrics.win_rate,
                    "profit_factor": metrics.profit_factor,
                    "sharpe_ratio": metrics.sharpe_ratio,
                    "max_drawdown": metrics.max_drawdown,
                    "ml_accuracy": metrics.ml_accuracy,
                    "timestamp": metrics.timestamp.isoformat(),
                },
            }
        )

        # Log performance improvements
        await self._log_performance_insights(metrics)

    def _calculate_sharpe_ratio(self, returns: list[float]) -> float:
        """Calculate Sharpe ratio"""
        if not returns:
            return 0

        mean_return = statistics.mean(returns)
        std_return = statistics.stdev(returns) if len(returns) > 1 else 0

        if std_return == 0:
            return 0

        # Assuming risk-free rate of 0 for simplicity
        return mean_return / std_return

    def _calculate_max_drawdown(self, returns: list[float]) -> float:
        """Calculate maximum drawdown"""
        if not returns:
            return 0

        cumulative = 1.0
        peak = 1.0
        max_dd = 0.0

        for ret in returns:
            cumulative *= 1 + ret / 100
            if cumulative > peak:
                peak = cumulative
            drawdown = (peak - cumulative) / peak
            max_dd = max(max_dd, drawdown)

        return max_dd * 100  # Convert to percentage

    def _calculate_risk_adjusted_return(self, returns: list[float]) -> float:
        """Calculate risk-adjusted return"""
        if not returns:
            return 0

        mean_return = statistics.mean(returns)
        std_return = statistics.stdev(returns) if len(returns) > 1 else 0

        if std_return == 0:
            return mean_return

        return mean_return / std_return

    def _calculate_ml_accuracy(self) -> float:
        """Calculate ML prediction accuracy"""
        if not self.ml_predictions:
            return 0

        correct_predictions = 0
        total_predictions = 0

        for pred in self.ml_predictions:
            if "actual_otimezone.utcome" in pred and "predicted_probability" in pred:
                predicted_success = pred["predicted_probability"] > 0.5
                actual_success = pred["actual_otimezone.utcome"]

                if predicted_success == actual_success:
                    correct_predictions += 1
                total_predictions += 1

        return correct_predictions / total_predictions if total_predictions > 0 else 0

    def _calculate_sentiment_accuracy(self) -> float:
        """Calculate sentiment prediction accuracy"""
        if not self.sentiment_predictions:
            return 0

        correct_predictions = 0
        total_predictions = 0

        for pred in self.sentiment_predictions:
            if "actual_otimezone.utcome" in pred and "sentiment_score" in pred:
                predicted_success = pred["sentiment_score"] > 0
                actual_success = pred["actual_otimezone.utcome"]

                if predicted_success == actual_success:
                    correct_predictions += 1
                total_predictions += 1

        return correct_predictions / total_predictions if total_predictions > 0 else 0

    def _calculate_enhanced_score_accuracy(self) -> float:
        """Calculate enhanced score accuracy"""
        if not self.trades:
            return 0

        correct_predictions = 0
        total_predictions = 0

        for trade in self.trades:
            if hasattr(trade, "enhanced_score"):
                predicted_success = trade.enhanced_score > 75
                actual_success = trade.success

                if predicted_success == actual_success:
                    correct_predictions += 1
                total_predictions += 1

        return correct_predictions / total_predictions if total_predictions > 0 else 0

    def _find_best_performing_hours(self) -> list[int]:
        """Find best performing trading hours"""
        if not self.trades:
            return []

        hourly_performance: dict[int, list[float]] = {}

        for trade in self.trades:
            hour = trade.timestamp.hour
            if hour not in hourly_performance:
                hourly_performance[hour] = []
            hourly_performance[hour].append(trade.profit_percentage)

        # Calculate average performance per hour
        avg_performance: dict[int, float] = {}
        for hour, profits in hourly_performance.items():
            avg_performance[hour] = statistics.mean(profits)

        # Return top 3 hours
        sorted_hours = sorted(avg_performance.items(), key=lambda x: x[1], reverse=True)
        return [hour for hour, _ in sorted_hours[:3]]

    def _find_best_performing_symbols(self) -> list[str]:
        """Find best performing symbols"""
        if not self.trades:
            return []

        symbol_performance: dict[str, list[float]] = {}

        for trade in self.trades:
            symbol = trade.symbol
            if symbol not in symbol_performance:
                symbol_performance[symbol] = []
            symbol_performance[symbol].append(trade.profit_percentage)

        # Calculate average performance per symbol
        avg_performance: dict[str, float] = {}
        for symbol, profits in symbol_performance.items():
            avg_performance[symbol] = statistics.mean(profits)

        # Return top 5 symbols
        sorted_symbols = sorted(avg_performance.items(), key=lambda x: x[1], reverse=True)
        return [symbol for symbol, _ in sorted_symbols[:5]]

    async def _log_performance_insights(self, metrics: PerformanceMetrics):
        """Log performance insights and recommendations"""
        logger.info("=== PERFORMANCE INSIGHTS ===")
        logger.info(f"Win Rate: {metrics.win_rate:.2%}")
        logger.info(f"Profit Factor: {metrics.profit_factor:.2f}")
        logger.info(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        logger.info(f"ML Accuracy: {metrics.ml_accuracy:.2%}")
        logger.info(f"Sentiment Accuracy: {metrics.sentiment_accuracy:.2%}")
        logger.info(f"Best Hours: {metrics.best_performing_hours}")
        logger.info(f"Best Symbols: {metrics.best_performing_symbols}")

        # Performance recommendations
        if metrics.win_rate < 0.7:
            logger.warning("âš ï¸ Win rate below 70% - Consider tightening signal criteria")

        if metrics.profit_factor < 1.5:
            logger.warning("âš ï¸ Profit factor below 1.5 - Review risk management")

        if metrics.ml_accuracy < 0.6:
            logger.warning("âš ï¸ ML accuracy below 60% - Retrain models")

        if metrics.sentiment_accuracy < 0.5:
            logger.warning("âš ï¸ Sentiment accuracy below 50% - Review sentiment sources")

    def get_current_performance(self) -> PerformanceMetrics | None:
        """Get current performance metrics"""
        if self.performance_history:
            return self.performance_history[-1]
        return None

    def get_performance_trend(self, days: int = 30) -> dict[str, Any]:
        """Get performance trend over specified days"""
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_trades = [t for t in self.trades if t.timestamp > cutoff_date]

        if not recent_trades:
            return {"error": "No recent trades found"}

        recent_win_rate = len([t for t in recent_trades if t.success]) / len(recent_trades)
        recent_avg_profit = statistics.mean([t.profit_percentage for t in recent_trades])

        return {
            "period_days": days,
            "recent_trades": len(recent_trades),
            "recent_win_rate": recent_win_rate,
            "recent_avg_profit": recent_avg_profit,
            "trend": (
                "improving"
                if recent_win_rate > 0.75
                else "stable" if recent_win_rate > 0.65 else "declining"
            ),
        }


# Global performance tracker instance
performance_tracker = AdvancedPerformanceTracker()


