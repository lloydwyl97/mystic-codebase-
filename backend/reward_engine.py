# reward_engine.py
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from models import Strategy, Trade, StrategyPerformance
from db_logger import get_session
import datetime
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

DB_URL = "sqlite:////data/trading_memory.db"
engine = create_engine(DB_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)


def evaluate_strategies(min_trades: int = 5, days: int = 7) -> Dict[str, Any]:
    """
    Evaluate all strategies and update their performance metrics

    Args:
        min_trades: Minimum number of trades required for evaluation
        days: Number of days to look back for evaluation

    Returns:
        Dict containing evaluation results
    """
    session = get_session()
    try:
        strategies = session.query(Strategy).filter_by(is_active=True).all()
        now = datetime.datetime.timezone.utcnow()
        cutoff_date = now - datetime.timedelta(days=days)

        evaluation_results = {
            "total_strategies": len(strategies),
            "evaluated_strategies": 0,
            "updated_strategies": 0,
            "strategy_details": [],
        }

        for strat in strategies:
            # Get recent trades for this strategy
            trades = (
                session.query(Trade)
                .filter(Trade.strategy_id == strat.id)
                .filter(Trade.timestamp >= cutoff_date)
                .filter(Trade.exit_price.isnot(None))  # Only completed trades
                .all()
            )

            if len(trades) >= min_trades:
                evaluation_results["evaluated_strategies"] += 1

                # Calculate performance metrics
                total_profit = sum(t.profit for t in trades if t.profit is not None)
                win_count = sum(1 for t in trades if t.success)
                avg_profit = total_profit / len(trades) if trades else 0.0
                win_rate = win_count / len(trades) if trades else 0.0

                # Calculate additional metrics
                profits = [t.profit for t in trades if t.profit is not None]
                max_profit = max(profits) if profits else 0.0
                max_loss = min(profits) if profits else 0.0

                # Update strategy stats
                strat.win_rate = round(win_rate, 4)
                strat.avg_profit = round(avg_profit, 4)
                strat.trades_executed = len(trades)
                strat.total_profit = round(total_profit, 4)
                strat.updated_at = now

                # Create performance record
                performance = StrategyPerformance(
                    strategy_id=strat.id,
                    strategy_name=strat.name,
                    date=now,
                    win_rate=win_rate,
                    avg_profit=avg_profit,
                    total_trades=len(trades),
                    total_profit=total_profit,
                    max_drawdown=abs(max_loss) if max_loss < 0 else 0.0,
                    sharpe_ratio=(calculate_sharpe_ratio(profits) if profits else 0.0),
                )
                session.add(performance)

                evaluation_results["updated_strategies"] += 1

                strategy_detail = {
                    "id": strat.id,
                    "name": strat.name,
                    "win_rate": win_rate,
                    "avg_profit": avg_profit,
                    "total_profit": total_profit,
                    "trades_count": len(trades),
                    "max_profit": max_profit,
                    "max_loss": max_loss,
                }
                evaluation_results["strategy_details"].append(strategy_detail)

                logger.info(
                    f"Updated: {strat.name} | Win: {win_rate:.2%} | Profit: {avg_profit:.2f} | Total: {total_profit:.2f}"
                )
            else:
                logger.debug(
                    f"Strategy {strat.name} has only {len(trades)} trades, skipping evaluation"
                )

        session.commit()
        logger.info(
            f"Strategy evaluation completed: {evaluation_results['updated_strategies']}/{evaluation_results['total_strategies']} strategies updated"
        )
        return evaluation_results

    except Exception as e:
        session.rollback()
        logger.error(f"Strategy evaluation failed: {e}")
        return {"error": str(e)}
    finally:
        session.close()


def calculate_sharpe_ratio(profits: List[float], risk_free_rate: float = 0.02) -> float:
    """
    Calculate Sharpe ratio for a list of profits

    Args:
        profits: List of profit values
        risk_free_rate: Risk-free rate (default 2% annual)

    Returns:
        float: Sharpe ratio
    """
    if not profits or len(profits) < 2:
        return 0.0

    import numpy as np

    returns = np.array(profits)
    avg_return = np.mean(returns)
    std_return = np.std(returns)

    if std_return == 0:
        return 0.0

    # Annualize (assuming daily returns)
    sharpe = (avg_return - risk_free_rate / 365) / std_return * np.sqrt(365)
    return round(sharpe, 4)


def get_top_performers(top_n: int = 5, min_trades: int = 5) -> List[Dict[str, Any]]:
    """
    Get top performing strategies

    Args:
        top_n: Number of top strategies to return
        min_trades: Minimum trades required

    Returns:
        List of top performing strategies
    """
    session = get_session()
    try:
        strategies = (
            session.query(Strategy)
            .filter(Strategy.trades_executed >= min_trades)
            .filter(Strategy.is_active)
            .order_by(Strategy.win_rate.desc(), Strategy.avg_profit.desc())
            .limit(top_n)
            .all()
        )

        return [
            {
                "id": strat.id,
                "name": strat.name,
                "win_rate": strat.win_rate,
                "avg_profit": strat.avg_profit,
                "total_profit": strat.total_profit,
                "trades_executed": strat.trades_executed,
            }
            for strat in strategies
        ]
    finally:
        session.close()


def get_poor_performers(min_trades: int = 5, max_win_rate: float = 0.4) -> List[Dict[str, Any]]:
    """
    Get poorly performing strategies that might need to be deactivated

    Args:
        min_trades: Minimum trades required
        max_win_rate: Maximum win rate to be considered poor

    Returns:
        List of poor performing strategies
    """
    session = get_session()
    try:
        strategies = (
            session.query(Strategy)
            .filter(Strategy.trades_executed >= min_trades)
            .filter(Strategy.win_rate <= max_win_rate)
            .filter(Strategy.is_active)
            .order_by(Strategy.win_rate.asc())
            .all()
        )

        return [
            {
                "id": strat.id,
                "name": strat.name,
                "win_rate": strat.win_rate,
                "avg_profit": strat.avg_profit,
                "total_profit": strat.total_profit,
                "trades_executed": strat.trades_executed,
            }
            for strat in strategies
        ]
    finally:
        session.close()


def deactivate_strategy(strategy_id: int) -> bool:
    """
    Deactivate a strategy (mark as inactive)

    Args:
        strategy_id: ID of strategy to deactivate

    Returns:
        bool: True if successful
    """
    session = get_session()
    try:
        strategy = session.query(Strategy).filter_by(id=strategy_id).first()
        if strategy:
            strategy.is_active = False
            strategy.updated_at = datetime.datetime.timezone.utcnow()
            session.commit()
            logger.info(f"Deactivated strategy: {strategy.name}")
            return True
        return False
    except Exception as e:
        session.rollback()
        logger.error(f"Failed to deactivate strategy {strategy_id}: {e}")
        return False
    finally:
        session.close()


def get_strategy_performance_history(strategy_id: int, days: int = 30) -> List[Dict[str, Any]]:
    """
    Get performance history for a strategy

    Args:
        strategy_id: Strategy ID
        days: Number of days to look back

    Returns:
        List of performance records
    """
    session = get_session()
    try:
        cutoff_date = datetime.datetime.timezone.utcnow() - datetime.timedelta(days=days)

        performances = (
            session.query(StrategyPerformance)
            .filter(StrategyPerformance.strategy_id == strategy_id)
            .filter(StrategyPerformance.date >= cutoff_date)
            .order_by(StrategyPerformance.date.desc())
            .all()
        )

        return [
            {
                "date": perf.date.isoformat() if perf.date else None,
                "win_rate": perf.win_rate,
                "avg_profit": perf.avg_profit,
                "total_trades": perf.total_trades,
                "total_profit": perf.total_profit,
                "max_drawdown": perf.max_drawdown,
                "sharpe_ratio": perf.sharpe_ratio,
            }
            for perf in performances
        ]
    finally:
        session.close()


def run_daily_evaluation() -> Dict[str, Any]:
    """
    Run daily strategy evaluation (can be called by scheduler)

    Returns:
        Dict with evaluation results
    """
    logger.info("Starting daily strategy evaluation...")
    results = evaluate_strategies(min_trades=3, days=1)

    # Get top and poor performers
    top_performers = get_top_performers(top_n=3, min_trades=5)
    poor_performers = get_poor_performers(min_trades=5, max_win_rate=0.4)

    results["top_performers"] = top_performers
    results["poor_performers"] = poor_performers

    logger.info(
        f"Daily evaluation completed. Top performers: {len(top_performers)}, Poor performers: {len(poor_performers)}"
    )
    return results


