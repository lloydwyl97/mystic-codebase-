#!/usr/bin/env python3
"""
Binance US Autobuy Report Generator
Comprehensive reporting for SOLUSDT, BTCUSDT, ETHUSDT, AVAXUSDT autobuy system
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

from backend.endpoints.autobuy.autobuy_config import get_config
from binance_us_autobuy import autobuy_system

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TradeReport:
    """Individual trade report"""

    trade_id: str
    symbol: str
    timestamp: str
    amount_usd: float
    price: float
    quantity: float
    confidence: float
    signals: List[str]
    status: str
    profit_loss: Optional[float] = None
    execution_time: Optional[str] = None


@dataclass
class PairReport:
    """Trading pair performance report"""

    symbol: str
    name: str
    total_trades: int
    successful_trades: int
    failed_trades: int
    total_volume: float
    total_profit_loss: float
    success_rate: float
    avg_trade_amount: float
    avg_confidence: float
    best_trade: Optional[TradeReport]
    worst_trade: Optional[TradeReport]
    recent_signals: int
    active_trades: int


@dataclass
class SystemReport:
    """Complete system report"""

    report_time: str
    system_uptime: str
    total_trades: int
    successful_trades: int
    failed_trades: int
    total_volume: float
    total_profit_loss: float
    success_rate: float
    active_trades: int
    trading_enabled: bool
    emergency_stop: bool
    pairs_performance: Dict[str, PairReport]
    recent_trades: List[TradeReport]
    system_health: Dict[str, Any]
    recommendations: List[str]


class AutobuyReporter:
    """Comprehensive reporter for the autobuy system"""

    def __init__(self):
        self.config = get_config()
        self.reports_dir = "reports"
        os.makedirs(self.reports_dir, exist_ok=True)

    def generate_trade_report(self, trade_data: Dict[str, Any]) -> TradeReport:
        """Generate a trade report from trade data"""
        return TradeReport(
            trade_id=trade_data.get("order_id", "unknown"),
            symbol=trade_data.get("symbol", "unknown"),
            timestamp=trade_data.get("timestamp", ""),
            amount_usd=trade_data.get("amount_usd", 0.0),
            price=trade_data.get("price", 0.0),
            quantity=trade_data.get("quantity", 0.0),
            confidence=trade_data.get("confidence", 0.0),
            signals=trade_data.get("signals", []),
            status=trade_data.get("status", "unknown"),
            profit_loss=trade_data.get("profit_loss"),
            execution_time=trade_data.get("execution_time"),
        )

    def generate_pair_report(self, symbol: str) -> PairReport:
        """Generate a report for a specific trading pair"""
        # Get all trades for this symbol
        symbol_trades = [
            trade for trade in autobuy_system.trade_history if trade.get("symbol") == symbol
        ]

        if not symbol_trades:
            return PairReport(
                symbol=symbol,
                name=(
                    self.config.get_pair_config(symbol).name
                    if self.config.get_pair_config(symbol)
                    else symbol
                ),
                total_trades=0,
                successful_trades=0,
                failed_trades=0,
                total_volume=0.0,
                total_profit_loss=0.0,
                success_rate=0.0,
                avg_trade_amount=0.0,
                avg_confidence=0.0,
                best_trade=None,
                worst_trade=None,
                recent_signals=len(autobuy_system.signal_history.get(symbol, [])),
                active_trades=(1 if symbol in autobuy_system.active_trades else 0),
            )

        # Calculate statistics
        total_trades = len(symbol_trades)
        successful_trades = len([t for t in symbol_trades if t.get("status") == "executed"])
        failed_trades = total_trades - successful_trades
        total_volume = sum(t.get("amount_usd", 0) for t in symbol_trades)
        total_profit_loss = sum(t.get("profit_loss", 0) for t in symbol_trades)
        success_rate = (successful_trades / total_trades * 100) if total_trades > 0 else 0
        avg_trade_amount = total_volume / total_trades if total_trades > 0 else 0
        avg_confidence = (
            sum(t.get("confidence", 0) for t in symbol_trades) / total_trades
            if total_trades > 0
            else 0
        )

        # Find best and worst trades
        profitable_trades = [t for t in symbol_trades if t.get("profit_loss", 0) > 0]
        loss_trades = [t for t in symbol_trades if t.get("profit_loss", 0) < 0]

        best_trade = None
        if profitable_trades:
            best_trade_data = max(profitable_trades, key=lambda x: x.get("profit_loss", 0))
            best_trade = self.generate_trade_report(best_trade_data)

        worst_trade = None
        if loss_trades:
            worst_trade_data = min(loss_trades, key=lambda x: x.get("profit_loss", 0))
            worst_trade = self.generate_trade_report(worst_trade_data)

        return PairReport(
            symbol=symbol,
            name=(
                self.config.get_pair_config(symbol).name
                if self.config.get_pair_config(symbol)
                else symbol
            ),
            total_trades=total_trades,
            successful_trades=successful_trades,
            failed_trades=failed_trades,
            total_volume=total_volume,
            total_profit_loss=total_profit_loss,
            success_rate=success_rate,
            avg_trade_amount=avg_trade_amount,
            avg_confidence=avg_confidence,
            best_trade=best_trade,
            worst_trade=worst_trade,
            recent_signals=len(autobuy_system.signal_history.get(symbol, [])),
            active_trades=1 if symbol in autobuy_system.active_trades else 0,
        )

    def generate_system_report(self) -> SystemReport:
        """Generate a complete system report"""
        # Calculate overall statistics
        total_trades = autobuy_system.total_trades
        successful_trades = autobuy_system.successful_trades
        failed_trades = autobuy_system.failed_trades
        total_volume = autobuy_system.total_volume
        active_trades = len(autobuy_system.active_trades)

        # Calculate total profit/loss
        total_profit_loss = sum(
            trade.get("profit_loss", 0) for trade in autobuy_system.trade_history
        )

        success_rate = (successful_trades / total_trades * 100) if total_trades > 0 else 0

        # Generate pair reports
        pairs_performance = {}
        for symbol in self.config.get_enabled_pairs():
            pairs_performance[symbol] = self.generate_pair_report(symbol)

        # Get recent trades
        recent_trades = []
        for trade_data in autobuy_system.trade_history[-20:]:  # Last 20 trades
            recent_trades.append(self.generate_trade_report(trade_data))

        # System health assessment
        system_health = self.assess_system_health()

        # Generate recommendations
        recommendations = self.generate_recommendations()

        return SystemReport(
            report_time=datetime.now(timezone.utc).isoformat(),
            system_uptime="N/A",  # Would need to track system start time
            total_trades=total_trades,
            successful_trades=successful_trades,
            failed_trades=failed_trades,
            total_volume=total_volume,
            total_profit_loss=total_profit_loss,
            success_rate=success_rate,
            active_trades=active_trades,
            trading_enabled=self.config.trading_enabled,
            emergency_stop=self.config.emergency_stop,
            pairs_performance=pairs_performance,
            recent_trades=recent_trades,
            system_health=system_health,
            recommendations=recommendations,
        )

    def assess_system_health(self) -> Dict[str, Any]:
        """Assess overall system health"""
        health_score = 100

        # Check success rate
        if autobuy_system.total_trades > 0:
            success_rate = (autobuy_system.successful_trades / autobuy_system.total_trades) * 100
            if success_rate < 70:
                health_score -= 20
            elif success_rate < 85:
                health_score -= 10

        # Check for too many failed trades
        if autobuy_system.failed_trades > autobuy_system.successful_trades:
            health_score -= 30

        # Check for emergency stop
        if self.config.emergency_stop:
            health_score -= 50

        # Check trading enabled
        if not self.config.trading_enabled:
            health_score -= 20

        # Determine health status
        if health_score >= 80:
            status = "Excellent"
        elif health_score >= 60:
            status = "Good"
        elif health_score >= 40:
            status = "Fair"
        else:
            status = "Poor"

        return {
            "score": health_score,
            "status": status,
            "issues": self.identify_issues(),
            "last_update": datetime.now(timezone.utc).isoformat(),
        }

    def identify_issues(self) -> List[str]:
        """Identify potential issues with the system"""
        issues = []

        if self.config.emergency_stop:
            issues.append("Emergency stop is active")

        if not self.config.trading_enabled:
            issues.append("Trading is disabled")

        if autobuy_system.failed_trades > autobuy_system.successful_trades:
            issues.append("High failure rate detected")

        if autobuy_system.total_trades > 0:
            success_rate = (autobuy_system.successful_trades / autobuy_system.total_trades) * 100
            if success_rate < 70:
                issues.append(f"Low success rate: {success_rate:.1f}%")

        # Check for pairs with no recent activity
        for symbol in self.config.get_enabled_pairs():
            if symbol not in autobuy_system.signal_history:
                issues.append(f"No signals for {symbol}")

        return issues

    def generate_recommendations(self) -> List[str]:
        """Generate recommendations for system improvement"""
        recommendations = []

        # Success rate recommendations
        if autobuy_system.total_trades > 0:
            success_rate = (autobuy_system.successful_trades / autobuy_system.total_trades) * 100
            if success_rate < 70:
                recommendations.append("Consider increasing minimum confidence threshold")
                recommendations.append("Review signal generation parameters")
            elif success_rate > 90:
                recommendations.append("Consider increasing trade frequency")

        # Volume recommendations
        if autobuy_system.total_volume < 1000:
            recommendations.append("Consider increasing trade amounts for better returns")

        # Pair-specific recommendations
        for symbol in self.config.get_enabled_pairs():
            pair_report = self.generate_pair_report(symbol)
            if pair_report.total_trades == 0:
                recommendations.append(f"Monitor {symbol} for trading opportunities")
            elif pair_report.success_rate < 60:
                recommendations.append(f"Review {symbol} trading parameters")

        # General recommendations
        if len(autobuy_system.active_trades) == 0:
            recommendations.append("No active trades - system may be too conservative")

        if len(autobuy_system.active_trades) >= self.config.risk_config.max_concurrent_trades:
            recommendations.append("Maximum concurrent trades reached - consider increasing limit")

        return recommendations

    def save_report(self, report: SystemReport, filename: Optional[str] = None) -> str:
        """Save report to file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"autobuy_report_{timestamp}.json"

        filepath = os.path.join(self.reports_dir, filename)

        with open(filepath, "w") as f:
            json.dump(asdict(report), f, indent=2, default=str)

        logger.info(f"ðŸ“„ Report saved to {filepath}")
        return filepath

    def print_report(self, report: SystemReport):
        """Print a formatted report to console"""
        print("\n" + "=" * 80)
        print("ðŸš€ BINANCE US AUTOBUY SYSTEM REPORT")
        print("=" * 80)

        # Overall Summary
        print("\nðŸ“Š OVERALL SUMMARY:")
        print(f"   Total Trades: {report.total_trades}")
        print(f"   Successful: {report.successful_trades}")
        print(f"   Failed: {report.failed_trades}")
        print(f"   Success Rate: {report.success_rate:.1f}%")
        print(f"   Total Volume: ${report.total_volume:,.2f}")
        print(f"   Total P&L: ${report.total_profit_loss:,.2f}")
        print(f"   Active Trades: {report.active_trades}")

        # System Health
        health = report.system_health
        print("\nðŸ¥ SYSTEM HEALTH:")
        print(f"   Status: {health['status']} ({health['score']}/100)")
        if health["issues"]:
            print(f"   Issues: {', '.join(health['issues'])}")

        # Pair Performance
        print("\nðŸ“ˆ PAIR PERFORMANCE:")
        for symbol, pair_report in report.pairs_performance.items():
            status = (
                "âœ…"
                if pair_report.success_rate >= 70
                else "âš ï¸" if pair_report.success_rate >= 50 else "âŒ"
            )
            print(f"   {status} {pair_report.name} ({symbol}):")
            print(
                f"      Trades: {pair_report.total_trades}, Success: {pair_report.success_rate:.1f}%"
            )
            print(
                f"      Volume: ${pair_report.total_volume:,.2f}, P&L: ${pair_report.total_profit_loss:,.2f}"
            )

        # Recent Trades
        if report.recent_trades:
            print("\nðŸ•’ RECENT TRADES:")
            for trade in report.recent_trades[-5:]:  # Last 5 trades
                status = "âœ…" if trade.status == "executed" else "âŒ"
                print(
                    f"   {status} {trade.symbol}: ${trade.amount_usd} @ ${trade.price} ({trade.confidence:.1f}%)"
                )

        # Recommendations
        if report.recommendations:
            print("\nðŸ’¡ RECOMMENDATIONS:")
            for rec in report.recommendations:
                print(f"   â€¢ {rec}")

        print("\n" + "=" * 80)


def generate_and_save_report() -> str:
    """Generate and save a complete report"""
    reporter = AutobuyReporter()
    report = reporter.generate_system_report()

    # Print to console
    reporter.print_report(report)

    # Save to file
    filepath = reporter.save_report(report)

    return filepath


if __name__ == "__main__":
    print("ðŸ“Š Generating Binance US Autobuy Report...")
    report_file = generate_and_save_report()
    print(f"âœ… Report generated and saved to: {report_file}")


