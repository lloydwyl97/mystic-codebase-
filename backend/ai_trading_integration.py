import asyncio
import logging
from datetime import datetime

from ai_auto_learner import AIAutoLearner
from ai_mode_controller import AITradingController
from backup_utils import snapshot
from chart_generator import plot_performance_over_time
from daily_summary import send_daily_summary
from notifier import send_performance_alert, send_trade_alert
from simulation_logger import SimulationLogger
from stagnation_detector import check_performance_plateau, detect_stagnation
from strategy_tagger import get_strategy_confidence, tag_trade


class AITradingIntegration:
    def __init__(self):
        self.controller = AITradingController()
        self.logger = SimulationLogger()
        self.learner = AIAutoLearner()
        self.is_running = False

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger_instance = logging.getLogger(__name__)

    async def start_trading_loop(self):
        """Main trading loop that runs continuously"""
        self.is_running = True
        self.logger_instance.info("Starting AI Trading Integration...")

        while self.is_running:
            try:
                # Check for stagnation
                detect_stagnation()

                # Evaluate and adapt AI strategy
                self.learner.evaluate_and_adapt()

                # Simulate a trade (replace with real market data)
                await self.simulate_trade_cycle()

                # Generate performance charts
                plot_performance_over_time()

                # Wait before next cycle
                await asyncio.sleep(60)  # 1 minute intervals

            except Exception as e:
                self.logger_instance.error(f"Error in trading loop: {e}")
                await asyncio.sleep(30)

    async def simulate_trade_cycle(self):
        """Simulate a complete trade cycle"""
        # Simulate market data
        symbol = "ETHUSDT"
        current_price = 1824.55  # Replace with real price feed
        recent_prices = [1820, 1822, 1825, 1823, 1824.55]

        # Tag the trade strategy
        strategy = tag_trade(current_price, recent_prices)

        # Get mystic signals (placeholder)
        mystic_signals = {
            "tesla_369": 0.75,
            "faerie_star": 0.82,
            "lagos_alignment": 0.68,
        }

        # Calculate confidence
        pattern = analyze_trade_pattern(recent_prices)
        confidence = get_strategy_confidence(pattern, mystic_signals)

        # Simulate profit
        simulated_profit = 5.30 if confidence > 0.7 else -2.15

        # Check if we should execute
        if self.controller.should_execute_trade(simulated_profit):
            # Real trade execution would go here
            self.logger_instance.info(f"Executing trade: {symbol} BUY @ ${current_price}")
            send_trade_alert(symbol, "BUY", current_price, simulated_profit)

            # Send performance alert for high-confidence trades
            if confidence > 0.8:
                send_performance_alert(simulated_profit, 1)  # avg_profit, total_trades
        else:
            # Log simulated trade
            self.logger.log_trade(
                symbol=symbol,
                action="BUY",
                price=current_price,
                confidence=confidence,
                simulated_profit=simulated_profit,
                strategy=strategy,
                mystic_signals=str(mystic_signals),
            )
            self.logger_instance.info(f"Simulated trade: {symbol} BUY @ ${current_price}")

    def stop_trading(self):
        """Stop the trading loop"""
        self.is_running = False
        self.logger_instance.info(f"AI Trading Integration stopped at {datetime.now()}")

    async def run_daily_tasks(self):
        """Run daily maintenance tasks"""
        while self.is_running:
            try:
                # Send daily summary
                send_daily_summary()

                # Backup files
                snapshot()

                # Check performance plateau
                check_performance_plateau()

                # Wait 24 hours
                await asyncio.sleep(86400)

            except Exception as e:
                self.logger_instance.error(f"Error in daily tasks: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour on error


def analyze_trade_pattern(prices: list[float]) -> dict[str, str]:
    """Analyze trade pattern and return strategy insights"""
    if len(prices) < 3:
        return {"pattern": "insufficient_data"}

    current_price = prices[-1]
    prev_price = prices[-2]
    price_change = ((current_price - prev_price) / prev_price) * 100

    pattern = {
        "trend": ("up" if price_change > 0 else "down" if price_change < 0 else "sideways"),
        "strength": "strong" if abs(price_change) > 2 else "weak",
        "volatility": (
            "high" if len(prices) > 5 and max(prices) - min(prices) > current_price * 0.1 else "low"
        ),
    }

    return pattern


