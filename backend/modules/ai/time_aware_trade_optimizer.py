"""
Time-Aware Trade Optimizer for Mystic AI Trading Platform
Analyzes historical price data to identify optimal trading windows and timing patterns.
"""

import logging
import numpy as np
from typing import Dict, List, Any
from datetime import datetime, timezone
import sys
import os

# Add backend to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.modules.ai.persistent_cache import PersistentCache

logger = logging.getLogger(__name__)


class TimeAwareTradeOptimizer:
    def __init__(self):
        """Initialize time-aware trade optimizer with analysis parameters"""
        self.cache = PersistentCache()

        # Time analysis parameters
        self.time_windows = {
            'intraday': 24,  # hours
            'weekly': 168,    # hours (7 days)
            'monthly': 720    # hours (30 days)
        }

        # Volatility analysis parameters
        self.volatility_threshold = 0.02  # 2% volatility threshold
        self.volume_surge_threshold = 1.5  # 50% volume increase
        self.min_data_points = 100

        # Time optimization parameters
        self.hour_granularity = 1  # 1-hour intervals
        self.confidence_threshold = 0.6
        self.min_trades_for_pattern = 5

        # Trading session definitions
        self.trading_sessions = {
            'asian': (0, 8),    # 00:00-08:00 UTC
            'london': (8, 16),  # 08:00-16:00 UTC
            'new_york': (13, 21)  # 13:00-21:00 UTC
        }

        logger.info("âœ… TimeAwareTradeOptimizer initialized")

    def _get_historical_data(self, symbol: str, hours: int = 168) -> List[Dict[str, Any]]:
        """Get historical price data from cache"""
        try:
            # Get price history from cache
            price_history = self.cache.get_price_history('aggregated', symbol, limit=hours)

            if not price_history or len(price_history) < self.min_data_points:
                logger.warning(f"Insufficient historical data for {symbol}: {len(price_history) if price_history else 0} records")
                return []

            # Convert to structured format
            historical_data = []
            for data in price_history:
                try:
                    timestamp = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
                    price = float(data['price'])
                    volume = float(data.get('volume', 0))

                    historical_data.append({
                        'timestamp': timestamp,
                        'price': price,
                        'volume': volume
                    })
                except (ValueError, KeyError) as e:
                    logger.warning(f"Skipping invalid data point: {e}")
                    continue

            # Sort by timestamp
            historical_data.sort(key=lambda x: x['timestamp'])

            return historical_data

        except Exception as e:
            logger.error(f"Failed to get historical data for {symbol}: {e}")
            return []

    def _calculate_intraday_volatility(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate intraday volatility patterns"""
        try:
            if len(data) < 24:
                return {"hourly_volatility": {}, "peak_hours": []}

            # Group data by hour
            hourly_data = {}
            for entry in data:
                hour = entry['timestamp'].hour
                if hour not in hourly_data:
                    hourly_data[hour] = []
                hourly_data[hour].append(entry['price'])

            # Calculate volatility for each hour
            hourly_volatility = {}
            peak_hours = []

            for hour, prices in hourly_data.items():
                if len(prices) < 2:
                    continue

                # Calculate price changes
                price_changes = []
                for i in range(1, len(prices)):
                    change = abs(prices[i] - prices[i-1]) / prices[i-1]
                    price_changes.append(change)

                # Calculate average volatility
                if price_changes:
                    avg_volatility = np.mean(price_changes)
                    hourly_volatility[hour] = avg_volatility

                    # Check if this hour has high volatility
                    if avg_volatility > self.volatility_threshold:
                        peak_hours.append(hour)

            return {
                "hourly_volatility": hourly_volatility,
                "peak_hours": sorted(peak_hours)
            }

        except Exception as e:
            logger.error(f"Failed to calculate intraday volatility: {e}")
            return {"hourly_volatility": {}, "peak_hours": []}

    def _identify_volume_surges(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify volume surge patterns"""
        try:
            if len(data) < 24:
                return []

            # Group data by hour
            hourly_volumes = {}
            for entry in data:
                hour = entry['timestamp'].hour
                if hour not in hourly_volumes:
                    hourly_volumes[hour] = []
                hourly_volumes[hour].append(entry['volume'])

            # Calculate average volume for each hour
            avg_volumes = {}
            for hour, volumes in hourly_volumes.items():
                if volumes:
                    avg_volumes[hour] = np.mean(volumes)

            # Calculate overall average volume
            all_volumes = [v for volumes in hourly_volumes.values() for v in volumes]
            overall_avg = np.mean(all_volumes) if all_volumes else 0

            # Identify surge hours
            surge_hours = []
            for hour, avg_volume in avg_volumes.items():
                if avg_volume > (overall_avg * self.volume_surge_threshold):
                    surge_hours.append({
                        'hour': hour,
                        'avg_volume': avg_volume,
                        'surge_factor': avg_volume / overall_avg if overall_avg > 0 else 1.0
                    })

            return sorted(surge_hours, key=lambda x: x['surge_factor'], reverse=True)

        except Exception as e:
            logger.error(f"Failed to identify volume surges: {e}")
            return []

    def _analyze_historical_trades(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze historical trade patterns"""
        try:
            if len(data) < self.min_trades_for_pattern:
                return {"entry_times": [], "exit_times": [], "successful_hours": []}

            # Mock historical trade analysis (in real implementation, would analyze actual trades)
            # For now, we'll simulate based on price movements
            entry_times = []
            exit_times = []
            successful_hours = []

            # Analyze price movements to identify potential entry/exit points
            for i in range(1, len(data)):
                price_change = (data[i]['price'] - data[i-1]['price']) / data[i-1]['price']
                hour = data[i]['timestamp'].hour

                # Simulate entry signals (price increases)
                if price_change > 0.01:  # 1% increase
                    entry_times.append(hour)
                    successful_hours.append(hour)

                # Simulate exit signals (price decreases)
                if price_change < -0.01:  # 1% decrease
                    exit_times.append(hour)
                    successful_hours.append(hour)

            # Count occurrences
            entry_counts = {}
            exit_counts = {}
            success_counts = {}

            for hour in entry_times:
                entry_counts[hour] = entry_counts.get(hour, 0) + 1

            for hour in exit_times:
                exit_counts[hour] = exit_counts.get(hour, 0) + 1

            for hour in successful_hours:
                success_counts[hour] = success_counts.get(hour, 0) + 1

            # Get most common hours
            best_entry_hours = sorted(entry_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            best_exit_hours = sorted(exit_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            most_successful_hours = sorted(success_counts.items(), key=lambda x: x[1], reverse=True)[:5]

            return {
                "entry_times": [hour for hour, count in best_entry_hours],
                "exit_times": [hour for hour, count in best_exit_hours],
                "successful_hours": [hour for hour, count in most_successful_hours]
            }

        except Exception as e:
            logger.error(f"Failed to analyze historical trades: {e}")
            return {"entry_times": [], "exit_times": [], "successful_hours": []}

    def _calculate_optimal_windows(self, volatility_analysis: Dict[str, Any],
                                 volume_surges: List[Dict[str, Any]],
                                 trade_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate optimal trading windows"""
        try:
            optimal_windows = {
                'best_entry_hours': [],
                'best_exit_hours': [],
                'avoid_hours': [],
                'confidence': 0.0
            }

            # Combine signals from different analyses
            entry_signals = {}
            exit_signals = {}

            # Volatility-based signals
            peak_hours = volatility_analysis.get('peak_hours', [])
            for hour in peak_hours:
                entry_signals[hour] = entry_signals.get(hour, 0) + 1
                exit_signals[hour] = exit_signals.get(hour, 0) + 1

            # Volume surge signals
            surge_hours = [surge['hour'] for surge in volume_surges]
            for hour in surge_hours:
                entry_signals[hour] = entry_signals.get(hour, 0) + 2  # Higher weight

            # Historical trade signals
            successful_hours = trade_analysis.get('successful_hours', [])
            for hour in successful_hours:
                entry_signals[hour] = entry_signals.get(hour, 0) + 1
                exit_signals[hour] = exit_signals.get(hour, 0) + 1

            # Determine best entry hours
            if entry_signals:
                sorted_entries = sorted(entry_signals.items(), key=lambda x: x[1], reverse=True)
                optimal_windows['best_entry_hours'] = [hour for hour, score in sorted_entries[:3]]

            # Determine best exit hours
            if exit_signals:
                sorted_exits = sorted(exit_signals.items(), key=lambda x: x[1], reverse=True)
                optimal_windows['best_exit_hours'] = [hour for hour, score in sorted_exits[:3]]

            # Determine hours to avoid (low activity)
            all_hours = set(range(24))
            active_hours = set(entry_signals.keys()) | set(exit_signals.keys())
            optimal_windows['avoid_hours'] = list(all_hours - active_hours)

            # Calculate confidence
            total_signals = len(entry_signals) + len(exit_signals)
            if total_signals > 0:
                optimal_windows['confidence'] = min(1.0, total_signals / 20.0)  # Normalize

            return optimal_windows

        except Exception as e:
            logger.error(f"Failed to calculate optimal windows: {e}")
            return {
                'best_entry_hours': [],
                'best_exit_hours': [],
                'avoid_hours': [],
                'confidence': 0.0
            }

    def _format_time_decision(self, symbol: str, optimal_windows: Dict[str, Any]) -> Dict[str, Any]:
        """Format time decision for output"""
        try:
            # Convert hours to UTC time strings
            best_entry_times = []
            best_exit_times = []

            for hour in optimal_windows['best_entry_hours']:
                time_str = f"{hour:02d}:00 UTC"
                best_entry_times.append(time_str)

            for hour in optimal_windows['best_exit_hours']:
                time_str = f"{hour:02d}:00 UTC"
                best_exit_times.append(time_str)

            # Select primary times
            primary_entry = best_entry_times[0] if best_entry_times else "14:00 UTC"
            primary_exit = best_exit_times[0] if best_exit_times else "21:00 UTC"

            return {
                'symbol': symbol,
                'best_entry': primary_entry,
                'best_exit': primary_exit,
                'confidence': optimal_windows['confidence'],
                'alternative_entries': best_entry_times[1:] if len(best_entry_times) > 1 else [],
                'alternative_exits': best_exit_times[1:] if len(best_exit_times) > 1 else [],
                'avoid_hours': optimal_windows['avoid_hours'],
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to format time decision: {e}")
            return {
                'symbol': symbol,
                'best_entry': "14:00 UTC",
                'best_exit': "21:00 UTC",
                'confidence': 0.0,
                'alternative_entries': [],
                'alternative_exits': [],
                'avoid_hours': [],
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

    def get_optimal_trade_times(self, exchange: str, symbol: str) -> Dict[str, Any]:
        """Get optimal trade times for a symbol"""
        try:
            logger.info(f"â° Analyzing optimal trade times for {symbol}")

            # Get historical data
            historical_data = self._get_historical_data(symbol, hours=self.time_windows['weekly'])

            if not historical_data:
                return {
                    'symbol': symbol,
                    'success': False,
                    'error': 'Insufficient historical data',
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }

            # Perform analyses
            volatility_analysis = self._calculate_intraday_volatility(historical_data)
            volume_surges = self._identify_volume_surges(historical_data)
            trade_analysis = self._analyze_historical_trades(historical_data)

            # Calculate optimal windows
            optimal_windows = self._calculate_optimal_windows(
                volatility_analysis, volume_surges, trade_analysis
            )

            # Format decision
            time_decision = self._format_time_decision(symbol, optimal_windows)

            # Store optimization result in cache
            optimization_id = f"time_opt_{symbol}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            self.cache.store_signal(
                signal_id=optimization_id,
                symbol=symbol,
                signal_type="TIME_OPTIMIZATION",
                confidence=optimal_windows['confidence'],
                strategy="time_aware_optimization",
                metadata=time_decision
            )

            logger.info(f"âœ… Time optimization complete for {symbol}: {time_decision['best_entry']} - {time_decision['best_exit']}")

            return {
                'symbol': symbol,
                'success': True,
                'time_decision': time_decision,
                'analysis_summary': {
                    'volatility_peaks': len(volatility_analysis.get('peak_hours', [])),
                    'volume_surges': len(volume_surges),
                    'historical_patterns': len(trade_analysis.get('successful_hours', []))
                },
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to get optimal trade times for {symbol}: {e}")
            return {
                'symbol': symbol,
                'success': False,
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

    def optimize_all_symbols(self) -> Dict[str, Any]:
        """Optimize trade times for all available symbols"""
        try:
            logger.info("ðŸ”„ Starting optimization for all symbols")

            # Get list of symbols (mock implementation)
            symbols = ['BTC-USD', 'ETH-USD', 'ADA-USD', 'DOT-USD', 'LINK-USD']

            results = {}
            successful_optimizations = 0

            for symbol in symbols:
                try:
                    result = self.get_optimal_trade_times('aggregated', symbol)
                    results[symbol] = result

                    if result.get('success', False):
                        successful_optimizations += 1

                except Exception as e:
                    logger.error(f"Failed to optimize {symbol}: {e}")
                    results[symbol] = {
                        'symbol': symbol,
                        'success': False,
                        'error': str(e)
                    }

            return {
                'success': True,
                'total_symbols': len(symbols),
                'successful_optimizations': successful_optimizations,
                'results': results,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to optimize all symbols: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

    def get_optimization_history(self, symbol: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get time optimization history for a symbol"""
        try:
            # Get recent time optimization signals from cache
            signals = self.cache.get_signals_by_type("TIME_OPTIMIZATION", limit=limit)

            # Filter by symbol
            symbol_optimizations = [
                signal for signal in signals
                if signal.get("symbol") == symbol
            ]

            return symbol_optimizations

        except Exception as e:
            logger.error(f"Failed to get optimization history: {e}")
            return []

    def get_optimizer_status(self) -> Dict[str, Any]:
        """Get current optimizer status"""
        try:
            return {
                "service": "TimeAwareTradeOptimizer",
                "status": "active",
                "time_windows": self.time_windows,
                "parameters": {
                    "volatility_threshold": self.volatility_threshold,
                    "volume_surge_threshold": self.volume_surge_threshold,
                    "min_data_points": self.min_data_points,
                    "confidence_threshold": self.confidence_threshold
                },
                "trading_sessions": self.trading_sessions,
                "capabilities": {
                    "intraday_volatility_analysis": True,
                    "volume_surge_detection": True,
                    "historical_trade_analysis": True,
                    "optimal_window_calculation": True
                }
            }

        except Exception as e:
            logger.error(f"Failed to get optimizer status: {e}")
            return {"success": False, "error": str(e)}


# Global time-aware trade optimizer instance
time_aware_trade_optimizer = TimeAwareTradeOptimizer()


def get_time_aware_trade_optimizer() -> TimeAwareTradeOptimizer:
    """Get the global time-aware trade optimizer instance"""
    return time_aware_trade_optimizer


if __name__ == "__main__":
    # Test the time-aware trade optimizer
    optimizer = TimeAwareTradeOptimizer()
    print(f"âœ… TimeAwareTradeOptimizer initialized: {optimizer}")

    # Test optimal trade times
    result = optimizer.get_optimal_trade_times('coinbase', 'BTC-USD')
    print(f"Optimal trade times: {result}")

    # Test status
    status = optimizer.get_optimizer_status()
    print(f"Optimizer status: {status['status']}")


