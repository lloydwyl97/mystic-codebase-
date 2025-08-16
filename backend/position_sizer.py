def calculate_position_size(capital_usdt, strategy_win_rate, volatility=0.02, risk_per_trade=0.01):
    """
    Calculates dynamic position size.
    - capital_usdt: total capital available
    - strategy_win_rate: float from 0.0 to 1.0
    - volatility: recent volatility (default 2%)
    - risk_per_trade: capital risked per trade (1% default)
    """
    risk_factor = min(max(strategy_win_rate, 0.1), 0.9)
    volatility_adjusted = max(volatility, 0.01)

    position_size = (capital_usdt * risk_per_trade * risk_factor) / volatility_adjusted
    return round(position_size, 4)


def get_strategy_volatility(strategy_name, lookback_days=30):
    """Calculate recent volatility for a strategy"""
    # This would connect to your trade logger
    # For now, return a default volatility
    return 0.02


def size_position_for_strategy(strategy_name, capital_usdt, win_rate):
    """Complete position sizing for a specific strategy"""
    volatility = get_strategy_volatility(strategy_name)
    return calculate_position_size(capital_usdt, win_rate, volatility)


class PositionSizer:
    """Position sizing class for capital allocator integration"""

    def __init__(self):
        """Initialize the PositionSizer with default configuration"""
        self.default_risk_per_trade = 0.01  # 1% default risk
        self.max_position_size = 0.25  # 25% maximum position size
        self.min_position_size = 0.001  # 0.1% minimum position size
        self.volatility_lookback_days = 30
        self.kelly_criterion_max = 0.25  # Maximum Kelly Criterion percentage

    def calculate_kelly_criterion(self, win_rate, avg_win, avg_loss):
        """Calculate Kelly Criterion percentage"""
        if avg_loss == 0:
            return 0.0

        kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        return max(0.0, min(kelly, 0.25))  # Cap at 25%

    def calculate_position_size(
        self,
        capital_usdt,
        strategy_win_rate,
        volatility=0.02,
        risk_per_trade=0.01,
    ):
        """Calculate position size using the main function"""
        return calculate_position_size(capital_usdt, strategy_win_rate, volatility, risk_per_trade)


