class TradeMemory:
    """
    In-memory trade state manager for live trading.
    Stores, updates, and retrieves open/closed trades.
    """

    def __init__(self):
        # trade_id -> trade dict
        self.trades = {}
        self.open_trades = set()
        self.closed_trades = set()

    def add_trade(self, trade_id, trade_data):
        """Add a new trade to memory and mark as open."""
        self.trades[trade_id] = trade_data.copy()
        self.open_trades.add(trade_id)
        self.closed_trades.discard(trade_id)

    def close_trade(self, trade_id, close_data=None):
        """Mark a trade as closed and update its data."""
        if trade_id in self.trades:
            if close_data:
                self.trades[trade_id].update(close_data)
            self.open_trades.discard(trade_id)
            self.closed_trades.add(trade_id)

    def get_open_trades(self):
        """Return a list of open trade dicts."""
        return [self.trades[tid] for tid in self.open_trades if tid in self.trades]

    def reset_memory(self):
        """Clear all trade memory (use with caution)."""
        self.trades.clear()
        self.open_trades.clear()
        self.closed_trades.clear()


