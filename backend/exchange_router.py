import os


class ExchangeRouter:
    def __init__(self):
        self.exchange = os.getenv("EXCHANGE", "binance").lower()
        self.binance_coins = [
            "BTCUSDT",
            "ETHUSDT",
            "ADAUSDT",
            "SOLUSDT",
            "DOTUSDT",
            "LINKUSDT",
            "MATICUSDT",
            "AVAXUSDT",
            "UNIUSDT",
            "ATOMUSDT",
        ]
        self.coinbase_coins = [
            "BTC-USD",
            "ETH-USD",
            "ADA-USD",
            "SOL-USD",
            "DOT-USD",
            "LINK-USD",
            "MATIC-USD",
            "AVAX-USD",
            "UNI-USD",
            "ATOM-USD",
        ]

    def get_active_exchange(self):
        return self.exchange

    def get_coin_list(self):
        if self.exchange == "binance":
            return self.binance_coins
        elif self.exchange == "coinbase":
            return self.coinbase_coins
        else:
            raise ValueError(f"Unsupported exchange: {self.exchange}")

    def format_symbol(self, base: str, quote: str = "USDT") -> str:
        if self.exchange == "binance":
            return f"{base.upper()}{quote.upper()}"
        elif self.exchange == "coinbase":
            return f"{base.upper()}-{quote.upper()}"
        else:
            raise ValueError("Unsupported exchange formatting")

    def is_binance(self):
        return self.exchange == "binance"

    def is_coinbase(self):
        return self.exchange == "coinbase"
