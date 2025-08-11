#!/usr/bin/env python3
"""
Supported Coins Configuration
Defines which coins are supported by each exchange to prevent API errors
"""

# ✅ STEP 1: Filter Supported Coins per Exchange - User specified pairs
SUPPORTED_COINS = {
    "coinbase": [
        # Coinbase pairs (format: BASE-USD)
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
    ],
    "binance": [
        # Binance pairs (format: BASEUSDT)
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
    ],
}

# Trading pairs mapping for different exchanges
TRADING_PAIRS = {
    "coinbase": [
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
    ],
    "binance": [
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
    ],
}


def is_supported(coin: str, exchange: str) -> bool:
    """Check if a coin is supported by the specified exchange"""
    coin_upper = coin.upper()
    supported_pairs = SUPPORTED_COINS.get(exchange, [])

    # Check if the coin is directly in the list
    if coin_upper in supported_pairs:
        return True

    # Check if the coin is part of any trading pair
    for pair in supported_pairs:
        if exchange == "coinbase":
            # Coinbase format: "BTC-USD" -> check if "BTC" matches
            if pair.startswith(f"{coin_upper}-"):
                return True
        elif exchange == "binance":
            # Binance format: "BTCUSDT" -> check if "BTC" matches
            if pair.startswith(f"{coin_upper}"):
                return True

    return False


def get_supported_coins(exchange: str) -> list[str]:
    """Get list of supported coins for the specified exchange"""
    return SUPPORTED_COINS.get(exchange, [])


def get_trading_pairs(exchange: str) -> list[str]:
    """Get list of trading pairs for the specified exchange"""
    return TRADING_PAIRS.get(exchange, [])


def get_all_supported_coins() -> dict[str, list[str]]:
    """Get all supported coins for all exchanges"""
    return SUPPORTED_COINS.copy()


def get_all_trading_pairs() -> dict[str, list[str]]:
    """Get all trading pairs for all exchanges"""
    return TRADING_PAIRS.copy()
