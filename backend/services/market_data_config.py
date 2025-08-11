SUPPORTED_COINS = [
    "BTC",
    "ETH",
    "ADA",
    "SOL",
    "DOT",
    "LINK",
    "MATIC",
    "AVAX",
    "UNI",
    "ATOM",
]

EXCHANGES = ["coinbase", "binance"]

API_PROVIDERS = ["coingecko", "coinpaprika", "coincap"]

RATE_LIMITS = {
    "coingecko": 30,  # Free tier
    "coinpaprika": 25,  # Approx safe threshold
    "coincap": 15,  # Free tier estimate
}
