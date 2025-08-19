from __future__ import annotations

EXCHANGE_TOP4: dict[str, list[str]] = {
    "coinbase": ["BTC-USD", "ETH-USD", "ADA-USD", "SOL-USD"],
    "binanceus": ["BTC-USD", "ETH-USD", "ADA-USD", "SOL-USD"],
    "kraken": ["BTC-USD", "ETH-USD", "ADA-USD", "SOL-USD"],
}


def normalize_symbol_to_dash(symbol: str) -> str:
    s = symbol.upper().replace("/", "-")
    if s.endswith("-USDT"):
        s = s.replace("-USDT", "-USD")
    return s


def to_exchange_symbol(exchange: str, symbol_dash: str) -> str:
    x = exchange.lower()
    base, quote = symbol_dash.upper().split("-")
    if x == "binanceus":
        q = "USDT" if quote == "USD" else quote
        return f"{base}{q}"
    if x == "coinbase":
        return f"{base}-{quote}"
    if x == "kraken":
        b = "XBT" if base == "BTC" else base
        return f"{b}{quote}"
    return symbol_dash


def is_top4(exchange: str, symbol_dash: str) -> bool:
    dash = normalize_symbol_to_dash(symbol_dash)
    return dash in EXCHANGE_TOP4.get(exchange.lower(), [])




