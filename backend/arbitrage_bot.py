import ccxt

binance = ccxt.binance()
coinbase = ccxt.coinbasepro()


def fetch_prices(symbol="BTC/USDT"):
    b_price = binance.fetch_ticker(symbol)["last"]
    c_price = coinbase.fetch_ticker(symbol)["last"]
    return b_price, c_price


def check_arbitrage():
    b, c = fetch_prices()
    spread = abs(b - c)
    if spread > 20:  # adjust threshold
        print(f"[ARBITRAGE] Spread Detected! Binance: {b}, Coinbase: {c} | Î” = {spread}")


