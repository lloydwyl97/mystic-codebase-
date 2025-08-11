# price_poller.py
import httpx
import asyncio
import json
import csv
import os
from itertools import cycle
from collections import defaultdict
from datetime import datetime

BINANCE = {
    "BTC": "https://api.binance.us/api/v3/ticker/price?symbol=BTCUSDT",
    "ETH": "https://api.binance.us/api/v3/ticker/price?symbol=ETHUSDT",
    "SOL": "https://api.binance.us/api/v3/ticker/price?symbol=SOLUSDT",
    "ADA": "https://api.binance.us/api/v3/ticker/price?symbol=ADAUSDT",
}
COINBASE = {
    "BTC": "https://api.coinbase.com/v2/prices/BTC-USD/spot",
    "ETH": "https://api.coinbase.com/v2/prices/ETH-USD/spot",
    "SOL": "https://api.coinbase.com/v2/prices/SOL-USD/spot",
    "ADA": "https://api.coinbase.com/v2/prices/ADA-USD/spot",
}

CALLS = cycle(
    [("Binance", sym, url) for sym, url in BINANCE.items()]
    + [("Coinbase", sym, url) for sym, url in COINBASE.items()]
)

shared_file = "shared_data.json"
log_file = "price_log.csv"
latest_prices = defaultdict(dict)
previous_prices = {}


async def poll():
    async with httpx.AsyncClient() as client:
        while True:
            source, symbol, url = next(CALLS)
            try:
                resp = await client.get(url, timeout=10)
                data = await resp.json()
                price = float(data.get("price") or data.get("ask") or "0")
                ts = datetime.timezone.utcnow().isoformat()

                # Calculate % change
                prev = previous_prices.get((source, symbol))
                change = (price - prev) / prev * 100 if prev else 0.0
                previous_prices[(source, symbol)] = price

                # Update shared data
                latest_prices[source][symbol] = {
                    "price": price,
                    "symbol": symbol,
                    "source": source,
                    "timestamp": ts,
                    "change_pct": round(change, 2),
                }

                # Save JSON snapshot
                with open(shared_file, "w") as f:
                    json.dump(latest_prices, f, indent=2)

                # Log to CSV
                with open(log_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([ts, source, symbol, price, round(change, 2)])

                print(f"[{source}] {symbol}: ${price} ({change:+.2f}%)")

            except Exception as e:
                print(f"[{source}] {symbol} ERROR: {e}")
            await asyncio.sleep(30)


if __name__ == "__main__":
    print("⏳ Starting upgraded polling...")
    # Create log file with headers if not exists
    if not os.path.exists(log_file):
        with open(log_file, "w", newline="") as f:
            csv.writer(f).writerow(
                ["timestamp", "source", "symbol", "price", "change_pct"]
            )
    asyncio.run(poll())
