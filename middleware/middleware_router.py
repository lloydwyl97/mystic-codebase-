#!/usr/bin/env python3
"""
Mystic Middleware Router
------------------------
Handles live data polling, pre-processing, API throttling,
and redistributing to AI, backend, and alerts via Redis pub/sub.
"""

import asyncio
import json
import logging
import os
import time

import aiohttp
import redis.asyncio as aioredis

# Config
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
COINGECKO_URL = "https://api.coingecko.com/api/v3/simple/price"
COINBASE_URL = "https://api.coinbase.com/v2/prices/{}-USD/spot"
BINANCE_URL = "https://api.binance.us/api/v3/ticker/price?symbol={}"

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("middleware-router")

WATCHLIST = ["ETH", "SOL", "ADA", "AVAX"]


class MiddlewareRouter:
    """
    Live Middleware Router for polling and publishing market data.
    """
    def __init__(self):
        self.redis_url = REDIS_URL
        self.watchlist = WATCHLIST
        self.coingecko_url = COINGECKO_URL
        self.coinbase_url = COINBASE_URL
        self.binance_url = BINANCE_URL
        self.logger = logger

    async def connect_redis_with_retry(self):
        retries = 5
        delay = 3
        for attempt in range(retries):
            try:
                redis = aioredis.from_url(self.redis_url, decode_responses=True)
                await redis.ping()
                self.logger.info("Connected to Redis.")
                return redis
            except Exception as e:
                self.logger.warning(
                    f"Redis not ready (attempt {attempt+1}/{retries}): {e}"
                )
                await asyncio.sleep(delay)
        raise ConnectionError("Could not connect to Redis after retries.")

    async def fetch_json(self, session, url):
        try:
            async with session.get(url, timeout=10) as response:
                return await response.json()
        except Exception as e:
            self.logger.warning(f"Fetch error: {url} - {e}")
            return {}

    async def poll_loop(self):
        redis = await self.connect_redis_with_retry()
        async with aiohttp.ClientSession() as session:
            while True:
                market_data = {}
                timestamp = time.time()

                for symbol in self.watchlist:
                    symbol_pair = symbol + "USDT"

                    # Binance
                    b = await self.fetch_json(
                        session, self.binance_url.format(symbol_pair)
                    )
                    if "price" in b:
                        market_data[f"{symbol}_binance"] = float(b["price"])

                    # Coinbase
                    c = await self.fetch_json(
                        session, self.coinbase_url.format(symbol)
                    )
                    if "data" in c and "amount" in c["data"]:
                        market_data[f"{symbol}_coinbase"] = float(c["data"]["amount"])

                # CoinGecko additional data source (every 60s)
                if int(timestamp) % 60 == 0:
                    g = await self.fetch_json(
                        session,
                        self.coingecko_url + (
                            "?ids=ethereum,solana,cardano,avalanche-2&vs_currencies=usd"
                        ),
                    )
                    for k, v in g.items():
                        market_data[f"{k}_gecko"] = v.get("usd")

                if market_data:
                    payload = {"timestamp": timestamp, "data": market_data}
                    await redis.publish("mystic:livefeed", json.dumps(payload))
                    self.logger.info(
                        f"[Livefeed] Published {len(market_data)} items"
                    )

                await asyncio.sleep(5)

    def start(self):
        asyncio.run(self.poll_loop())


# Entry
if __name__ == "__main__":
    router = MiddlewareRouter()
    router.start()
