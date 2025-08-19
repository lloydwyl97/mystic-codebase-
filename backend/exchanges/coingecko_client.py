from __future__ import annotations

from datetime import datetime, timezone

import aiohttp

from backend.models.market_types import OHLCV, Ticker  # type: ignore[import-not-found]
from backend.utils.symbols import normalize_symbol_to_dash  # type: ignore[import-not-found]


class CoinGeckoClient:
    name = "coingecko"

    def __init__(self) -> None:
        self.api_base = "https://api.coingecko.com/api/v3"

    async def get_simple_price(self, ids: list[str]) -> dict[str, Ticker]:
        if not ids:
            return {}
        joined = ",".join(ids)
        url = (
            f"{self.api_base}/simple/price?ids={joined}&vs_currencies=usd&include_24hr_change=true&include_last_updated_at=true"
        )
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as resp:
                data = await resp.json()
        ts_now = int(datetime.now(timezone.utc).timestamp() * 1000)
        out: dict[str, Ticker] = {}
        for cid, v in data.items():
            price = float(v.get("usd", 0) or 0)
            ts = int(v.get("last_updated_at", ts_now)) * 1000
            out[cid] = Ticker(exchange=self.name, symbol=cid.upper(), price=price, bid=None, ask=None, ts=ts)
        return out

    async def get_ohlcv(self, coingecko_id: str, days: int = 7) -> list[OHLCV]:
        url = f"{self.api_base}/coins/{coingecko_id}/market_chart?vs_currency=usd&days={days}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as resp:
                data = await resp.json()
        prices = data.get("prices", [])
        out: list[OHLCV] = []
        for ts, price in prices:
            out.append(
                OHLCV(
                    exchange=self.name,
                    symbol=normalize_symbol_to_dash(coingecko_id.upper()),
                    ts=int(ts),
                    open=float(price),
                    high=float(price),
                    low=float(price),
                    close=float(price),
                    volume=0.0,
                )
            )
        return out




