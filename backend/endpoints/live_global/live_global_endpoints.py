"""
Live Global Market Data Endpoints

Provides real-time global market data and statistics.
All data is live and connected to real market sources.
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from ai.persistent_cache import get_persistent_cache

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/live/global")
async def get_global_market_data() -> Dict[str, Any]:
    """
    Get comprehensive global market data

    Returns live global market statistics including:
    - Total market cap
    - 24h volume
    - Market dominance
    - Top gainers/losers
    - Market sentiment
    """
    try:
        # Get live data from persistent cache
        cache = get_persistent_cache()

        # Get data from all sources
        coingecko_data = cache.get_coingecko()
        binance_data = cache.get_binance()
        coinbase_data = cache.get_coinbase()

        # Calculate global market statistics
        total_market_cap = 0
        total_volume_24h = 0
        total_price_change_24h = 0
        coin_count = 0

        # Process CoinGecko data for global stats
        for coin_id, coin_data in coingecko_data.items():
            if isinstance(coin_data, dict):
                market_cap = coin_data.get("market_cap", 0)
                volume_24h = coin_data.get("volume_24h", 0)
                price_change_24h = coin_data.get("price_change_24h", 0)

                total_market_cap += market_cap
                total_volume_24h += volume_24h
                total_price_change_24h += price_change_24h
                coin_count += 1

        # Calculate average 24h change
        avg_24h_change = total_price_change_24h / coin_count if coin_count > 0 else 0

        # Get top gainers and losers
        gainers = []
        losers = []

        for coin_id, coin_data in coingecko_data.items():
            if isinstance(coin_data, dict):
                symbol = coin_data.get("symbol", coin_id.upper())
                price = coin_data.get("price", 0)
                price_change_24h = coin_data.get("price_change_24h", 0)
                market_cap = coin_data.get("market_cap", 0)

                coin_info = {
                    "symbol": symbol,
                    "price": price,
                    "price_change_24h": price_change_24h,
                    "market_cap": market_cap,
                    "volume_24h": coin_data.get("volume_24h", 0),
                }

                if price_change_24h > 0:
                    gainers.append(coin_info)
                elif price_change_24h < 0:
                    losers.append(coin_info)

        # Sort by absolute change and take top 5
        gainers.sort(key=lambda x: abs(x["price_change_24h"]), reverse=True)
        losers.sort(key=lambda x: abs(x["price_change_24h"]), reverse=True)

        top_gainers = gainers[:5]
        top_losers = losers[:5]

        # Calculate market dominance (Bitcoin and Ethereum)
        btc_dominance = 0
        eth_dominance = 0

        btc_data = coingecko_data.get("bitcoin", {})
        eth_data = coingecko_data.get("ethereum", {})

        if isinstance(btc_data, dict) and isinstance(eth_data, dict):
            btc_market_cap = btc_data.get("market_cap", 0)
            eth_market_cap = eth_data.get("market_cap", 0)

            if total_market_cap > 0:
                btc_dominance = (btc_market_cap / total_market_cap) * 100
                eth_dominance = (eth_market_cap / total_market_cap) * 100

        # Market sentiment based on price changes
        positive_coins = len(
            [
                c
                for c in coingecko_data.values()
                if isinstance(c, dict) and c.get("price_change_24h", 0) > 0
            ]
        )
        negative_coins = len(
            [
                c
                for c in coingecko_data.values()
                if isinstance(c, dict) and c.get("price_change_24h", 0) < 0
            ]
        )

        sentiment = (
            "bullish"
            if positive_coins > negative_coins
            else "bearish" if negative_coins > positive_coins else "neutral"
        )

        # Calculate fear and greed index (simplified)
        fear_greed_score = 50  # Neutral baseline
        if avg_24h_change > 5:
            fear_greed_score = 75  # Greed
        elif avg_24h_change > 2:
            fear_greed_score = 65  # Greed
        elif avg_24h_change < -5:
            fear_greed_score = 25  # Fear
        elif avg_24h_change < -2:
            fear_greed_score = 35  # Fear

        fear_greed_label = (
            "Extreme Fear"
            if fear_greed_score <= 25
            else (
                "Fear"
                if fear_greed_score <= 45
                else (
                    "Neutral"
                    if fear_greed_score <= 55
                    else "Greed" if fear_greed_score <= 75 else "Extreme Greed"
                )
            )
        )

        # Get active exchanges
        active_exchanges = []
        if binance_data:
            active_exchanges.append("Binance")
        if coinbase_data:
            active_exchanges.append("Coinbase")

        return {
            "global_market_data": {
                "total_market_cap": total_market_cap,
                "total_volume_24h": total_volume_24h,
                "average_change_24h": avg_24h_change,
                "total_coins": coin_count,
                "active_exchanges": active_exchanges,
                "last_updated": datetime.now().isoformat(),
            },
            "market_dominance": {
                "bitcoin": btc_dominance,
                "ethereum": eth_dominance,
                "others": 100 - btc_dominance - eth_dominance,
            },
            "top_performers": {"gainers": top_gainers, "losers": top_losers},
            "market_sentiment": {
                "overall_sentiment": sentiment,
                "positive_coins": positive_coins,
                "negative_coins": negative_coins,
                "fear_greed_score": fear_greed_score,
                "fear_greed_label": fear_greed_label,
            },
            "live_data": True,
            "timestamp": time.time(),
            "source": "coingecko_binance_coinbase",
        }

    except Exception as e:
        logger.error(f"Error getting global market data: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting global market data: {str(e)}",
        )


@router.get("/live/global/summary")
async def get_global_market_summary() -> Dict[str, Any]:
    """
    Get simplified global market summary

    Returns key global market metrics for quick overview
    """
    try:
        # Get the full global data
        global_data = await get_global_market_data()

        # Extract key metrics
        market_data = global_data["global_market_data"]
        dominance = global_data["market_dominance"]
        sentiment = global_data["market_sentiment"]

        return {
            "total_market_cap": market_data["total_market_cap"],
            "total_volume_24h": market_data["total_volume_24h"],
            "average_change_24h": market_data["average_change_24h"],
            "bitcoin_dominance": dominance["bitcoin"],
            "ethereum_dominance": dominance["ethereum"],
            "market_sentiment": sentiment["overall_sentiment"],
            "fear_greed_score": sentiment["fear_greed_score"],
            "fear_greed_label": sentiment["fear_greed_label"],
            "live_data": True,
            "timestamp": time.time(),
        }

    except Exception as e:
        logger.error(f"Error getting global market summary: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting global market summary: {str(e)}",
        )
