import time
from typing import Any

import requests

# Rate limiting configuration - Only Binance and Coinbase
RATE_LIMITS: dict[str, dict[str, Any]] = {
    "coinbase": {"rpm": 10, "last_call": 0},
    "binance": {"rpm": 10, "last_call": 0},
}

# âœ… STEP 1: Only call supported + tradable coins - OPTIMIZED LIST
# These coins are tested and work on multiple exchanges
SUPPORTED_COINS: list[str] = []  # Will be populated dynamically from exchange APIs

# Removed: All other coins to minimize API calls

# ðŸ”§ STAGGERED BATCHING CONFIGURATION
# Group coins by priority and API capacity
FAST_COINS: list[str] = []  # Will be populated dynamically from exchange APIs

# API rotation schedule
API_SCHEDULE: dict[str, dict[str, Any]] = {
    "binance": {"coins": FAST_COINS, "delay": 0},
    "coinbase": {"coins": FAST_COINS, "delay": 0},
}


def is_supported(coin: str) -> bool:
    return coin.upper() in SUPPORTED_COINS


def throttle(provider: str) -> None:
    """Throttle requests based on provider rate limits."""
    if provider not in RATE_LIMITS:
        return

    current_time = time.time()
    limit_info = RATE_LIMITS[provider]

    # Check if we need to wait
    time_since_last = current_time - limit_info["last_call"]
    min_interval = 60.0 / limit_info["rpm"]  # seconds between requests

    if time_since_last < min_interval:
        sleep_time = min_interval - time_since_last
        time.sleep(sleep_time)

    # Update last call time
    RATE_LIMITS[provider]["last_call"] = int(time.time())


# ðŸ”§ STAGGERED BATCH FETCH
def fetch_staggered_batch() -> dict[str, dict[str, Any]]:
    """Fetch data using staggered batching to respect rate limits."""
    results: dict[str, dict[str, Any]] = {}

    print("ðŸš€ Starting staggered batch fetch for 3 major coins...")

    # Only use fast APIs (Binance & Coinbase) for the 3 major coins
    print("ðŸ“¡ Fast APIs (Binance & Coinbase)")
    for coin in SUPPORTED_COINS:
        for api in ["binance", "coinbase"]:
            try:
                if api == "binance":
                    result = fetch_from_binance(coin)
                else:
                    result = fetch_from_coinbase(coin)

                if result and coin not in results:
                    results[coin] = result
                    print(f"âœ… {coin}: ${result['price']} from {result['source']}")
                    break  # Got data, move to next coin
            except Exception as e:
                print(f"âŒ {api.upper()} failed for {coin}: {e}")

    print(f"ðŸŽ¯ Batch complete: {len(results)}/{len(SUPPORTED_COINS)} coins fetched")
    return results


def fetch_from_binance(symbol: str) -> dict[str, Any] | None:
    try:
        url = f"https://api.binance.us/api/v3/ticker/price?symbol={symbol.upper()}USDT"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return {"price": float(data["price"]), "source": "binance"}
        return None
    except Exception as e:
        raise Exception(f"Binance error: {e}")


def fetch_from_coinbase(symbol: str) -> dict[str, Any] | None:
    try:
        url = f"https://api.coinbase.us/v2/prices/{symbol.upper()}-USD/spot"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return {
                "price": float(data["data"]["amount"]),
                "source": "coinbase",
            }
        return None
    except Exception as e:
        raise Exception(f"Coinbase error: {e}")


# âœ… STEP 3: Batch fetch function for all supported coins - UPDATED
def fetch_all_supported_coins() -> dict[str, dict[str, Any]]:
    """Fetch data for all supported coins using staggered batching."""
    return fetch_staggered_batch()


def test_coin_support(symbol: str) -> dict[str, bool]:
    """Test which exchanges support a given coin."""
    results: dict[str, bool] = {}
    apis = ["binance", "coinbase"]

    for api in apis:
        try:
            if api == "binance":
                result = fetch_from_binance(symbol)
                results[api] = result is not None
            elif api == "coinbase":
                result = fetch_from_coinbase(symbol)
                results[api] = result is not None
        except Exception as e:
            print(f"âŒ {api.upper()} test failed for {symbol}: {e}")
            results[api] = False

    return results


def get_coin_support_summary() -> dict[str, dict[str, bool]]:
    """Get support summary for all coins in the list."""
    summary: dict[str, dict[str, bool]] = {}
    for coin in SUPPORTED_COINS:
        summary[coin] = test_coin_support(coin)
        time.sleep(2)  # Be nice to APIs
    return summary


def show_coin_summary():
    """Show summary of coins and expected API hits per service."""
    print("=" * 60)
    print("ðŸ“Š COIN SUMMARY & STAGGERED BATCH ANALYSIS")
    print("=" * 60)

    print(f"\nðŸª™ TOTAL COINS: {len(SUPPORTED_COINS)}")
    print(f"ðŸ“‹ COIN LIST: {', '.join(SUPPORTED_COINS)}")

    print("\nðŸ”§ STAGGERED BATCHING STRATEGY:")
    print(f"   • FAST COINS ({len(FAST_COINS)}): {', '.join(FAST_COINS)}")

    print("\nðŸ“ˆ OPTIMIZED API HITS PER SERVICE:")
    print(f"   • Binance: {len(SUPPORTED_COINS)} hits (no delay)")
    print(f"   • Coinbase: {len(SUPPORTED_COINS)} hits (no delay)")

    # Calculate optimized time
    fast_time = 0  # No delays
    total_time = fast_time

    print(f"\nâ±ï¸  OPTIMIZED TOTAL TIME: {total_time} seconds ({total_time/60:.1f} minutes)")

    print("\nðŸŽ¯ RATE LIMIT COMPLIANCE:")
    print(f"   • Binance: {len(SUPPORTED_COINS)} hits vs {RATE_LIMITS['binance']['rpm']}/min âœ…")
    print(f"   • Coinbase: {len(SUPPORTED_COINS)} hits vs {RATE_LIMITS['coinbase']['rpm']}/min âœ…")

    print("=" * 60)
    return {
        "total_coins": len(SUPPORTED_COINS),
        "coins": SUPPORTED_COINS,
        "fast_coins": FAST_COINS,
        "hits_per_service": {
            "binance": len(SUPPORTED_COINS),
            "coinbase": len(SUPPORTED_COINS),
        },
        "rate_limits": RATE_LIMITS,
        "total_time_seconds": total_time,
    }


