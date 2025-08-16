def tag_trade(price: float, recent_prices: list) -> str:
    if not recent_prices:
        return "unknown"

    last = recent_prices[-1]
    if price > max(recent_prices):
        return "breakout"
    elif price < min(recent_prices):
        return "breakdown"
    elif price > last:
        return "uptrend"
    elif price < last:
        return "downtrend"
    else:
        return "consolidation"


def analyze_trade_pattern(prices: list, volumes: list = None) -> dict:
    """Analyze trade pattern and return strategy insights"""
    if len(prices) < 3:
        return {"pattern": "insufficient_data"}

    current_price = prices[-1]
    prev_price = prices[-2]
    price_change = ((current_price - prev_price) / prev_price) * 100

    pattern = {
        "trend": ("up" if price_change > 0 else "down" if price_change < 0 else "sideways"),
        "strength": "strong" if abs(price_change) > 2 else "weak",
        "volatility": (
            "high" if len(prices) > 5 and max(prices) - min(prices) > current_price * 0.1 else "low"
        ),
    }

    return pattern


def get_strategy_confidence(pattern: dict, mystic_signals: dict = None) -> float:
    """Calculate confidence based on pattern and mystic signals"""
    base_confidence = 0.5

    # Pattern confidence
    if pattern.get("trend") == "up" and pattern.get("strength") == "strong":
        base_confidence += 0.2
    elif pattern.get("trend") == "down":
        base_confidence -= 0.1

    # Mystic signals boost
    if mystic_signals:
        if mystic_signals.get("tesla_369", 0) > 0.7:
            base_confidence += 0.1
        if mystic_signals.get("faerie_star", 0) > 0.7:
            base_confidence += 0.1

    return min(1.0, max(0.0, base_confidence))


