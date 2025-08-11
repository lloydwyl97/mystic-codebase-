"""
AI Signals - Ranked Signals

Generates ranked trading signals based on multiple factors
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

from .persistent_cache import get_persistent_cache

logger = logging.getLogger("ai_signals")


def signal_scorer() -> List[str]:
    """Score and rank trading opportunities"""
    try:
        cache = get_persistent_cache()
        scored = []

        # Get cached data using correct methods
        binance_data = cache.get_binance()
        coingecko_data = cache.get_coingecko()

        # Process top coins by volume
        if binance_data:
            for symbol, data in list(binance_data.items())[:20]:
                try:
                    price = float(data.get("price", 0))
                    volume = float(data.get("volume", 0))
                    change_24h = float(data.get("change_24h", 0))

                    if price > 0 and volume > 1000000:  # Only coins with significant volume
                        # Calculate score based on volume and price momentum
                        volume_score = min(volume / 10000000, 50)  # Max 50 points for volume
                        momentum_score = max(change_24h * 2, 0)  # Positive momentum gets points
                        base_score = 30  # Base score for all coins

                        total_score = int(base_score + volume_score + momentum_score)
                        if total_score > 0:
                            scored.append(f"{symbol} SCORE {total_score}")
                except Exception as e:
                    logger.warning(f"Error processing {symbol}: {e}")
                    continue

        # Add some CoinGecko coins if available
        if coingecko_data:
            for coin_id, data in list(coingecko_data.items())[:10]:
                try:
                    price = float(data.get("current_price", 0))
                    volume = float(data.get("total_volume", 0))
                    change_24h = float(data.get("price_change_percentage_24h", 0))

                    if price > 0 and volume > 500000:
                        volume_score = min(volume / 5000000, 40)
                        momentum_score = max(change_24h * 1.5, 0)
                        base_score = 25

                        total_score = int(base_score + volume_score + momentum_score)
                        if total_score > 0:
                            scored.append(f"{coin_id.upper()} SCORE {total_score}")
                except Exception as e:
                    logger.warning(f"Error processing CoinGecko {coin_id}: {e}")
                    continue

        # Sort by score (highest first)
        scored.sort(key=lambda x: int(x.split()[-1]), reverse=True)

        logger.info(f"✅ Generated {len(scored)} scored signals")
        return scored[:10]  # Return top 10

    except Exception as e:
        logger.error(f"❌ Signal scorer error: {e}")
        return []


def risk_adjusted_signals() -> List[Dict[str, Any]]:
    """Generate risk-adjusted trading signals"""
    try:
        cache = get_persistent_cache()
        signals = []

        binance_data = cache.get_binance()

        for symbol, data in list(binance_data.items())[:15]:
            try:
                price = float(data.get("price", 0))
                volume = float(data.get("volume", 0))
                change_24h = float(data.get("change_24h", 0))

                if price > 0 and volume > 500000:
                    # Calculate risk metrics
                    volatility = abs(change_24h) / 100
                    volume_stability = min(volume / 10000000, 1.0)

                    # Risk score (lower is better)
                    risk_score = max(volatility * 100 - volume_stability * 20, 10)

                    # Reward potential
                    reward_potential = max(change_24h, 0) * 2

                    # Overall score
                    score = max(100 - risk_score + reward_potential, 20)

                    recommendation = "BUY" if score > 70 else "HOLD" if score > 40 else "SELL"

                    signals.append(
                        {
                            "symbol": symbol,
                            "price": price,
                            "recommendation": recommendation,
                            "score": int(score),
                            "risk_score": int(risk_score),
                            "reward_potential": int(reward_potential),
                            "volatility": round(volatility, 3),
                            "volume_stability": round(volume_stability, 3),
                        }
                    )
            except Exception as e:
                logger.warning(f"Error processing risk signal for {symbol}: {e}")
                continue

        # Sort by score
        signals.sort(key=lambda x: x["score"], reverse=True)

        logger.info(f"✅ Generated {len(signals)} risk-adjusted signals")
        return signals[:8]

    except Exception as e:
        logger.error(f"❌ Risk-adjusted signals error: {e}")
        return []


def technical_signals() -> List[Dict[str, Any]]:
    """Generate technical analysis signals"""
    try:
        cache = get_persistent_cache()
        signals = []

        binance_data = cache.get_binance()

        for symbol, data in list(binance_data.items())[:20]:
            try:
                price = float(data.get("price", 0))
                volume = float(data.get("volume", 0))
                change_24h = float(data.get("change_24h", 0))

                if price > 0 and volume > 300000:
                    # Technical indicators simulation
                    rsi = 50 + (change_24h * 2)  # Simulated RSI
                    macd = change_24h * 1.5  # Simulated MACD

                    # Determine action based on technical indicators
                    if rsi > 70 and macd > 0:
                        action = "SELL"
                        confidence = 75
                        strength = "STRONG"
                        sentiment = "BEARISH"
                    elif rsi < 30 and macd < 0:
                        action = "BUY"
                        confidence = 80
                        strength = "STRONG"
                        sentiment = "BULLISH"
                    elif change_24h > 5:
                        action = "BUY"
                        confidence = 65
                        strength = "MEDIUM"
                        sentiment = "BULLISH"
                    elif change_24h < -5:
                        action = "SELL"
                        confidence = 60
                        strength = "MEDIUM"
                        sentiment = "BEARISH"
                    else:
                        action = "HOLD"
                        confidence = 50
                        strength = "WEAK"
                        sentiment = "NEUTRAL"

                    # Calculate targets
                    target = price * (1 + (change_24h / 100) * 1.5)
                    stop_loss = price * (1 - abs(change_24h / 100) * 0.8)

                    signals.append(
                        {
                            "symbol": symbol,
                            "action": action,
                            "confidence": confidence,
                            "price": price,
                            "target": round(target, 6),
                            "stop_loss": round(stop_loss, 6),
                            "strength": strength,
                            "reasoning": (
                                f"RSI: {rsi:.1f}, MACD: {macd:.2f}, 24h change: {change_24h:.2f}%"
                            ),
                            "sentiment": sentiment,
                        }
                    )
            except Exception as e:
                logger.warning(f"Error processing technical signal for {symbol}: {e}")
                continue

        # Sort by confidence
        signals.sort(key=lambda x: x["confidence"], reverse=True)

        logger.info(f"✅ Generated {len(signals)} technical signals")
        return signals[:10]

    except Exception as e:
        logger.error(f"❌ Technical signals error: {e}")
        return []


def market_strength_signals() -> Dict[str, Any]:
    """Analyze overall market strength"""
    try:
        cache = get_persistent_cache()

        binance_data = cache.get_binance()
        coingecko_data = cache.get_coingecko()

        strong_coins = 0
        weak_coins = 0
        total_coins = 0

        # Analyze Binance data
        for symbol, data in binance_data.items():
            try:
                change_24h = float(data.get("change_24h", 0))
                volume = float(data.get("volume", 0))

                if volume > 100000:  # Only consider coins with significant volume
                    total_coins += 1
                    if change_24h > 2:
                        strong_coins += 1
                    elif change_24h < -2:
                        weak_coins += 1
            except (ValueError, TypeError, KeyError) as e:
                logger.debug(f"Failed to process market data: {e}")
                continue

        # Analyze CoinGecko data
        for coin_id, data in coingecko_data.items():
            try:
                change_24h = float(data.get("price_change_percentage_24h", 0))
                volume = float(data.get("total_volume", 0))

                if volume > 50000:
                    total_coins += 1
                    if change_24h > 2:
                        strong_coins += 1
                    elif change_24h < -2:
                        weak_coins += 1
            except (ValueError, TypeError, KeyError) as e:
                logger.debug(f"Failed to process CoinGecko data: {e}")
                continue

        # Calculate percentages
        if total_coins > 0:
            strong_percentage = (strong_coins / total_coins) * 100
            weak_percentage = (weak_coins / total_coins) * 100
        else:
            strong_percentage = 0
            weak_percentage = 0

        # Determine recommendation
        if strong_percentage > 60:
            recommendation = "BULLISH - Strong market momentum"
        elif weak_percentage > 60:
            recommendation = "BEARISH - Market weakness detected"
        elif strong_percentage > weak_percentage:
            recommendation = "CAUTIOUSLY BULLISH - Mixed signals"
        else:
            recommendation = "NEUTRAL - Balanced market conditions"

        result = {
            "market_strength": round(strong_percentage, 1),
            "market_weakness": round(weak_percentage, 1),
            "total_coins_analyzed": total_coins,
            "strong_coins": strong_coins,
            "weak_coins": weak_coins,
            "recommendation": recommendation,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(
            f"✅ Market strength analysis: {strong_percentage:.1f}% strong, {weak_percentage:.1f}% weak"
        )
        return result

    except Exception as e:
        logger.error(f"❌ Market strength signals error: {e}")
        return {
            "market_strength": 0,
            "market_weakness": 0,
            "total_coins_analyzed": 0,
            "strong_coins": 0,
            "weak_coins": 0,
            "recommendation": "ERROR - Unable to analyze market",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


def trend_analysis() -> Dict[str, Any]:
    """Analyze market trends"""
    try:
        cache = get_persistent_cache()

        binance_data = cache.get_binance()

        # Calculate trend metrics
        positive_changes = 0
        negative_changes = 0
        total_volume = 0
        avg_change = 0
        changes = []

        for symbol, data in binance_data.items():
            try:
                change_24h = float(data.get("change_24h", 0))
                volume = float(data.get("volume", 0))

                if volume > 100000:
                    changes.append(change_24h)
                    total_volume += volume

                    if change_24h > 0:
                        positive_changes += 1
                    elif change_24h < 0:
                        negative_changes += 1
            except (ValueError, TypeError, KeyError) as e:
                logger.debug(f"Failed to process trend data: {e}")
                continue

        if changes:
            avg_change = sum(changes) / len(changes)

        # Determine trend
        if avg_change > 3 and positive_changes > negative_changes * 1.5:
            trend = "STRONG_UPTREND"
            summary = "Market showing strong upward momentum with high volume"
        elif avg_change > 1 and positive_changes > negative_changes:
            trend = "UPTREND"
            summary = "Market trending upward with moderate momentum"
        elif avg_change < -3 and negative_changes > positive_changes * 1.5:
            trend = "STRONG_DOWNTREND"
            summary = "Market showing strong downward pressure"
        elif avg_change < -1 and negative_changes > positive_changes:
            trend = "DOWNTREND"
            summary = "Market trending downward"
        else:
            trend = "SIDEWAYS"
            summary = "Market moving sideways with mixed signals"

        result = {
            "trend": trend,
            "summary": summary,
            "average_change": round(avg_change, 2),
            "positive_coins": positive_changes,
            "negative_coins": negative_changes,
            "total_volume": total_volume,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(f"✅ Trend analysis: {trend} - {summary}")
        return result

    except Exception as e:
        logger.error(f"❌ Trend analysis error: {e}")
        return {
            "trend": "UNKNOWN",
            "summary": "Unable to analyze trends",
            "average_change": 0,
            "positive_coins": 0,
            "negative_coins": 0,
            "total_volume": 0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


def mystic_oracle() -> Dict[str, Any]:
    """Mystic Oracle - Advanced market predictions"""
    try:
        cache = get_persistent_cache()

        # Get market data for analysis
        binance_data = cache.get_binance()

        # Analyze market patterns
        high_volume_coins = 0
        momentum_coins = 0
        total_analyzed = 0

        for symbol, data in binance_data.items():
            try:
                volume = float(data.get("volume", 0))
                change_24h = float(data.get("change_24h", 0))

                if volume > 1000000:  # High volume threshold
                    high_volume_coins += 1

                if change_24h > 5:  # Strong momentum
                    momentum_coins += 1

                total_analyzed += 1
            except (ValueError, TypeError, KeyError) as e:
                logger.debug(f"Failed to process oracle data: {e}")
                continue

        # Generate prediction based on patterns
        if high_volume_coins > total_analyzed * 0.3 and momentum_coins > total_analyzed * 0.2:
            prediction = "BULLISH BREAKOUT IMMINENT - High volume and momentum suggest strong upward movement"
            confidence = 85
            timeframe = "24-48 hours"
        elif high_volume_coins > total_analyzed * 0.2:
            prediction = "ACCUMULATION PHASE - High volume indicates institutional interest"
            confidence = 75
            timeframe = "1-2 weeks"
        elif momentum_coins > total_analyzed * 0.15:
            prediction = "MOMENTUM BUILDING - Positive price action across multiple assets"
            confidence = 70
            timeframe = "3-7 days"
        else:
            prediction = "CONSOLIDATION - Market stabilizing after recent movements"
            confidence = 60
            timeframe = "1-3 days"

        result = {
            "prediction": prediction,
            "confidence": confidence,
            "timeframe": timeframe,
            "high_volume_coins": high_volume_coins,
            "momentum_coins": momentum_coins,
            "total_analyzed": total_analyzed,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(f"✅ Mystic Oracle: {prediction}")
        return result

    except Exception as e:
        logger.error(f"❌ Mystic Oracle error: {e}")
        return {
            "prediction": "Unable to generate prediction",
            "confidence": 0,
            "timeframe": "Unknown",
            "high_volume_coins": 0,
            "momentum_coins": 0,
            "total_analyzed": 0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


def get_trading_status() -> Dict[str, Any]:
    """Get current trading system status"""
    try:
        cache = get_persistent_cache()

        # Check cache status
        binance_symbols = len(cache.get_binance())
        coingecko_coins = len(cache.get_coingecko())

        # Determine trading status
        if binance_symbols > 50 and coingecko_coins > 100:
            trading_enabled = True
            status = "ACTIVE"
        else:
            trading_enabled = False
            status = "INSUFFICIENT_DATA"

        result = {
            "trading_enabled": trading_enabled,
            "status": status,
            "cache_status": {
                "binance_symbols": binance_symbols,
                "coingecko_coins": coingecko_coins,
                "last_updated": cache.get_last_update(),
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(
            f"✅ Trading status: {status} - {binance_symbols} Binance pairs, {coingecko_coins} CoinGecko coins"
        )
        return result

    except Exception as e:
        logger.error(f"❌ Trading status error: {e}")
        return {
            "trading_enabled": False,
            "status": "ERROR",
            "cache_status": {
                "binance_symbols": 0,
                "coingecko_coins": 0,
                "last_updated": "Unknown",
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


def get_trade_summary() -> Dict[str, Any]:
    """Get trading performance summary"""
    try:
        # Simulate trading performance based on signal quality
        signals = signal_scorer()
        risk_signals = risk_adjusted_signals()

        # Calculate simulated performance
        total_trades = len(signals) + len(risk_signals)
        if total_trades > 0:
            # Simulate win rate based on signal quality
            high_quality_signals = len(
                [s for s in signals if "SCORE" in s and int(s.split()[-1]) > 70]
            )
            win_rate = min((high_quality_signals / total_trades) * 100, 95)

            # Simulate PnL based on win rate
            total_pnl = (win_rate - 50) * 2  # Positive if win rate > 50%
        else:
            win_rate = 0
            total_pnl = 0

        result = {
            "total_trades": total_trades,
            "win_rate": round(win_rate, 1),
            "total_pnl": round(total_pnl, 2),
            "high_quality_signals": len(
                [s for s in signals if "SCORE" in s and int(s.split()[-1]) > 70]
            ),
            "risk_adjusted_signals": len(risk_signals),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(
            f"✅ Trade summary: {total_trades} trades, {win_rate:.1f}% win rate, {total_pnl:.2f} PnL"
        )
        return result

    except Exception as e:
        logger.error(f"❌ Trade summary error: {e}")
        return {
            "total_trades": 0,
            "win_rate": 0,
            "total_pnl": 0,
            "high_quality_signals": 0,
            "risk_adjusted_signals": 0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
