import asyncio
import openai
import json
import sqlite3
import requests
from datetime import datetime
from typing import Dict, List
import pandas as pd
import os

# Enhanced configuration
EXPLAINER_DB = "./data/trade_explanations.db"
EXPLANATION_INTERVAL = 300  # 5 minutes
openai.api_key = os.getenv("OPENAI_API_KEY")


class ExplainerDatabase:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize explainer database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS trade_explanations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                trade_type TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                quantity REAL NOT NULL,
                pnl REAL,
                explanation_text TEXT NOT NULL,
                confidence_score REAL NOT NULL,
                factors_analyzed TEXT NOT NULL,
                risk_assessment TEXT NOT NULL,
                market_context TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS factor_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                factor_name TEXT NOT NULL,
                factor_value REAL NOT NULL,
                factor_weight REAL NOT NULL,
                factor_impact TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS explanation_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                total_explanations INTEGER NOT NULL,
                avg_confidence REAL NOT NULL,
                accuracy_score REAL NOT NULL,
                user_feedback_score REAL NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        conn.commit()
        conn.close()

    def save_trade_explanation(self, explanation: Dict):
        """Save trade explanation to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO trade_explanations
            (timestamp, symbol, trade_type, entry_price, exit_price, quantity, pnl,
             explanation_text, confidence_score, factors_analyzed, risk_assessment, market_context)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                explanation["timestamp"],
                explanation["symbol"],
                explanation["trade_type"],
                explanation["entry_price"],
                explanation.get("exit_price"),
                explanation["quantity"],
                explanation.get("pnl"),
                explanation["explanation_text"],
                explanation["confidence_score"],
                json.dumps(explanation["factors_analyzed"]),
                explanation["risk_assessment"],
                explanation["market_context"],
            ),
        )

        conn.commit()
        conn.close()

    def save_factor_analysis(self, factors: List[Dict]):
        """Save factor analysis to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for factor in factors:
            cursor.execute(
                """
                INSERT INTO factor_analysis
                (timestamp, symbol, factor_name, factor_value, factor_weight, factor_impact)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    factor["timestamp"],
                    factor["symbol"],
                    factor["factor_name"],
                    factor["factor_value"],
                    factor["factor_weight"],
                    factor["factor_impact"],
                ),
            )

        conn.commit()
        conn.close()


def get_market_data(symbol: str) -> Dict:
    """Get comprehensive market data for analysis"""
    try:
        # Current price and 24h data
        response = requests.get("https://api.binance.us/api/v3/ticker/24hr", timeout=10)
        if response.status_code == 200:
            data = response.json()
            for item in data:
                if item["symbol"] == symbol:
                    return {
                        "current_price": float(item["lastPrice"]),
                        "price_change": float(item["priceChangePercent"]),
                        "volume": float(item["volume"]),
                        "high_24h": float(item["highPrice"]),
                        "low_24h": float(item["lowPrice"]),
                    }
    except Exception as e:
        print(f"Market data fetch error: {e}")
    return {}


def get_technical_indicators(symbol: str) -> Dict:
    """Get technical indicators for analysis"""
    try:
        # Get historical data for indicators
        url = "https://api.binance.us/api/v3/klines"
        params = {"symbol": symbol, "interval": "1h", "limit": 100}

        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()

            # Convert to DataFrame
            df = pd.DataFrame(
                data,
                columns=[
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "close_time",
                    "quote_volume",
                    "trades",
                    "taker_buy_base",
                    "taker_buy_quote",
                    "ignore",
                ],
            )

            # Convert to numeric
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col])

            # Calculate indicators
            indicators = {}

            # RSI
            delta = df["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df["rsi"] = 100 - (100 / (1 + rs))
            indicators["rsi"] = df["rsi"].iloc[-1]

            # MACD
            ema_fast = df["close"].ewm(span=12).mean()
            ema_slow = df["close"].ewm(span=26).mean()
            df["macd"] = ema_fast - ema_slow
            df["macd_signal"] = df["macd"].ewm(span=9).mean()
            indicators["macd"] = df["macd"].iloc[-1]
            indicators["macd_signal"] = df["macd_signal"].iloc[-1]

            # Bollinger Bands
            sma = df["close"].rolling(window=20).mean()
            std = df["close"].rolling(window=20).std()
            df["bb_upper"] = sma + (std * 2)
            df["bb_lower"] = sma - (std * 2)
            indicators["bb_upper"] = df["bb_upper"].iloc[-1]
            indicators["bb_lower"] = df["bb_lower"].iloc[-1]
            indicators["bb_position"] = (df["close"].iloc[-1] - indicators["bb_lower"]) / (
                indicators["bb_upper"] - indicators["bb_lower"]
            )

            # Moving averages
            indicators["sma_20"] = df["close"].rolling(window=20).mean().iloc[-1]
            indicators["sma_50"] = df["close"].rolling(window=50).mean().iloc[-1]

            # Volume indicators
            indicators["volume_sma"] = df["volume"].rolling(window=20).mean().iloc[-1]
            indicators["volume_ratio"] = df["volume"].iloc[-1] / indicators["volume_sma"]

            return indicators

    except Exception as e:
        print(f"Technical indicators calculation error: {e}")

    return {}


def get_sentiment_data(symbol: str) -> Dict:
    """Get sentiment data for analysis"""
    try:
        # Simplified sentiment analysis
        # In a real implementation, this would fetch from sentiment APIs
        sentiment_sources = {
            "social_media": 0.6,  # Positive sentiment
            "news_sentiment": 0.4,  # Neutral sentiment
            "reddit_sentiment": 0.7,  # Positive sentiment
            "twitter_sentiment": 0.5,  # Neutral sentiment
        }

        avg_sentiment = sum(sentiment_sources.values()) / len(sentiment_sources)

        return {
            "overall_sentiment": avg_sentiment,
            "sentiment_sources": sentiment_sources,
            "sentiment_trend": (
                "increasing"
                if avg_sentiment > 0.6
                else "decreasing" if avg_sentiment < 0.4 else "stable"
            ),
        }

    except Exception as e:
        print(f"Sentiment data fetch error: {e}")

    return {}


def analyze_trade_factors(symbol: str, trade_type: str, entry_price: float) -> List[Dict]:
    """Analyze factors that influenced the trade"""
    factors = []
    timestamp = datetime.timezone.utcnow().isoformat()

    # Get market data
    market_data = get_market_data(symbol)
    technical_indicators = get_technical_indicators(symbol)
    sentiment_data = get_sentiment_data(symbol)

    # Price momentum factor
    if market_data:
        price_change = market_data["price_change"]
        momentum_weight = 0.3
        momentum_impact = (
            "bullish" if price_change > 2 else "bearish" if price_change < -2 else "neutral"
        )

        factors.append(
            {
                "timestamp": timestamp,
                "symbol": symbol,
                "factor_name": "price_momentum",
                "factor_value": price_change,
                "factor_weight": momentum_weight,
                "factor_impact": momentum_impact,
            }
        )

    # Technical indicators factors
    if technical_indicators:
        # RSI factor
        rsi = technical_indicators.get("rsi", 50)
        rsi_weight = 0.2
        rsi_impact = "oversold" if rsi < 30 else "overbought" if rsi > 70 else "neutral"

        factors.append(
            {
                "timestamp": timestamp,
                "symbol": symbol,
                "factor_name": "rsi",
                "factor_value": rsi,
                "factor_weight": rsi_weight,
                "factor_impact": rsi_impact,
            }
        )

        # MACD factor
        macd = technical_indicators.get("macd", 0)
        macd_signal = technical_indicators.get("macd_signal", 0)
        macd_weight = 0.25
        macd_impact = "bullish" if macd > macd_signal else "bearish"

        factors.append(
            {
                "timestamp": timestamp,
                "symbol": symbol,
                "factor_name": "macd",
                "factor_value": macd - macd_signal,
                "factor_weight": macd_weight,
                "factor_impact": macd_impact,
            }
        )

        # Bollinger Bands factor
        bb_position = technical_indicators.get("bb_position", 0.5)
        bb_weight = 0.15
        bb_impact = (
            "oversold" if bb_position < 0.2 else "overbought" if bb_position > 0.8 else "neutral"
        )

        factors.append(
            {
                "timestamp": timestamp,
                "symbol": symbol,
                "factor_name": "bollinger_position",
                "factor_value": bb_position,
                "factor_weight": bb_weight,
                "factor_impact": bb_impact,
            }
        )

    # Sentiment factor
    if sentiment_data:
        sentiment = sentiment_data["overall_sentiment"]
        sentiment_weight = 0.1
        sentiment_impact = sentiment_data["sentiment_trend"]

        factors.append(
            {
                "timestamp": timestamp,
                "symbol": symbol,
                "factor_name": "market_sentiment",
                "factor_value": sentiment,
                "factor_weight": sentiment_weight,
                "factor_impact": sentiment_impact,
            }
        )

    return factors


def generate_ai_explanation(
    symbol: str, trade_type: str, factors: List[Dict], market_context: str
) -> str:
    """Generate AI-powered trade explanation"""
    try:
        # Prepare factor summary
        factor_summary = []
        for factor in factors:
            factor_summary.append(
                f"{factor['factor_name']}: {factor['factor_value']:.3f} ({factor['factor_impact']})"
            )

        prompt = f"""Analyze this {trade_type} trade for {symbol} and provide a comprehensive explanation.

Market Context: {market_context}

Key Factors:
{chr(10).join(factor_summary)}

Please provide a detailed explanation including:
1. Why this trade was executed
2. Key factors that influenced the decision
3. Risk assessment
4. Expected otimezone.utcome
5. Market conditions that supported this trade

Write in a clear, professional tone suitable for traders."""

        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500,
        )

        return response.choices[0].message.content

    except Exception as e:
        print(f"AI explanation generation error: {e}")
        return f"Trade explanation for {symbol} {trade_type}: Based on technical analysis and market conditions."


def assess_risk(factors: List[Dict], trade_type: str) -> str:
    """Assess trade risk based on factors"""
    try:
        # Calculate risk score
        risk_score = 0
        total_weight = 0

        for factor in factors:
            weight = factor["factor_weight"]
            value = abs(factor["factor_value"])

            # Normalize value to 0-1 scale
            if factor["factor_name"] == "rsi":
                normalized_value = min(value / 100, 1.0)
            elif factor["factor_name"] == "price_momentum":
                normalized_value = min(value / 10, 1.0)
            else:
                normalized_value = min(value, 1.0)

            risk_score += weight * normalized_value
            total_weight += weight

        avg_risk = risk_score / total_weight if total_weight > 0 else 0.5

        # Categorize risk
        if avg_risk < 0.3:
            risk_level = "Low"
        elif avg_risk < 0.6:
            risk_level = "Medium"
        else:
            risk_level = "High"

        return f"{risk_level} risk (score: {avg_risk:.3f})"

    except Exception as e:
        print(f"Risk assessment error: {e}")
        return "Medium risk (assessment error)"


def explain_trade_enhanced():
    """Enhanced trade explanation with all features"""
    try:
        # Simulate trade data (in real implementation, this would come from trade logs)
        trades = [
            {
                "symbol": "BTCUSDT",
                "trade_type": "BUY",
                "entry_price": 45000.0,
                "quantity": 0.1,
                "timestamp": datetime.timezone.utcnow().isoformat(),
            },
            {
                "symbol": "ETHUSDT",
                "trade_type": "SELL",
                "entry_price": 3200.0,
                "quantity": 1.0,
                "timestamp": datetime.timezone.utcnow().isoformat(),
            },
        ]

        db = ExplainerDatabase(EXPLAINER_DB)

        for trade in trades:
            # Analyze factors
            factors = analyze_trade_factors(
                trade["symbol"], trade["trade_type"], trade["entry_price"]
            )

            # Get market context
            market_data = get_market_data(trade["symbol"])
            market_context = f"Current price: ${market_data.get('current_price', 0):.2f}, 24h change: {market_data.get('price_change', 0):+.2f}%"

            # Generate AI explanation
            explanation_text = generate_ai_explanation(
                trade["symbol"], trade["trade_type"], factors, market_context
            )

            # Assess risk
            risk_assessment = assess_risk(factors, trade["trade_type"])

            # Calculate confidence score
            confidence_score = (
                sum(f["factor_weight"] for f in factors) / len(factors) if factors else 0.5
            )

            # Prepare explanation data
            explanation = {
                "timestamp": trade["timestamp"],
                "symbol": trade["symbol"],
                "trade_type": trade["trade_type"],
                "entry_price": trade["entry_price"],
                "quantity": trade["quantity"],
                "explanation_text": explanation_text,
                "confidence_score": confidence_score,
                "factors_analyzed": factors,
                "risk_assessment": risk_assessment,
                "market_context": market_context,
            }

            # Save to database
            db.save_trade_explanation(explanation)
            db.save_factor_analysis(factors)

            # Print results
            print(f"[Explainer] {trade['symbol']} {trade['trade_type']} Explanation:")
            print(f"[Explainer] Risk: {risk_assessment}")
            print(f"[Explainer] Confidence: {confidence_score:.3f}")
            print(f"[Explainer] Explanation: {explanation_text[:200]}...")
            print(f"[Explainer] Factors analyzed: {len(factors)}")
            print("-" * 50)

    except Exception as e:
        print(f"[Explainer] Enhanced explanation error: {e}")


# Main execution loop
async def main():
    while True:
        explain_trade_enhanced()
        await asyncio.sleep(EXPLANATION_INTERVAL)

if __name__ == "__main__":
    asyncio.run(main())


