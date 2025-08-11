import os
import openai
import time
import json
import requests
import hashlib
import random
from datetime import datetime
from typing import Dict, Optional
import ast

STRATEGY_DIR = "./generated_modules"
INTERVAL_HOURS = 3
openai.api_key = os.getenv("OPENAI_API_KEY")

# Enhanced configuration
STRATEGY_TEMPLATES = {
    "rsi_strategy": (
        """
def rsi_strategy(df):
    import pandas as pd

    # Calculate RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # Generate signals
    df['signal'] = 0
    df.loc[df['rsi'] < 30, 'signal'] = 1  # Buy when oversold
    df.loc[df['rsi'] > 70, 'signal'] = -1  # Sell when overbought

    return df
"""
    ),
    "macd_strategy": (
        """
def macd_strategy(df):
    import pandas as pd

    # Calculate MACD
    ema_fast = df['close'].ewm(span=12).mean()
    ema_slow = df['close'].ewm(span=26).mean()
    df['macd'] = ema_fast - ema_slow
    df['macd_signal'] = df['macd'].ewm(span=9).mean()

    # Generate signals
    df['signal'] = 0
    df.loc[df['macd'] > df['macd_signal'], 'signal'] = 1  # Buy on crossover
    df.loc[df['macd'] < df['macd_signal'], 'signal'] = -1  # Sell on crossover

    return df
"""
    ),
    "bollinger_strategy": (
        """
def bollinger_strategy(df):
    import pandas as pd

    # Calculate Bollinger Bands
    sma = df['close'].rolling(window=20).mean()
    std = df['close'].rolling(window=20).std()
    df['bb_upper'] = sma + (std * 2)
    df['bb_lower'] = sma - (std * 2)

    # Generate signals
    df['signal'] = 0
    df.loc[df['close'] < df['bb_lower'], 'signal'] = 1  # Buy at lower band
    df.loc[df['close'] > df['bb_upper'], 'signal'] = -1  # Sell at upper band

    return df
"""
    ),
}


def get_live_market_data() -> Dict[str, float]:
    """Get live market data for context"""
    try:
        # Binance API for live prices
        response = requests.get("https://api.binance.us/api/v3/ticker/24hr", timeout=10)
        if response.status_code == 200:
            data = response.json()
            # Get top cryptocurrencies
            top_coins = {}
            for item in data:
                if item["symbol"] in [
                    "BTCUSDT",
                    "ETHUSDT",
                    "ADAUSDT",
                    "DOTUSDT",
                ]:
                    top_coins[item["symbol"]] = {
                        "price": float(item["lastPrice"]),
                        "change": float(item["priceChangePercent"]),
                        "volume": float(item["volume"]),
                    }
            return top_coins
    except Exception as e:
        print(f"Market data fetch error: {e}")
    return {}


def validate_strategy_code(code: str) -> bool:
    """Validate generated strategy code"""
    try:
        # Basic syntax check
        ast.parse(code)

        # Check for required imports
        if "import pandas" not in code:
            return False

        # Check for function definition
        if "def " not in code:
            return False

        # Check for signal generation
        if "signal" not in code:
            return False

        return True
    except SyntaxError:
        return False


def generate_strategy_hash(code: str) -> str:
    """Generate hash for strategy versioning"""
    return hashlib.md5(code.encode()).hexdigest()[:8]


def save_strategy_version(code: str, filename: str, metadata: Dict) -> None:
    """Save strategy with version control"""
    # Save the strategy file
    filepath = os.path.join(STRATEGY_DIR, filename)
    with open(filepath, "w") as f:
        f.write(code)

    # Save metadata
    metadata_file = filepath.replace(".py", "_metadata.json")
    metadata["hash"] = generate_strategy_hash(code)
    metadata["created_at"] = datetime.timezone.utcnow().isoformat()
    metadata["filename"] = filename

    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)


def run_basic_backtest(filename: str) -> Dict:
    """Run real backtest on generated strategy using live historical data"""
    try:
        import pandas as pd

        # Fetch historical price data from Binance
        symbol = "BTCUSDT"
        url = f"https://api.binance.us/api/v3/klines?symbol={symbol}&interval=1h&limit=500"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        klines = response.json()
        closes = [float(k[4]) for k in klines]
        df = pd.DataFrame({"close": closes})
        # Dynamically import the generated strategy
        import importlib.util
        import os

        strategy_path = os.path.join(STRATEGY_DIR, filename)
        spec = importlib.util.spec_from_file_location("strategy_mod", strategy_path)
        strategy_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(strategy_mod)
        # Run the strategy function
        df = strategy_mod.rsi_strategy(df) if hasattr(strategy_mod, "rsi_strategy") else df
        # Calculate winrate and profit
        trades = df["signal"].diff().fillna(0)
        buy_signals = trades[trades == 1].index
        sell_signals = trades[trades == -1].index
        profit = 0
        for buy, sell in zip(buy_signals, sell_signals):
            profit += df["close"][sell] - df["close"][buy]
        winrate = len(buy_signals) / max(len(trades), 1)
        return {
            "winrate": winrate,
            "total_trades": len(buy_signals),
            "profit": profit,
            "max_drawdown": float(df["close"].max() - df["close"].min()),
            "sharpe_ratio": 0,  # Placeholder, can be calculated with returns
            "backtest_date": datetime.timezone.utcnow().isoformat(),
        }
    except Exception as e:
        return {"error": str(e)}


def generate_prompt(template: Optional[str] = None) -> str:
    """Generate enhanced prompt with market context"""
    market_data = get_live_market_data()
    market_context = ""

    if market_data:
        context_parts = []
        for symbol, data in market_data.items():
            context_parts.append(f"{symbol}: ${data['price']:.2f} ({data['change']:+.2f}%)")
        market_context = f"Current market: {', '.join(context_parts)}"

    base_prompt = f"""You are an advanced crypto quant trader. Create a new Python trading strategy function.

Market Context: {market_context}

Requirements:
- Use pandas for technical indicators (implement RSI, MACD, Bollinger Bands manually)
- Include proper error handling
- Return buy/sell signals based on technical analysis
- Include position sizing logic
- Add stop-loss and take-profit levels
- Use modern Python syntax (f-strings, type hints)
- Strategy should be profitable and risk-managed

Return only the complete Python function with no explanations."""

    if template:
        base_prompt += f"\n\nUse this template as inspiration:\n{template}"

    return base_prompt


def generate_strategy_enhanced():
    """Enhanced strategy generation with all features"""
    try:
        # Get market context
        market_data = get_live_market_data()

        # Choose template randomly
        template = None
        if random.random() < 0.3:  # 30% chance to use template
            template_name = random.choice(list(STRATEGY_TEMPLATES.keys()))
            template = STRATEGY_TEMPLATES[template_name]

        # Generate strategy
        prompt = generate_prompt(template)
        client = openai.OpenAI()
        res = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1000,
        )

        code = res.choices[0].message.content

        # Validate code
        if not validate_strategy_code(code):
            print("[LLM] Generated code failed validation, retrying...")
            return

        # Create filename with timestamp
        ts = datetime.timezone.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"strategy_llm_{ts}.py"

        # Prepare metadata
        metadata = {
            "model": "gpt-4",
            "market_context": market_data,
            "template_used": template is not None,
            "validation_passed": True,
            "prompt_length": len(prompt),
        }

        # Save strategy with version control
        save_strategy_version(code, filename, metadata)

        # Run basic backtest
        backtest_results = run_basic_backtest(filename)
        metadata["backtest_results"] = backtest_results

        # Update metadata with backtest results
        metadata_file = os.path.join(STRATEGY_DIR, filename.replace(".py", "_metadata.json"))
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"[LLM] Enhanced strategy saved: {filename}")
        print(f"[LLM] Backtest results: Winrate {backtest_results.get('winrate', 0):.3f}")

    except Exception as e:
        print(f"[LLM] Enhanced generation error: {e}")


# Main execution loop

while True:
    try:
        generate_strategy_enhanced()
    except Exception as e:
        print(f"[LLM] Enhanced Gen Error: {e}")
    time.sleep(INTERVAL_HOURS * 3600)
