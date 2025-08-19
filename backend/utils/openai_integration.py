"""
OpenAI Integration for Strategy Descriptions
===========================================

Generates rich, human-readable descriptions for evolved strategies using OpenAI's GPT models.
"""

import json
import logging
import os
from typing import Any

import requests

logger = logging.getLogger(__name__)

# OpenAI API configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
MODEL = "gpt-3.5-turbo"  # Can be upgraded to gpt-4 for better descriptions


def generate_openai_description(
    strategy: dict[str, Any],
    parent: str = "",
    backtest_results: dict[str, Any] = None,
) -> str:
    """
    Generate a rich strategy description using OpenAI GPT.

    Args:
        strategy: Strategy configuration dictionary
        parent: Parent strategy filename (if evolved from another)
        backtest_results: Backtest results if available

    Returns:
        Rich description of the strategy
    """
    if not OPENAI_API_KEY:
        logger.warning("OpenAI API key not configured, using fallback description")
        return generate_fallback_description(strategy, parent)

    try:
        # Prepare the prompt
        prompt = create_strategy_prompt(strategy, parent, backtest_results)

        # Call OpenAI API
        response = call_openai_api(prompt)

        if response and response.get("choices"):
            description = response["choices"][0]["message"]["content"].strip()
            return description
        else:
            logger.error("OpenAI API returned invalid response")
            return generate_fallback_description(strategy, parent)

    except Exception as e:
        logger.error(f"OpenAI API call failed: {e}")
        return generate_fallback_description(strategy, parent)


def create_strategy_prompt(
    strategy: dict[str, Any], parent: str, backtest_results: dict[str, Any]
) -> str:
    """Create a detailed prompt for OpenAI to generate strategy description."""

    strategy_type = strategy.get("strategy_type", "unknown")
    params = strategy.get("parameters", strategy)

    prompt = f"""You are an expert quantitative trading analyst. Analyze this trading strategy and provide a clear, professional description.

Strategy Type: {strategy_type}
Parameters: {json.dumps(params, indent=2)}

{f"Parent Strategy: {parent}" if parent else "This is a newly generated strategy (no parent)"}

{f"Backtest Results: Win Rate: {backtest_results.get('win_rate', 0):.1%}, "
 f"Profit: {backtest_results.get('total_profit', 0):.2f}%, "
 f"Trades: {backtest_results.get('num_trades', 0)}" if backtest_results else ""}

Please provide a 2-3 sentence description that explains:
1. What this strategy does
2. Key parameters and their significance
3. Expected market conditions it performs well in
4. Any notable improvements from the parent strategy (if applicable)

Write in a professional, technical tone suitable for trading documentation."""

    return prompt


def call_openai_api(prompt: str) -> dict[str, Any] | None:
    """Make API call to OpenAI."""

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert quantitative trading analyst specializing in "
                    "algorithmic trading strategies."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 200,
        "temperature": 0.7,
    }

    try:
        response = requests.post(
            OPENAI_API_URL, headers=headers, json=data, timeout=30
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"OpenAI API request failed: {e}")
        return None


def generate_fallback_description(strategy: dict[str, Any], parent: str = "") -> str:
    """Generate a fallback description when OpenAI is not available."""

    strategy_type = strategy.get("strategy_type", "unknown")
    params = strategy.get("parameters", strategy)

    desc = f"This {strategy_type} strategy uses "

    if strategy_type == "breakout":
        desc += f"a {params.get('lookback_period', 'N/A')}-period lookback to detect breakouts with {params.get('entry_threshold', 'N/A')}x threshold. "
        desc += f"Stop loss at {params.get('stop_loss', 'N/A')}% and take profit at {params.get('take_profit', 'N/A')}%. "
    elif strategy_type == "ema_crossover":
        desc += f"EMA crossover with fast EMA ({params.get('fast_ema', 'N/A')}) and slow EMA ({params.get('slow_ema', 'N/A')}). "
        desc += f"Risk management: {params.get('stop_loss', 'N/A')}% stop loss, {params.get('take_profit', 'N/A')}% take profit. "
    elif strategy_type == "rsi_threshold":
        desc += f"RSI thresholds: buy below {params.get('rsi_buy', 'N/A')}, sell above {params.get('rsi_sell', 'N/A')}. "
        desc += f"Lookback period: {params.get('lookback_period', 'N/A')}. "
    else:
        desc += f"parameters: {', '.join(f'{k}={v}' for k, v in params.items())}. "

    if parent:
        desc += f"Evolved from {parent} with AI-optimized parameters. "

    desc += "Auto-generated by the Mystic AI Evolution Engine."

    return desc


def is_openai_available() -> bool:
    """Check if OpenAI integration is available."""
    return bool(OPENAI_API_KEY)


# Global function for easy access
def generate_strategy_description(
    strategy: dict[str, Any],
    parent: str = "",
    backtest_results: dict[str, Any] = None,
) -> str:
    """Main function to generate strategy description."""
    return generate_openai_description(strategy, parent, backtest_results)


