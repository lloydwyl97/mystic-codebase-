"""
Tokenomics Analyzer Module

Provides utilities for analyzing token economics and calculating risk scores.
Integrates with the main trading system for token evaluation and risk assessment.
"""

import logging
from typing import Any

# Configure logging
logger = logging.getLogger(__name__)


def evaluate_tokenomics(
    emission_rate: float,
    unlock_schedule: list[float],
    supply_cap: float,
    current_supply: float,
) -> float:
    """
    Evaluate tokenomics and calculate a risk score.

    Args:
        emission_rate: Annual token emission rate
        unlock_schedule: List of scheduled token unlocks
        supply_cap: Maximum total supply
        current_supply: Current circulating supply

    Returns:
        float: Risk score (0-100, higher is better)

    Raises:
        ValueError: If inputs are invalid
    """
    try:
        # Validate inputs
        if emission_rate < 0:
            raise ValueError("Emission rate must be non-negative")

        if not unlock_schedule or any(x < 0 for x in unlock_schedule):
            raise ValueError("Unlock schedule must contain non-negative values")

        if supply_cap <= 0:
            raise ValueError("Supply cap must be positive")

        if current_supply <= 0 or current_supply > supply_cap:
            raise ValueError("Current supply must be positive and not exceed supply cap")

        # Original logic preserved
        inflation = (emission_rate / current_supply) * 100
        unlock_risk = sum(unlock_schedule) / supply_cap
        score = 100 - (inflation * 2 + unlock_risk * 100)

        # Ensure score is within bounds
        score = max(0, min(100, score))

        final_score = round(score, 2)

        logger.info(f"[TOKENOMICS] Calculated risk score: {final_score}/100")
        logger.debug(f"[TOKENOMICS] Inflation: {inflation:.2f}%, Unlock risk: {unlock_risk:.4f}")

        return final_score

    except Exception as e:
        error_msg = f"[TOKENOMICS] Error evaluating tokenomics: {str(e)}"
        logger.error(error_msg)
        raise


def analyze_token_metrics(
    token_symbol: str,
    emission_rate: float,
    unlock_schedule: list[float],
    supply_cap: float,
    current_supply: float,
    market_cap: float | None = None,
) -> dict[str, Any]:
    """
    Comprehensive tokenomics analysis.

    Args:
        token_symbol: Token symbol/name
        emission_rate: Annual token emission rate
        unlock_schedule: List of scheduled token unlocks
        supply_cap: Maximum total supply
        current_supply: Current circulating supply
        market_cap: Current market capitalization (optional)

    Returns:
        dict: Comprehensive tokenomics analysis
    """
    try:
        logger.info(f"[TOKENOMICS] Analyzing tokenomics for {token_symbol}")

        # Calculate risk score
        risk_score = evaluate_tokenomics(emission_rate, unlock_schedule, supply_cap, current_supply)

        # Calculate additional metrics
        inflation_rate = (emission_rate / current_supply) * 100
        unlock_risk = sum(unlock_schedule) / supply_cap
        circulating_ratio = current_supply / supply_cap

        # Determine risk level
        if risk_score >= 80:
            risk_level = "LOW"
        elif risk_score >= 60:
            risk_level = "MEDIUM"
        elif risk_score >= 40:
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"

        analysis = {
            "token_symbol": token_symbol,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "inflation_rate": round(inflation_rate, 2),
            "unlock_risk": round(unlock_risk, 4),
            "circulating_ratio": round(circulating_ratio, 4),
            "emission_rate": emission_rate,
            "supply_cap": supply_cap,
            "current_supply": current_supply,
            "market_cap": market_cap,
            "analysis_timestamp": (
                logger.handlers[0].formatter.formatTime(
                    logging.LogRecord("", 0, "", 0, "", (), None)
                )
                if logger.handlers
                else None
            ),
        }

        logger.info(f"[TOKENOMICS] Analysis completed for {token_symbol}: {risk_level} risk")
        return analysis

    except Exception as e:
        error_msg = f"[TOKENOMICS] Error analyzing {token_symbol}: {str(e)}"
        logger.error(error_msg)
        return {
            "token_symbol": token_symbol,
            "error": str(e),
            "risk_score": 0,
            "risk_level": "ERROR",
        }


def compare_tokenomics(tokens: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Compare tokenomics across multiple tokens.

    Args:
        tokens: List of token analysis dictionaries

    Returns:
        dict: Comparison results
    """
    try:
        logger.info(f"[TOKENOMICS] Comparing {len(tokens)} tokens")

        if not tokens:
            return {"error": "No tokens provided for comparison"}

        # Sort by risk score (best first)
        sorted_tokens = sorted(tokens, key=lambda x: x.get("risk_score", 0), reverse=True)

        # Calculate averages
        avg_risk_score = sum(t.get("risk_score", 0) for t in tokens) / len(tokens)
        avg_inflation = sum(t.get("inflation_rate", 0) for t in tokens) / len(tokens)

        comparison = {
            "total_tokens": len(tokens),
            "average_risk_score": round(avg_risk_score, 2),
            "average_inflation_rate": round(avg_inflation, 2),
            "best_token": sorted_tokens[0] if sorted_tokens else None,
            "worst_token": sorted_tokens[-1] if sorted_tokens else None,
            "ranked_tokens": sorted_tokens,
        }

        logger.info(
            f"[TOKENOMICS] Comparison completed. Best: {comparison['best_token']['token_symbol'] if comparison['best_token'] else 'N/A'}"
        )
        return comparison

    except Exception as e:
        error_msg = f"[TOKENOMICS] Error comparing tokens: {str(e)}"
        logger.error(error_msg)
        return {"error": str(e)}


# Integration with main trading system
def integrate_with_trading_system() -> bool:
    """
    Integrate tokenomics analyzer with the main trading system.

    Returns:
        bool: True if integration successful
    """
    try:
        logger.info("[TOKENOMICS] Integrating with trading system...")

        # Test the analyzer
        test_analysis = analyze_token_metrics(
            "TEST",
            emission_rate=1000000,
            unlock_schedule=[1e6, 2e6, 2e6],
            supply_cap=1e9,
            current_supply=5e8,
        )

        logger.info(f"[TOKENOMICS] Integration test completed: {test_analysis['risk_score']}/100")
        return True

    except Exception as e:
        logger.error(f"[TOKENOMICS] Integration failed: {str(e)}")
        return False


if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Test the tokenomics analyzer
    score = evaluate_tokenomics(
        emission_rate=1000000,
        unlock_schedule=[1e6, 2e6, 2e6],
        supply_cap=1e9,
        current_supply=5e8,
    )
    print(f"[TOKENOMICS] Risk Score: {score}/100")

    # Test comprehensive analysis
    analysis = analyze_token_metrics(
        "TEST_TOKEN",
        emission_rate=1000000,
        unlock_schedule=[1e6, 2e6, 2e6],
        supply_cap=1e9,
        current_supply=5e8,
    )
    print(f"[TOKENOMICS] Analysis: {analysis}")


