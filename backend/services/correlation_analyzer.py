"""
Advanced Correlation Analysis System
Tracks relationships between assets and sectors for diversification
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict

logger = logging.getLogger(__name__)


@dataclass
class CorrelationSignal:
    """Correlation analysis result"""

    btc_correlation: float
    eth_correlation: float
    gold_correlation: float
    sp500_correlation: float
    dollar_correlation: float
    oil_correlation: float
    sector_correlation: float
    overall_correlation_score: float
    diversification_opportunity: bool
    hedge_recommendation: str
    timestamp: datetime


class AdvancedCorrelationAnalyzer:
    """Advanced correlation analysis for crypto markets"""

    def __init__(self):
        self.correlation_history: Dict[str, Any] = {}
        self.sector_weights = {
            "defi": 0.25,
            "layer1": 0.30,
            "layer2": 0.15,
            "meme": 0.10,
            "gaming": 0.10,
            "privacy": 0.10,
        }

    async def analyze_correlations(self, symbol: str, price: float) -> CorrelationSignal:
        """Analyze correlations with major assets and indicators"""
        try:
            # Get real correlation data from correlation service
            from backend.services.correlation_service import get_correlation_service

            correlation_service = get_correlation_service()
            correlations = await correlation_service.get_correlations(symbol, price)

            # Calculate overall correlation score (lower is better for diversification)
            overall_score = (
                abs(correlations.get("btc_correlation", 0.5)) * 0.3
                + abs(correlations.get("eth_correlation", 0.5)) * 0.25
                + abs(correlations.get("sp500_correlation", 0.0)) * 0.2
                + abs(correlations.get("dollar_correlation", -0.3)) * 0.15
                + abs(correlations.get("oil_correlation", 0.0)) * 0.1
            )

            # Diversification opportunity
            diversification_opportunity = overall_score < 0.4

            # Hedge recommendation
            hedge_recommendation = self._generate_hedge_recommendation(
                correlations.get("btc_correlation", 0.5),
                correlations.get("eth_correlation", 0.5),
                correlations.get("sp500_correlation", 0.0),
                correlations.get("dollar_correlation", -0.3),
            )

            return CorrelationSignal(
                btc_correlation=correlations.get("btc_correlation", 0.5),
                eth_correlation=correlations.get("eth_correlation", 0.5),
                gold_correlation=correlations.get("gold_correlation", 0.0),
                sp500_correlation=correlations.get("sp500_correlation", 0.0),
                dollar_correlation=correlations.get("dollar_correlation", -0.3),
                oil_correlation=correlations.get("oil_correlation", 0.0),
                sector_correlation=correlations.get("sector_correlation", 0.5),
                overall_correlation_score=overall_score,
                diversification_opportunity=diversification_opportunity,
                hedge_recommendation=hedge_recommendation,
                timestamp=datetime.now(),
            )

        except Exception as e:
            logger.error(f"Error in correlation analysis: {e}")
            return CorrelationSignal(
                btc_correlation=0.5,
                eth_correlation=0.5,
                gold_correlation=0.0,
                sp500_correlation=0.0,
                dollar_correlation=-0.3,
                oil_correlation=0.0,
                sector_correlation=0.5,
                overall_correlation_score=0.5,
                diversification_opportunity=False,
                hedge_recommendation="Hold - Insufficient data",
                timestamp=datetime.now(),
            )

    def _calculate_sector_correlation(self, symbol: str) -> float:
        """Calculate correlation within the same sector"""
        try:
            # Get real sector correlation from sector analysis service
            from backend.services.sector_analysis_service import (
                get_sector_analysis_service,
            )

            sector_service = get_sector_analysis_service()
            sector = self._get_symbol_sector(symbol)
            correlation = sector_service.get_sector_correlation(sector)

            return correlation

        except Exception as e:
            logger.error(f"Error calculating sector correlation: {e}")
            return 0.5

    def _get_symbol_sector(self, symbol: str) -> str:
        """Get sector for a given symbol"""
        symbol_lower = symbol.lower()

        # DeFi tokens
        if any(token in symbol_lower for token in ["uni", "aave", "comp", "sushi", "curve"]):
            return "defi"
        # Layer 1
        elif any(token in symbol_lower for token in ["btc", "eth", "ada", "sol", "dot"]):
            return "layer1"
        # Layer 2
        elif any(token in symbol_lower for token in ["matic", "arb", "op", "imx"]):
            return "layer2"
        # Meme coins
        elif any(token in symbol_lower for token in ["doge", "shib", "pepe", "floki"]):
            return "meme"
        # Gaming
        elif any(token in symbol_lower for token in ["axs", "mana", "sand", "enj"]):
            return "gaming"
        # Privacy
        elif any(token in symbol_lower for token in ["xmr", "zec", "dash"]):
            return "privacy"
        else:
            return "defi"  # Default

    def _generate_hedge_recommendation(
        self,
        btc_corr: float,
        eth_corr: float,
        sp500_corr: float,
        dollar_corr: float,
    ) -> str:
        """Generate hedging recommendation based on correlations"""
        if btc_corr > 0.8 and eth_corr > 0.7:
            return "Consider stablecoins or gold for BTC/ETH correlation hedge"
        elif sp500_corr > 0.5:
            return "Consider crypto as hedge against traditional markets"
        elif dollar_corr < -0.5:
            return "Strong USD hedge - good for portfolio diversification"
        elif btc_corr < 0.4 and eth_corr < 0.4:
            return "Low correlation with majors - good diversification"
        else:
            return "Standard crypto allocation recommended"

    def get_correlation_summary(self, symbol: str) -> Dict[str, Any]:
        """Get correlation analysis summary"""
        return {
            "symbol": symbol,
            "sector": self._get_symbol_sector(symbol),
            "sector_weight": self.sector_weights.get(self._get_symbol_sector(symbol), 0.1),
            "analysis_active": True,
            "hedge_tracking": True,
            "diversification_scoring": True,
        }


# Global correlation analyzer instance
correlation_analyzer = AdvancedCorrelationAnalyzer()


