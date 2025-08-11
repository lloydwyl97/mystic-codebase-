"""
Indicators Fetcher Service
Handles fetching technical indicators
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class IndicatorsFetcher:
    def __init__(self):
        self.available_indicators = [
            "rsi",
            "macd",
            "bollinger_bands",
            "moving_averages",
        ]

    async def fetch_indicators(
        self, symbol: str, timeframe: str = "1d"
    ) -> Optional[Dict[str, Any]]:
        """Fetch technical indicators for a given symbol"""
        try:
            # Get real technical indicators from technical analysis service
            from services.technical_analysis_service import (
                get_technical_analysis_service,
            )

            ta_service = get_technical_analysis_service()
            indicators = await ta_service.get_indicators(symbol, timeframe)

            return indicators
        except Exception as e:
            logger.error(f"Error fetching indicators for {symbol}: {e}")
            return None


# Global instance
indicators_fetcher = IndicatorsFetcher()
