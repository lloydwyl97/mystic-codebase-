"""
Exchange Integration Service

Provides exchange integration functionality for the Mystic Trading Platform.
"""

from exchange_integration import ExchangeManager, OrderRequest

# Create global exchange manager instance
exchange_manager = ExchangeManager()

__all__ = ["OrderRequest", "exchange_manager"]


