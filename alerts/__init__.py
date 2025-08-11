"""
Alerts Module
============

This module provides alert and notification functionality for the Mystic AI Trading Platform.
"""

__version__ = "1.0.0"
__author__ = "Mystic AI Team"

from .notify_bot import AlertManager, NotificationBot

__all__ = ["AlertManager", "NotificationBot"]
