"""
Pydantic Models for Mystic Trading

Simple data models for market data and trading signals without database dependencies.
"""

import uuid
from datetime import datetime, timezone

from pydantic import BaseModel, Field


class MarketBase(BaseModel):
    """Base market data model."""

    model_config = {"protected_namespaces": ("settings_",)}

    name: str = Field(..., description="Market name")
    symbol: str = Field(..., description="Market symbol")
    price: float = Field(..., description="Current price")
    change_24h: float = Field(..., description="24-hour price change")


class Market(MarketBase):
    """Market model with ID and timestamps."""

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique market ID",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last update timestamp",
    )


class MarketUpdate(BaseModel):
    """Market update model."""

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique update ID",
    )
    market_id: str = Field(..., description="Associated market ID")
    price: float = Field(..., description="Updated price")
    change_24h: float = Field(..., description="24-hour price change")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Update timestamp",
    )


class Indicator(BaseModel):
    """Technical indicator model."""

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique indicator ID",
    )
    market_id: str = Field(..., description="Associated market ID")
    name: str = Field(..., description="Indicator name")
    description: str = Field(..., description="Indicator description")
    timeframes: str = Field(..., description="Supported timeframes as JSON string")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last update timestamp",
    )


class TradingSignal(BaseModel):
    """Trading signal model."""

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique signal ID",
    )
    symbol: str = Field(..., description="Trading symbol")
    signal_type: str = Field(..., description="Signal type (buy/sell/hold)")
    price: float | None = Field(None, description="Signal price")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Signal timestamp",
    )
    confidence: float | None = Field(None, description="Signal confidence (0-1)")


class UserSettings(BaseModel):
    """User settings model."""

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique settings ID",
    )
    user_id: str = Field(..., description="User ID")
    setting_name: str = Field(..., description="Setting name")
    setting_value: str | None = Field(None, description="Setting value")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last update timestamp",
    )


