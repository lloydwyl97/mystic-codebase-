"""
Social Trading Features

Includes copy trading, leaderboards, social features, and community trading.
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class TraderRank(Enum):
    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"
    PLATINUM = "platinum"
    DIAMOND = "diamond"


class CopyMode(Enum):
    PERCENTAGE = "percentage"
    FIXED_AMOUNT = "fixed_amount"
    PROPORTIONAL = "proportional"


@dataclass
class TraderProfile:
    trader_id: str
    username: str
    display_name: str
    rank: TraderRank
    total_followers: int
    total_following: int
    total_trades: int
    win_rate: float
    total_pnl: float
    avg_monthly_return: float
    max_drawdown: float
    sharpe_ratio: float
    risk_score: float
    bio: str
    avatar_url: str
    is_verified: bool
    is_pro_trader: bool
    created_at: datetime
    last_active: datetime
    tags: List[str] = field(default_factory=list)
    achievements: List[str] = field(default_factory=list)


@dataclass
class CopyTradeConfig:
    copy_id: str
    follower_id: str
    trader_id: str
    copy_mode: CopyMode
    allocation_percentage: float
    fixed_amount: float
    max_allocation: float
    auto_copy: bool
    copy_stop_loss: bool
    copy_take_profit: bool
    risk_multiplier: float
    enabled: bool
    created_at: datetime
    last_copied: datetime


@dataclass
class TradeSignal:
    signal_id: str
    trader_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    confidence: float
    reasoning: str
    timestamp: datetime
    followers_count: int
    likes_count: int
    comments_count: int
    is_public: bool
    tags: List[str] = field(default_factory=list)


@dataclass
class LeaderboardEntry:
    trader_id: str
    username: str
    rank: int
    total_pnl: float
    win_rate: float
    total_trades: int
    sharpe_ratio: float
    followers_count: int
    period: str  # 'daily', 'weekly', 'monthly', 'all_time'


class SocialTradingManager:
    """Manages social trading features and copy trading."""

    def __init__(self):
        self.traders = {}
        self.copy_configs = {}
        self.trade_signals = {}
        self.followers = {}  # trader_id -> [follower_ids]
        self.following = {}  # follower_id -> [trader_ids]
        self.leaderboards = {}
        self.achievements = {}

    def create_trader_profile(
        self, user_id: str, username: str, display_name: str
    ) -> TraderProfile:
        """Create a new trader profile."""
        trader_id = str(uuid.uuid4())

        profile = TraderProfile(
            trader_id=trader_id,
            username=username,
            display_name=display_name,
            rank=TraderRank.BRONZE,
            total_followers=0,
            total_following=0,
            total_trades=0,
            win_rate=0.0,
            total_pnl=0.0,
            avg_monthly_return=0.0,
            max_drawdown=0.0,
            sharpe_ratio=0.0,
            risk_score=0.0,
            bio="",
            avatar_url="",
            is_verified=False,
            is_pro_trader=False,
            created_at=datetime.now(timezone.timezone.utc),
            last_active=datetime.now(timezone.timezone.utc),
        )

        self.traders[trader_id] = profile
        self.followers[trader_id] = []
        self.following[trader_id] = []

        logger.info(f"Created trader profile: {username} (ID: {trader_id})")
        return profile

    def follow_trader(self, follower_id: str, trader_id: str) -> bool:
        """Follow a trader."""
        if trader_id not in self.traders or follower_id not in self.traders:
            return False

        if follower_id not in self.followers[trader_id]:
            self.followers[trader_id].append(follower_id)
            self.traders[trader_id].total_followers += 1

        if trader_id not in self.following[follower_id]:
            self.following[follower_id].append(trader_id)
            self.traders[follower_id].total_following += 1

        logger.info(f"User {follower_id} started following trader {trader_id}")
        return True

    def unfollow_trader(self, follower_id: str, trader_id: str) -> bool:
        """Unfollow a trader."""
        if trader_id in self.followers and follower_id in self.followers[trader_id]:
            self.followers[trader_id].remove(follower_id)
            self.traders[trader_id].total_followers -= 1

        if follower_id in self.following and trader_id in self.following[follower_id]:
            self.following[follower_id].remove(trader_id)
            self.traders[follower_id].total_following -= 1

        logger.info(f"User {follower_id} unfollowed trader {trader_id}")
        return True

    def create_copy_trade_config(
        self,
        follower_id: str,
        trader_id: str,
        copy_mode: CopyMode,
        allocation_percentage: float = 0.0,
        fixed_amount: float = 0.0,
        max_allocation: float = 1000.0,
    ) -> CopyTradeConfig:
        """Create a copy trading configuration."""
        copy_id = str(uuid.uuid4())

        config = CopyTradeConfig(
            copy_id=copy_id,
            follower_id=follower_id,
            trader_id=trader_id,
            copy_mode=copy_mode,
            allocation_percentage=allocation_percentage,
            fixed_amount=fixed_amount,
            max_allocation=max_allocation,
            auto_copy=True,
            copy_stop_loss=True,
            copy_take_profit=True,
            risk_multiplier=1.0,
            enabled=True,
            created_at=datetime.now(timezone.timezone.utc),
            last_copied=datetime.now(timezone.timezone.utc),
        )

        self.copy_configs[copy_id] = config
        logger.info(f"Created copy trade config: {copy_id}")
        return config

    def update_copy_trade_config(self, copy_id: str, **kwargs) -> bool:
        """Update copy trading configuration."""
        if copy_id not in self.copy_configs:
            return False

        config = self.copy_configs[copy_id]
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        logger.info(f"Updated copy trade config: {copy_id}")
        return True

    def disable_copy_trade(self, copy_id: str) -> bool:
        """Disable copy trading."""
        if copy_id in self.copy_configs:
            self.copy_configs[copy_id].enabled = False
            logger.info(f"Disabled copy trade config: {copy_id}")
            return True
        return False

    async def create_trade_signal(
        self,
        trader_id: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        confidence: float,
        reasoning: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        is_public: bool = True,
        tags: List[str] = None,
    ) -> TradeSignal:
        """Create a trade signal for followers to copy."""
        signal_id = str(uuid.uuid4())

        signal = TradeSignal(
            signal_id=signal_id,
            trader_id=trader_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            reasoning=reasoning,
            timestamp=datetime.now(timezone.timezone.utc),
            followers_count=0,
            likes_count=0,
            comments_count=0,
            is_public=is_public,
            tags=tags or [],
        )

        self.trade_signals[signal_id] = signal

        # Auto-copy to followers
        if is_public:
            await self._auto_copy_signal(signal)

        logger.info(f"Created trade signal: {signal_id} by trader {trader_id}")
        return signal

    async def _auto_copy_signal(self, signal: TradeSignal):
        """Automatically copy signal to followers."""
        trader_id = signal.trader_id
        followers = self.followers.get(trader_id, [])

        for follower_id in followers:
            # Find active copy configs for this trader
            copy_configs = [
                config
                for config in self.copy_configs.values()
                if config.follower_id == follower_id
                and config.trader_id == trader_id
                and config.enabled
                and config.auto_copy
            ]

            for config in copy_configs:
                try:
                    await self._execute_copy_trade(signal, config)
                except Exception as e:
                    logger.error(f"Error copying trade for follower {follower_id}: {str(e)}")

    async def _execute_copy_trade(self, signal: TradeSignal, config: CopyTradeConfig):
        """Execute a copy trade based on configuration."""
        # Calculate copy amount
        if config.copy_mode == CopyMode.PERCENTAGE:
            # This would need portfolio value from the follower
            copy_amount = 1000 * (config.allocation_percentage / 100)  # Simplified
        elif config.copy_mode == CopyMode.FIXED_AMOUNT:
            copy_amount = config.fixed_amount
        else:  # PROPORTIONAL
            copy_amount = signal.quantity * config.risk_multiplier

        # Cap at max allocation
        copy_amount = min(copy_amount, config.max_allocation)

        # Calculate copy quantity
        copy_quantity = copy_amount / signal.price

        # Create copy order
        copy_order = {
            "symbol": signal.symbol,
            "side": signal.side,
            "quantity": copy_quantity,
            "price": signal.price,
            "stop_loss": signal.stop_loss if config.copy_stop_loss else None,
            "take_profit": (signal.take_profit if config.copy_take_profit else None),
            "follower_id": config.follower_id,
            "original_signal_id": signal.signal_id,
            "copy_config_id": config.copy_id,
        }

        # Here you would actually place the order with the exchange
        logger.info(f"Executing copy trade: {copy_order}")

        # Update last copied timestamp
        config.last_copied = datetime.now(timezone.timezone.utc)

    def like_signal(self, user_id: str, signal_id: str) -> bool:
        """Like a trade signal."""
        if signal_id in self.trade_signals:
            self.trade_signals[signal_id].likes_count += 1
            logger.info(f"User {user_id} liked signal {signal_id}")
            return True
        return False

    def unlike_signal(self, user_id: str, signal_id: str) -> bool:
        """Unlike a trade signal."""
        if signal_id in self.trade_signals:
            self.trade_signals[signal_id].likes_count = max(
                0, self.trade_signals[signal_id].likes_count - 1
            )
            logger.info(f"User {user_id} unliked signal {signal_id}")
            return True
        return False

    def add_comment_to_signal(self, user_id: str, signal_id: str, comment: str) -> bool:
        """Add comment to a trade signal."""
        if signal_id in self.trade_signals:
            self.trade_signals[signal_id].comments_count += 1
            logger.info(f"User {user_id} commented on signal {signal_id}")
            return True
        return False

    def get_trader_feed(self, trader_id: str, limit: int = 50) -> List[TradeSignal]:
        """Get trade signals from followed traders."""
        following = self.following.get(trader_id, [])
        signals = []

        for followed_trader in following:
            trader_signals = [
                signal
                for signal in self.trade_signals.values()
                if signal.trader_id == followed_trader and signal.is_public
            ]
            signals.extend(trader_signals)

        # Sort by timestamp (newest first)
        signals.sort(key=lambda x: x.timestamp, reverse=True)
        return signals[:limit]

    def get_trader_signals(self, trader_id: str, limit: int = 50) -> List[TradeSignal]:
        """Get all signals from a specific trader."""
        signals = [
            signal
            for signal in self.trade_signals.values()
            if signal.trader_id == trader_id and signal.is_public
        ]
        signals.sort(key=lambda x: x.timestamp, reverse=True)
        return signals[:limit]


class LeaderboardManager:
    """Manages leaderboards and rankings."""

    def __init__(self):
        self.leaderboards = {
            "daily": [],
            "weekly": [],
            "monthly": [],
            "all_time": [],
        }
        self.last_updated = {}

    def update_leaderboard(self, period: str = "all_time"):
        """Update leaderboard for a specific period."""
        if period not in self.leaderboards:
            return

        # Get real trader performance from database
        try:
            from database import get_db_connection

            conn = get_db_connection()
            cursor = conn.cursor()

            # Query real trader performance data
            cursor.execute(
                """
                SELECT
                    trader_id, username, total_pnl, win_rate, total_trades,
                    sharpe_ratio, followers_count
                FROM trader_performance
                WHERE period = ?
                ORDER BY total_pnl DESC
            """,
                (period,),
            )

            rows = cursor.fetchall()
            conn.close()

            if not rows:
                logger.warning(f"No trader performance data found for period {period}")
                return

            # Convert to leaderboard format
            sorted_traders = []
            for row in rows:
                sorted_traders.append(
                    {
                        "trader_id": row[0],
                        "username": row[1],
                        "total_pnl": row[2],
                        "win_rate": row[3],
                        "total_trades": row[4],
                        "sharpe_ratio": row[5],
                        "followers_count": row[6],
                    }
                )

        except Exception as e:
            logger.error(f"Error getting real trader performance: {e}")
            return

        # Create leaderboard entries
        leaderboard = []
        for rank, trader in enumerate(sorted_traders, 1):
            entry = LeaderboardEntry(
                trader_id=trader["trader_id"],
                username=trader["username"],
                rank=rank,
                total_pnl=trader["total_pnl"],
                win_rate=trader["win_rate"],
                total_trades=trader["total_trades"],
                sharpe_ratio=trader["sharpe_ratio"],
                followers_count=trader["followers_count"],
                period=period,
            )
            leaderboard.append(entry)

        self.leaderboards[period] = leaderboard
        self.last_updated[period] = datetime.now(timezone.timezone.utc)

        logger.info(f"Updated {period} leaderboard with {len(leaderboard)} traders")

    def get_leaderboard(self, period: str = "all_time", limit: int = 100) -> List[LeaderboardEntry]:
        """Get leaderboard for a specific period."""
        if period not in self.leaderboards:
            return []

        return self.leaderboards[period][:limit]

    def get_trader_rank(self, trader_id: str, period: str = "all_time") -> Optional[int]:
        """Get rank of a specific trader."""
        leaderboard = self.leaderboards.get(period, [])
        for entry in leaderboard:
            if entry.trader_id == trader_id:
                return entry.rank
        return None


class AchievementSystem:
    """Gamification system with achievements and badges."""

    def __init__(self):
        self.achievements = {
            "first_trade": {
                "name": "First Trade",
                "description": "Complete your first trade",
                "icon": "ðŸŽ¯",
                "points": 10,
            },
            "profit_master": {
                "name": "Profit Master",
                "description": "Achieve 100% profit",
                "icon": "ðŸ’°",
                "points": 50,
            },
            "winning_streak": {
                "name": "Winning Streak",
                "description": "Win 10 trades in a row",
                "icon": "ðŸ”¥",
                "points": 100,
            },
            "social_butterfly": {
                "name": "Social Butterfly",
                "description": "Gain 100 followers",
                "icon": "ðŸ¦‹",
                "points": 25,
            },
            "copy_trader": {
                "name": "Copy Trader",
                "description": "Copy your first trade",
                "icon": "ðŸ“‹",
                "points": 15,
            },
            "risk_manager": {
                "name": "Risk Manager",
                "description": "Maintain positive PnL for 30 days",
                "icon": "ðŸ›¡ï¸",
                "points": 75,
            },
            "volume_king": {
                "name": "Volume King",
                "description": "Trade $100,000 in volume",
                "icon": "ðŸ“Š",
                "points": 200,
            },
            "diamond_hands": {
                "name": "Diamond Hands",
                "description": "Hold a position for 30 days",
                "icon": "ðŸ’Ž",
                "points": 30,
            },
        }

        self.user_achievements = {}  # user_id -> [achievement_ids]
        self.user_points = {}  # user_id -> total_points

    def check_achievements(self, user_id: str, user_stats: Dict[str, Any]) -> List[str]:
        """Check and award achievements based on user stats."""
        earned_achievements = []

        # Check first trade
        if user_stats.get(
            "total_trades", 0
        ) >= 1 and "first_trade" not in self.user_achievements.get(user_id, []):
            earned_achievements.append("first_trade")

        # Check profit master
        if user_stats.get(
            "total_pnl", 0
        ) >= 100 and "profit_master" not in self.user_achievements.get(user_id, []):
            earned_achievements.append("profit_master")

        # Check winning streak
        if user_stats.get(
            "current_winning_streak", 0
        ) >= 10 and "winning_streak" not in self.user_achievements.get(user_id, []):
            earned_achievements.append("winning_streak")

        # Check social butterfly
        if user_stats.get(
            "total_followers", 0
        ) >= 100 and "social_butterfly" not in self.user_achievements.get(user_id, []):
            earned_achievements.append("social_butterfly")

        # Award achievements
        for achievement_id in earned_achievements:
            self._award_achievement(user_id, achievement_id)

        return earned_achievements

    def _award_achievement(self, user_id: str, achievement_id: str):
        """Award an achievement to a user."""
        if user_id not in self.user_achievements:
            self.user_achievements[user_id] = []

        if achievement_id not in self.user_achievements[user_id]:
            self.user_achievements[user_id].append(achievement_id)

            # Award points
            points = self.achievements[achievement_id]["points"]
            self.user_points[user_id] = self.user_points.get(user_id, 0) + points

            logger.info(
                f"Awarded achievement '{achievement_id}' to user {user_id} (+{points} points)"
            )

    def get_user_achievements(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all achievements for a user."""
        user_achievements = self.user_achievements.get(user_id, [])
        return [self.achievements[achievement_id] for achievement_id in user_achievements]

    def get_user_points(self, user_id: str) -> int:
        """Get total points for a user."""
        return self.user_points.get(user_id, 0)


# Global instances
social_trading_manager = SocialTradingManager()
leaderboard_manager = LeaderboardManager()
achievement_system = AchievementSystem()


