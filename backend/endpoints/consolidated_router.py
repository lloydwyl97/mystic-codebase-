"""
Consolidated Router
Single router loading system that organizes all endpoints properly
Replaces the chaotic multiple router loading systems
"""

import logging

from fastapi import APIRouter

# settings may be used by downstream includes; keep import local when needed

logger = logging.getLogger(__name__)

# Create main consolidated router
consolidated_router = APIRouter()


def load_all_endpoints() -> None:
    """Load all organized endpoints into the consolidated router"""

    # Core System Endpoints
    try:
        from backend.endpoints.core.system_endpoints import router as core_router

        consolidated_router.include_router(core_router)
        logger.info("âœ… Loaded core system endpoints")
    except ImportError as e:
        logger.warning(f"âš ï¸ Core system endpoints not available: {e}")

    # Trading Endpoints
    try:
        from backend.endpoints.trading.portfolio_endpoints import (
            router as portfolio_router,
        )

        consolidated_router.include_router(portfolio_router)
        logger.info("âœ… Loaded portfolio endpoints")
    except ImportError as e:
        logger.warning(f"âš ï¸ Portfolio endpoints not available: {e}")

    try:
        from backend.endpoints.trading.autobuy_endpoints import (
            router as autobuy_router,
        )

        consolidated_router.include_router(autobuy_router)
        logger.info("âœ… Loaded autobuy endpoints")
    except ImportError as e:
        logger.warning(f"âš ï¸ Autobuy endpoints not available: {e}")

    # Market Endpoints
    try:
        from backend.endpoints.market.market_data_endpoints import (
            router as market_router,
        )

        consolidated_router.include_router(market_router)
        logger.info("âœ… Loaded market data endpoints")
    except ImportError as e:
        logger.warning(f"âš ï¸ Market data endpoints not available: {e}")

    # AI Endpoints
    try:
        from backend.endpoints.ai.ai_strategy_endpoints import router as ai_router

        consolidated_router.include_router(ai_router)
        logger.info("âœ… Loaded AI strategy endpoints")
    except ImportError as e:
        logger.warning(f"âš ï¸ AI strategy endpoints not available: {e}")

    # Experimental Endpoints
    try:
        from backend.endpoints.experimental.advanced_tech_endpoints import router as experimental_router

        consolidated_router.include_router(experimental_router)
        logger.info("âœ… Loaded experimental endpoints")
    except ImportError as e:
        logger.warning(f"âš ï¸ Experimental endpoints not available: {e}")

    # Legacy endpoints that still need to be organized
    # These will be moved to proper folders in future phases

    # Live Trading Endpoints
    try:
        from backend.endpoints.live_trading_endpoints import (
            router as live_trading_router,
        )

        consolidated_router.include_router(live_trading_router)
        logger.info("âœ… Loaded live trading endpoints")
    except ImportError as e:
        logger.warning(f"âš ï¸ Live trading endpoints not available: {e}")

    # Live Data Endpoints (orders, balances, trades)
    try:
        from backend.endpoints.live_data_endpoints import (
            router as live_data_router,
        )

        # Canonical mount
        consolidated_router.include_router(live_data_router)
        # API-prefixed alias for UI stability (/api/live/...)
        consolidated_router.include_router(live_data_router, prefix="/api")
        logger.info("âœ… Loaded live data endpoints (with /api alias)")
    except ImportError as e:
        logger.warning(f"âš ï¸ Live data endpoints not available: {e}")

    # AI Strategy Endpoints (legacy)
    try:
        from backend.endpoints.ai_strategies.ai_strategy_endpoints import (
            router as legacy_ai_router,
        )

        consolidated_router.include_router(legacy_ai_router)
        logger.info("âœ… Loaded legacy AI strategy endpoints")
    except ImportError as e:
        logger.warning(f"âš ï¸ Legacy AI strategy endpoints not available: {e}")

    # Enhanced API Endpoints
    try:
        from backend.endpoints.enhanced_api.enhanced_api_endpoints import (
            router as enhanced_api_router,
        )

        consolidated_router.include_router(enhanced_api_router)
        logger.info("âœ… Loaded enhanced API endpoints")
    except ImportError as e:
        logger.warning(f"âš ï¸ Enhanced API endpoints not available: {e}")

    # Crypto Autoengine Endpoints (router has internal '/api' prefix; may produce double-API paths when mounted)
    try:
        from crypto_autoengine_api import router as crypto_router

        consolidated_router.include_router(crypto_router)
        logger.info("âœ… Loaded crypto autoengine endpoints")
    except ImportError as e:
        logger.warning(f"âš ï¸ Crypto autoengine endpoints not available: {e}")

    # AI Supercomputing Endpoints
    try:
        from backend.endpoints.ai_supercomputing.ai_supercomputing_endpoints import (
            router as ai_supercomputing_router,
        )

        consolidated_router.include_router(ai_supercomputing_router)
        logger.info("âœ… Loaded AI supercomputing endpoints")
    except ImportError as e:
        logger.warning(f"âš ï¸ AI supercomputing endpoints not available: {e}")

    # AI Heartbeat Endpoint
    try:
        from backend.endpoints.ai.ai_heartbeat_endpoints import router as ai_heartbeat_router

        consolidated_router.include_router(ai_heartbeat_router)
        logger.info("âœ… Loaded AI heartbeat endpoint")
    except ImportError as e:
        logger.warning(f"âš ï¸ AI heartbeat endpoint not available: {e}")

    # Missing endpoints that need to be implemented
    try:
        from backend.endpoints.dashboard_missing.dashboard_missing_endpoints import (
            router as missing_router,
        )

        consolidated_router.include_router(missing_router)
        logger.info("âœ… Loaded missing dashboard endpoints")
    except ImportError as e:
        logger.warning(f"âš ï¸ Missing dashboard endpoints not available: {e}")

    # Ensure generic candles endpoint is available under main path
    try:
        # Import module then read 'router' attribute dynamically to satisfy type checkers
        from backend.endpoints.missing import missing_endpoints as missing_mod  # type: ignore[no-redef]
        generic_missing_router = getattr(missing_mod, "router", None)
        if generic_missing_router is not None:
            consolidated_router.include_router(generic_missing_router)  # type: ignore[arg-type]
            logger.info("âœ… Loaded generic missing endpoints (includes /market/candles)")
        else:
            raise ImportError("router not found in missing_endpoints")
    except ImportError as e:
        logger.warning(f"âš ï¸ Generic missing endpoints not available: {e}")

    try:
        from backend.endpoints.portfolio_missing.portfolio_missing_endpoints import (
            router as portfolio_missing_router,
        )

        consolidated_router.include_router(portfolio_missing_router)
        logger.info("âœ… Loaded missing portfolio endpoints")
    except ImportError as e:
        logger.warning(f"âš ï¸ Missing portfolio endpoints not available: {e}")

    # Experimental endpoints (legacy) - using different prefix to avoid conflicts
    try:
        from backend.endpoints.experimental_endpoints import (
            router as legacy_experimental_router,
        )

        consolidated_router.include_router(legacy_experimental_router)
        logger.info("âœ… Loaded legacy experimental endpoints")
    except ImportError as e:
        logger.warning(f"âš ï¸ Legacy experimental endpoints not available: {e}")

    # Phase 5 endpoints
    try:
        from backend.endpoints.phase5_endpoints import router as phase5_router

        consolidated_router.include_router(phase5_router)
        logger.info("âœ… Loaded Phase 5 endpoints")
    except ImportError as e:
        logger.warning(f"âš ï¸ Phase 5 endpoints not available: {e}")

    # Trading endpoints (legacy)
    try:
        from backend.endpoints.trading_endpoints import router as legacy_trading_router

        consolidated_router.include_router(legacy_trading_router)
        logger.info("âœ… Loaded legacy trading endpoints")
    except ImportError as e:
        logger.warning(f"âš ï¸ Legacy trading endpoints not available: {e}")

    # Market endpoints (legacy)
    try:
        from backend.endpoints.market_endpoints import router as legacy_market_router

        consolidated_router.include_router(legacy_market_router)
        logger.info("âœ… Loaded legacy market endpoints")
    except ImportError as e:
        logger.warning(f"âš ï¸ Legacy market endpoints not available: {e}")

    # Bot endpoints
    try:
        from backend.endpoints.bot_endpoints import router as bot_router

        consolidated_router.include_router(bot_router)
        logger.info("âœ… Loaded bot endpoints")
    except ImportError as e:
        logger.warning(f"âš ï¸ Bot endpoints not available: {e}")

    # Social endpoints
    try:
        from backend.endpoints.social_endpoints import router as social_router

        consolidated_router.include_router(social_router)
        logger.info("âœ… Loaded social endpoints")
    except ImportError as e:
        logger.warning(f"âš ï¸ Social endpoints not available: {e}")

    # Notification endpoints
    try:
        from backend.endpoints.notification_endpoints import (
            router as notification_router,
        )

        consolidated_router.include_router(notification_router)
        logger.info("âœ… Loaded notification endpoints")
    except ImportError as e:
        logger.warning(f"âš ï¸ Notification endpoints not available: {e}")

    # Analytics endpoints
    try:
        from backend.endpoints.analytics_endpoints import router as analytics_router

        consolidated_router.include_router(analytics_router)
        logger.info("âœ… Loaded analytics endpoints")
    except ImportError as e:
        logger.warning(f"âš ï¸ Analytics endpoints not available: {e}")

    # Advanced analytics endpoints
    try:
        from backend.endpoints.analytics_advanced_endpoints import (
            router as advanced_analytics_router,
        )

        consolidated_router.include_router(advanced_analytics_router)
        logger.info("âœ… Loaded advanced analytics endpoints")
    except ImportError as e:
        logger.warning(f"âš ï¸ Advanced analytics endpoints not available: {e}")

    # Signal advanced endpoints
    try:
        from backend.endpoints.signal_advanced_endpoints import (
            router as signal_advanced_router,
        )

        consolidated_router.include_router(signal_advanced_router)
        logger.info("âœ… Loaded signal advanced endpoints")
    except ImportError as e:
        logger.warning(f"âš ï¸ Signal advanced endpoints not available: {e}")

    # Auto trading endpoints
    try:
        from backend.endpoints.auto_trading_endpoints import (
            router as auto_trading_router,
        )

        consolidated_router.include_router(auto_trading_router)
        logger.info("âœ… Loaded auto trading endpoints")
    except ImportError as e:
        logger.warning(f"âš ï¸ Auto trading endpoints not available: {e}")

    # Bot management endpoints
    try:
        from backend.endpoints.bot_management_endpoints import (
            router as bot_management_router,
        )

        consolidated_router.include_router(bot_management_router)
        logger.info("âœ… Loaded bot management endpoints")
    except ImportError as e:
        logger.warning(f"âš ï¸ Bot management endpoints not available: {e}")

    # Misc endpoints
    try:
        from backend.endpoints.misc_endpoints import router as misc_router

        consolidated_router.include_router(misc_router)
        logger.info("âœ… Loaded misc endpoints")
    except ImportError as e:
        logger.warning(f"âš ï¸ Misc endpoints not available: {e}")

    # WebSocket endpoints
    try:
        from backend.endpoints.websocket_endpoints import router as websocket_router

        consolidated_router.include_router(websocket_router)
        logger.info("âœ… Loaded WebSocket endpoints")
    except ImportError as e:
        logger.warning(f"âš ï¸ WebSocket endpoints not available: {e}")

    # Additional mounts requested
    try:
        from backend.endpoints.portfolio.transactions_endpoints import router as transactions_router
        consolidated_router.include_router(transactions_router)
        logger.info("âœ… Loaded portfolio transactions endpoints")
    except ImportError as e:
        logger.warning(f"âš ï¸ Portfolio transactions endpoints not available: {e}")

    try:
        from backend.endpoints.ai.ai_leaderboard_endpoints import router as ai_leaderboard_router
        consolidated_router.include_router(ai_leaderboard_router)
        logger.info("âœ… Loaded AI leaderboard endpoints")
    except ImportError as e:
        logger.warning(f"âš ï¸ AI leaderboard endpoints not available: {e}")

    try:
        from backend.endpoints.compat.trading_alias_endpoints import router as trading_alias_router
        consolidated_router.include_router(trading_alias_router)
        logger.info("âœ… Loaded trading alias endpoints")
    except ImportError as e:
        logger.warning(f"âš ï¸ Trading alias endpoints not available: {e}")

    try:
        from backend.endpoints.portfolio.risk_metrics_endpoints import router as risk_metrics_router
        consolidated_router.include_router(risk_metrics_router)
        logger.info("âœ… Loaded portfolio risk metrics endpoints")
    except ImportError as e:
        logger.warning(f"âš ï¸ Portfolio risk metrics endpoints not available: {e}")

    try:
        from backend.endpoints.signals.whale_alerts_endpoints import router as whale_alerts_router
        consolidated_router.include_router(whale_alerts_router)
        logger.info("âœ… Loaded whale alerts endpoints")
    except ImportError as e:
        logger.warning(f"âš ï¸ Whale alerts endpoints not available: {e}")

    logger.info(
        f"âœ… Consolidated router loaded with {len(consolidated_router.routes)} total routes"
    )


# Load all endpoints when module is imported
load_all_endpoints()

# Export the consolidated router
router = consolidated_router



