"""
Consolidated Router
Single router loading system that organizes all endpoints properly
Replaces the chaotic multiple router loading systems
"""

import logging

from fastapi import APIRouter

logger = logging.getLogger(__name__)

# Create main consolidated router
consolidated_router = APIRouter()


def load_all_endpoints() -> None:
    """Load all organized endpoints into the consolidated router"""

    # Core System Endpoints
    try:
        from endpoints.core.system_endpoints import router as core_router

        consolidated_router.include_router(core_router)
        logger.info("✅ Loaded core system endpoints")
    except ImportError as e:
        logger.warning(f"⚠️ Core system endpoints not available: {e}")

    # Trading Endpoints
    try:
        from endpoints.trading.portfolio_endpoints import (
            router as portfolio_router,
        )

        consolidated_router.include_router(portfolio_router)
        logger.info("✅ Loaded portfolio endpoints")
    except ImportError as e:
        logger.warning(f"⚠️ Portfolio endpoints not available: {e}")

    try:
        from endpoints.trading.autobuy_endpoints import (
            router as autobuy_router,
        )

        consolidated_router.include_router(autobuy_router)
        logger.info("✅ Loaded autobuy endpoints")
    except ImportError as e:
        logger.warning(f"⚠️ Autobuy endpoints not available: {e}")

    # Market Endpoints
    try:
        from endpoints.market.market_data_endpoints import (
            router as market_router,
        )

        consolidated_router.include_router(market_router)
        logger.info("✅ Loaded market data endpoints")
    except ImportError as e:
        logger.warning(f"⚠️ Market data endpoints not available: {e}")

    # AI Endpoints
    try:
        from endpoints.ai.ai_strategy_endpoints import router as ai_router

        consolidated_router.include_router(ai_router)
        logger.info("✅ Loaded AI strategy endpoints")
    except ImportError as e:
        logger.warning(f"⚠️ AI strategy endpoints not available: {e}")

    # Experimental Endpoints
    try:
        from endpoints.experimental.advanced_tech_endpoints import router as experimental_router

        consolidated_router.include_router(experimental_router)
        logger.info("✅ Loaded experimental endpoints")
    except ImportError as e:
        logger.warning(f"⚠️ Experimental endpoints not available: {e}")

    # Legacy endpoints that still need to be organized
    # These will be moved to proper folders in future phases

    # Live Trading Endpoints
    try:
        from endpoints.live_trading_endpoints import (
            router as live_trading_router,
        )

        consolidated_router.include_router(live_trading_router)
        logger.info("✅ Loaded live trading endpoints")
    except ImportError as e:
        logger.warning(f"⚠️ Live trading endpoints not available: {e}")

    # Live Data Endpoints (orders, balances, trades)
    try:
        from endpoints.live_data_endpoints import (
            router as live_data_router,
        )

        consolidated_router.include_router(live_data_router)
        logger.info("✅ Loaded live data endpoints")
    except ImportError as e:
        logger.warning(f"⚠️ Live data endpoints not available: {e}")

    # AI Strategy Endpoints (legacy)
    try:
        from endpoints.ai_strategies.ai_strategy_endpoints import (
            router as legacy_ai_router,
        )

        consolidated_router.include_router(legacy_ai_router)
        logger.info("✅ Loaded legacy AI strategy endpoints")
    except ImportError as e:
        logger.warning(f"⚠️ Legacy AI strategy endpoints not available: {e}")

    # Enhanced API Endpoints
    try:
        from endpoints.enhanced_api.enhanced_api_endpoints import (
            router as enhanced_api_router,
        )

        consolidated_router.include_router(enhanced_api_router)
        logger.info("✅ Loaded enhanced API endpoints")
    except ImportError as e:
        logger.warning(f"⚠️ Enhanced API endpoints not available: {e}")

    # Crypto Autoengine Endpoints (router has internal '/api' prefix; results in '/api/api/*' when mounted)
    try:
        from crypto_autoengine_api import router as crypto_router

        consolidated_router.include_router(crypto_router)
        logger.info("✅ Loaded crypto autoengine endpoints")
    except ImportError as e:
        logger.warning(f"⚠️ Crypto autoengine endpoints not available: {e}")

    # AI Supercomputing Endpoints
    try:
        from endpoints.ai_supercomputing.ai_supercomputing_endpoints import (
            router as ai_supercomputing_router,
        )

        consolidated_router.include_router(ai_supercomputing_router)
        logger.info("✅ Loaded AI supercomputing endpoints")
    except ImportError as e:
        logger.warning(f"⚠️ AI supercomputing endpoints not available: {e}")

    # AI Heartbeat Endpoint
    try:
        from endpoints.ai.ai_heartbeat_endpoints import router as ai_heartbeat_router

        consolidated_router.include_router(ai_heartbeat_router)
        logger.info("✅ Loaded AI heartbeat endpoint")
    except ImportError as e:
        logger.warning(f"⚠️ AI heartbeat endpoint not available: {e}")

    # Missing endpoints that need to be implemented
    try:
        from endpoints.dashboard_missing.dashboard_missing_endpoints import (
            router as missing_router,
        )

        consolidated_router.include_router(missing_router)
        logger.info("✅ Loaded missing dashboard endpoints")
    except ImportError as e:
        logger.warning(f"⚠️ Missing dashboard endpoints not available: {e}")

    try:
        from endpoints.portfolio_missing.portfolio_missing_endpoints import (
            router as portfolio_missing_router,
        )

        consolidated_router.include_router(portfolio_missing_router)
        logger.info("✅ Loaded missing portfolio endpoints")
    except ImportError as e:
        logger.warning(f"⚠️ Missing portfolio endpoints not available: {e}")

    # Experimental endpoints (legacy) - using different prefix to avoid conflicts
    try:
        from endpoints.experimental_endpoints import (
            router as legacy_experimental_router,
        )

        consolidated_router.include_router(legacy_experimental_router)
        logger.info("✅ Loaded legacy experimental endpoints")
    except ImportError as e:
        logger.warning(f"⚠️ Legacy experimental endpoints not available: {e}")

    # Phase 5 endpoints
    try:
        from endpoints.phase5_endpoints import router as phase5_router

        consolidated_router.include_router(phase5_router)
        logger.info("✅ Loaded Phase 5 endpoints")
    except ImportError as e:
        logger.warning(f"⚠️ Phase 5 endpoints not available: {e}")

    # Trading endpoints (legacy)
    try:
        from endpoints.trading_endpoints import router as legacy_trading_router

        consolidated_router.include_router(legacy_trading_router)
        logger.info("✅ Loaded legacy trading endpoints")
    except ImportError as e:
        logger.warning(f"⚠️ Legacy trading endpoints not available: {e}")

    # Market endpoints (legacy)
    try:
        from endpoints.market_endpoints import router as legacy_market_router

        consolidated_router.include_router(legacy_market_router)
        logger.info("✅ Loaded legacy market endpoints")
    except ImportError as e:
        logger.warning(f"⚠️ Legacy market endpoints not available: {e}")

    # Bot endpoints
    try:
        from endpoints.bot_endpoints import router as bot_router

        consolidated_router.include_router(bot_router)
        logger.info("✅ Loaded bot endpoints")
    except ImportError as e:
        logger.warning(f"⚠️ Bot endpoints not available: {e}")

    # Social endpoints
    try:
        from endpoints.social_endpoints import router as social_router

        consolidated_router.include_router(social_router)
        logger.info("✅ Loaded social endpoints")
    except ImportError as e:
        logger.warning(f"⚠️ Social endpoints not available: {e}")

    # Notification endpoints
    try:
        from endpoints.notification_endpoints import (
            router as notification_router,
        )

        consolidated_router.include_router(notification_router)
        logger.info("✅ Loaded notification endpoints")
    except ImportError as e:
        logger.warning(f"⚠️ Notification endpoints not available: {e}")

    # Analytics endpoints
    try:
        from endpoints.analytics_endpoints import router as analytics_router

        consolidated_router.include_router(analytics_router)
        logger.info("✅ Loaded analytics endpoints")
    except ImportError as e:
        logger.warning(f"⚠️ Analytics endpoints not available: {e}")

    # Advanced analytics endpoints
    try:
        from endpoints.analytics_advanced_endpoints import (
            router as advanced_analytics_router,
        )

        consolidated_router.include_router(advanced_analytics_router)
        logger.info("✅ Loaded advanced analytics endpoints")
    except ImportError as e:
        logger.warning(f"⚠️ Advanced analytics endpoints not available: {e}")

    # Signal advanced endpoints
    try:
        from endpoints.signal_advanced_endpoints import (
            router as signal_advanced_router,
        )

        consolidated_router.include_router(signal_advanced_router)
        logger.info("✅ Loaded signal advanced endpoints")
    except ImportError as e:
        logger.warning(f"⚠️ Signal advanced endpoints not available: {e}")

    # Auto trading endpoints
    try:
        from endpoints.auto_trading_endpoints import (
            router as auto_trading_router,
        )

        consolidated_router.include_router(auto_trading_router)
        logger.info("✅ Loaded auto trading endpoints")
    except ImportError as e:
        logger.warning(f"⚠️ Auto trading endpoints not available: {e}")

    # Bot management endpoints
    try:
        from endpoints.bot_management_endpoints import (
            router as bot_management_router,
        )

        consolidated_router.include_router(bot_management_router)
        logger.info("✅ Loaded bot management endpoints")
    except ImportError as e:
        logger.warning(f"⚠️ Bot management endpoints not available: {e}")

    # Misc endpoints
    try:
        from endpoints.misc_endpoints import router as misc_router

        consolidated_router.include_router(misc_router)
        logger.info("✅ Loaded misc endpoints")
    except ImportError as e:
        logger.warning(f"⚠️ Misc endpoints not available: {e}")

    # WebSocket endpoints
    try:
        from endpoints.websocket_endpoints import router as websocket_router

        consolidated_router.include_router(websocket_router)
        logger.info("✅ Loaded WebSocket endpoints")
    except ImportError as e:
        logger.warning(f"⚠️ WebSocket endpoints not available: {e}")

    logger.info(
        f"✅ Consolidated router loaded with {len(consolidated_router.routes)} total routes"
    )


# Load all endpoints when module is imported
load_all_endpoints()

# Export the consolidated router
router = consolidated_router
