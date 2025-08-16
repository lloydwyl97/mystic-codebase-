import logging

from fastapi import APIRouter, HTTPException

from backend.routes.live_data_manager import get_live_data_manager
from backend.services.market_data import MarketDataService
from backend.services.notification import get_notification_service

# Import market data service and manager
from backend.services.service_manager import service_manager

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize market data service and manager
market_data_service = MarketDataService()
live_data_manager = get_live_data_manager(market_data_service)

# Initialize notification service for error reporting
notification_service = get_notification_service(
    None
)  # Will be properly initialized with redis_client later


@router.get("/live/supported")
async def get_supported_symbols():
    """Get list of supported symbols for live data."""
    try:
        # Check service manager health
        service_health = service_manager.get_health_status()

        result = live_data_manager.get_supported_symbols_data()

        # Add service health info to response
        result["service_health"] = service_health

        return result
    except Exception as e:
        logger.error(f"Error getting supported symbols: {str(e)}")
        # Send notification for critical errors
        try:
            await notification_service.send_notification(
                "Live Data Error",
                f"Failed to get supported symbols: {str(e)}",
                "error",
            )
        except Exception as notification_error:
            logger.error(f"Failed to send notification: {str(notification_error)}")
            pass  # Don't fail if notification fails
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get supported symbols: {str(e)}",
        )


@router.get("/live/enhanced")
async def get_enhanced_live_data():
    """Get enhanced live data for all supported coins."""
    try:
        return live_data_manager.get_enhanced_live_data()
    except Exception as e:
        logger.error(f"Error getting enhanced live data: {str(e)}")
        # Send notification for critical errors
        try:
            await notification_service.send_notification(
                "Live Data Error",
                f"Failed to get enhanced live data: {str(e)}",
                "error",
            )
        except Exception as notification_error:
            logger.error(f"Failed to send notification: {str(notification_error)}")
            pass  # Don't fail if notification fails
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get enhanced live data: {str(e)}",
        )


@router.get("/live/all")
async def get_all_live_data():
    """Get all live data for supported coins."""
    try:
        return live_data_manager.get_all_live_data()
    except Exception as e:
        logger.error(f"Error getting all live data: {str(e)}")
        # Send notification for critical errors
        try:
            await notification_service.send_notification(
                "Live Data Error",
                f"Failed to get all live data: {str(e)}",
                "error",
            )
        except Exception as notification_error:
            logger.error(f"Failed to send notification: {str(notification_error)}")
            pass  # Don't fail if notification fails
        raise HTTPException(status_code=500, detail=f"Failed to get all live data: {str(e)}")


@router.get("/live/{symbol}")
async def get_live_data(symbol: str):
    """Get live data for a specific symbol."""
    try:
        return live_data_manager.get_live_data_for_symbol(symbol)
    except ValueError as e:
        logger.error(f"Error getting live data for {symbol}: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting live data for {symbol}: {str(e)}")
        # Send notification for critical errors
        try:
            await notification_service.send_notification(
                "Live Data Error",
                f"Failed to get live data for {symbol}: {str(e)}",
                "error",
            )
        except Exception as notification_error:
            logger.error(f"Failed to send notification: {str(notification_error)}")
            pass  # Don't fail if notification fails
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get live data for {symbol}: {str(e)}",
        )


@router.get("/supported")
async def get_supported_coins():
    """Get list of supported coins."""
    try:
        return live_data_manager.get_supported_coins_data()
    except Exception as e:
        logger.error(f"Error getting supported coins: {str(e)}")
        # Send notification for critical errors
        try:
            await notification_service.send_notification(
                "Live Data Error",
                f"Failed to get supported coins: {str(e)}",
                "error",
            )
        except Exception as notification_error:
            logger.error(f"Failed to send notification: {str(notification_error)}")
            pass  # Don't fail if notification fails
        raise HTTPException(status_code=500, detail=f"Failed to get supported coins: {str(e)}")


@router.get("/coin-support")
async def get_coin_support_info():
    """Get detailed coin support information."""
    try:
        return live_data_manager.get_coin_support_info()
    except Exception as e:
        logger.error(f"Error getting coin support info: {str(e)}")
        # Send notification for critical errors
        try:
            await notification_service.send_notification(
                "Live Data Error",
                f"Failed to get coin support info: {str(e)}",
                "error",
            )
        except Exception:
            pass  # Don't fail if notification fails
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get coin support info: {str(e)}",
        )


@router.get("/summary")
async def get_coin_summary():
    """Get summary of coin data."""
    try:
        return live_data_manager.get_coin_summary()
    except Exception as e:
        logger.error(f"Error getting coin summary: {str(e)}")
        # Send notification for critical errors
        try:
            await notification_service.send_notification(
                "Live Data Error",
                f"Failed to get coin summary: {str(e)}",
                "error",
            )
        except Exception:
            pass  # Don't fail if notification fails
        raise HTTPException(status_code=500, detail=f"Failed to get coin summary: {str(e)}")


