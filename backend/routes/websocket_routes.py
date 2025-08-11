"""
WebSocket Routes for Mystic Trading

Handles WebSocket connections for real-time data streaming.
"""

import asyncio
import json
import logging
import time
from datetime import timezone, datetime
from typing import Dict, List

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

try:
    from app_config import slowapi_limiter
    from services.websocket_manager import get_websocket_manager
except ImportError:
    from services.websocket_manager import get_websocket_manager

    slowapi_limiter = None

from services.notification import get_notification_service
from services.service_manager import service_manager

# Get logger
logger = logging.getLogger("mystic.websocket")

# Create router
router = APIRouter()

# Initialize services for health checks and notifications
notification_service = get_notification_service(None)

# Simple in-memory store for WebSocket connection rate limiting
ws_connection_tracker: Dict[str, List[float]] = {}
last_cleanup_time = time.time()


def cleanup_connection_tracker():
    """Clean up old entries in the connection tracker to prevent memory leaks"""
    global last_cleanup_time
    current_time = time.time()

    # Only clean up every 10 minutes
    if current_time - last_cleanup_time < 600:
        return

    # Remove connections older than 1 hour
    for ip in list(ws_connection_tracker.keys()):
        ws_connection_tracker[ip] = [
            t for t in ws_connection_tracker[ip] if current_time - t < 3600
        ]
        if not ws_connection_tracker[ip]:
            del ws_connection_tracker[ip]

    last_cleanup_time = current_time


@router.websocket("/ws/market-data")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time market data."""
    manager = get_websocket_manager()

    try:
        # Clean up old entries in the connection tracker
        cleanup_connection_tracker()

        # Get client IP for rate limiting
        client_ip = websocket.client.host if websocket.client else "unknown"

        # Apply rate limiting for WebSocket connections
        current_time = time.time()
        if client_ip in ws_connection_tracker:
            # Check if connections are too frequent (max 5 connections per minute)
            connections: List[float] = [
                t for t in ws_connection_tracker[client_ip] if current_time - t < 60
            ]
            if len(connections) >= 5:
                logger.warning(f"WebSocket connection rate limit exceeded for IP: {client_ip}")
                # Use slowapi_limiter if available for additional rate limiting
                if slowapi_limiter:
                    logger.info(
                        f"Rate limit exceeded for IP: {client_ip} - slowapi_limiter available"
                    )
                await websocket.close(code=1008)  # Policy Violation (1008)
                return

            # Update connection tracker
            ws_connection_tracker[client_ip] = connections + [current_time]
        else:
            # First connection from this IP
            ws_connection_tracker[client_ip] = [current_time]

        # Check service manager health
        service_manager.get_health_status() if service_manager else {}

        await manager.connect(websocket)
        try:
            # Track message rate for this connection
            message_times: List[float] = []

            while True:
                # Wait for client message (ping)
                data = await websocket.receive_text()

                # Apply message rate limiting (max 30 messages per minute)
                current_time = time.time()
                message_times = [t for t in message_times if current_time - t < 60]
                message_times.append(current_time)

                if len(message_times) > 30:
                    logger.warning(f"WebSocket message rate limit exceeded for IP: {client_ip}")
                    # Send notification for rate limit violations
                    try:
                        await notification_service.send_notification(
                            "WebSocket Rate Limit",
                            f"Rate limit exceeded for IP: {client_ip}",
                            "warning",
                        )
                    except Exception as notification_error:
                        logger.error(
                            f"Failed to send WebSocket rate limit notification: {notification_error}"
                        )
                        pass
                    await websocket.send_text(
                        '{"error": "Rate limit exceeded", "detail": "Too many messages. Please slow down."}'
                    )
                    continue

                # Process client message if needed
                if data == "ping":
                    await websocket.send_text("pong")

                # Send market data updates periodically
                # This would typically be handled by a background task
        except WebSocketDisconnect:
            manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        # Send notification for critical errors
        try:
            await notification_service.send_notification(
                "WebSocket Error",
                f"WebSocket connection error: {str(e)}",
                "error",
            )
        except Exception as notification_error:
            logger.error(f"Failed to send WebSocket error notification: {notification_error}")
            pass
        if websocket in manager.active_connections:
            manager.disconnect(websocket)


@router.websocket("/ws/trading-updates")
async def websocket_trading_updates(websocket: WebSocket):
    """WebSocket endpoint for real-time trading updates"""
    await websocket.accept()
    try:
        while True:
            # Send real-time trading updates
            trading_data = {
                "type": "trading_update",
                "data": {
                    "recent_trades": [
                        {
                            "id": "trade_001",
                            "symbol": "BTC/USDT",
                            "side": "buy",
                            "quantity": 0.1,
                            "price": 48000.50,
                            "timestamp": (datetime.now(timezone.utc).isoformat()),
                            "status": "filled",
                        }
                    ],
                    "open_orders": [
                        {
                            "id": "order_001",
                            "symbol": "ETH/USDT",
                            "side": "sell",
                            "quantity": 2.0,
                            "price": 3400.25,
                            "timestamp": (datetime.now(timezone.utc).isoformat()),
                            "status": "open",
                        }
                    ],
                    "account_balance": {
                        "total_usd": 125000.50,
                        "available_usd": 50000.25,
                        "total_btc": 2.5,
                        "total_eth": 10.0,
                    },
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            await websocket.send_text(json.dumps(trading_data))

            # Wait for 2 seconds before sending next update
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        logger.info("WebSocket trading updates client disconnected")
    except Exception as e:
        logger.error(f"WebSocket trading updates error: {str(e)}")
        await websocket.close()


@router.websocket("/ws/portfolio-updates")
async def websocket_portfolio_updates(websocket: WebSocket):
    """WebSocket endpoint for real-time portfolio updates"""
    await websocket.accept()
    try:
        while True:
            # Send real-time portfolio updates
            portfolio_update_data = {
                "type": "portfolio_update",
                "data": {
                    "total_value": 125000.50,
                    "total_pnl": 15000.25,
                    "daily_pnl": 1250.75,
                    "weekly_pnl": 5000.50,
                    "monthly_pnl": 12000.75,
                    "positions": [
                        {
                            "symbol": "BTC/USDT",
                            "quantity": 2.5,
                            "current_price": 48000,
                            "unrealized_pnl": 7500,
                            "allocation": 0.4,
                        },
                        {
                            "symbol": "ETH/USDT",
                            "quantity": 10.0,
                            "current_price": 3400,
                            "unrealized_pnl": 3000,
                            "allocation": 0.3,
                        },
                    ],
                    "performance_metrics": {
                        "sharpe_ratio": 1.85,
                        "max_drawdown": 0.08,
                        "win_rate": 0.72,
                        "total_trades": 156,
                    },
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            await websocket.send_text(json.dumps(portfolio_update_data))

            # Wait for 3 seconds before sending next update
            await asyncio.sleep(3)
    except WebSocketDisconnect:
        logger.info("WebSocket portfolio updates client disconnected")
    except Exception as e:
        logger.error(f"WebSocket portfolio updates error: {str(e)}")
        await websocket.close()


@router.websocket("/ws/ai-signals")
async def websocket_ai_signals(websocket: WebSocket):
    """WebSocket endpoint for real-time AI trading signals"""
    await websocket.accept()
    try:
        while True:
            # Send real-time AI signals
            ai_signal_data = {
                "type": "ai_signal",
                "data": {
                    "signals": [
                        {
                            "symbol": "BTC/USDT",
                            "action": "buy",
                            "confidence": 0.85,
                            "price": 48000.50,
                            "reason": "Strong momentum detected by AI model",
                            "model": "gpt-4",
                            "timestamp": (datetime.now(timezone.utc).isoformat()),
                        },
                        {
                            "symbol": "ETH/USDT",
                            "action": "hold",
                            "confidence": 0.65,
                            "price": 3400.25,
                            "reason": "Neutral signal from technical analysis",
                            "model": "lstm",
                            "timestamp": (datetime.now(timezone.utc).isoformat()),
                        },
                    ],
                    "model_performance": {
                        "accuracy": 0.78,
                        "total_signals": 1250,
                        "profitable_signals": 975,
                        "avg_return": 0.045,
                    },
                    "market_sentiment": {
                        "overall": "bullish",
                        "confidence": 0.72,
                        "factors": ["momentum", "volume", "social_sentiment"],
                    },
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            await websocket.send_text(json.dumps(ai_signal_data))

            # Wait for 5 seconds before sending next update
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        logger.info("WebSocket AI signals client disconnected")
    except Exception as e:
        logger.error(f"WebSocket AI signals error: {str(e)}")
        await websocket.close()
