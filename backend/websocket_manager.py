"""
WebSocket Connection Manager for Mystic Trading

Manages WebSocket connections and provides broadcasting capabilities.
"""

import logging
from typing import Any

from fastapi import WebSocket

# Get logger
logger = logging.getLogger("mystic.websocket")


class WebSocketConnectionManager:
    """Manages WebSocket connections and provides broadcasting capabilities."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        """Accept and register a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        total_connections = len(self.active_connections)
        logger.info(f"WebSocket client connected. Total connections: {total_connections}")

    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        self.active_connections.remove(websocket)
        total_connections = len(self.active_connections)
        logger.info(f"WebSocket client disconnected. Total connections: {total_connections}")

    async def broadcast(self, message: str):
        """Broadcast a message to all connected clients."""
        disconnected: list[WebSocket] = []

        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error broadcasting message: {str(e)}")
                # Mark for removal
                disconnected.append(connection)

        # Remove broken connections
        for connection in disconnected:
            if connection in self.active_connections:
                self.active_connections.remove(connection)

    async def send_to_client(self, websocket: WebSocket, message: str):
        """Send a message to a specific client."""
        try:
            await websocket.send_text(message)
            return True
        except Exception as e:
            logger.error(f"Error sending message to client: {str(e)}")
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
            return False

    async def broadcast_json(self, message: dict[str, Any]):
        """Broadcast a JSON message to all connected clients."""
        disconnected: list[WebSocket] = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting JSON message: {str(e)}")
                disconnected.append(connection)
        for connection in disconnected:
            if connection in self.active_connections:
                self.active_connections.remove(connection)


# Global WebSocket connection manager instance
websocket_manager = WebSocketConnectionManager()


def get_websocket_manager():
    """Get the global websocket manager instance"""
    return websocket_manager


class WebSocketManager:
    pass


