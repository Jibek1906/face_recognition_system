"""
WebSocket manager for broadcasting real-time track updates to frontend clients.
"""
import json
import logging
import threading
import asyncio
from fastapi import WebSocket
from typing import List

logger = logging.getLogger(__name__)

class TracksManager:
    """Manages WebSocket connections for real-time track broadcasting"""
    
    def __init__(self):
        self.connections: List[WebSocket] = []
        self.lock = threading.Lock()
    
    async def connect(self, websocket: WebSocket):
        """Accept and register a new WebSocket connection"""
        await websocket.accept()
        with self.lock:
            self.connections.append(websocket)
        logger.info(f"✅ Track WebSocket connected. Total: {len(self.connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        with self.lock:
            if websocket in self.connections:
                self.connections.remove(websocket)
        logger.info(f"❌ Track WebSocket disconnected. Remaining: {len(self.connections)}")
    
    async def broadcast_tracks(self, tracks_data: dict):
        """Broadcast track updates to all connected clients"""
        if not self.connections:
            return
        
        message = json.dumps(tracks_data)
        dead_connections = []
        
        # Send to all connections
        for connection in self.connections[:]:  # Copy to avoid modification issues
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.debug(f"Failed to send to WebSocket client: {e}")
                dead_connections.append(connection)
        
        # Clean up dead connections
        for conn in dead_connections:
            self.disconnect(conn)

# Global instance
tracks_manager = TracksManager()
