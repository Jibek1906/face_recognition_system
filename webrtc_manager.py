import asyncio
import logging
import cv2
import numpy as np
from typing import Optional, Dict
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder, MediaRelay
from av import VideoFrame
import threading
import time

logger = logging.getLogger(__name__)

class VideoTransformTrack(VideoStreamTrack):
    """
    A video stream track that transforms frames from an another track.
    Actually we just want to intercept requests. 
    However, for RECEIVING video from browser, we handle the track event on PC.
    """
    kind = "video"

    def __init__(self, track, transform):
        super().__init__()  # don't forget this!
        self.track = track
        self.transform = transform

    async def recv(self):
        frame = await self.track.recv()
        return frame

class WebRTCManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(WebRTCManager, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        if self.initialized:
            return
        self.initialized = True
        self.pcs: Dict[str, RTCPeerConnection] = {}
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.relay = MediaRelay()
        self.connection_active = False
        self.last_frame_time = 0

    async def handle_offer(self, sdp: str, type: str):
        """
        Handle a WebRTC offer from a client.
        """
        logger.info("WebRTCManager.handle_offer called: received offer from browser")
        pc = RTCPeerConnection()
        pc_id = "default" # Only support one active connection for now for simplicity, or generate UUID
        
        # Close existing if any (single stream policy for now)
        if pc_id in self.pcs:
            logger.info("Closing existing WebRTC connection")
            await self.pcs[pc_id].close()
        
        self.pcs[pc_id] = pc

        @pc.on("datachannel")
        def on_datachannel(channel):
            pass

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info(f"WebRTC Connection state: {pc.connectionState}")
            if pc.connectionState == "failed":
                await pc.close()
                self.connection_active = False
            elif pc.connectionState == "closed":
                self.connection_active = False
            elif pc.connectionState == "connected":
                self.connection_active = True

        @pc.on("track")
        def on_track(track):
            logger.info(f"WebRTC Track received: {track.kind}")
            if track.kind == "video":
                # Start a consumer for this track
                asyncio.ensure_future(self._consume_track(track))
            
            @track.on("ended")
            async def on_ended():
                logger.info("WebRTC Track ended")
                self.connection_active = False

        # Set remote description
        offer = RTCSessionDescription(sdp=sdp, type=type)
        await pc.setRemoteDescription(offer)

        # Create answer
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        }

    async def _consume_track(self, track):
        """
        Continuously reads frames from the track and updates latest_frame.
        """
        logger.info("WebRTCManager._consume_track started: waiting for frames from browser")
        self.connection_active = True
        frame_count = 0
        last_log_time = time.time()
        
        while True:
            try:
                # Get frame from track
                # This needs to be called frequently to keep the buffer clean
                frame = await track.recv()
                # logger.debug("WebRTCManager._consume_track: received a frame from browser")
                
                # Convert to numpy/opencv
                img = frame.to_ndarray(format="bgr24")
                with self.frame_lock:
                    self.latest_frame = img
                    self.last_frame_time = time.time()
                
                frame_count += 1
                
                # Log frame rate every 5 seconds
                current_time = time.time()
                if current_time - last_log_time >= 5.0:
                    fps = frame_count / (current_time - last_log_time)
                    logger.info(f"WebRTC receiving frames: {fps:.1f} fps")
                    frame_count = 0
                    last_log_time = current_time
                
            except Exception as e:
                logger.warning(f"WebRTC track consumption stop/error: {e}")
                break
                
        self.connection_active = False
        logger.info("Stopped consuming WebRTC video track")

    def get_latest_frame(self):
        """
        Called by FrameGrabber.
        """
        with self.frame_lock:
            # ✅ Reduced timeout from 5.0s to 1.0s for faster failure detection
            if self.latest_frame is None:
                return None
            
            time_since_last = time.time() - self.last_frame_time
            if time_since_last > 1.0:
                logger.debug(f"WebRTC frame is stale ({time_since_last:.1f}s old)")
                return None
            
            return self.latest_frame

# Global Singleton
GlobalWebRTCManager = WebRTCManager()
