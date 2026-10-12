import asyncio
import cv2
import av
import logging
import ssl
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiohttp import ClientSession, TCPConnector
import threading

logger = logging.getLogger(__name__)

class ExternalWebRTCClient:
    def __init__(self, url, on_frame_callback):
        """
        Initialize WebRTC client for RTSPtoWeb.
        
        Args:
            url: Player page URL (e.g., https://server/pages/player/webrtc/camera1/0)
            on_frame_callback: Callback function that receives each frame (numpy array)
        """
        self.url = url
        self.on_frame_callback = on_frame_callback
        self.pc = None
        self._task = None
        self.loop = None
        self._thread = None
        self._running = False
        self._frames_received = 0
        
        # Extract API endpoint from player URL
        self.api_endpoint = self._convert_to_api_endpoint(url)
        logger.info(f"ExternalWebRTCClient initialized with API endpoint: {self.api_endpoint}")

    def _convert_to_api_endpoint(self, player_url):
        """
        Convert player page URL to WebRTC API endpoint.
        
        Example:
            Input:  https://server/pages/player/webrtc/camera1/0
            Output: https://server/stream/camera1/channel/0/webrtc
        """
        try:
            # Parse the URL to extract base, stream_id, and channel_id
            # Format: {base}/pages/player/webrtc/{stream_id}/{channel_id}
            parts = player_url.split('/pages/player/webrtc/')
            if len(parts) != 2:
                logger.warning(f"Unexpected URL format: {player_url}, using as-is")
                return player_url
            
            base_url = parts[0]
            stream_info = parts[1].split('/')
            
            if len(stream_info) >= 2:
                stream_id = stream_info[0]
                channel_id = stream_info[1]
                api_endpoint = f"{base_url}/stream/{stream_id}/channel/{channel_id}/webrtc"
                logger.info(f"Converted URL: {player_url} -> {api_endpoint}")
                return api_endpoint
            else:
                logger.warning(f"Could not parse stream/channel from URL: {player_url}")
                return player_url
        except Exception as e:
            logger.error(f"Error parsing URL: {e}, using original URL")
            return player_url

    async def start(self):
        """Start the WebRTC connection."""
        self._running = True
        self.loop = asyncio.get_event_loop()
        
        try:
            logger.info("Starting ExternalWebRTC client...")
            
            # Configure ICE servers (STUN/TURN) - required for WebRTC
            from aiortc import RTCConfiguration, RTCIceServer
            ice_servers = [
                RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
                RTCIceServer(urls=["stun:stun1.l.google.com:19302"]),
            ]
            config = RTCConfiguration(iceServers=ice_servers)
            
            # Create peer connection with ICE configuration
            self.pc = RTCPeerConnection(configuration=config)
            logger.info(f"RTCPeerConnection created with {len(ice_servers)} ICE servers")

            # Add a transceiver to receive video (RTSPtoWeb is sending video to us)
            # This is critical - without it, the SDP won't have proper media sections
            self.pc.addTransceiver("video", direction="recvonly")
            logger.info("Added recvonly video transceiver")

            @self.pc.on("track")
            def on_track(track):
                logger.info(f"Track received: {track.kind}")
                if track.kind == "video":
                    logger.info("Video track received, starting frame loop")
                    async def recv_video():
                        frame_count = 0
                        try:
                            while self._running:
                                try:
                                    frame = await track.recv()
                                    img = frame.to_ndarray(format="bgr24")
                                    
                                    if self.on_frame_callback:
                                        self.on_frame_callback(img)
                                    
                                    frame_count += 1
                                    self._frames_received = frame_count
                                    
                                    if frame_count % 100 == 0:
                                        logger.info(f"Received {frame_count} frames from WebRTC stream")
                                        
                                except Exception as e:
                                    if self._running:
                                        logger.error(f"Error receiving frame: {e}")
                                    break
                        except Exception as e:
                            logger.error(f"Frame reception loop error: {e}")
                    
                    asyncio.create_task(recv_video())

            # Create SSL context that allows self-signed certificates
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            logger.warning("SSL certificate verification disabled for self-signed certificates")

            # Create connector with custom SSL context
            connector = TCPConnector(ssl=ssl_context)
            
            async with ClientSession(connector=connector) as session:
                logger.info("Creating WebRTC offer...")
                offer = await self.pc.createOffer()
                await self.pc.setLocalDescription(offer)
                
                logger.info(f"Sending offer to {self.api_endpoint}")
                try:
                    async with session.post(
                        self.api_endpoint, 
                        data=offer.sdp, 
                        headers={"Content-Type": "application/sdp"},
                        timeout=10
                    ) as resp:
                        if resp.status != 200:
                            logger.error(f"Server returned status {resp.status}: {await resp.text()}")
                            return
                        
                        answer_sdp = await resp.text()
                        logger.info("Received SDP answer from server")
                        
                except Exception as e:
                    logger.error(f"Failed to send offer to server: {e}")
                    return
                
                # Set remote description
                await self.pc.setRemoteDescription(RTCSessionDescription(sdp=answer_sdp, type="answer"))
                logger.info("WebRTC peer connection established")
                
                # Keep alive - wait until stopped
                while self._running:
                    await asyncio.sleep(1)
                    
        except Exception as e:
            logger.error(f"WebRTC client error: {e}", exc_info=True)
        finally:
            logger.info("WebRTC client shutting down")
            if self.pc:
                await self.pc.close()

    def run(self):
        """Start client in a separate thread (alternative method)."""
        self._thread = threading.Thread(target=self._run_in_thread, daemon=True)
        self._thread.start()

    def _run_in_thread(self):
        """Run the asyncio event loop in a separate thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self.loop = loop
        try:
            loop.run_until_complete(self.start())
        except Exception as e:
            logger.error(f"Thread execution error: {e}")
        finally:
            loop.close()

    def stop(self):
        """Stop the WebRTC client gracefully."""
        logger.info(f"Stopping ExternalWebRTCClient (received {self._frames_received} frames total)")
        self._running = False
        
        # Close peer connection in the event loop
        if self.pc and self.loop and self.loop.is_running():
            fut = asyncio.run_coroutine_threadsafe(self.pc.close(), self.loop)
            try:
                fut.result(timeout=5)
            except Exception as e:
                logger.error(f"Error closing peer connection: {e}")
        elif self.pc:
            # Fallback: try to close synchronously
            try:
                loop = asyncio.get_event_loop()
                loop.run_until_complete(self.pc.close())
            except Exception as e:
                logger.error(f"Error in fallback close: {e}")
        
        if self._task:
            self._task.cancel()