import cv2
import logging
import time
import os
from face_app.core.config.env import (
   REOLINK_IP, REOLINK_PORT, REOLINK_USERNAME, REOLINK_PASSWORD, REOLINK_CHANNEL,
   REOLINK_STREAM_QUALITY
)


logger = logging.getLogger(__name__)


class ReolinkCamera:
   """Class for working with Reolink camera via RTSP"""


   def __init__(self):
       self.ip = REOLINK_IP
       self.port = REOLINK_PORT
       self.username = REOLINK_USERNAME
       self.password = REOLINK_PASSWORD
       self.channel = REOLINK_CHANNEL
       self.quality = REOLINK_STREAM_QUALITY

       logger.info(f"Reolink Camera initialized: {self.ip}")


   def get_stream_url(self, quality: str = "HD") -> str:
       """
       Get RTSP stream URL
       quality: "HD" (main stream)
                "SD" (sub stream)
       """
       # Reolink RTSP format:
       # Main Stream: rtsp://admin:password@ip:554/h264Preview_01_main
       # Sub Stream:  rtsp://admin:password@ip:554/h264Preview_01_sub

       stream_type = "main" if quality == "HD" else "sub"

       # Channel usually formatted as 01, 02 etc.
       channel_str = f"{self.channel:02d}"

       url = f"rtsp://{self.username}:{self.password}@{self.ip}:{self.port}/h264Preview_{channel_str}_{stream_type}"

       return url


   def get_video_capture(self, quality: str = "HD"):
       """
       Open video stream from camera
       quality: "SD" or "HD"
       """
       try:
           # ✅ DIAGNOSTIC: Check network first
           response = os.system(f"ping -c 1 -W 1 {self.ip} > /dev/null 2>&1")

           if response != 0:
               logger.warning(f"⚠️ Reolink IP {self.ip} ping failed (might be blocked), trying RTSP anyway...")
           else:
               logger.info(f"✓ Reolink IP {self.ip} is reachable")


           stream_url = self.get_stream_url(quality)
           logger.info(f"Opening Reolink stream ({quality})...")
           # Mask password for logging
           masked_url = stream_url.replace(self.password, "****")
           logger.info(f"Stream URL: {masked_url}")

           # Force TCP for RTSP to prevent packet loss / decoding errors
           # This is critical for preventing libavcodec crashes on Reolink streams
           os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

           # Use CAP_FFMPEG explicitly
           cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)

           # Set buffer size to minimum to reduce latency
           cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

           if not cap.isOpened():
               logger.error("Failed to open Reolink video stream (check IP/Auth/Network)")
               return None

           # Wait a bit for the first frame
           time.sleep(1.0)

           ret, frame = cap.read()
           if ret and frame is not None:
               logger.info(f"✓ Reolink stream opened successfully! Frame: {frame.shape}")
               return cap
           else:
               logger.error("Cannot read frames from Reolink stream")
               cap.release()
               return None

       except Exception as e:
           logger.error(f"Reolink stream open error: {e}", exc_info=True)
           return None


def test_reolink_camera():
   """Function for testing connection"""
   logging.basicConfig(level=logging.INFO)
   print("\n" + "="*70)
   print("REOLINK CAMERA CONNECTION TEST")
   print("="*70)

   camera = ReolinkCamera()

   # Test SD (Sub) Stream
   print("\n[1/2] Testing SD Stream (Sub)...")
   cap = camera.get_video_capture("SD")
   if cap:
       print("✓ SD Stream working")
       cap.release()
   else:
       print("✗ SD Stream failed")

   # Test HD (Main) Stream
   print("\n[2/2] Testing HD Stream (Main)...")
   cap = camera.get_video_capture("HD")
   if cap:
       print("✓ HD Stream working")
       cap.release()
   else:
       print("✗ HD Stream failed")


if __name__ == "__main__":
   test_reolink_camera()
