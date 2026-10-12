import cv2
import time
import os

from face_app.core.config.env import SECOND_RTSP_URL, SECOND_RTSP_USERNAME, SECOND_RTSP_PASSWORD, SECOND_RTSP_IP, SECOND_RTSP_PORT, SECOND_RTSP_CHANNEL, SECOND_RTSP_STREAM_QUALITY

rtsp_url = SECOND_RTSP_URL if SECOND_RTSP_URL else f"rtsp://{SECOND_RTSP_USERNAME}:{SECOND_RTSP_PASSWORD}@{SECOND_RTSP_IP}:{SECOND_RTSP_PORT}//h264Preview_{SECOND_RTSP_CHANNEL:02d}_main"

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer+discardcorrupt|flags;low_delay|analyzeduration;500000|probesize;500000|stimeout;5000000|max_delay;500000|reorder_queue_size;0"

cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
if not cap.isOpened():
    print("Cannot open RTSP")
    exit(1)

for _ in range(10):
    cap.grab()

ret, frame = cap.retrieve()
if ret:
    print(f"Frame captured: {frame.shape}")
    cv2.imwrite("debug_frame.jpg", frame)
else:
    print("Failed to retrieve frame")

cap.release()
