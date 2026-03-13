import os
from dotenv import load_dotenv

load_dotenv()

DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "1234")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "facerecognition")

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# 🔥 КРИТИЧНО: Снижены пороги распознавания, но сейчас мы ПОВЫШАЕМ их, чтобы избежать ложных совпадений
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.65))  # INCREASED from 0.60 to 0.65 for stricter recognition

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 9100))

VIDEO_FPS = 20
CAMERA_INDEX = 0

DISPLAY_WIDTH = int(os.getenv("DISPLAY_WIDTH", 960))
DISPLAY_HEIGHT = int(os.getenv("DISPLAY_HEIGHT", 540))

PROCESS_EVERY_N_FRAMES = int(os.getenv("PROCESS_EVERY_N_FRAMES", 3)) # OPTIMIZED: Process every 3rd frame for RTSP performance
DRAW_EVERY_N_FRAMES = int(os.getenv("DRAW_EVERY_N_FRAMES", 1))
STREAM_FPS = int(os.getenv("STREAM_FPS", 15))

IMOU_APP_ID = os.getenv("IMOU_APP_ID", "")
IMOU_APP_SECRET = os.getenv("IMOU_APP_SECRET", "")
IMOU_DEVICE_ID = os.getenv("IMOU_DEVICE_ID", "8H0902APAZC373D")
IMOU_CHANNEL_ID = os.getenv("IMOU_CHANNEL_ID", "0")

CAMERA_SOURCE = os.getenv("CAMERA_SOURCE", "none")  # Options: none, local, imou, reolink

IMOU_STREAM_QUALITY = os.getenv("IMOU_STREAM_QUALITY", "HD")

# Reolink Camera Settings
REOLINK_IP = os.getenv("REOLINK_IP", "192.168.31.100")
REOLINK_PORT = int(os.getenv("REOLINK_PORT", 554))
REOLINK_USERNAME = os.getenv("REOLINK_USERNAME", "admin")
REOLINK_PASSWORD = os.getenv("REOLINK_PASSWORD", "1234")
REOLINK_CHANNEL = int(os.getenv("REOLINK_CHANNEL", 1))
REOLINK_STREAM_QUALITY = os.getenv("REOLINK_STREAM_QUALITY", "HD")

# Second RTSP Stream Settings
SECOND_RTSP_URL = os.getenv("SECOND_RTSP_URL", "")
SECOND_RTSP_IP = os.getenv("SECOND_RTSP_IP", "192.168.1.65")
SECOND_RTSP_PORT = int(os.getenv("SECOND_RTSP_PORT", 554))
SECOND_RTSP_USERNAME = os.getenv("SECOND_RTSP_USERNAME", "admin")
SECOND_RTSP_PASSWORD = os.getenv("SECOND_RTSP_PASSWORD", "")
SECOND_RTSP_CHANNEL = int(os.getenv("SECOND_RTSP_CHANNEL", 1))
SECOND_RTSP_STREAM_QUALITY = os.getenv("SECOND_RTSP_STREAM_QUALITY", "HD")

print("IMOU_APP_ID:", IMOU_APP_ID, repr(IMOU_APP_ID))
print("IMOU_APP_SECRET:", IMOU_APP_SECRET, repr(IMOU_APP_SECRET))
IMOU_REGION = os.getenv("IMOU_REGION", "sg")
print("IMOU_REGION:", IMOU_REGION, repr(IMOU_REGION))
print(f"✅ STREAM QUALITY: {IMOU_STREAM_QUALITY}")
print(f"✅ PROCESSING RESOLUTION: {DISPLAY_WIDTH}x{DISPLAY_HEIGHT}")
print(f"✅ OPTIMIZATION: Process every {PROCESS_EVERY_N_FRAMES} frames")
print(f"✅ OPTIMIZATION: Draw every {DRAW_EVERY_N_FRAMES} frames")
print(f"✅ OPTIMIZATION: Stream target {STREAM_FPS} FPS")
print(f"🔥 SIMILARITY_THRESHOLD: {SIMILARITY_THRESHOLD} (СНИЖЕН для лучшего распознавания)")