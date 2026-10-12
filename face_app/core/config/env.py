"""Environment configuration.

This is the new home for what historically lived in `face_system/config.py`.

Important: we keep behavior compatible with the old module (including prints)
because many modules import config at import-time.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


# Path: face_system/face_app/core/config/env.py
# parents[0]=config, [1]=core, [2]=face_app, [3]=face_system
_BASE_DIR = Path(__file__).resolve().parents[3]  # .../face_system
# Public alias: modules that used to locate repo-root resources via their own
# os.path.dirname(__file__) (e.g. cert.pem, admin.html, the ML weights dir)
# broke once they moved deeper into face_app/ - they should resolve paths
# against this instead of recomputing their own depth-dependent __file__ math.
BASE_DIR = _BASE_DIR
_DOTENV_PATH = _BASE_DIR / ".env"

# Prefer explicit .env near `run.py` to avoid surprises when cwd differs.
load_dotenv(dotenv_path=_DOTENV_PATH, override=False)


DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "1234")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "facerecognition")

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# 🔥 КРИТИЧНО: Снижены пороги распознавания, но сейчас мы ПОВЫШАЕМ их, чтобы избежать ложных совпадений
SIMILARITY_THRESHOLD = float(
    os.getenv("SIMILARITY_THRESHOLD", 0.70)
)  # INCREASED to 0.70 for stricter recognition to prevent false positives

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 9100))
EXTERNAL_API_KEY = os.getenv("EXTERNAL_API_KEY", "face-annotation-secret-2026")
CAM_AGENT_API_KEY = os.getenv("CAM_AGENT_API_KEY", "face-cam-agent-secret-2026")

VIDEO_FPS = 20
CAMERA_INDEX = 0

DISPLAY_WIDTH = int(os.getenv("DISPLAY_WIDTH", 960))
DISPLAY_HEIGHT = int(os.getenv("DISPLAY_HEIGHT", 540))

PROCESS_EVERY_N_FRAMES = int(
    os.getenv("PROCESS_EVERY_N_FRAMES", 3)
)  # OPTIMIZED: Process every 3rd frame for RTSP performance
DRAW_EVERY_N_FRAMES = int(os.getenv("DRAW_EVERY_N_FRAMES", 1))
STREAM_FPS = int(os.getenv("STREAM_FPS", 20))

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

# Legacy prints (kept for backwards compatibility / current ops expectations)
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
