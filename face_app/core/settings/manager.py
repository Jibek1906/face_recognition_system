"""Settings manager.

This is the new home for what historically lived in `face_system/settings_manager.py`.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Dict


# Path: face_system/face_app/core/settings/manager.py
# parents[0]=settings, [1]=core, [2]=face_app, [3]=face_system
_BASE_DIR = Path(__file__).resolve().parents[3]  # .../face_system

# Keep settings.json in the project root (next to run.py) for compatibility.
SETTINGS_FILE = str(_BASE_DIR / "settings.json")


DEFAULT_SETTINGS: Dict[str, Any] = {
    "similarity_threshold": 0.55,
    "duplicate_threshold": 0.92,
    "min_frames_to_create": 3,
    "min_face_size": 80,
    "min_detection_confidence": 0.45,
    "camera_source": "imou",  # Default source
    "camera_quality": "HD",  # HD/SD for RTSP-like sources

    # Overlay/OSD
    "draw_boxes_enabled": True,  # enable/disable green/orange rectangles
    "draw_timer_enabled": True,  # enable/disable timer label for recognized persons
    "orange_box_enabled": True,
    "orange_box_min_frames": 5,
    "show_unknown_boxes": True,
    "display_min_hits": 4,
    "match_confidence_margin": 0.05,
    "tracker_max_age": 60,
    "tracker_max_misses_active": 15,
    "tracker_iou_threshold": 0.15,
    "tracker_sim_threshold": 0.50,
    "tracker_min_hits_to_confirm": 3,
    "body_match_threshold": 0.55,
    "body_embedding_interval": 1.5,
    "body_embedding_budget": 2,
    "min_face_recognition_width": 80,

    # Pipeline tuning
    "processing_every_n_frames": 1,  # 1 = process every frame (default)
    "drawing_every_n_frames": 1,  # 1 = draw every processed frame
    "recognition_enqueue_interval_frames": 1,  # 1 = enqueue recognition every processed frame per track
    "recognition_min_hits": 5,

    # New person creation thresholds (RecognitionThread)
    "new_person_min_hits": 10,
    "new_person_min_confidence": 0.65,

    # Event-gating: reduce heavy detection when idle, boost right after alarm/events
    "event_gating_enabled": True,
    "event_gating_activate_on_tracks": True,
    "event_gating_active_after_event_seconds": 12.0,
    "event_gating_active_after_track_seconds": 2.0,
    "event_gating_skip_detection_when_idle": True,
    "event_gating_idle_processing_every_n_frames": 12,
    "event_gating_active_processing_every_n_frames": 1,

    # FTP snapshot ingestion (/ftp/process_image)
    # These snapshots can arrive frequently; keep them cheap by resizing and skipping duplicates.
    "ftp_snapshot_save_enabled": True,
    "ftp_snapshot_save_dir": "/var/www/face_recognition/face_system/face_appkuma",
    "ftp_snapshot_keep_last": 10,
    "ftp_snapshot_jpeg_quality": 85,
    "ftp_processing_resize_width": 640,
    "ftp_skip_similar_enabled": True,
    # Mean absolute diff between 32x32 grayscale signatures (0..255). Lower => more aggressive skipping.
    "ftp_skip_similar_threshold": 2.0,

    # Expensive optional features
    # When enabled, OSNet (torchreid) will be loaded during /faces/{id}/photos upload.
    # Keep disabled for low-resource deployments.
    "enable_body_embedding_on_photo_upload": False,

    # ── External Identification Integration ──────────────────────────────────
    # URL стороннего сервера, который принимает фото и возвращает ФИО и ID
    "external_api_url": "http://127.0.0.1:8000/dummy_external",
    # Включить/выключить автоматическую отправку
    "external_integration_enabled": True,
    # Минимальная дисперсия Лапласиана для определения резкости фото
    # Чем выше — тем строже фильтр. 80-150 — хороший диапазон.
    "blur_threshold": 100.0,
    # Минимальная резкость лица для записи нового человека в базу данных
    # Если sharpness < этого значения — система ждёт более чёткого кадра
    "new_person_min_sharpness": 120.0,

    # Legacy/compat aliases (some modules still read these keys)
}


class SettingsManager:
    _instance = None
    _lock = threading.RLock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(SettingsManager, cls).__new__(cls)
                cls._instance._settings = {}
                cls._instance._version = 0
                cls._instance.load_settings()
        return cls._instance

    def load_settings(self):
        with self._lock:
            if os.path.exists(SETTINGS_FILE):
                try:
                    with open(SETTINGS_FILE, "r") as f:
                        saved_settings = json.load(f)
                        # Merge with defaults to ensure all keys exist
                        self._settings = {**DEFAULT_SETTINGS, **saved_settings}
                except Exception as e:
                    print(f"Error loading settings: {e}. Using defaults.")
                    self._settings = DEFAULT_SETTINGS.copy()
            else:
                self._settings = DEFAULT_SETTINGS.copy()
                self.save_settings()

            # bump version on any (re)load
            self._version += 1

    def save_settings(self):
        try:
            with open(SETTINGS_FILE, "w") as f:
                json.dump(self._settings, f, indent=2)
        except Exception as e:
            print(f"Error saving settings: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._settings.get(key, default)

    def get_all(self) -> Dict[str, Any]:
        with self._lock:
            return self._settings.copy()

    def get_version(self) -> int:
        """Monotonic version incremented on every load/update/reset."""
        with self._lock:
            return int(self._version)

    def update(self, new_settings: Dict[str, Any]):
        with self._lock:
            for k, v in new_settings.items():
                if k in DEFAULT_SETTINGS:  # only allow known settings
                    expected_type = type(DEFAULT_SETTINGS[k])

                    # Allow int -> float
                    if expected_type == float and isinstance(v, int):
                        v = float(v)
                    # Allow float -> int if it looks integral
                    if expected_type == int and isinstance(v, float):
                        if abs(v - int(v)) < 1e-9:
                            v = int(v)
                    # Allow 0/1 -> bool
                    if expected_type == bool and isinstance(v, (int, float)):
                        v = bool(int(v))

                    if isinstance(v, expected_type):
                        self._settings[k] = v
                    else:
                        print(
                            f"Warning: Invalid type for setting '{k}'. Expected {expected_type}, got {type(v)}"
                        )
            self.save_settings()
            self._version += 1

    def reset_to_defaults(self):
        with self._lock:
            self._settings = DEFAULT_SETTINGS.copy()
            self.save_settings()
            self._version += 1


# Create a global instance for easy import
settings = SettingsManager()
