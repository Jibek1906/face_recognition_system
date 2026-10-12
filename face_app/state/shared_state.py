"""Thread-safe shared in-memory state.

Moved from legacy `face_system/shared_state.py`.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Dict, List, Optional

import numpy as np


class SharedState:
    """Thread-safe storage for the latest frame and active tracks.

    Acts as the single source of truth for the Stream/API thread.
    """

    FRAME_QUEUE_SIZE = 5

    def __init__(self):
        self.lock = threading.Lock()
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_frame_time: float = 0.0
        self.latest_annotated_frame: Optional[np.ndarray] = None  # Frame drawn precisely with its own tracks
        self.latest_annotated_time: float = 0.0
        self.frame_queue = deque(maxlen=self.FRAME_QUEUE_SIZE)
        self.frame_id = 0
        self.active_tracks: Dict = {}  # {track_id: ValidatedTrackInfo}
        self.resolved_identities: Dict = {}  # {track_id: (person_id, name, timestamp)}
        self.status_message = "Initializing..."
        self.fps = 0.0
        self.today_seconds_cache: Dict[int, int] = {}  # {person_id: total_seconds_today}

        # Event-gating: temporarily boost detection frequency after an external event
        # (e.g. FTP alarm snapshot) or when tracks appear.
        self.event_active_until: float = 0.0

    def trigger_event(self, active_for_seconds: float):
        """Marks the pipeline as "active" for a period of time.

        Used by FTP/alarm triggers to temporarily increase detection frequency.
        """
        try:
            dur = float(active_for_seconds)
        except Exception:
            dur = 0.0
        if dur <= 0:
            return
        now = time.time()
        with self.lock:
            self.event_active_until = max(self.event_active_until, now + dur)

    def is_event_active(self) -> bool:
        now = time.time()
        with self.lock:
            return now < float(self.event_active_until)

    def update_frame(self, frame: np.ndarray):
        with self.lock:
            self.latest_frame = frame
            self.latest_frame_time = time.time()
            self.frame_queue.append(frame)
            self.frame_id += 1

    def get_latest_frame_age_seconds(self) -> float:
        """Returns age of latest raw frame in seconds (inf if never received)."""
        with self.lock:
            ts = float(self.latest_frame_time or 0.0)
        if ts <= 0:
            return float("inf")
        return time.time() - ts

    def get_frame(self):
        with self.lock:
            if self.frame_queue:
                frame = self.frame_queue.popleft()
                return frame.copy()
            if self.latest_frame is not None:
                # If queue is empty, return last frame (may cause short freeze)
                return self.latest_frame.copy()
            return None

    def get_annotated_frame(self, max_age: Optional[float] = None, fallback_to_latest: bool = False):
        with self.lock:
            annotated = self.latest_annotated_frame
            if annotated is not None:
                if max_age is None or (time.time() - self.latest_annotated_time) <= max_age:
                    return annotated.copy()
                # Annotation exists but stale
                if not fallback_to_latest:
                    return None
            if fallback_to_latest and self.latest_frame is not None:
                return self.latest_frame.copy()
            if annotated is not None and max_age is None:
                # No freshness requirement and annotation exists
                return annotated.copy()
            if self.latest_frame is not None:
                return self.latest_frame.copy()
            return None

    def set_annotated_frame(self, frame: np.ndarray):
        with self.lock:
            self.latest_annotated_frame = frame
            self.latest_annotated_time = time.time()

    def is_annotation_fresh(self, max_age: float) -> bool:
        with self.lock:
            if self.latest_annotated_frame is None:
                return False
            return (time.time() - self.latest_annotated_time) <= max_age

    def publish_identity(self, track_id, person_id, name):
        """Called by RecognitionThread to notify DetectionThread."""
        with self.lock:
            self.resolved_identities[track_id] = (person_id, name, time.time())

    def consume_identities(self):
        """Called by DetectionThread to get new identifications."""
        with self.lock:
            data = self.resolved_identities.copy()
            self.resolved_identities.clear()
            return data

    def get_today_time_str(self, person_id):
        with self.lock:
            seconds = self.today_seconds_cache.get(person_id, 0)
        hh = seconds // 3600
        mm = (seconds % 3600) // 60
        ss = seconds % 60
        if hh > 0:
            return f"{hh:02}:{mm:02}:{ss:02}"
        return f"{mm:02}:{ss:02}"

    def increment_today_time(self, person_id, seconds_to_add=1):
        with self.lock:
            self.today_seconds_cache.setdefault(person_id, 0)
            self.today_seconds_cache[person_id] += seconds_to_add

    def set_today_time(self, person_id, seconds):
        with self.lock:
            self.today_seconds_cache[person_id] = seconds

    def update_tracks(self, tracks: List):
        """Updates the lightweight display list for the frontend."""
        with self.lock:
            self.active_tracks = {}
            for t in tracks:
                self.active_tracks[t.track_id] = {
                    "bbox": t.bbox,
                    "face_bbox": getattr(t, "face_bbox", None),
                    "person_id": t.person_id,
                    "state": t.state.name,
                    "name": getattr(t, "display_name", "Unknown"),
                    "visit_id": getattr(t, "visit_id", None),
                    "face_img": t.best_face_img,
                    "hits": t.hits,
                    "misses": t.misses,
                    "first_seen_time": getattr(t, "first_seen_time", time.time()),
                }

    def get_active_tracks(self):
        with self.lock:
            return self.active_tracks.copy()

    def set_status(self, msg: str):
        with self.lock:
            self.status_message = msg

    def get_status(self):
        with self.lock:
            return self.status_message
