import threading
import time
from typing import Dict, List, Optional
import cv2
import numpy as np
from collections import deque

class SharedState:
    """
    Thread-safe storage for the latest frame and active tracks.
    Acts as the single source of truth for the Stream/API thread.
    """
    FRAME_QUEUE_SIZE = 5

    def __init__(self):
        self.lock = threading.Lock()
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_annotated_frame: Optional[np.ndarray] = None # 🔥 Frame drawn precisely with its own tracks
        self.frame_queue = deque(maxlen=self.FRAME_QUEUE_SIZE)
        self.frame_id = 0
        self.active_tracks: Dict = {} # {track_id: ValidatedTrackInfo}
        self.resolved_identities: Dict = {} # {track_id: (person_id, name, timestamp)}
        self.status_message = "Initializing..."
        self.fps = 0.0
        self.today_seconds_cache: Dict[int, int] = {} # {person_id: total_seconds_today}

    def update_frame(self, frame: np.ndarray):
        with self.lock:
            self.latest_frame = frame
            self.frame_queue.append(frame)
            self.frame_id += 1

    def get_frame(self):
        with self.lock:
            if self.frame_queue:
                frame = self.frame_queue.popleft()
                return frame.copy()
            elif self.latest_frame is not None:
                # If queue is empty, return last frame (may cause short freeze)
                return self.latest_frame.copy()
            return None

    def get_latest_frame(self):
        """Non-destructive read: always returns the most recent frame without
        consuming it from the queue.  Use this in the HLS streamer so it
        doesn't race with DetectionThread's get_frame() calls."""
        with self.lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
            return None

    def set_annotated_frame(self, frame: np.ndarray):
        with self.lock:
            self.latest_annotated_frame = frame

    def get_annotated_frame(self):
        with self.lock:
            if self.latest_annotated_frame is not None:
                return self.latest_annotated_frame.copy()
            # Fallback
            elif self.latest_frame is not None:
                return self.latest_frame.copy()
            return None

    def publish_identity(self, track_id, person_id, name):
        """Called by RecognitionThread to notify DetectionThread."""
        with self.lock:
            self.resolved_identities[track_id] = (person_id, name, time.time())

    def consume_identities(self):
        """Called by DetectionThread to get new identifications."""
        with self.lock:
            # Return copy or pop? Keeping it as a cache is better, but we need to limit size.
            # Pop is safe if DetectionThread is the only consumer.
            data = self.resolved_identities.copy()
            self.resolved_identities.clear() # Clear after consumption
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
            # Auto-initialize to 0 if this person hasn't been seen before this session
            # This fixes the frozen timer bug where increment was silently ignored
            self.today_seconds_cache.setdefault(person_id, 0)
            self.today_seconds_cache[person_id] += seconds_to_add

    def set_today_time(self, person_id, seconds):
        with self.lock:
            self.today_seconds_cache[person_id] = seconds


    def update_tracks(self, tracks: List):
        """
        Updates the lightweight display list for the frontend.
        processed_tracks: List of dicts ready for JSON serialization/Drawing
        """
        with self.lock:
            # Convert internal Track objects to simple dicts for UI
            self.active_tracks = {} 
            for t in tracks:
                # Basic info needed for drawing
                self.active_tracks[t.track_id] = {
                    "bbox": t.bbox,
                    "face_bbox": getattr(t, 'face_bbox', None), # ✅ ADDED
                    "person_id": t.person_id,
                    "state": t.state.name,
                    "name": getattr(t, 'display_name', "Unknown"), # Populated by Recognition Thread
                    "visit_id": getattr(t, 'visit_id', None),
                    "face_img": t.best_face_img, # For debugging/thumbnails
                    "hits": t.hits,
                    "misses": t.misses,
                    "first_seen_time": getattr(t, 'first_seen_time', time.time())
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
