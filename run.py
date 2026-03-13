import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
import cv2
import numpy as np
import logging
from datetime import datetime
from threading import Thread, Lock, Event
import sys
import signal
import time
import gc
import asyncio
import queue
from db import get_session, init_db
from models import Face, Visit, get_bishkek_time
from face_recognition import FaceRecognizer, BodyRecognizer
from config import (
    CAMERA_INDEX, VIDEO_FPS, DISPLAY_WIDTH, DISPLAY_HEIGHT,
    SIMILARITY_THRESHOLD, API_HOST, API_PORT, CAMERA_SOURCE,
    IMOU_APP_ID, IMOU_APP_SECRET, IMOU_DEVICE_ID, IMOU_CHANNEL_ID,
    PROCESS_EVERY_N_FRAMES, DRAW_EVERY_N_FRAMES, STREAM_FPS
)
from api import run_api
import threading
import os
import traceback

# New Architecture Imports
from tracker import Tracker, TrackState
from identity_manager import IdentityManager
from visit_manager import VisitManager
from shared_state import SharedState

# Dynamic Camera Imports
from reolink_camera import ReolinkCamera
from imou_camera import ImouCamera
from webrtc_manager import GlobalWebRTCManager
from external_webrtc_client import ExternalWebRTCClient

# Prepare Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress noisy logs
logging.getLogger('aiortc').setLevel(logging.WARNING)
logging.getLogger('aioice').setLevel(logging.WARNING)
logging.getLogger('av').setLevel(logging.WARNING)

# --- WORKER THREADS ---

class CaptureThread(Thread):
    """
    Dedicated thread for grabbing frames from the camera/stream.
    Pushes the latest frame to a size-limited queue to ensure freshness.
    """
    def __init__(self, source, frame_queue: queue.Queue, stop_event: Event, shared_state=None, quality="HD"):
        super().__init__(name="CaptureThread", daemon=True)
        self.source = source
        self.quality = quality  # ✅ NEW: Track quality for RTSP streams
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self.shared_state = shared_state # ✅ Store shared state for direct updates
        self.source = source
        self.quality = quality  # ✅ NEW: Track quality for RTSP streams
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self.cap = None
        self.external_webrtc_client = None
        self.lock = Lock()
        self.connected = False
        
    def run(self):
        logger.info(f"🚀 CaptureThread started. Source: {self.source}")
        self._init_camera()
        last_frame_time = time.time()
        while not self.stop_event.is_set():
            try:
                frame_processed = False
                if self.source == "webrtc":
                    frame = GlobalWebRTCManager.get_latest_frame()
                    if frame is not None:
                        self._enqueue_frame(frame)
                        frame_processed = True
                    else:
                        time.sleep(0.01)
                        
                elif self.source == "external_webrtc":
                    time.sleep(0.1)
                    
                elif self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret:
                        self._enqueue_frame(frame)
                        frame_processed = True
                    else:
                        logger.warning("CaptureThread: Failed to read frame")
                        self._reconnect()
                else:
                    self._reconnect()
                if frame_processed:
                    last_frame_time = time.time()
                    logger.debug(f"CaptureThread: Frame captured at {last_frame_time}")
                # Watchdog: log if no frame for 10s
                if time.time() - last_frame_time > 10:
                    logger.error("CaptureThread: No frame captured for >10s! Possible deadlock.")
                    last_frame_time = time.time()
            except Exception as e:
                logger.error(f"Capture loop error: {e}")
                time.sleep(1)
        
        self._release_resources()
        logger.info("CaptureThread stopped.")

    def _enqueue_frame(self, frame):
        # Resize if huge to prevent pipeline clog (optional, handled in config?)
        # For now, raw frame.
        
        # ✅ PERFORMANCE FIX: Update Shared State IMMEDIATELY for smooth display
        if hasattr(self, 'shared_state') and self.shared_state:
            self.shared_state.update_frame(frame)

        # ✅ LIFO FIX: Drop ALL old frames in the queue to ensure zero latency (WebRTC-like)
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
                
        self.frame_queue.put(frame)

    def _init_camera(self):
        with self.lock:
            if self.cap: self.cap.release()
            
            if self.source == "imou":
                try:
                    cam = ImouCamera()
                    self.cap = cam.get_video_capture(quality="HD")
                except Exception as e:
                    logger.error(f"Imou init failed: {e}")
            elif self.source == "reolink":
                try:
                    cam = ReolinkCamera()
                    self.cap = cam.get_video_capture(quality="HD")
                except Exception as e:
                    logger.error(f"Reolink init failed: {e}")
            elif self.source in ["rtsp", "second_rtsp"]:
                # ✅ OPTIMIZED: Second RTSP stream with low-latency configuration
                import config
                stream_type = "sub" if self.quality == "SD" else "main"
                if getattr(config, "SECOND_RTSP_URL", ""):
                    rtsp_url = config.SECOND_RTSP_URL
                    logger.info(f"Connecting to Second RTSP (custom URL): {rtsp_url}")
                else:
                    rtsp_url = f"rtsp://{config.SECOND_RTSP_USERNAME}:{config.SECOND_RTSP_PASSWORD}@{config.SECOND_RTSP_IP}:{config.SECOND_RTSP_PORT}//h264Preview_{config.SECOND_RTSP_CHANNEL:02d}_{stream_type}"
                    logger.info(f"Connecting to Second RTSP: {rtsp_url.replace(config.SECOND_RTSP_PASSWORD, '***')}")
                
                # ✅ CRITICAL: Configure for low-latency RTSP streaming
                self.cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer - CRITICAL for reducing lag
                self.cap.set(cv2.CAP_PROP_FPS, 15)  # Limit FPS to reduce processing load
                
                # Additional RTSP optimization flags
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|buffer_size;1024000|max_delay;0"
            elif self.source == "external_webrtc":
                 # This starts its own async loop/thread usually
                 pass 
            elif self.source == "webrtc":
                pass
            elif self.source and self.source != "none":
                # Local
                self.cap = cv2.VideoCapture(CAMERA_INDEX)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_WIDTH)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_HEIGHT)
                self.cap.set(cv2.CAP_PROP_FPS, VIDEO_FPS)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # ✅ ADDED buffer limit for 0 latency
            
            if (self.cap and self.cap.isOpened()) or self.source in ["webrtc", "external_webrtc"]:
                self.connected = True
                logger.info(f"✓ Camera initialized: {self.source}")
            else:
                self.connected = False

    def switch_source(self, new_source):
        logger.info(f"🔄 CaptureThread switching source: {self.source} -> {new_source}")
        self.source = new_source
        self._init_camera()

    def _reconnect(self):
        if self.source == "none":
            time.sleep(1.0) # Just wait if no source
            return
            
        logger.warning("Reconnecting camera...")
        time.sleep(2)
        self._init_camera()

    def _release_resources(self):
         if self.cap: self.cap.release()
         if self.external_webrtc_client: self.external_webrtc_client.stop()


class DetectionThread(Thread):
    """
    Pulls frames, runs Body/Face Detection, Updates Tracker.
    Pushes crop tasks to Recognition Queue.
    """
    def __init__(self, frame_queue: queue.Queue, recognition_queue: queue.Queue, 
                 shared_state: SharedState, tracker: Tracker, identity_manager: IdentityManager, stop_event: Event):
        super().__init__(name="DetectionThread", daemon=True)
        self.frame_queue = frame_queue
        self.recognition_queue = recognition_queue
        self.shared_state = shared_state
        self.stop_event = stop_event
        
        # Models
        self.face_recognizer = FaceRecognizer()
        self.body_recognizer = BodyRecognizer()
        
        # Shared Components
        self.tracker = tracker
        self.identity_manager = identity_manager  # ✅ For display name lookup
        self.frame_count = 0

    def run(self):
        logger.info("🚀 DetectionThread started.")
        last_frame_time = time.time()
        while not self.stop_event.is_set():
            try:
                try:
                    frame = self.frame_queue.get(timeout=0.05)
                except queue.Empty:
                    # Watchdog: log if no frame for 10s
                    if time.time() - last_frame_time > 10:
                        logger.error("DetectionThread: No frame received for >10s! Possible deadlock.")
                        last_frame_time = time.time()
                    continue
                
                last_frame_time = time.time()
                logger.debug(f"DetectionThread: Frame processed at {last_frame_time}")
                self.frame_count += 1
                
                # 0. Update Shared State with Latest Frame -> MOVED TO CaptureThread for speed
                # self.shared_state.update_frame(frame) 

                # 0.5 Update Tracks with Async Results
                resolved = self.shared_state.consume_identities()
                if resolved:
                    for track in self.tracker.tracks:
                        if track.track_id in resolved:
                            pid, name, ts = resolved[track.track_id]
                            track.person_id = pid
                            track.display_name = name # Attach name to track object for SharedState to see
                            logger.info(f"✅ Track {track.track_id} updated with Identity P{pid} ({name})")

                self.frame_count += 1

                # 1. We process EVERY frame we get from the queue.
                # Because the queue is maxsize=1 (LIFO), if DetectionThread is busy, 
                # CaptureThread naturally drops frames. We don't need artificial skipping anymore.

                # 2. Detect with higher body threshold to avoid background things
                bodies = self.body_recognizer.detect_bodies(frame, conf_threshold=0.55)
                faces = self.face_recognizer.detect_faces(frame)
                
                # 3. Form Detections List
                detections = []
                # Simple IoU matching for Face <-> Body association
                # ... (Logic from original run.py adapted below) ...
                
                # We need to map bodies to faces to create a "Person" detection
                # For simplicity in this refactor, we stick to the existing logic 
                # but cleaned up.
                
                # Iterate bodies, find matching face
                used_faces = set()
                for bx, by, bw, bh in bodies:
                    body_crop = frame[by:by+bh, bx:bx+bw]
                    # Body Embedding (Wait... is this fast enough for Detection Thread?
                    # Original code did it here. OSNet is relatively fast (~15ms).
                    # Let's keep it here for now to maintain logic, or move to Rec Queue if slow.)
                    # Optimization: Only extract if we strictly need it for ReID.
                    # For now, proceed.
                    
                    # Match with face
                    best_face = None
                    best_iou = 0
                    for i, face in enumerate(faces):
                        if i in used_faces: continue
                        # Calculate Intersection
                        fx, fy, fw, fh = face['x'], face['y'], face['w'], face['h']
                        
                        xA = max(bx, fx)
                        yA = max(by, fy)
                        xB = min(bx+bw, fx+fw)
                        yB = min(by+bh, fy+fh)
                        interArea = max(0, xB - xA) * max(0, yB - yA)
                        if interArea > 0:
                            faceArea = fw * fh
                            iou = interArea / faceArea
                            if iou > 0.5: # Face inside body
                                best_face = face
                                best_iou = iou
                                used_faces.add(i)
                                break
                    
                    if best_face:
                        det = {
                            'x': bx, 'y': by, 'w': bw, 'h': bh,
                            'face_img': best_face['face_img'],
                            'embedding': best_face.get('embedding'),
                            'confidence': best_face.get('confidence', 0.8),
                            'face_bbox': (best_face['x'], best_face['y'], best_face['w'], best_face['h'])
                        }
                        detections.append(det)

                # Add remaining faces (unassigned) - Ensure they have the correct structure
                for i, face in enumerate(faces):
                    if i not in used_faces:
                        # Ensure face_bbox is set for standalone faces
                        face_det = {
                            'x': face['x'], 'y': face['y'], 'w': face['w'], 'h': face['h'], # Box is the face itself here
                            'face_bbox': (face['x'], face['y'], face['w'], face['h']),
                            'face_img': face.get('face_img'),
                            'embedding': face.get('embedding'),
                            'confidence': face.get('confidence', 0.8)
                        }
                        detections.append(face_det)

                # 4. Update Tracker
                tracks = self.tracker.update(detections, frame.shape)
                
                # 5. Push Unknown/Unverified tracks to Recognition Queue
                # STICKY ID RULE: only queue tracks that do NOT yet have a person_id.
                # Once an identity is confirmed, we NEVER re-queue — prevents id swapping.
                current_timestamp = time.time()
                for track in tracks:
                    # ✅ STICKY: skip entirely if already identified
                    if track.person_id is not None:
                        continue

                    if track.state != TrackState.CONFIRMED:
                        continue
                        
                    # ✅ TEMPORAL SMOOTHING: Require minimum hits before recognition
                    if track.hits < 5:
                        continue

                    # Filter distant people: if face width is too small, skip recognition and creation
                    distant = False
                    if track.face_bbox:
                         fx, fy, fw, fh = track.face_bbox
                         if fw < 100:  # Reject recognition and creation for faces < 100px width
                              distant = True

                    if distant:
                        track.display_name = ""
                        continue

                    if track.best_face_img is not None:
                         avg_emb = track.get_average_embedding()
                         rec_task = {
                             'track_id': track.track_id,
                             'face_img': track.best_face_img.copy(),
                             'embedding': avg_emb,
                             'confidence': track.best_face_score,
                             'hits': track.hits,
                             'person_id': None,  # Always None here — sticky check done above
                             'timestamp': current_timestamp
                         }
                         # Non-blocking put: drop if queue is full (old task), we'll retry next frame
                         try:
                             self.recognition_queue.put_nowait(rec_task)
                         except queue.Full:
                             pass
                
                # 6. Set display_name for ALL identified tracks  ✅ FIX
                for track in tracks:
                    if track.person_id:
                        info = self.identity_manager.get_identity_info(track.person_id)
                        if info:
                            track.display_name = info['name']
                        else:
                            track.display_name = f"Client {track.person_id}"
                    elif not hasattr(track, 'display_name') or track.display_name in ('Unknown', None):
                        # Only set Analyzing if we didn't explicitly blank it (distant faces)
                        if getattr(track, 'display_name', None) != '':
                            track.display_name = 'Analyzing...'
                
                # 7. Push Tracks to Shared State
                # We do this EVERY frame (or detection frame) so UI is snappy
                self.shared_state.update_tracks(tracks)

                # 8. ✅ STREAM PERFORMANCE FIX: Draw the annotated frame HERE in DetectionThread,
                # immediately after we have the latest frame + tracks, then push to shared_state.
                # The HLS streamer just reads get_annotated_frame() — zero drawing, zero lock fight.
                import video_state
                fs = getattr(video_state, 'global_face_system', None)
                if fs is not None:
                    annotated = frame.copy()
                    fs.draw_overlay(annotated)
                    self.shared_state.set_annotated_frame(annotated)
                
            except Exception as e:
                logger.error(f"Detection step error: {e}")
                traceback.print_exc()

class RecognitionThread(Thread):
    """
    Worker that consumes face crops and identifies them against DB/Embeddings.
    Updates the IdentityManager Main Cache.
    """
    def __init__(self, recognition_queue: queue.Queue, shared_state: SharedState, 
                 identity_manager: IdentityManager, stop_event: Event):
        super().__init__(name="RecognitionThread", daemon=True)
        self.queue = recognition_queue
        self.shared_state = shared_state
        self.stop_event = stop_event
        self.identity_manager = identity_manager
        self.face_recognizer = FaceRecognizer() # Need access to embedding extraction if not provided

    def run(self):
        logger.info("🚀 RecognitionThread started.")
        # Load embeddings once
        self.identity_manager.load_embeddings_background()
        
        while not self.stop_event.is_set():
            try:
                try:
                    # ✅ FIX: Faster timeout to keep loop spinning
                    task = self.queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                
                track_id = task['track_id']
                face_img = task['face_img']
                embedding = task.get('embedding')
                hits = task.get('hits', 0)
                confidence = task.get('confidence', 0.0)
                known_pid = task.get('person_id') # Identify if track already matched
                
                # 1. Extract Embedding (if not passed)
                if embedding is None:
                    embedding = self.face_recognizer.get_embedding(face_img)
                
                if embedding is not None:
                     # 2. Identify — only publish if score meets threshold
                     from config import SIMILARITY_THRESHOLD
                     pid, score = self.identity_manager.find_best_match(embedding)

                     if pid and score >= SIMILARITY_THRESHOLD:
                         # ✅ THRESHOLD GUARD: only accept confident matches
                         info = self.identity_manager.get_identity_info(pid)
                         name = info['name']
                         self.shared_state.publish_identity(track_id, pid, name)
                         logger.info(f"✅ Recognized {name} (Track {track_id}, Score: {score:.2f})")

                     elif pid:
                         # Score was too low — log but do NOT publish anything
                         logger.debug(f"⚠️ Match below threshold Track {track_id} → P{pid} (Score: {score:.2f} < {SIMILARITY_THRESHOLD:.2f}), skipping")

                     else:
                         # 3. Handle NEW Person
                         # Criteria: Safe hits, decent confidence, and no DB match found at all.
                         # Note: known_pid is always None here because sticky ID tracks are never queued.
                         # ✅ FIX: hits >= 10 and confidence >= 0.65 to avoid saving blurry/distant faces.
                         should_create = (hits >= 10 and confidence >= 0.65)

                         if should_create:
                             logger.info(f"🆕 Creating new person from Track {track_id} (Hits: {hits}, Confidence: {confidence:.2f})")
                             new_pid, new_name = self.identity_manager.create_new_person(
                                 track_id=track_id,
                                 embedding=embedding,
                                 face_img=face_img,
                                 hits=hits
                             )
                             if new_pid:
                                 self.shared_state.publish_identity(track_id, new_pid, new_name)
                                 logger.info(f"✨ Created & Recognized {new_name} (Track {track_id})")

            except Exception as e:
                logger.error(f"Recognition error: {e}")

class VisitThread(Thread):
    def __init__(self, shared_state: SharedState, stop_event: Event):
        super().__init__(name="VisitThread", daemon=True)
        self.shared_state = shared_state
        self.stop_event = stop_event
        # ✅ STRICT CLOSING: Close visit immediately (1s) after person leaves
        self.visit_manager = VisitManager(ttl=3.0, grace_period=0.5) # Raised TTL to 3.0
        
    def run(self):
        logger.info("🚀 VisitThread started.")
        while not self.stop_event.is_set():
            time.sleep(1.0) # Check every second
            try:
                tracks = self.shared_state.get_active_tracks() # These are dicts, not objects
                # VisitManager expects objects or IDs. 
                # Refactoring VisitManager to accept list of dicts or just person_ids?
                
                # Check logic:
                # Iterate active tracks. If person_id present -> Update Visit.
                # If person_id missing -> Do nothing.
                
                current_active_person_ids = []
                for tid, info in tracks.items():
                    pid = info.get('person_id')
                    visit_id = info.get('visit_id')
                    
                    if pid:
                        # Update visit (keep alive)
                        self.visit_manager.update_visit(
                            person_id=pid,
                            bbox=info['bbox'],
                            display_name=info['name'],
                            timestamp=time.time()
                        )
                        # ✅ FIX: Increment the actual timer ticking logic for UI display
                        # Calculate elapsed time accurately since we check every second
                        self.shared_state.increment_today_time(pid, 1)
                        current_active_person_ids.append(pid)
                        
                # Check for left people
                # Pass SET of active person_ids
                active_pids_set = set(current_active_person_ids)
                self.visit_manager.check_left(active_pids_set)
                
            except Exception as e:
                logger.error(f"Visit loop error: {e}")
                
        self.visit_manager.shutdown()


class MultiThreadedFaceSystem:
    def __init__(self):
        self.stop_event = Event()
        self.shared_state = SharedState()
        
        # Queues
        # ✅ FIX: Make frame_queue size 1 so we ONLY ever have the absolute latest frame. No buffer bloat.
        self.frame_queue = queue.Queue(maxsize=1)
        self.recognition_queue = queue.Queue(maxsize=100) # Buffer for burst requests
        
        # Components
        self.tracker = Tracker(
            max_age=60, # INCREASED to 60 to persist tracks properly for 1-2 frames
            iou_threshold=0.3, # How close boxes must be to be considered the same person
            sim_threshold=SIMILARITY_THRESHOLD, # ✅ NEW: Enforce global similarity threshold!
            min_hits_for_candidate=3
        )
        self.identity_manager = IdentityManager()
        self.recognizer = FaceRecognizer() # For API usage
        self.body_recognizer = BodyRecognizer() # For API usage
        
        # Threads
        self.capture_thread = None 
        self.detect_thread = None
        self.rec_thread = None
        self.visit_thread = None
        
        # Auto-detect headless mode if no DISPLAY is set on Linux
        # This prevents cv2.imshow from crashing the application on headless servers.
        default_headless = "1" if sys.platform.startswith("linux") and not os.getenv("DISPLAY") else "0"
        self.headless = os.getenv("HEADLESS", default_headless) != "0"
        
        # Initialize video_state early
        import video_state
        video_state.global_face_system = self
        
        self.current_camera_source = CAMERA_SOURCE # Track current source
        self.current_quality = "HD"
        self.frame_lock = self.shared_state.lock # Alias for API
        self.cache_lock = Lock() # For currently_visible cache
        self.currently_visible_cache = []

    def switch_camera(self, source, quality="HD"):
        """Called by API to switch camera."""
        logger.info(f"🔄 System switching camera to {source} ({quality})")
        if self.capture_thread:
            self.capture_thread.switch_source(source)
            self.current_camera_source = source
            self.current_quality = quality
            return True
        return False

    def start(self):
        # 1. Start Capture
        self.capture_thread = CaptureThread(
            source=CAMERA_SOURCE, 
            frame_queue=self.frame_queue,
            stop_event=self.stop_event,
            shared_state=self.shared_state # ✅ Pass for smooth display
        )
        self.capture_thread.start()
        
        # 2. Start Recognition Worker
        self.rec_thread = RecognitionThread(
            recognition_queue=self.recognition_queue,
            shared_state=self.shared_state,
            identity_manager=self.identity_manager,
            stop_event=self.stop_event
        )
        self.rec_thread.start()
        
        # 3. Start Detection (The Brain)
        self.detect_thread = DetectionThread(
            frame_queue=self.frame_queue,
            recognition_queue=self.recognition_queue,
            shared_state=self.shared_state,
            tracker=self.tracker,
            identity_manager=self.identity_manager,  # ✅ Pass for name lookup
            stop_event=self.stop_event
        )
        self.detect_thread.start()
        
        # 4. Start Visits
        self.visit_thread = VisitThread(
            shared_state=self.shared_state, 
            stop_event=self.stop_event
        )
        self.visit_thread.start()
        
        # 5. Main Thread -> Stream / UI
        self._run_main_ui()

    def draw_overlay(self, frame):
        tracks = self.shared_state.get_active_tracks()
        
        # Pre-calculate known boxes for NMS to avoid overlapping UI boxes
        known_boxes = [info.get('face_bbox') or info['bbox'] for info in tracks.values() if info['person_id']]

        for tid, info in tracks.items():
            # ✅ ALWAYS USE FACE BBOX (smaller, more accurate)
            display_bbox = info.get('face_bbox')
            
            # GHOSTING FIX: If no face_bbox, DO NOT DRAW.
            # This hides "Body Only" tracks which might be false positives or back-view only.
            if display_bbox is None:
                continue

            # Only fallback to body bbox if no face detected
            if display_bbox is None:
                display_bbox = info['bbox']

            # Remove velocity extrapolation since Tracker now handles EMA smoothing
            display_bbox = (int(display_bbox[0]), int(display_bbox[1]), int(display_bbox[2]), int(display_bbox[3]))
            
            name = info['name']
            state = info['state']
            
            # ✅ Fix duplicate ghosting in local UI viewing (don't draw if heavily missed > 2 frames)
            if info.get('misses', 0) > 2:
                continue
                
            # ✅ Don't show "Analyzing..." for fleeting tracks or overlapping tracks
            if not info['person_id']:
                if info.get('hits', 0) < 10:
                    continue
                    
                # Spatial NMS against recognized people
                should_draw = True
                x1, y1, w1, h1 = display_bbox
                for k_box in known_boxes:
                    if k_box:
                        x2, y2, w2, h2 = k_box
                        xi1, yi1 = max(x1, x2), max(y1, y2)
                        xi2, yi2 = min(x1+w1, x2+w2), min(y1+h1, y2+h2)
                        inter_area = max(0, xi2-xi1) * max(0, yi2-yi1)
                        iou = inter_area / ((w1*h1) + (w2*h2) - inter_area + 1e-6)
                        if iou > 0.35:
                            should_draw = False
                            break
                if not should_draw:
                    continue
            
            # Color based on state
            color = (0, 255, 0) if state == "CONFIRMED" else (0, 255, 255)
            if info['person_id']:
                 color = (0, 255, 0) # Known
            else:
                 color = (0, 165, 255) # Unknown/Recognizing
            
            x, y, w, h = int(display_bbox[0]), int(display_bbox[1]), int(display_bbox[2]), int(display_bbox[3])
            
            # ✅ Draw smaller, tighter box around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Calculate Duration Timer from Shared State directly so it's accurate and continues incrementally
            timer_str = "00:00"
            # ✅ Only show label for recognized clients — distant/unknown = orange box only, NO text
            person_id = info.get('person_id')
            if person_id:
                timer_str = self.shared_state.get_today_time_str(person_id)
                display_text = f"[ Client - {timer_str} ]"
                (text_width, text_height), baseline = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                text_y = max(y, text_height + 15)
                cv2.rectangle(frame, (x, text_y - text_height - 15), (x + text_width, text_y), color, cv2.FILLED)
                cv2.putText(frame, display_text, (x, text_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            # else: no text — just the orange box drawn above

    def _run_main_ui(self):
        logger.info("🎨 Main UI/Stream Loop started.")
        init_db() # Ensure DB is ready
        
        # Inject self into video_state for API access
        import video_state
        video_state.global_face_system = self
        
        # Start API in background thread
        logger.info("📡 Starting API Thread...")
        api_thread = Thread(target=run_api, daemon=True)
        api_thread.start()

        last_drawn_frame_id = -1

        try:
            while not self.stop_event.is_set():
                if self.headless:
                    time.sleep(1)
                    continue
                    
                # Local Debug Display
                current_frame_id = self.shared_state.frame_id
                if current_frame_id == last_drawn_frame_id:
                    time.sleep(0.01) # Wait for new frame
                    continue
                    
                display_frame = self.shared_state.get_annotated_frame()
                if display_frame is not None:
                    last_drawn_frame_id = current_frame_id
                    
                    if not self.headless:
                        cv2.imshow("Multi-Threaded Face System", display_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            self.stop_event.set()
                        
                else:
                    time.sleep(0.01)
                    
        except KeyboardInterrupt:
            logger.info("Stopping...")
            self.stop_event.set()
        finally:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    # Ensure HLS output directory exists
    import os
    hls_dir = "/tmp/hls"
    if not os.path.exists(hls_dir):
        os.makedirs(hls_dir, exist_ok=True)

    # Start HLS streamer (backend-annotated frames to HLS)
    from hls_streamer import start_hls_streamer
    start_hls_streamer()

    system = MultiThreadedFaceSystem()
    system.start()
