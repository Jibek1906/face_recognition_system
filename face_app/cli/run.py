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
from face_app.db.session import get_session, init_db
from face_app.db.models import Face, Visit, get_bishkek_time
from face_app.vision.recognition.face_recognition import FaceRecognizer, BodyRecognizer
from face_app.core.config.env import (
    CAMERA_INDEX, VIDEO_FPS, DISPLAY_WIDTH, DISPLAY_HEIGHT,
    SIMILARITY_THRESHOLD, API_HOST, API_PORT, CAMERA_SOURCE,
    IMOU_APP_ID, IMOU_APP_SECRET, IMOU_DEVICE_ID, IMOU_CHANNEL_ID,
    PROCESS_EVERY_N_FRAMES, DRAW_EVERY_N_FRAMES, STREAM_FPS
)
from face_app.api.app import run_api
import threading
import os
import traceback
from face_app.core.settings.manager import settings

# Optional: start FTP server (alarm snapshots -> /ftp/process_image)
def _start_ftp_server_if_enabled():
    """Starts ftp_server.py in a background thread.

    Enabled by default (can be disabled via env START_FTP_SERVER=0).
    """
    try:
        _logger = logging.getLogger(__name__)
        enabled = str(os.getenv("START_FTP_SERVER", "1")).strip() != "0"
        if not enabled:
            _logger.info("[FTP] START_FTP_SERVER=0 -> FTP server will NOT be started")
            return

        from face_app.streaming.cameras.ftp_server import run_ftp_server, file_processor_worker, cleanup_old_files

        # Save decoded snapshots for verification, keep last 10
        debug_dir = os.environ.setdefault(
            "FTP_DEBUG_SAVE_DIR",
            "/var/www/face_recognition/face_system/face_appkuma",
        )
        os.environ.setdefault("FTP_SAVE_JPEG", "1")
        os.environ.setdefault("FTP_DEBUG_KEEP_LAST", "10")

        try:
            os.makedirs(debug_dir, exist_ok=True)
        except Exception as e:
            _logger.warning(f"[FTP] Could not create debug dir {debug_dir}: {e}")

        threading.Thread(target=file_processor_worker, daemon=True, name="FTPFileWorker").start()
        threading.Thread(target=run_ftp_server, daemon=True, name="FTPServer").start()
        threading.Thread(target=cleanup_old_files, daemon=True, name="FTPCleanup").start()
        _logger.info("[FTP] FTP server started in background thread (port 2121)")
    except Exception as e:
        logging.getLogger(__name__).error(f"[FTP] Failed to start FTP server: {e}")

# New Architecture Imports
from face_app.vision.tracking.tracker import Tracker, TrackState
from face_app.domain.identities.identity_manager import IdentityManager
from face_app.domain.visits.visit_manager import VisitManager
from face_app.state.shared_state import SharedState

# Dynamic Camera Imports
from face_app.streaming.cameras.reolink_camera import ReolinkCamera
from face_app.streaming.cameras.imou_camera import ImouCamera
from face_app.streaming.webrtc_manager import GlobalWebRTCManager
from face_app.streaming.cameras.external_webrtc_client import ExternalWebRTCClient

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
        self._force_reconnect_requested = False
        
    def run(self):
        logger.info(f"🚀 CaptureThread started. Source: {self.source}")
        self._init_camera()
        last_frame_time = time.time()
        while not self.stop_event.is_set():
            try:
                if self._force_reconnect_requested:
                    self._force_reconnect_requested = False
                    logger.warning("CaptureThread: force reconnect requested (e.g. restart button)")
                    self._reconnect()
                    last_frame_time = time.time()
                    continue

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
                    # ✅ Use grab()+retrieve() instead of read() — grab() is non-blocking
                    # and returns False instantly on corrupt HEVC frames instead of hanging.
                    grabbed = self.cap.grab()
                    if grabbed:
                        ret, frame = self.cap.retrieve()
                        if ret and frame is not None:
                            self._enqueue_frame(frame)
                            frame_processed = True
                            self._consecutive_grab_fails = 0
                        else:
                            # retrieve() decoded nothing — skip silently
                            self._consecutive_grab_fails = getattr(self, '_consecutive_grab_fails', 0) + 1
                    else:
                        self._consecutive_grab_fails = getattr(self, '_consecutive_grab_fails', 0) + 1
                    
                    # If we fail too many consecutive grabs, reconnect
                    if getattr(self, '_consecutive_grab_fails', 0) > 30:
                        logger.warning(f"CaptureThread: {self._consecutive_grab_fails} consecutive grab failures, reconnecting...")
                        self._consecutive_grab_fails = 0
                        self._reconnect()
                else:
                    self._reconnect()
                if frame_processed:
                    last_frame_time = time.time()
                    logger.debug(f"CaptureThread: Frame captured at {last_frame_time}")
                # Watchdog: if no frames for a while, force reconnect.
                if time.time() - last_frame_time > 5:
                    logger.error("CaptureThread: No frame captured for >5s! Forcing reconnect...")
                    self._reconnect()
                    last_frame_time = time.time()
            except Exception as e:
                logger.error(f"Capture loop error: {e}")
                time.sleep(1)
        
        self._release_resources()
        logger.info("CaptureThread stopped.")

    def _enqueue_frame(self, frame):
        # Resize if huge to prevent pipeline clog
        if frame is not None:
            import cv2
            from face_app.core.config.env import DISPLAY_WIDTH, DISPLAY_HEIGHT
            h, w = frame.shape[:2]
            if w > DISPLAY_WIDTH or h > DISPLAY_HEIGHT:
                frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_LINEAR)
        
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
                    self.cap = cam.get_video_capture(quality=self.quality)
                except Exception as e:
                    logger.error(f"Imou init failed: {e}")
            elif self.source == "reolink":
                try:
                    cam = ReolinkCamera()
                    self.cap = cam.get_video_capture(quality=self.quality)
                except Exception as e:
                    logger.error(f"Reolink init failed: {e}")
            elif self.source in ["rtsp", "second_rtsp"]:
                # ✅ OPTIMIZED: Second RTSP stream with low-latency configuration
                import face_app.core.config.env as config
                stream_type = "sub" if self.quality == "SD" else "main"
                if getattr(config, "SECOND_RTSP_URL", ""):
                    rtsp_url = config.SECOND_RTSP_URL
                    logger.info(f"Connecting to Second RTSP (custom URL): {rtsp_url}")
                else:
                    rtsp_url = f"rtsp://{config.SECOND_RTSP_USERNAME}:{config.SECOND_RTSP_PASSWORD}@{config.SECOND_RTSP_IP}:{config.SECOND_RTSP_PORT}//h264Preview_{config.SECOND_RTSP_CHANNEL:02d}_{stream_type}"
                    logger.info(f"Connecting to Second RTSP: {rtsp_url.replace(config.SECOND_RTSP_PASSWORD, '***')}")
                
                # ✅ CRITICAL: MUST set env BEFORE creating VideoCapture — OpenCV reads it at construction time only!
                # We force TCP + low-latency + aggressive corruption discard.
                # Also, we disable reordering/extra buffering to reduce the chance of pipeline freezes.
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                    "rtsp_transport;tcp"
                    "|fflags;nobuffer+discardcorrupt"
                    "|flags;low_delay"
                    "|analyzeduration;500000"
                    "|probesize;500000"
                    "|stimeout;5000000"
                    "|max_delay;500000"
                    "|reorder_queue_size;0"
                )
                
                self.cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer - CRITICAL for reducing lag

                # Some OpenCV builds accept CAP_PROP_OPEN_TIMEOUT_MSEC / CAP_PROP_READ_TIMEOUT_MSEC.
                # Best-effort: don't fail if not supported.
                try:
                    self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
                    self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
                except Exception:
                    pass
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

    def force_reconnect(self):
        """
        Вызывается из другого потока (например, из API-запроса по кнопке рестарта).
        Не делает reconnect напрямую (это гонка с самим циклом run(), который в это же
        время может читать self.cap в другом потоке) - только ставит флаг, а фактический
        reconnect выполняет сам run() в своём потоке.
        """
        self._force_reconnect_requested = True

    def _release_resources(self):
         if self.cap: self.cap.release()
         if self.external_webrtc_client: self.external_webrtc_client.stop()


class InferenceThread(Thread):
    """
    ✅ PERFORMANCE FIX: Вынесли InsightFace/YOLO инференс в отдельный поток.
    DetectionThread больше НЕ БЛОКИРУЕТСЯ на 200мс ML-инференса.
    
    Схема:
    CaptureThread → frame_queue → DetectionThread (быстро: оверлей + трекер)
                                      ↓ (каждые N кадров, non-blocking)
                              InferenceQueue → InferenceThread (медленно: InsightFace)
                                      ↓
                              inference_result_queue → DetectionThread (применяет результаты)
    """
    def __init__(self, inference_queue: queue.Queue, result_queue: queue.Queue,
                 stop_event: Event, enable_body_recognition: bool = True):
        super().__init__(name="InferenceThread", daemon=True)
        self.inference_queue = inference_queue
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.settings = settings

        self.enable_body_recognition = enable_body_recognition
        self.face_recognizer = FaceRecognizer()
        self.body_recognizer = BodyRecognizer() if enable_body_recognition else None

    def run(self):
        logger.info("🚀 InferenceThread started.")
        while not self.stop_event.is_set():
            try:
                try:
                    task = self.inference_queue.get(timeout=0.05)
                except queue.Empty:
                    continue

                frame = task['frame']
                frame_id = task['frame_id']

                # ✅ КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: уменьшаем кадр для детекции.
                # InsightFace работает на 640px внутри — давать ему 1280x720 бессмысленно и медленно.
                # Детектируем на 640x360, координаты масштабируем обратно.
                det_w, det_h = 640, 360
                orig_h, orig_w = frame.shape[:2]
                scale_x = orig_w / det_w
                scale_y = orig_h / det_h

                try:
                    small_frame = cv2.resize(frame, (det_w, det_h), interpolation=cv2.INTER_LINEAR)
                except Exception:
                    small_frame = frame
                    scale_x = scale_y = 1.0

                _t0 = time.time()

                # --- Body detection ---
                bodies_raw = []
                if self.enable_body_recognition and self.body_recognizer is not None:
                    try:
                        bodies_raw = self.body_recognizer.detect_bodies(small_frame, conf_threshold=0.45)
                    except Exception as e:
                        logger.debug(f"Body detection error: {e}")

                # Scale body boxes back to original coords
                bodies = []
                for bx, by, bw, bh in bodies_raw:
                    bodies.append([
                        int(bx * scale_x), int(by * scale_y),
                        int(bw * scale_x), int(bh * scale_y)
                    ])

                # --- Face detection ---
                faces_raw = []
                try:
                    faces_raw = self.face_recognizer.detect_faces(small_frame)
                except Exception as e:
                    logger.debug(f"Face detection error: {e}")

                # Scale face boxes back to original coords
                faces = []
                for face in faces_raw:
                    # Scale bbox coords
                    sx = int(face['x'] * scale_x)
                    sy = int(face['y'] * scale_y)
                    sw = int(face['w'] * scale_x)
                    sh = int(face['h'] * scale_y)
                    # Re-crop face_img from original full-resolution frame for quality
                    face_img = frame[sy:sy+sh, sx:sx+sw] if frame is not None else face.get('face_img')
                    faces.append({
                        'x': sx, 'y': sy, 'w': sw, 'h': sh,
                        'face_img': face_img,
                        'confidence': face.get('confidence', 0.8),
                        'embedding': face.get('embedding'),
                        'pose_info': face.get('pose_info'),
                    })

                _det_ms = (time.time() - _t0) * 1000

                # Push result back to DetectionThread
                try:
                    self.result_queue.put_nowait({
                        'frame_id': frame_id,
                        'frame': frame,
                        'faces': faces,
                        'bodies': bodies,
                        'det_ms': _det_ms,
                    })
                except queue.Full:
                    pass  # DetectionThread will use last known tracks

            except Exception as e:
                logger.error(f"InferenceThread error: {e}")


class DetectionThread(Thread):
    """
    ✅ НОВАЯ АРХИТЕКТУРА: DetectionThread теперь НЕ блокируется на инференсе.
    
    Он делает 3 вещи на каждом кадре:
    1. Рисует оверлей с ПОСЛЕДНИМИ ИЗВЕСТНЫМИ треками (мгновенно)
    2. Каждые N кадров отправляет задачу в InferenceThread (non-blocking put_nowait)
    3. Применяет результаты инференса когда они готовы (из result_queue)
    
    Поток HLS теперь получает свежий annotated_frame на каждом кадре камеры,
    а не только когда InsightFace заканчивает работу.
    """
    def __init__(self, frame_queue: queue.Queue, recognition_queue: queue.Queue, 
                 shared_state: SharedState, tracker: Tracker, identity_manager: IdentityManager,
                 stop_event: Event, inference_queue: queue.Queue, inference_result_queue: queue.Queue):
        super().__init__(name="DetectionThread", daemon=True)
        self.frame_queue = frame_queue
        self.recognition_queue = recognition_queue
        self.shared_state = shared_state
        self.stop_event = stop_event
        self.settings = settings
        
        # Honor env: ENABLE_BODY_RECOGNITION=false
        self.enable_body_recognition = str(os.getenv("ENABLE_BODY_RECOGNITION", "true")).strip().lower() not in (
            "0", "false", "no"
        )
        
        # Shared Components
        self.tracker = tracker
        self.identity_manager = identity_manager
        self.frame_count = 0
        self.body_embedding_interval = float(self.settings.get('body_embedding_interval', 1.5))
        self.body_embedding_budget = max(1, int(self.settings.get('body_embedding_budget', 2)))
        self.min_face_recognition_width = int(self.settings.get('min_face_recognition_width', 40))

        # Recognition enqueue throttling
        self.recognition_enqueue_interval_frames = int(self.settings.get('recognition_enqueue_interval_frames', 2))
        self.recognition_min_hits = int(self.settings.get('recognition_min_hits', 6))
        self._last_enqueued_frame_for_track = {}

        # Inference pipeline
        self.inference_queue = inference_queue
        self.inference_result_queue = inference_result_queue
        self._last_inference_frame_id = -1
        self._last_detection_sent_frame = -999

    def _build_detections(self, frame, faces, bodies):
        """Build detections list from faces and bodies (same logic as before)."""
        detections = []
        used_faces = set()

        for bx, by, bw, bh in bodies:
            body_crop = frame[by:by+bh, bx:bx+bw]
            best_face = None
            best_face_iou = 0.0
            for i, face in enumerate(faces):
                if i in used_faces:
                    continue
                fx, fy, fw, fh = face['x'], face['y'], face['w'], face['h']
                xA = max(bx, fx); yA = max(by, fy)
                xB = min(bx+bw, fx+fw); yB = min(by+bh, fy+fh)
                inter_area = max(0, xB-xA) * max(0, yB-yA)
                if inter_area <= 0: continue
                face_area = fw * fh
                if face_area <= 0: continue
                iou = inter_area / face_area
                if iou > 0.2 and iou > best_face_iou:
                    best_face = face; best_face_iou = iou; used_faces.add(i)

            if best_face is not None:
                detections.append({
                    'x': bx, 'y': by, 'w': bw, 'h': bh,
                    'face_img': best_face['face_img'],
                    'embedding': best_face.get('embedding'),
                    'confidence': best_face.get('confidence', 0.8),
                    'face_bbox': (best_face['x'], best_face['y'], best_face['w'], best_face['h']),
                    'body_embedding': None,
                    'body_crop': body_crop if body_crop.size > 0 else None,
                    'pose_info': best_face.get('pose_info'),
                })
            else:
                detections.append({
                    'x': bx, 'y': by, 'w': bw, 'h': bh,
                    'face_img': None, 'embedding': None, 'confidence': 0.5,
                    'face_bbox': None, 'body_embedding': None,
                    'body_crop': body_crop if body_crop.size > 0 else None,
                    'pose_info': {'angle_category': 'back'},
                })

        for i, face in enumerate(faces):
            if i in used_faces:
                continue
            detections.append({
                'x': face['x'], 'y': face['y'], 'w': face['w'], 'h': face['h'],
                'face_bbox': (face['x'], face['y'], face['w'], face['h']),
                'face_img': face.get('face_img'),
                'embedding': face.get('embedding'),
                'confidence': face.get('confidence', 0.8),
                'body_embedding': None, 'body_crop': None,
                'pose_info': face.get('pose_info'),
            })
        return detections

    def run(self):
        logger.info("🚀 DetectionThread started.")
        last_frame_time = time.time()
        from face_app.state import video_state

        while not self.stop_event.is_set():
            try:
                # --- 1. Get latest frame (non-blocking after first frame) ---
                try:
                    frame = self.frame_queue.get(timeout=0.05)
                except queue.Empty:
                    if time.time() - last_frame_time > 10:
                        logger.error("DetectionThread: No frame received for >10s!")
                        last_frame_time = time.time()
                    continue

                last_frame_time = time.time()
                self.frame_count += 1

                # --- Event gating: when idle, run heavy inference less often ---
                # This reduces CPU/GPU load drastically while still keeping the stream smooth,
                # because overlay drawing happens on every frame.
                try:
                    gating_enabled = bool(self.settings.get('event_gating_enabled', True))
                    activate_on_tracks = bool(self.settings.get('event_gating_activate_on_tracks', True))
                    active_after_track = float(self.settings.get('event_gating_active_after_track_seconds', 2.0))
                    idle_every_n = max(1, int(self.settings.get('event_gating_idle_processing_every_n_frames', 12)))
                    active_every_n = max(1, int(self.settings.get('event_gating_active_processing_every_n_frames', 1)))
                except Exception:
                    gating_enabled = True
                    activate_on_tracks = True
                    active_after_track = 2.0
                    idle_every_n = 12
                    active_every_n = 1

                is_active_window = False
                if gating_enabled:
                    # Triggered by FTP alarm snapshots
                    if getattr(self.shared_state, 'is_event_active', None) and self.shared_state.is_event_active():
                        is_active_window = True
                    # Triggered by currently visible tracks
                    if not is_active_window and activate_on_tracks:
                        try:
                            has_tracks = any(
                                (t.state == TrackState.CONFIRMED and t.misses < 5)
                                for t in getattr(self.tracker, 'tracks', [])
                            )
                            if has_tracks:
                                # Extend active window a bit to avoid oscillation
                                self.shared_state.trigger_event(active_after_track)
                                is_active_window = True
                        except Exception:
                            pass

                # Choose effective inference period
                if gating_enabled and bool(self.settings.get('event_gating_skip_detection_when_idle', True)):
                    effective_every_n = active_every_n if is_active_window else idle_every_n
                else:
                    effective_every_n = max(1, int(self.settings.get('processing_every_n_frames', 4)))

                # --- 2. Apply latest async identity resolutions ---
                resolved = self.shared_state.consume_identities()
                if resolved:
                    for track in self.tracker.tracks:
                        if track.track_id in resolved:
                            pid, name, ts = resolved[track.track_id]
                            track.person_id = pid
                            track.display_name = name
                            logger.info(f"✅ Track {track.track_id} updated with Identity P{pid} ({name})")

                # --- 3. Apply inference results if ready (non-blocking) ---
                inference_result = None
                try:
                    inference_result = self.inference_result_queue.get_nowait()
                except queue.Empty:
                    pass

                if inference_result is not None:
                    inf_frame = inference_result['frame']
                    faces = inference_result['faces']
                    bodies = inference_result['bodies']
                    _det_ms = inference_result['det_ms']

                    if self.frame_count % 60 == 0:
                        logger.info(f"📊 Detection [frame {self.frame_count}]: {len(faces)} faces, {len(bodies)} bodies, inf_time={_det_ms:.0f}ms")

                    # Update tracker settings from settings.json
                    try:
                        self.tracker.min_face_size = max(5, int(self.settings.get('min_face_size', self.tracker.min_face_size)))
                        self.tracker.iou_threshold = float(self.settings.get('tracker_iou_threshold', self.tracker.iou_threshold))
                        self.tracker.sim_threshold = float(self.settings.get('tracker_sim_threshold', self.tracker.sim_threshold))
                        self.tracker.min_hits = int(self.settings.get('tracker_min_hits_to_confirm', self.tracker.min_hits))
                        self.tracker.max_age = int(self.settings.get('tracker_max_age', self.tracker.max_age))
                        self.recognition_enqueue_interval_frames = int(self.settings.get('recognition_enqueue_interval_frames', self.recognition_enqueue_interval_frames))
                        self.recognition_min_hits = int(self.settings.get('recognition_min_hits', self.recognition_min_hits))
                        self.min_face_recognition_width = int(self.settings.get('min_face_recognition_width', self.min_face_recognition_width))
                    except Exception:
                        pass

                    detections = self._build_detections(inf_frame, faces, bodies)
                    tracks = self.tracker.update(detections, inf_frame.shape)

                    # Queue recognition for unidentified tracks
                    current_timestamp = time.time()
                    for track in tracks:
                        if track.person_id is not None: continue
                        if track.state != TrackState.CONFIRMED: continue
                        if track.hits < self.recognition_min_hits: continue
                        try:
                            last_f = self._last_enqueued_frame_for_track.get(track.track_id, -10**9)
                            if (self.frame_count - last_f) < self.recognition_enqueue_interval_frames: continue
                        except Exception: pass
                        if track.face_bbox:
                            _, _, fw, _ = track.face_bbox
                            if fw < self.min_face_recognition_width: continue
                        if track.best_face_img is None: continue

                        rec_task = {
                            'track_id': track.track_id,
                            'face_img': track.best_face_img.copy(),
                            'embedding': track.get_average_embedding(),
                            'body_embedding': track.get_average_body_embedding(),
                            'confidence': track.best_face_score,
                            'hits': track.hits,
                            'frontal_hits': getattr(track, 'frontal_hits', 0),
                            'sharpness': getattr(track, 'best_face_sharpness', 0.0),
                            'accumulated_faces': getattr(track, 'accumulated_faces', []),
                            'person_id': None,
                            'timestamp': current_timestamp,
                        }
                        try:
                            self.recognition_queue.put_nowait(rec_task)
                            self._last_enqueued_frame_for_track[track.track_id] = self.frame_count
                        except queue.Full:
                            pass

                    # Update display names
                    for track in tracks:
                        if track.person_id:
                            info = self.identity_manager.get_identity_info(track.person_id)
                            track.display_name = info['name'] if info else f"Client {track.person_id}"
                        elif not hasattr(track, 'display_name') or track.display_name in ('Unknown', None):
                            if getattr(track, 'display_name', None) != '':
                                track.display_name = 'Гость'

                    self.shared_state.update_tracks(tracks)


                # --- 4. Send frame to InferenceThread every N frames (non-blocking) ---
                if (self.frame_count - self._last_detection_sent_frame) >= effective_every_n:
                    try:
                        # Drop old task if InferenceThread is busy (LIFO behaviour)
                        while not self.inference_queue.empty():
                            try: self.inference_queue.get_nowait()
                            except queue.Empty: break
                        self.inference_queue.put_nowait({
                            'frame': frame.copy(),
                            'frame_id': self.frame_count,
                        })
                        self._last_detection_sent_frame = self.frame_count
                    except queue.Full:
                        pass  # InferenceThread busy, skip this frame

                # --- 5. ✅ ALWAYS update currently_visible_cache on EVERY frame ---
                try:
                    fs = getattr(video_state, 'global_face_system', None)
                    if fs is not None:
                        visible = []
                        _track_counter = 0
                        for t in self.tracker.tracks:
                            if t.state != TrackState.CONFIRMED and t.hits < 1: continue
                            if t.misses > 5: continue  # ✅ Increased to 5 for better retention of poorly visible faces
                            _track_counter += 1
                            bbox = getattr(t, 'face_bbox', None) or t.bbox

                            # ✅ NEW: Include live face snapshot (base64) for UI display
                            # This allows showing "who is in frame" BEFORE DB write
                            live_face_b64 = None
                            try:
                                best_img = getattr(t, 'best_face_img', None)
                                if best_img is not None and best_img.size > 0:
                                    import cv2 as _cv2
                                    import base64 as _b64
                                    # Resize to max 160px for performance
                                    _h, _w = best_img.shape[:2]
                                    _max_dim = 160
                                    if _h > _max_dim or _w > _max_dim:
                                        _scale = min(_max_dim / _h, _max_dim / _w)
                                        _ns = (int(_w * _scale), int(_h * _scale))
                                        _thumb = _cv2.resize(best_img, _ns, interpolation=_cv2.INTER_AREA)
                                    else:
                                        _thumb = best_img
                                    _, _buf = _cv2.imencode('.jpg', _thumb, [_cv2.IMWRITE_JPEG_QUALITY, 75])
                                    live_face_b64 = _b64.b64encode(_buf).decode('utf-8')
                            except Exception:
                                pass

                            visible.append({
                                'person_id': t.person_id,
                                'track_id': t.track_id,
                                'bbox': bbox,
                                'face_bbox': getattr(t, 'face_bbox', None),
                                'display_name': getattr(t, 'display_name', 'Гость'),
                                'last_seen_time': getattr(t, 'last_seen_time', time.time()),
                                'live_face_b64': live_face_b64,  # ✅ Real-time face snapshot
                                'temp_label': f'Client {_track_counter}',  # ✅ Temp label for unidentified
                                'hits': getattr(t, 'hits', 0),
                                'frontal_hits': getattr(t, 'frontal_hits', 0),
                            })
                        with fs.cache_lock:
                            fs.currently_visible_cache = visible
                except Exception:
                    pass

                # --- 6. ✅ ALWAYS draw overlay on EVERY frame (не ждём инференса!) ---
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
                frontal_hits = task.get('frontal_hits', 0)
                accumulated_faces = task.get('accumulated_faces', [])
                confidence = task.get('confidence', 0.0)
                known_pid = task.get('person_id') # Identify if track already matched
                
                # 1. Extract Embedding (if not passed)
                if embedding is None:
                    embedding = self.face_recognizer.get_embedding(face_img)

                if embedding is None and face_img is not None:
                    embedding = self.face_recognizer.get_embedding_enhanced(face_img)

                if embedding is None and face_img is not None:
                    embedding = self.face_recognizer.get_embedding_with_mbd(face_img)

                from face_app.core.settings.manager import settings
                from face_app.core.config.env import SIMILARITY_THRESHOLD

                # 2. Identify — only publish if score meets threshold
                current_threshold = float(settings.get('similarity_threshold', SIMILARITY_THRESHOLD))
                pid = None
                score = 0.0

                if embedding is not None:
                    pid, score = self.identity_manager.find_best_match(embedding)
                elif task.get('body_embedding') is not None:
                    pid, score = self.identity_manager.find_best_match(
                        None,
                        task.get('body_embedding'),
                        is_back_view=True,
                    )

                if pid and score >= current_threshold:
                    info = self.identity_manager.get_identity_info(pid)
                    name = info['name']
                    self.shared_state.publish_identity(track_id, pid, name)
                    logger.info(f"✅ Recognized {name} (Track {track_id}, Score: {score:.2f})")

                else:
                    # 3. Handle NEW Person (or match score below threshold)
                    if pid:
                        logger.debug(
                            f"⚠️ Match below threshold Track {track_id} → P{pid} (Score: {score:.2f} < {current_threshold:.2f}), treating as potential new person"
                        )

                    new_person_min_hits = int(settings.get('new_person_min_hits', 5))
                    new_person_min_confidence = float(settings.get('new_person_min_confidence', 0.65))
                    new_person_min_sharpness = float(settings.get('new_person_min_sharpness', 40.0))

                    require_frontal = bool(settings.get('require_frontal_face', True))
                    min_frontal = int(settings.get('min_frontal_frames', 3))

                    frontal_ok = True
                    if require_frontal and frontal_hits < min_frontal:
                        frontal_ok = False
                        logger.debug(
                            f"Skipping new person creation for Track {track_id}: not enough frontal frames ({frontal_hits}/{min_frontal})"
                        )

                    sharpness = task.get('sharpness', 0.0)
                    is_sharp = (sharpness >= new_person_min_sharpness)
                    if not is_sharp:
                        logger.debug(
                            f"Skipping new person creation for Track {track_id}: face not sharp enough ({sharpness:.1f} < {new_person_min_sharpness:.1f})"
                        )

                    should_create = (hits >= new_person_min_hits and confidence >= new_person_min_confidence and frontal_ok and is_sharp)

                    if should_create:
                        logger.info(
                            f"🆕 Creating new person from Track {track_id} (Hits: {hits}, Confidence: {confidence:.2f}, Sharpness: {sharpness:.1f})"
                        )
                        new_pid, new_name = self.identity_manager.create_new_person(
                            track_id=track_id,
                            embedding=embedding,
                            face_img=face_img,
                            body_embedding=task.get('body_embedding'),
                            hits=hits,
                            accumulated_faces=accumulated_faces,
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
        # ✅ PERF: Inference pipeline queues (maxsize=1 = always latest frame, no backlog)
        self.inference_queue = queue.Queue(maxsize=1)
        self.inference_result_queue = queue.Queue(maxsize=1)
        
        # Components
        self.tracker = Tracker(
            max_age=60, # INCREASED to 60 to persist tracks properly for 1-2 frames
            iou_threshold=0.3, # How close boxes must be to be considered the same person
            sim_threshold=SIMILARITY_THRESHOLD, # ✅ NEW: Enforce global similarity threshold!
            min_hits_for_candidate=3
        )
        self.identity_manager = IdentityManager()
        self.recognizer = FaceRecognizer() # For API usage

        # Honor env: ENABLE_BODY_RECOGNITION=false (avoid heavy torchreid load)
        enable_body = str(os.getenv("ENABLE_BODY_RECOGNITION", "true")).strip().lower() not in (
            "0", "false", "no"
        )
        self.body_recognizer = BodyRecognizer() if enable_body else None
        
        # Threads
        self.capture_thread = None 
        self.detect_thread = None
        self.inference_thread = None  # ✅ NEW: separate ML inference thread
        self.rec_thread = None
        self.visit_thread = None
        
        # Auto-detect headless mode if no DISPLAY is set on Linux
        # This prevents cv2.imshow from crashing the application on headless servers.
        default_headless = "1" if sys.platform.startswith("linux") and not os.getenv("DISPLAY") else "0"
        self.headless = os.getenv("HEADLESS", default_headless) != "0"
        
        # Initialize video_state early
        from face_app.state import video_state
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
        enable_body = str(os.getenv("ENABLE_BODY_RECOGNITION", "true")).strip().lower() not in (
            "0", "false", "no"
        )

        # 1. Start Capture
        self.capture_thread = CaptureThread(
            source=CAMERA_SOURCE, 
            frame_queue=self.frame_queue,
            stop_event=self.stop_event,
            shared_state=self.shared_state
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

        # 3. ✅ Start InferenceThread (InsightFace / YOLO — heavy ML work)
        self.inference_thread = InferenceThread(
            inference_queue=self.inference_queue,
            result_queue=self.inference_result_queue,
            stop_event=self.stop_event,
            enable_body_recognition=enable_body,
        )
        self.inference_thread.start()
        
        # 4. Start DetectionThread (overlay + tracker, non-blocking)
        self.detect_thread = DetectionThread(
            frame_queue=self.frame_queue,
            recognition_queue=self.recognition_queue,
            shared_state=self.shared_state,
            tracker=self.tracker,
            identity_manager=self.identity_manager,
            stop_event=self.stop_event,
            inference_queue=self.inference_queue,
            inference_result_queue=self.inference_result_queue,
        )
        self.detect_thread.start()
        
        # 5. Start Visits
        self.visit_thread = VisitThread(
            shared_state=self.shared_state, 
            stop_event=self.stop_event
        )
        self.visit_thread.start()
        
        # 6. Main Thread -> Stream / UI
        self._run_main_ui()


    def draw_overlay(self, frame):
        tracks = self.shared_state.get_active_tracks()
        
        # Pre-calculate known boxes for NMS to avoid overlapping UI boxes
        known_boxes = [info.get('face_bbox') or info['bbox'] for info in tracks.values() if info['person_id']]

        draw_boxes = settings.get('draw_boxes_enabled', True)
        draw_timer = settings.get('draw_timer_enabled', True)

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
                if info.get('hits', 0) < 3:
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
            if draw_boxes:
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Calculate Duration Timer from Shared State directly so it's accurate and continues incrementally
            if draw_timer:
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
        from face_app.state import video_state
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


def main():
    # Ensure HLS output directory exists
    import os
    hls_dir = "/tmp/hls"
    if not os.path.exists(hls_dir):
        os.makedirs(hls_dir, exist_ok=True)

    # Start HLS streamer (backend-annotated frames to HLS)
    from face_app.streaming.hls_streamer import start_hls_streamer
    start_hls_streamer()

    # Start FTP alarm snapshots receiver (in background)
    _start_ftp_server_if_enabled()

    system = MultiThreadedFaceSystem()
    system.start()


if __name__ == "__main__":
    main()
