import logging
import time
import numpy as np
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed

from models import Face, Visit
from face_recognition import FaceRecognizer, BodyRecognizer
from tracker import Tracker, TrackState
from identity_manager import IdentityManager
from visit_manager import VisitManager
from config import SIMILARITY_THRESHOLD

logger = logging.getLogger(__name__)

class RecognitionService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RecognitionService, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def initialize(self):
        if self.initialized:
            return
        
        logger.info("🚀 Initializing RecognitionService...")
        
        # 1. Core Components
        self.recognizer = FaceRecognizer()
        self.body_recognizer = BodyRecognizer()
        self.tracker = Tracker(
            max_age=90,
            iou_threshold=0.35, # Synced with Tracker defaults
            sim_threshold=0.55,
            min_hits_for_candidate=3 # Synced with Tracker defaults
        )
        self.identity_manager = IdentityManager()
        self.visit_manager = VisitManager(
            min_confirm_frames=3,
            ttl=5.0,
            grace_period=1.0
        )
        
        self.embedding_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="embedding")
        self.async_stats = {'concurrent_tasks': 0, 'max_concurrent': 0}
        
        self.latest_raw_detections = []
        self.latest_body_count = 0
        
        self.initialized = True
        logger.info("✓ RecognitionService initialized")

    def process_snapshot(self, frame: np.ndarray):
        """
        Process a single frame (snapshot) from the frontend.
        Returns: list of detections with names and coordinates.
        """
        start_time = time.time()
        
        # 1. DETECT FACES FIRST (primary source)
        faces = self.recognizer.detect_faces(frame)
        
        # 🛡️ GHOST FILTER: Remove small face detections
        H, W = frame.shape[:2]
        frame_area = W * H
        
        # Filter Faces (< 0.8% of screen) - INCREASED from 0.5%
        faces = [f for f in faces if (f['w'] * f['h']) > (frame_area * 0.008)]
        
        final_detections = []
        
        # ✅ PRIMARY LOGIC: Process each FACE detection
        for face in faces:
            fx, fy, fw, fh = face['x'], face['y'], face['w'], face['h']
            
            # Apply shrink for tighter box
            shrink = 0.88  # 88% of face box
            new_fw, new_fh = int(fw * shrink), int(fh * shrink)
            offset_x, offset_y = (fw - new_fw) // 2, (fh - new_fh) // 2
            
            det = {
                'x': fx + offset_x,
                'y': fy + offset_y,
                'w': new_fw,
                'h': new_fh,
                'embedding': face.get('embedding'),
                'body_embedding': None,  # Will try to get body embedding
                'confidence': face.get('confidence', 0.5),
                'name': None,
                'person_id': None,
                'face_img': face.get('face_img')
            }
            
            # Try to extract body region for body embedding
            # Expand face box to estimate body region
            body_x = max(0, fx - int(fw * 0.5))
            body_y = fy  # Start from face top
            body_w = min(W - body_x, int(fw * 2))  # 2x face width
            body_h = min(H - body_y, int(fh * 4))  # 4x face height
            
            if body_w > 50 and body_h > 100:
                body_crop = frame[body_y:body_y+body_h, body_x:body_x+body_w]
                if body_crop.size > 0:
                    try:
                        body_emb = self.body_recognizer.get_body_embedding(body_crop)
                        det['body_embedding'] = body_emb
                    except:
                        pass
            
            final_detections.append(det)

        logger.debug(f"Detected {len(final_detections)} faces")
        self.latest_raw_detections = final_detections

        # 2. TRACK
        tracks = self.tracker.update(final_detections, frame.shape)
        
        # 3. IDENTIFY
        self.identity_manager.resolve_identities(tracks)
        
        # 4. VISIT
        for track in tracks:
            if track.person_id:
                info = self.identity_manager.get_identity_info(track.person_id)
                self.visit_manager.update_visit(
                    track.person_id,
                    bbox=track.bbox,
                    display_name=info['name'],
                    timestamp=getattr(track, 'last_seen_time', time.time())
                )
        
        self.visit_manager.check_left(tracks)

        # Build Response
        results = []
        H, W = frame.shape[:2]  # Frame dimensions for normalization
        
        for track in tracks:
             if track.state == TrackState.LOST and (time.time() - track.last_seen_time > 1.0):
                 continue

             info = {}
             if track.person_id:
                 meta = self.identity_manager.get_identity_info(track.person_id)
                 name = meta['name']
                 is_employee = meta['is_employee']
             else:
                 name = "Analyzing..."
                 is_employee = False
             
             # ✅ FIX: Get actual similarity from track best_face_score or match score
             similarity = 0.0
             if hasattr(track, 'best_match_score') and track.best_match_score:
                 similarity = track.best_match_score
             elif hasattr(track, '_best_score_this_frame'):
                 similarity = track._best_score_this_frame
             elif hasattr(track, 'best_face_score'):
                 similarity = track.best_face_score
             
             # Face Image (Base64) for Sidebar
             face_b64 = ""
             if hasattr(track, 'best_face_img') and track.best_face_img is not None:
                 try:
                    face_b64 = self.recognizer.face_to_base64(track.best_face_img)
                 except: pass

             # ✅ ADD: Normalized coordinates for resolution-independent rendering
             x, y, w, h = track.bbox
             bbox_norm = [x / W, y / H, w / W, h / H]

             results.append({
                 "track_id": track.track_id,
                 "bbox": [int(b) for b in track.bbox],
                 "bbox_norm": bbox_norm,  # ✅ For smooth rendering
                 "name": name,
                 "is_employee": is_employee,
                 "person_id": track.person_id,
                 "similarity": round(similarity, 2),
                 "confidence": round(similarity, 2),  # Alias for compatibility
                 "face_base64": face_b64,
                 "recognized": track.person_id is not None
             })
             
        logger.debug(f"Snapshot processed in {time.time() - start_time:.3f}s. {len(results)} tracks.")
        return results

    def _extract_body_embedding_task(self, body_crop, body_idx, body_box):
        try:
            body_emb = self.body_recognizer.get_body_embedding(body_crop)
            return body_idx, body_box, body_emb
        except Exception as e:
            logger.error(f"Async body embedding extraction failed: {e}")
            return body_idx, body_box, None

# Global Accessor
recognition_service = RecognitionService()
