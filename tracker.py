import numpy as np
import time
from enum import Enum
import logging

logger = logging.getLogger("tracker")

class TrackState(Enum):
    TENTATIVE = 1  # Just seen, waiting for confirmation
    CONFIRMED = 2  # Stable track
    LOST = 3       # Temporarily lost (occlusion)
    DELETED = 4    # Gone for good

class Track:
    _id_counter = 0

    def __init__(self, bbox, embedding, body_embedding, frame_img, min_hits_to_confirm=3):
        Track._id_counter += 1
        self.track_id = Track._id_counter
        self.min_hits_to_confirm = min_hits_to_confirm
        
        self.state = TrackState.TENTATIVE
        self.bbox = bbox # (x, y, w, h)
        self.face_bbox = None # ✅ ADDED for visualization
        self.embeddings = [embedding] if embedding is not None else []
        self.body_embeddings = [body_embedding] if body_embedding is not None else []
        
        # History
        self.first_seen_time = time.time()
        self.last_seen_time = time.time()
        self.age_frames = 1
        self.hits = 1
        self.misses = 0
        
        self.person_id = None  # Assigned by IdentityManager
        
        # Best face for recognition (highest confidence or largest size)
        self.best_face_img = frame_img
        self.best_face_score = 0.0 # To be updated from detection confidence
        self.best_body_score = 0.0 # ✅ ADDED for body confidence tracking
        self.latest_pose = None # ✅ ADDED
        
        # Velocity for smooth stream drawing
        self.dx = 0.0
        self.dy = 0.0
        self.last_update_time = time.time()
        self._last_bbox = bbox
        self._last_face_bbox = None

    def is_active(self):
        """Returns True if the track is strictly visible (present in last frame)."""
        return self.state == TrackState.CONFIRMED and self.misses < 20 # INCREASED 5->20 for better stability

    def update(self, bbox, embedding, body_embedding, score, frame_img, pose_info=None, face_bbox=None):
        self.bbox = bbox
        if face_bbox is not None:
             self.face_bbox = face_bbox
             self._last_face_bbox = self.face_bbox # For IOU

        if pose_info:
            self.latest_pose = pose_info
        if embedding is not None:
            self.embeddings.append(embedding)
            if len(self.embeddings) > 20: # Keep last 20 embeddings
                self.embeddings.pop(0)
        
        if body_embedding is not None:
            self.body_embeddings.append(body_embedding)
            if len(self.body_embeddings) > 20:
                self.body_embeddings.pop(0)

        # Calculate velocity for extrapolation
        now = time.time()
        dt = now - self.last_update_time
        if 0 < dt < 1.0:
            self.dx = (bbox[0] - self._last_bbox[0]) / dt
            self.dy = (bbox[1] - self._last_bbox[1]) / dt
        self.last_update_time = now
        self._last_bbox = bbox
        if face_bbox is not None:
             self._last_face_bbox = face_bbox
        
        self.last_seen_time = time.time()
        self.age_frames += 1
        self.hits += 1
        self.misses = 0
        
        if self.state == TrackState.TENTATIVE and self.hits >= self.min_hits_to_confirm:
            self.state = TrackState.CONFIRMED
        elif self.state == TrackState.LOST:
            self.state = TrackState.CONFIRMED

        # Keep best face image
        if score > self.best_face_score:
            self.best_face_score = score
            self.best_face_img = frame_img

    def mark_missed(self):
        self.misses += 1
        if self.state == TrackState.CONFIRMED:
            self.state = TrackState.LOST
        
    def get_average_embedding(self):
        if not self.embeddings:
            return None
        embs = np.array(self.embeddings)
        avg = np.mean(embs, axis=0)
        norm = np.linalg.norm(avg)
        return avg / (norm + 1e-6)

    def get_average_body_embedding(self):
        if not self.body_embeddings:
            return None
        embs = np.array(self.body_embeddings)
        avg = np.mean(embs, axis=0)
        norm = np.linalg.norm(avg)
        return avg / (norm + 1e-6)

    def iou(self, bbox, face_bbox=None):
        def _calc_iou(b1, b2):
            x1, y1, w1, h1 = b1
            x2, y2, w2, h2 = b2
            
            xi1 = max(x1, x2)
            yi1 = max(y1, y2)
            xi2 = min(x1+w1, x2+w2)
            yi2 = min(y1+h1, y2+h2)
            
            inter_area = max(0, xi2-xi1) * max(0, yi2-yi1)
            union_area = (w1*h1) + (w2*h2) - inter_area
            return inter_area / (union_area + 1e-6)

        # Priority 1: If both have face_bboxes, they are the most stable anchor
        if self.face_bbox is not None and face_bbox is not None:
            face_iou = _calc_iou(self.face_bbox, face_bbox)
            # Check body fallback just in case
            body_iou = _calc_iou(self.bbox, bbox)
            return max(face_iou, body_iou)

        return _calc_iou(self.bbox, bbox)

class Tracker:
    def __init__(self, max_age=60, iou_threshold=0.15, sim_threshold=0.6, min_hits_for_candidate=2):  # INCREASED from 30 to 60 for better persistence
        self.tracks = []
        self.max_age = max_age
        self.iou_threshold = iou_threshold # REDUCED 0.3 -> 0.15 for close group tracking
        self.sim_threshold = sim_threshold # Embedding similarity check
        self.min_hits = min_hits_for_candidate
        
        # Filtering config
        self.min_face_size = 5 # ✅ REDUCED from 10 to detect distant people
        self.min_aspect_ratio = 0.5 # Tall faces
        self.max_aspect_ratio = 3.0 # Wide faces (relaxed for bodies/distant)
        
    def _is_valid_detection(self, det):
        """
        ✅ IMPROVED: Stricter validation to prevent non-human false positives
        """
        x, y, w, h = det['x'], det['y'], det['w'], det['h']
        
        # 1. Minimum size check (reject tiny/distant detections)
        # ✅ FIX: Raised from 5 to 60. Distant small faces will have ORANGE tracking boxes 
        # but will NOT be recognized or saved to the DB until they get closer.
        if w < 60 or h < 60:
            return False
        
        # 2. Aspect ratio validation (human heads are roughly square to slightly tall)
        ratio = w / float(h)
        if ratio < self.min_aspect_ratio or ratio > self.max_aspect_ratio:
            return False
        
        # 3. ✅ NEW: Confidence threshold (reject low-confidence detections)
        confidence = det.get('confidence', 0.0)
        # RELAXED: Lowered to 0.20 to catch back-of-head/blurry faces
        if confidence < 0.20:
            return False
        
        # 4. ✅ NEW: Reject detections that are too large (likely error)
        # Using a very generous limit. A close person can easily be 1000px tall.
        max_dimension = max(w, h)
        if max_dimension > 1000:  # Changed from 3000 to 1000 to prevent entire-frame false positives
            return False
        
        # 5. ✅ NEW: For body-only detections, ensure we have body embedding
        # This helps distinguish real people from objects
        if det.get('embedding') is None and det.get('body_embedding') is None:
            # No face AND no body embedding = invalid
            return False
        
        return True

    def get_active_tracks_count(self):
        """Returns count of all currently confirmed tracks (people in frame)."""
        count = 0
        for track in self.tracks:
            # Count CONFIRMED tracks that are not too stale
            if track.state == TrackState.CONFIRMED and track.misses < 15:
                count += 1
        return count

    def update(self, detections, frame_shape):
        """
        detections: list of dict {'x', 'y', 'w', 'h', 'embedding', 'confidence', 'face_img'}
        """
        # 1. Filter detections
        valid_detections = [d for d in detections if self._is_valid_detection(d)]
        
        matched_track_indices = set()
        matched_det_indices = set()
        
        # 2. Match existing tracks (IOU + Embedding)
        # Priority: IOU > Embedding for short term tracking
        
        # First Pass: IOU
        for t_idx, track in enumerate(self.tracks):
            best_iou = 0
            best_det_idx = -1
            
            for d_idx, det in enumerate(valid_detections):
                if d_idx in matched_det_indices: continue
                bbox = (det['x'], det['y'], det['w'], det['h'])
                
                iou = track.iou(bbox, face_bbox=det.get('face_bbox'))
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = d_idx
            
            # Match found via IOU
            if best_iou > self.iou_threshold: # ✅ ADDED THRESHOLD CHECK
                d = valid_detections[best_det_idx]
                
                # ✅ CRITIAL FIX: Verify Similarity if both have embeddings
                # This prevents "Client 2" from stealing "Client 1"'s ID just by walking past them
                is_mismatch = False
                track_emb = track.get_average_embedding()
                det_emb = d.get('embedding')
                
                if track_emb is not None and det_emb is not None:
                     # Calculate similarity
                     sim = np.dot(track_emb, det_emb / (np.linalg.norm(det_emb) + 1e-6))
                     if sim < self.sim_threshold: # Prevent merging different people if tracking box overlaps
                         is_mismatch = True
                         logger.debug(f"IOU Match Rejected: Track {track.track_id} vs Det {best_det_idx} (Sim: {sim:.2f} < {self.sim_threshold})")
                
                if not is_mismatch:
                    track.update(
                        (d['x'], d['y'], d['w'], d['h']), 
                        d.get('embedding'), 
                        d.get('body_embedding'),
                        d.get('confidence', 0), 
                        d.get('face_img'),
                        pose_info=d.get('pose_info'), # ✅ PASS POSE
                        face_bbox=d.get('face_bbox')
                    )
                    matched_track_indices.add(t_idx)
                    matched_det_indices.add(best_det_idx)
        
        # Second Pass: Embedding for LOST tracks (recovery)
        # Only if track was recently lost and moved fast or camera jumps?
        # For now, strict IOU is safer for "sticky ID", but let's try embedding match for unclaimed
        for t_idx, track in enumerate(self.tracks):
            if t_idx in matched_track_indices: continue
            
            best_sim = 0
            best_det_idx = -1
            track_avg_emb = track.get_average_embedding()
            
            if track_avg_emb is None: continue
            
            for d_idx, det in enumerate(valid_detections):
                if d_idx in matched_det_indices: continue
                
                det_emb = det.get('embedding')
                if det_emb is None: continue
                
                # Normalize just in case
                det_emb = det_emb / (np.linalg.norm(det_emb) + 1e-6)
                
                sim = np.dot(track_avg_emb, det_emb)
                if sim > best_sim:
                    best_sim = sim
                    best_det_idx = d_idx
            
            if best_sim > self.sim_threshold and best_det_idx != -1:
                 # Match found via Embedding (Recovery)
                d = valid_detections[best_det_idx]
                track.update(
                    (d['x'], d['y'], d['w'], d['h']), 
                    d.get('embedding'), 
                    d.get('body_embedding'),
                    d.get('confidence', 0), 
                    d.get('face_img'),
                    pose_info=d.get('pose_info'), # ✅ PASS POSE
                    face_bbox=d.get('face_bbox')
                )
                matched_track_indices.add(t_idx)
                matched_det_indices.add(best_det_idx)
                # print(f"Recovered track {track.track_id} via embedding sim {best_sim:.2f}")

        # 3. Handle unassigned -> Missed
        for t_idx, track in enumerate(self.tracks):
            if t_idx not in matched_track_indices:
                track.mark_missed()

        # 4. Handle unclaimed detections -> New Tracks
        for d_idx, det in enumerate(valid_detections):
            if d_idx not in matched_det_indices:
                new_track = Track(
                    (det['x'], det['y'], det['w'], det['h']),
                    det.get('embedding'),
                    det.get('body_embedding'), # ✅ ADDED
                    det.get('face_img'),
                    min_hits_to_confirm=self.min_hits
                )
                new_track.face_bbox = det.get('face_bbox')
                new_track.best_face_score = det.get('confidence', 0.0)
                self.tracks.append(new_track)

        # 5. Cleanup Dead Tracks
        # Remove if dead for too long (max_age frames)
        self.tracks = [t for t in self.tracks if t.misses < self.max_age]
        
        # 6. Internally Deduplicate Overlapping Tracks (Self-Healing)
        # If two tracks overlap by >60%, they are highly likely identical.
        # We keep the one that is recognized, or older.
        tracks_to_drop = set()
        for i, t1 in enumerate(self.tracks):
             if i in tracks_to_drop: continue
             for j, t2 in enumerate(self.tracks):
                  if i >= j or j in tracks_to_drop: continue
                  
                  # Use face_bbox if available for precise matching
                  overlap = t1.iou(t2.bbox, face_bbox=t2.face_bbox)
                  if overlap > 0.70: # DECREASED from 0.85 to correctly fuse duplicates on same person
                      # Determine which to keep
                      if t2.person_id and not t1.person_id:
                          tracks_to_drop.add(i)
                          break # t1 is dead, stop checking t1 against others
                      elif t1.person_id and not t2.person_id:
                          tracks_to_drop.add(j)
                      elif t2.hits > t1.hits:
                          tracks_to_drop.add(i)
                          break
                      else:
                          tracks_to_drop.add(j)
                          
        self.tracks = [t for i, t in enumerate(self.tracks) if i not in tracks_to_drop]

        return self.tracks
