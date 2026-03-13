import threading
import time
import logging
from enum import Enum

logger = logging.getLogger("visit_manager")

class VisitState(Enum):
    NEW = 1
    CONFIRMED = 2
    ACTIVE = 3
    LEFT = 4

class Visit:
    def __init__(self, person_id, min_confirm_frames=3, ttl=10.0, grace_period=1.0, display_name=None):
        self.person_id = person_id
        self.state = VisitState.NEW
        self.start_time = time.time()
        self.last_seen = time.time()
        self.frames_with_id = 1
        self.min_confirm_frames = min_confirm_frames
        self.ttl = ttl
        self.grace_period = grace_period
        self.visit_id = f"{person_id}_{int(self.start_time)}"
        self.display_name = display_name
        self._last_bbox = None
        self.active = True

    def update(self, bbox=None, display_name=None, timestamp=None):
        self.last_seen = timestamp if timestamp is not None else time.time()
        if bbox is not None:
            self._last_bbox = bbox
        if display_name and not self.display_name:
            self.display_name = display_name
        if self.state == VisitState.NEW and self.frames_with_id >= self.min_confirm_frames:
            self.state = VisitState.CONFIRMED
            logger.info(f"[VISIT] confirm_visit: person_id={self.person_id} visit_id={self.visit_id}")
        elif self.state == VisitState.CONFIRMED:
            self.state = VisitState.ACTIVE
            logger.info(f"[VISIT] active_visit: person_id={self.person_id} visit_id={self.visit_id}")

    def increment_frame(self):
        self.frames_with_id += 1

    def is_strictly_visible(self):
        """Returns True if the person was seen recently (strict UI check)."""
        return (time.time() - self.last_seen) < 1.0 # REDUCED 10.0s->1.0s to fix UI lag (ghosts)

    def check_left(self, active_person_ids):
        # Only close if person_id is NOT in the active list
        now = time.time()
        
        if self.person_id in active_person_ids:
            return # Still here
            
        # If not active, check TTL
        if now - self.last_seen > self.ttl + self.grace_period:
            if self.state != VisitState.LEFT:
                self.state = VisitState.LEFT
                self.active = False
                logger.info(f"[VISIT] close_visit: person_id={self.person_id} visit_id={self.visit_id}")

    def _iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
        yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou

class VisitManager:
    def __init__(self, min_confirm_frames=3, ttl=5.0, grace_period=3.0): # ADJUSTED: Aggressive Closing
        self.active_visits = {}
        self.visit_db_ids = {}  # ✅ NEW: Track DB visit IDs {person_id: db_visit_id}
        self.lock = threading.Lock()
        self.min_confirm_frames = min_confirm_frames
        self.ttl = ttl
        self.grace_period = grace_period

    def update_visit(self, person_id, bbox=None, display_name=None, timestamp=None):
        if person_id is None:
            return
        
        # DB Update Logic (Throttled)
        should_db_update = False
        
        with self.lock:
            visit = self.active_visits.get(person_id)
            if not visit or visit.state == VisitState.LEFT:
                visit = Visit(person_id, self.min_confirm_frames, self.ttl, self.grace_period, display_name)
                self.active_visits[person_id] = visit
                logger.info(f"[VISIT] create_visit: person_id={person_id} visit_id={visit.visit_id}")
                should_db_update = True # New visit -> Create DB record immediately
                # ✅ NEW: Create Visit record in DB (ONE per session)
                self._create_visit_record(person_id)
            
            # If visit was "closing" but person returned within grace period -> Reactivate
            if visit.state == VisitState.LEFT:
                 visit.state = VisitState.ACTIVE
                 visit.active = True
                 logger.info(f"[VISIT] reactivated visit: person_id={person_id}")

            visit.increment_frame()
            visit.update(bbox=bbox, display_name=display_name, timestamp=timestamp)
            
            # ✅ CHANGED: Only update Face.last_seen periodically, no new Visit records
            if time.time() - getattr(visit, "last_db_update", 0) > 1.0:
                should_db_update = True
                visit.last_db_update = time.time()

        if should_db_update:
            self._update_face_last_seen(person_id)

    def _create_visit_record(self, person_id):
        """✅ NEW: Creates a single Visit record when person enters (session start)."""
        try:
            from db import get_session
            from models import Face, Visit as DBVisit, get_bishkek_time
            
            session = get_session()
            now = get_bishkek_time()
            
            face_id = self._get_face_id(session, person_id)
            if face_id:
                # Create ONE visit record for this session
                visit = DBVisit(
                    face_id=face_id,
                    timestamp=now,  # Session start time
                    end_time=None   # Will be set when person leaves
                )
                session.add(visit)
                session.commit()
                
                # Store DB visit ID for later update
                self.visit_db_ids[person_id] = visit.id
                
                # Also update Face.last_seen
                face = session.query(Face).filter(Face.id == face_id).first()
                if face:
                    face.last_seen = now
                    face.detection_count += 1
                    session.commit()
                    
                logger.info(f"[VISIT] DB record created: person_id={person_id}, visit_id={visit.id}")
        except Exception as e:
            logger.error(f"Failed to create visit record for P{person_id}: {e}")
        finally:
            if 'session' in locals(): 
                session.close()
    
    def _update_face_last_seen(self, person_id):
        """✅ CHANGED: Only updates Face.last_seen, does NOT create Visit records."""
        try:
            from db import get_session
            from models import Face, get_bishkek_time
            
            session = get_session()
            now = get_bishkek_time()
            
            face_id = self._get_face_id(session, person_id)
            if face_id:
                face = session.query(Face).filter(Face.id == face_id).first()
                if face:
                    face.last_seen = now
                    session.commit()
        except Exception as e:
            logger.error(f"Failed to update last_seen for P{person_id}: {e}")
        finally:
            if 'session' in locals(): 
                session.close()
    
    def _close_visit_record(self, person_id):
        """✅ NEW: Updates Visit record with end_time when person leaves."""
        try:
            visit_id = self.visit_db_ids.get(person_id)
            if not visit_id:
                return  # No DB record to close
            
            from db import get_session
            from models import Visit as DBVisit, get_bishkek_time
            
            session = get_session()
            now = get_bishkek_time()
            
            visit = session.query(DBVisit).filter(DBVisit.id == visit_id).first()
            if visit:
                visit.end_time = now
                session.commit()
                logger.info(f"[VISIT] DB record closed: person_id={person_id}, visit_id={visit_id}, duration={(now - visit.timestamp).total_seconds():.1f}s")
            
            # Remove from tracking
            del self.visit_db_ids[person_id]
        except Exception as e:
            logger.error(f"Failed to close visit record for P{person_id}: {e}")
        finally:
            if 'session' in locals(): 
                session.close()

    def _get_face_id(self, session, person_id):
        # Quick helper or query
        # Ideally we'd cache this mapping in VisitManager too, but IdentityManager has it.
        # We can query Face table by person_id
        from models import Face
        f = session.query(Face).filter(Face.person_id == person_id).first()
        return f.id if f else None

    def update_by_bbox(self, bbox, all_tracks, name_map):
        with self.lock:
            for visit in self.active_visits.values():
                if visit.state == VisitState.LEFT:
                    continue
                for track in all_tracks:
                    if track.bbox and visit._last_bbox:
                        if visit._iou(track.bbox, visit._last_bbox) > 0.3:
                            visit.last_seen = time.time()
                            if track.person_id and not visit.display_name and name_map and track.person_id in name_map:
                                visit.display_name = name_map[track.person_id]

    def check_left(self, active_person_ids):
        with self.lock:
            for person_id, visit in list(self.active_visits.items()):
                visit.check_left(active_person_ids)
                if visit.state == VisitState.LEFT:
                    # ✅ NEW: Close visit record in DB when person leaves
                    self._close_visit_record(person_id)
                    del self.active_visits[person_id]

    def get_active_visits(self, strict=False):
        with self.lock:
            visits = [
                v for v in self.active_visits.values()
                if v.state in (VisitState.NEW, VisitState.CONFIRMED, VisitState.ACTIVE) and v.active
            ]
            if strict:
                visits = [v for v in visits if v.is_strictly_visible()]
            return visits

    def shutdown(self):
        """Clean up resources."""
        logger.info("VisitManager shutting down")
        with self.lock:
            self.active_visits.clear()


