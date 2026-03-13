import logging
import json
import threading
import time
import numpy as np
from db import get_session
from models import Face, Visit, PersonPhoto, get_bishkek_time
from tracker import TrackState

logger = logging.getLogger("identity_manager")

class IdentityManager:
    def __init__(self):
        self.lock = threading.Lock() # ✅ Thread safety
        self.person_embeddings = {}   # {pid: [(angle, emb), ...]}
        self.person_body_embeddings = {} # {pid: [body_emb, ...]}
        self.person_is_employee = {}  # {pid: bool}
        self.person_id_to_name = {}   # {pid: name}
        self.person_to_face_id = {}   # {pid: face_id} (primary DB id)
        
        # Matrix Optimization
        self.face_matrix = None
        self.face_meta = []
        self.body_matrix = None
        self.body_meta = []
        
        # Configuration
        from config import SIMILARITY_THRESHOLD
        self.match_threshold_strict = SIMILARITY_THRESHOLD # Use threshold from config
        self.match_threshold_loose = SIMILARITY_THRESHOLD * 0.85
        self.new_person_threshold = 0.55 # RAISED: must be close to SIMILARITY_THRESHOLD to prevent wrong person creation
        
        self.min_frames_to_create = 3      # Need 3 frames to confirm identity before creating
        self.min_frames_to_match = 1
        
        self.embedding_history_size = 20
        self.last_reload_time = time.time()
        
        # ✅ DEDUPLICATION CACHE
        # {track_id: (pid, timestamp)} to prevent multiple creations for same track in short window
        self.recent_creations = {} 
        
        # Load initial data
        self.load_embeddings()

    def load_embeddings_background(self):
        """Starts load_embeddings in a separate thread."""
        t = threading.Thread(target=self.load_embeddings, daemon=True)
        t.start()
        logger.info("Started background embedding reload...")

    def load_embeddings(self):
        """Refreshes in-memory cache of embeddings from DB and builds optimization matrices."""
        session = get_session()
        try:
            faces = session.query(Face).all()
            
            # 1. Structures for logical lookups
            person_is_employee = {}
            person_id_to_name = {}
            person_to_face_id = {}
            
            # 2. Lists for Matrix construction
            face_embs_list = []
            face_meta_list = [] # (pid, angle)
            
            body_embs_list = []
            body_meta_list = [] # (pid)
            
            count = 0
            for face in faces:
                pid = face.person_id
                if not pid: continue
                
                person_is_employee[pid] = (face.status == 'employee')
                name = getattr(face, "full_name", None) or getattr(face, "name", None) or f"Client {pid}"
                person_id_to_name[pid] = name
                person_to_face_id[pid] = face.id

                # Primary Face Embedding
                try:
                    emb = np.array(json.loads(face.embedding), dtype=np.float32)
                    norm = np.linalg.norm(emb)
                    if norm > 0: 
                        emb /= norm
                        face_embs_list.append(emb)
                        face_meta_list.append((pid, 'frontal'))
                        count += 1
                except Exception:
                    pass
                
                # Additional Photos Embeddings
                for photo in getattr(face, "additional_photos", []):
                    try:
                        p_emb = np.array(json.loads(photo.embedding), dtype=np.float32)
                        p_norm = np.linalg.norm(p_emb)
                        if p_norm > 0: 
                            p_emb /= p_norm
                            angle = getattr(photo, "angle", "unknown")
                            face_embs_list.append(p_emb)
                            face_meta_list.append((pid, angle))
                            count += 1
                        
                        # Body embedding from photo
                        if photo.body_embedding:
                            b_emb = np.array(json.loads(photo.body_embedding), dtype=np.float32)
                            b_norm = np.linalg.norm(b_emb)
                            if b_norm > 0: 
                                b_emb /= b_norm
                                body_embs_list.append(b_emb)
                                body_meta_list.append(pid)
                    except Exception:
                        pass
                
                # Body embedding from Main Face
                if face.body_embedding:
                    try:
                        b_emb_main = np.array(json.loads(face.body_embedding), dtype=np.float32)
                        b_norm_m = np.linalg.norm(b_emb_main)
                        if b_norm_m > 0: 
                            b_emb_main /= b_norm_m
                            body_embs_list.append(b_emb_main)
                            body_meta_list.append(pid)
                    except:
                        pass
            
            # 3. Build Matrices
            with self.lock:
                self.person_is_employee = person_is_employee
                self.person_id_to_name = person_id_to_name
                self.person_to_face_id = person_to_face_id
                
                self.face_matrix = np.array(face_embs_list) if face_embs_list else None
                self.face_meta = face_meta_list
                
                self.body_matrix = np.array(body_embs_list) if body_embs_list else None
                self.body_meta = body_meta_list
            
            self.last_reload_time = time.time()
            logger.info(f"Loaded {len(face_meta_list)} face embeddings and {len(body_meta_list)} body embeddings using Matrix Optimization")
            
        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
        finally:
            session.close()

    def find_best_match(self, face_embedding, body_embedding=None, is_back_view=False):
        """
        Matrix-Optimized matching. 
        Calculates dot products against all embeddings in one go.
        """
        # Lock to ensure matrix and meta lists are consistent during reload
        with self.lock:
            best_pid = None
            best_score = 0.0
            
            best_body_pid = None
            best_body_score = 0.0
            
            # 1. Face Matching (Matrix Multiply)
            if face_embedding is not None and not is_back_view and self.face_matrix is not None:
             # score = vector * matrix.T
             scores = np.dot(self.face_matrix, face_embedding)
             best_idx = np.argmax(scores)
             best_face_score = float(scores[best_idx])
             
             if best_face_score > 0:
                 best_pid, _ = self.face_meta[best_idx]
                 best_score = best_face_score

            # 2. Body Matching (Matrix Multiply)
            if body_embedding is not None and self.body_matrix is not None:
                 b_scores = np.dot(self.body_matrix, body_embedding)
                 best_b_idx = np.argmax(b_scores)
                 best_body_score = float(b_scores[best_b_idx])
                 
                 if best_body_score > 0:
                     best_body_pid = self.body_meta[best_b_idx]

            # 3. Decision Logic
            if is_back_view:
                 # STRICT: Back views MUST match via body
                if best_body_score > 0.50:  # RAISED from 0.35 to prevent wrong body matches
                    return best_body_pid, best_body_score
                else:
                    return None, 0.0
            else:
                # Normal/front view
                # Use STRICT threshold from config (e.g. 0.60)
                if best_score > self.match_threshold_strict:
                    # Validation boost
                    if body_embedding is not None and best_body_pid == best_pid and best_body_score > 0.45:
                         combined_score = (best_score * 0.7 + best_body_score * 0.3)
                         return best_pid, combined_score
                    else:
                        return best_pid, best_score
                
                # Fallback for weak face — ONLY if body match is very strong
                elif best_body_score > 0.50:  # RAISED from 0.35 to prevent wrong body matches
                    return best_body_pid, best_body_score
            
            # ✅ FIX: If no match met threshold, return None (was returning best_pid with low score)
            return None, 0.0

    def resolve_identities(self, tracks):
        """
        Main entry point. Iterates tracks and assigns identities using a unique assignment strategy.
        Now allows identifying TENTATIVE tracks if they have a strong match.
        """
        
        # 0. PERIODIC OR ON-DEMAND RELOAD
        reload_needed = False
        if time.time() - getattr(self, "last_reload_time", 0) > 30.0:
            reload_needed = True
        
        # Check for flag file (instant reload after admin action)
        import os
        if os.path.exists("RELOAD_EMBEDDINGS.flag"):
            logger.info("🚩 Detected RELOAD_EMBEDDINGS.flag, reloading...")
            reload_needed = True
            try:
                os.remove("RELOAD_EMBEDDINGS.flag")
            except:
                pass

        if reload_needed:
            self.load_embeddings()

        # 1. Collect potential matches for ALL tracks (Confirmed OR Tentative)
        potential_matches = []
        for track in tracks:
            if track.state == TrackState.DELETED:
                continue
            
            # Reset score for this frame
            track._best_score_this_frame = 0.0
            
            # ✅ OPTIMIZATION: Skip expensive search for STABLE tracks
            # If track is confirmed, has an ID, and we are confident -> only check every 10 frames
            # This satisfies "check every once in a while without load"
            if (track.person_id is not None and 
                track.state == TrackState.CONFIRMED and 
                track.hits > 15 and 
                track.hits % 10 != 0):
                continue
            
            # ✅ STRICT STICKY LOGIC: Once identified, NEVER change the person_id while track is alive
            if track.person_id is not None:
                # We simply stick to the assigned logic
                continue

            # ✅ STABILITY CHECK: Do not assign IDs to unconfirmed tracks
            if track.state == TrackState.TENTATIVE or track.hits < 5:
                continue

            avg_emb = track.get_average_embedding()
            avg_body_emb = track.get_average_body_embedding()
            if avg_emb is None and avg_body_emb is None:
                continue
            
            # Identify pose
            is_back_view = False
            # Fallback to pose_info from detect_head_pose or virtual head logic
            angle = "frontal"
            if hasattr(track, 'latest_pose') and track.latest_pose:
                angle = track.latest_pose.get('angle_category', 'frontal')
                if angle in ('back', 'back_view', 'rear', 'затылок'):
                    is_back_view = True
            
            pid, score = self.find_best_match(avg_emb, avg_body_emb, is_back_view)
            
            # Optimization: Tentative tracks need STRONGER match to be identified immediately
            threshold = self.match_threshold_strict
            if track.state == TrackState.TENTATIVE:
                threshold = max(threshold, 0.5) 
            
            if score > threshold:
                potential_matches.append({
                    'track': track,
                    'pid': pid,
                    'score': score
                })
            else:
                # Store potential match for "ambiguous" check
                track._best_score_this_frame = score
                track.best_match_score = score  # ✅ NEW: preserved for frontend display
                if score > 0.1:
                    logger.debug(f"Track {track.track_id} match P{pid} below threshold: {score:.2f} < {threshold:.2f}")

        # 2. Assign unique IDs (Starting from best match)
        potential_matches.sort(key=lambda x: x['score'], reverse=True)
        
        # ✅ ONLY tracks that were seen IN THIS FRAME (misses == 0) block their IDs
        # BUGFIX: We MUST protect all assigned IDs, not just visible ones, to stop ID stealing
        assigned_pids = {t.person_id for t in tracks if t.person_id is not None}
        assigned_tracks = {t.track_id for t in tracks if t.person_id is not None}
        
        for match in potential_matches:
            if match['pid'] in assigned_pids or match['track'].track_id in assigned_tracks:
                continue

            logger.info(f"[IDENTITY] Assigning person_id={match['pid']} to track_id={match['track'].track_id} (score={match['score']:.2f}, hits={match['track'].hits})")
            match['track'].person_id = match['pid']
            assigned_pids.add(match['pid'])
            assigned_tracks.add(match['track'].track_id)
            logger.info(f"Identified Track {match['track'].track_id} as P{match['pid']} (score={match['score']:.2f}, hits={match['track'].hits})")

        # 3. Handle New Person Creation
        for track in tracks:
            # ✅ STRICTER: Need more frames and higher face quality to create new person
            if (track.state == TrackState.CONFIRMED and 
                track.person_id is None and
                track.hits >= 20 and  # Raised from 15 to 20 to ensure stability before creation
                track.best_face_score >= 0.70):  # Raised from 0.60 to 0.70 to prevent duplicate artifacts
                avg_emb = track.get_average_embedding()
                avg_body_emb = track.get_average_body_embedding()
                new_pid, _ = self.create_new_person(
                    track_id=track.track_id, 
                    embedding=avg_emb, 
                    face_img=track.best_face_img,
                    body_embedding=avg_body_emb,
                    hits=track.hits
                )
                if new_pid:
                    logger.info(f"[IDENTITY] Created NEW person_id={new_pid} for track_id={track.track_id} (Hits: {track.hits}, Score: {track.best_face_score:.2f})")
                    track.person_id = new_pid
                logger.info(f"Created NEW person P{track.person_id} from Track {track.track_id} (Hits: {track.hits}, Score: {track.best_face_score:.2f})")
        return

    def create_new_person(self, track_id, embedding, face_img, body_embedding=None, hits=0):
        """Creates a new Face entry in DB and updates memory. Returns (pid, name)."""
        
        # ✅ DEDUPLICATION CHECK
        # If we just created a person for this track, return it immediately
        # This prevents "Queue Buffer" race condition where 10 frames queue up and all trigger creation
        if track_id in self.recent_creations:
            pid, ts = self.recent_creations[track_id]
            if time.time() - ts < 30.0: # Valid for 30 seconds
                name = self.person_id_to_name.get(pid, f"Client {pid}")
                logger.info(f"♻️ Deduplicated creation for Track {track_id} -> Existing P{pid}")
                return pid, name
            else:
                del self.recent_creations[track_id]

        session = get_session()
        new_pid = None
        name = "Unknown"
        
        try:
            # Find next free Person ID
            max_pid = session.query(Face.person_id).order_by(Face.person_id.desc()).first()
            new_pid = (max_pid[0] + 1) if max_pid and max_pid[0] else 1
            name = f"Client {new_pid}"
            
            import cv2
            import base64
            photo_b64 = ""
            if face_img is not None:
                try:
                    # Use face_to_base64 from recognizer if available, or manual
                    _, buffer = cv2.imencode('.jpg', face_img)
                    photo_b64 = base64.b64encode(buffer).decode('utf-8')
                except:
                    pass
            
            # Handling back view / no face: save a dummy zero embedding if no face detected
            if embedding is None:
                embedding = np.zeros(512, dtype=np.float32)
                
            emb_json = json.dumps(embedding.tolist())
            
            # Prepare body embedding JSON if available
            body_emb_json = None
            if body_embedding is not None:
                body_emb_json = json.dumps(body_embedding.tolist())
            
            # ✅ CREATE NEW PERSON
            new_face = Face(
                person_id=new_pid,
                photo_base64=photo_b64,
                embedding=emb_json,
                body_embedding=body_emb_json,
                status='new',
                detection_count=hits,
                name=name,
                first_seen=get_bishkek_time(),
                last_seen=get_bishkek_time()
            )
            session.add(new_face)
            session.flush() # Get ID
            
            # ✅ SAVE BODY EMBEDDING for future ReID
            if body_embedding is not None or (embedding is not None and np.linalg.norm(embedding) > 0.01):
                pass # PersonPhoto creation logic can be here if needed for history
                # Skipping for now to keep it simple, Face table is primary.

            session.commit()
            
            # Update In-Memory Cache IMMEDIATELY
            with self.lock:
                self.person_embeddings[new_pid] = [("frontal", embedding)] if embedding is not None else []
                if body_embedding is not None:
                    self.person_body_embeddings[new_pid] = [body_embedding]
                self.person_to_face_id[new_pid] = new_face.id
                self.person_is_employee[new_pid] = False
                self.person_id_to_name[new_pid] = name
                
                # Update deduplication cache
                self.recent_creations[track_id] = (new_pid, time.time())
                
                # ✅ INCREMENTAL MATRIX UPDATE
                # Append new embedding to matrix so it can be found immediately
                if embedding is not None and np.linalg.norm(embedding) > 0:
                     # Normalize if not already
                     emb_norm = embedding / (np.linalg.norm(embedding) + 1e-6)
                     
                     if self.face_matrix is None:
                         self.face_matrix = np.array([emb_norm])
                         self.face_meta = [(new_pid, 'frontal')]
                     else:
                         self.face_matrix = np.vstack([self.face_matrix, emb_norm])
                         self.face_meta.append((new_pid, 'frontal'))
                
                if body_embedding is not None and np.linalg.norm(body_embedding) > 0:
                     b_emb_norm = body_embedding / (np.linalg.norm(body_embedding) + 1e-6)
                     
                     if self.body_matrix is None:
                         # Ensure 2D array
                         self.body_matrix = np.array([b_emb_norm])
                         self.body_meta = [new_pid]
                     else:
                         self.body_matrix = np.vstack([self.body_matrix, b_emb_norm])
                         self.body_meta.append(new_pid)
            
            return new_pid, name
            
        except Exception as e:
            logger.error(f"Failed to create new person: {e}")
            session.rollback()
            return None, None
        finally:
            session.close()

    def refresh_definitions(self):
        """Refreshes names and statuses without reloading heavy embeddings."""
        session = get_session()
        try:
            # Query only needed fields
            faces = session.query(Face.person_id, Face.name, Face.status).filter(Face.person_id != None).all()
            for pid, name, status in faces:
                self.person_is_employee[pid] = (status == 'employee')
                # Format name logic
                display_name = name if name else f"Client {pid}"
                if status == 'employee' and not name:
                    display_name = f"Employee {pid}"
                self.person_id_to_name[pid] = display_name
            logger.info("Refreshed person definitions")
        except Exception as e:
            logger.error(f"Failed to refresh definitions: {e}")
        finally:
            session.close()

    def get_identity_info(self, person_id):
        # Fallback if not found
        name = self.person_id_to_name.get(person_id, f"Client {person_id}")
        is_employee = self.person_is_employee.get(person_id, False) 
        return {
            "name": name,
            "is_employee": is_employee
        }

    def delete_person(self, person_id):
        """Immediately removes a person from in-memory structures."""
        with self.lock:
            if person_id in self.person_id_to_name:
                del self.person_id_to_name[person_id]
            if person_id in self.person_is_employee:
                del self.person_is_employee[person_id]
            if person_id in self.person_to_face_id:
                del self.person_to_face_id[person_id]
            if person_id in self.person_embeddings:
                del self.person_embeddings[person_id]
            if person_id in self.person_body_embeddings:
                del self.person_body_embeddings[person_id]
        
        
        # ✅ FORCE MATRIX REBUILD
        # We need to remove the deleted person from the search matrices so they aren't found again.
        # The easiest way is to reload definitions (fast) but matrices are heavy.
        # Let's do a targeted removal from memory structures first.
        
        # NOTE: Since matrix indices are tied to lists, simply deleting row 'i' shifts everything.
        # It's safer to just trigger a background reload or do a full reload if critical.
        # For immediate effect, we'll rely on the fact that person_embeddings[pid] is gone.
        # BUT the matrix still has the vector.
        
        # Trigger full reload to be safe (it's rare event)
        self.load_embeddings_background()
        
        logger.info(f"🗑️ Cleaned up P{person_id} from memory and triggered background reload")
