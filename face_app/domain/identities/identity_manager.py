import logging
import json
import threading
import time
import numpy as np
from face_app.db.session import get_session
from face_app.db.models import Face, Visit, PersonPhoto, get_bishkek_time
from face_app.vision.tracking.tracker import TrackState
from face_app.core.settings.manager import settings

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
        
        # Configuration (runtime-tunable via settings.json)
        # NOTE: thresholds are read dynamically in matching functions; these are just fallbacks.
        from face_app.core.config.env import SIMILARITY_THRESHOLD
        base_thr = float(settings.get('similarity_threshold', SIMILARITY_THRESHOLD))
        self.match_threshold_strict = base_thr
        self.match_threshold_loose = base_thr * 0.85
        self.new_person_threshold = float(settings.get('similarity_threshold', 0.55))
        
        self.min_frames_to_create = int(settings.get('min_frames_to_create', 3))
        self.min_frames_to_match = 1
        
        self.embedding_history_size = 20
        self.last_reload_time = time.time()
        
        # ✅ DEDUPLICATION CACHE
        # {track_id: (pid, timestamp)} to prevent multiple creations for same track in short window
        self.recent_creations = {} 
        
        # Load initial data
        self.load_embeddings()

    def _get_thresholds(self):
        """Read current thresholds from SettingsManager (live-tunable)."""
        from face_app.core.config.env import SIMILARITY_THRESHOLD
        try:
            strict = float(settings.get('similarity_threshold', SIMILARITY_THRESHOLD))
        except Exception:
            strict = SIMILARITY_THRESHOLD

        loose = strict * 0.85

        try:
            body_match_thr = float(settings.get('body_match_threshold', 0.55))
        except Exception:
            body_match_thr = 0.55

        # Validation boost threshold (body agreement for the same pid)
        body_validation_thr = 0.45

        return strict, loose, body_match_thr, body_validation_thr

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
        strict_thr, loose_thr, body_match_thr, body_validation_thr = self._get_thresholds()

        # Normalize incoming vectors so dot-product scores are stable.
        if face_embedding is not None:
            face_norm = np.linalg.norm(face_embedding)
            if face_norm > 0:
                face_embedding = face_embedding / face_norm
            else:
                face_embedding = None

        if body_embedding is not None:
            body_norm = np.linalg.norm(body_embedding)
            if body_norm > 0:
                body_embedding = body_embedding / body_norm
            else:
                body_embedding = None

        # Lock to ensure matrix and meta lists are consistent during reload
        with self.lock:
            best_pid = None
            best_score = 0.0
            
            best_body_pid = None
            best_body_score = 0.0

            # Ambiguity guard: if top-1 and top-2 are too close, we treat the match as unsafe.
            face_ambiguous = False
            
            # 1. Face Matching (Matrix Multiply)
            if face_embedding is not None and not is_back_view and self.face_matrix is not None:
                # score = matrix * vector
                scores = np.dot(self.face_matrix, face_embedding)

                if scores.size > 0:
                    # Aggregate by person_id so one person with many photos does not dominate.
                    top_k = min(8, int(scores.size))
                    top_indices = np.argpartition(scores, -top_k)[-top_k:]

                    pid_best_scores = {}
                    for idx in top_indices:
                        pid, _ = self.face_meta[int(idx)]
                        sc = float(scores[int(idx)])
                        if pid not in pid_best_scores or sc > pid_best_scores[pid]:
                            pid_best_scores[pid] = sc

                    if pid_best_scores:
                        sorted_candidates = sorted(pid_best_scores.items(), key=lambda x: x[1], reverse=True)
                        best_pid, best_score = sorted_candidates[0]

                        if len(sorted_candidates) > 1:
                            second_best = float(sorted_candidates[1][1])
                            margin = float(settings.get('match_confidence_margin', 0.05))
                            if (float(best_score) - second_best) < margin:
                                face_ambiguous = True

            # 2. Body Matching (Matrix Multiply)
            if body_embedding is not None and self.body_matrix is not None:
                b_scores = np.dot(self.body_matrix, body_embedding)
                if b_scores.size > 0:
                    top_k = min(8, int(b_scores.size))
                    top_indices = np.argpartition(b_scores, -top_k)[-top_k:]

                    pid_best_scores = {}
                    for idx in top_indices:
                        pid = self.body_meta[int(idx)]
                        sc = float(b_scores[int(idx)])
                        if pid not in pid_best_scores or sc > pid_best_scores[pid]:
                            pid_best_scores[pid] = sc

                    if pid_best_scores:
                        best_body_pid, best_body_score = max(pid_best_scores.items(), key=lambda x: x[1])

            # 3. Decision Logic
            if is_back_view:
                 # STRICT: Back views MUST match via body
                if best_body_score > body_match_thr:
                    return best_body_pid, best_body_score
                else:
                    return None, 0.0
            else:
                # Normal/front view
                # Use STRICT threshold from config (e.g. 0.60)
                if best_score > strict_thr and not face_ambiguous:
                    # Validation boost
                    if body_embedding is not None and best_body_pid == best_pid and best_body_score > body_validation_thr:
                         combined_score = (best_score * 0.7 + best_body_score * 0.3)
                         return best_pid, combined_score
                    else:
                        return best_pid, best_score

                if best_score > strict_thr and face_ambiguous:
                    logger.debug(
                        f"Ambiguous face match rejected: pid={best_pid}, score={float(best_score):.3f}"
                    )
                
                # IMPORTANT: do NOT fallback to body match if we have a face embedding.
                # Body ReID is much less specific and can cause wrong identity swaps.
                # Allow body-only matching only when face embedding is absent.
                elif face_embedding is None and best_body_score > body_match_thr:
                    return best_body_pid, float(best_body_score)
            
            # ✅ FIX: If no match met threshold, return None (was returning best_pid with low score)
            return None, 0.0

    def _find_raw_best_match(self, face_embedding):
        """Calculates the absolute best dot product score across the DB without any threshold logic."""
        if face_embedding is None or self.face_matrix is None:
            return None, 0.0

        face_norm = np.linalg.norm(face_embedding)
        if face_norm == 0:
            return None, 0.0
        face_embedding = face_embedding / face_norm

        with self.lock:
            scores = np.dot(self.face_matrix, face_embedding)
            if scores.size == 0:
                return None, 0.0
            
            # Aggregate by person_id to find the true max
            top_k = min(8, int(scores.size))
            top_indices = np.argpartition(scores, -top_k)[-top_k:]

            pid_best_scores = {}
            for idx in top_indices:
                pid, _ = self.face_meta[int(idx)]
                sc = float(scores[int(idx)])
                if pid not in pid_best_scores or sc > pid_best_scores[pid]:
                    pid_best_scores[pid] = sc

            if pid_best_scores:
                best_pid, best_score = max(pid_best_scores.items(), key=lambda x: x[1])
                return best_pid, best_score
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

        strict_thr, loose_thr, body_match_thr, body_validation_thr = self._get_thresholds()

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
            threshold = strict_thr
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
        new_person_min_hits = int(settings.get('new_person_min_hits', 10))
        new_person_min_confidence = float(settings.get('new_person_min_confidence', 0.65))
        # ✅ Минимальная резкость лица для создания новой записи (Laplacian variance)
        # Если фото размыто → ждём более чёткого кадра вместо записи плохого эмбеддинга
        new_person_min_sharpness = float(settings.get('new_person_min_sharpness', 60.0))

        for track in tracks:
            # ✅ STRICTER: Need more frames and higher face quality to create new person
            if (track.state == TrackState.CONFIRMED and 
                track.person_id is None and
                track.hits >= new_person_min_hits and  # Dynamic threshold from settings
                track.best_face_score >= new_person_min_confidence):  # Dynamic threshold from settings

                # ✅ BLUR CHECK: Не создаём запись если лицо размытое
                # Ждём пока человек не подойдёт ближе/повернётся чётко
                sharpness = getattr(track, 'best_face_sharpness', 0.0)
                if sharpness < new_person_min_sharpness:
                    logger.debug(
                        f"[IDENTITY] Track {track.track_id}: waiting for sharper face "
                        f"(sharpness={sharpness:.1f} < {new_person_min_sharpness})"
                    )
                    continue

                # ✅ POSE CHECK: Требуем СТРОГО фронтальное лицо
                # Если угол неизвестен или лицо боком — ждём
                latest_pose = getattr(track, 'latest_pose', None)
                if not latest_pose:
                    continue
                
                angle = latest_pose.get('angle_category', 'unknown')
                if angle != 'frontal':
                    logger.debug(
                        f"[IDENTITY] Track {track.track_id}: waiting for frontal pose "
                        f"(current={angle})"
                    )
                    continue
                
                min_frontal_frames = int(settings.get('min_frontal_frames', 5))
                frontal_hits = getattr(track, 'frontal_hits', 0)
                if frontal_hits < min_frontal_frames:
                    logger.debug(
                        f"[IDENTITY] Track {track.track_id}: waiting for more frontal frames "
                        f"({frontal_hits} < {min_frontal_frames})"
                    )
                    continue

                avg_emb = track.get_average_embedding()
                avg_body_emb = track.get_average_body_embedding()
                new_pid, _ = self.create_new_person(
                    track_id=track.track_id, 
                    embedding=avg_emb, 
                    face_img=track.best_face_img,
                    body_embedding=avg_body_emb,
                    hits=track.hits,
                    accumulated_faces=getattr(track, 'accumulated_faces', [])
                )
                if new_pid:
                    logger.info(f"[IDENTITY] Created NEW person_id={new_pid} for track_id={track.track_id} (Hits: {track.hits}, Score: {track.best_face_score:.2f}, Sharpness: {sharpness:.1f})")
                    track.person_id = new_pid
                logger.info(f"Created NEW person P{track.person_id} from Track {track.track_id} (Hits: {track.hits}, Score: {track.best_face_score:.2f})")
        return


    def create_new_person(self, track_id, embedding, face_img, body_embedding=None, hits=0, accumulated_faces=None):
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

        # ✅ EMBEDDING DEDUPLICATION: Before creating a new person, check if embedding
        # is already too similar to someone in the DB (to prevent duplicate profiles)
        # Threshold lowered to 0.72 to catch same-person with slightly different angle
        if embedding is not None and np.linalg.norm(embedding) > 0.1:
            dup_threshold = float(settings.get('duplicate_threshold', 0.60))
            dup_pid, dup_score = self._find_raw_best_match(embedding)
            if dup_pid is not None and dup_score >= dup_threshold:
                dup_name = self.person_id_to_name.get(dup_pid, f"Client {dup_pid}")
                logger.info(
                    f"🛑 Dedup: Track {track_id} matches existing P{dup_pid} "
                    f"({dup_name}) score={dup_score:.3f} >= {dup_threshold:.2f} → skipping creation"
                )
                # Register as recent creation so further calls are short-circuited
                self.recent_creations[track_id] = (dup_pid, time.time())
                return dup_pid, dup_name

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

            # ✅ STRICT PHOTO SELECTION FOR DB:
            # Pick the best FRONTAL photo from accumulated_faces (highest sharpness).
            # This prevents blurry/sideways photos from being saved as the profile picture.
            best_frontal_img = None
            best_frontal_sharpness = -1.0
            if accumulated_faces:
                for fd in accumulated_faces:
                    af_img = fd.get('img')
                    af_pose = fd.get('pose') or {}
                    af_angle = af_pose.get('angle_category', 'unknown')
                    if af_angle != 'frontal' or af_img is None:
                        continue
                    
                    af_score = fd.get('score', 0.0)
                    if af_score < float(settings.get('new_person_min_confidence', 0.75)):
                        continue
                    try:
                        gray = cv2.cvtColor(af_img, cv2.COLOR_BGR2GRAY)
                        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                        if sharpness > best_frontal_sharpness:
                            best_frontal_sharpness = sharpness
                            best_frontal_img = af_img
                    except Exception:
                        pass

            min_sharpness = float(settings.get('new_person_min_sharpness', 85.0))
            
            selected_photo = None
            if best_frontal_img is not None and best_frontal_sharpness >= min_sharpness:
                selected_photo = best_frontal_img
            
            if selected_photo is None:
                logger.warning(f"🛑 Skipping creation for Track {track_id} because no sharp frontal photo was found (max sharpness {best_frontal_sharpness:.1f} < {min_sharpness} or no frontal pose)")
                return None, None

            if selected_photo is not None:
                try:
                    _, buffer = cv2.imencode('.jpg', selected_photo)
                    photo_b64 = base64.b64encode(buffer).decode('utf-8')
                except Exception:
                    pass
            elif face_img is not None:
                try:
                    _, buffer = cv2.imencode('.jpg', face_img)
                    photo_b64 = base64.b64encode(buffer).decode('utf-8')
                except Exception:
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
                pass
            
            # ✅ SAVE ACCUMULATED PHOTOS (FRONTAL ONLY to DB for clean profile)
            # Only save frontal photos with sufficient sharpness to PersonPhoto table
            if accumulated_faces:
                for face_dict in accumulated_faces:
                    img = face_dict.get('img')
                    pose = face_dict.get('pose') or {}
                    if img is None: continue
                    # ✅ STRICT FILTER: only frontal, sharp photos go to DB
                    angle_cat = pose.get('angle_category', 'unknown')
                    if angle_cat != 'frontal':
                        logger.debug(f"Skipping non-frontal photo (angle={angle_cat}) for face_id={new_face.id}")
                        continue
                        
                    af_score = face_dict.get('score', 0.0)
                    if af_score < float(settings.get('new_person_min_confidence', 0.75)):
                        continue
                    try:
                        _gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        _sharp = cv2.Laplacian(_gray, cv2.CV_64F).var()
                        if _sharp < min_sharpness:
                            logger.debug(f"Skipping blurry photo (sharpness={_sharp:.1f}) for face_id={new_face.id}")
                            continue
                    except Exception:
                        pass
                    try:
                        _, buffer = cv2.imencode('.jpg', img)
                        pb64 = base64.b64encode(buffer).decode('utf-8')
                        p_photo = PersonPhoto(
                            face_id=new_face.id,
                            photo_base64=pb64,
                            embedding=emb_json,
                            angle=angle_cat if angle_cat != 'unknown' else 'frontal',
                            pose_pitch=pose.get('pitch', 0.0),
                            pose_yaw=pose.get('yaw', 0.0),
                            pose_roll=pose.get('roll', 0.0),
                            created_at=get_bishkek_time()
                        )
                        session.add(p_photo)
                    except Exception as e:
                        logger.error(f"Failed to save accumulated photo: {e}")

            session.commit()
            
            # ── Запускаем внешнюю идентификацию в фоне ───────────────────
            # Не блокирует пайплайн: проверит качество фото и отправит на сторонний сервер
            try:
                from face_app.integrations.external_identification import schedule_send
                schedule_send(new_face.id)
                logger.debug(f"[ExtInt] Запланирована отправка для face_id={new_face.id}")
            except Exception as ext_err:
                logger.warning(f"[ExtInt] Не удалось запустить внешнюю интеграцию: {ext_err}")
            
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