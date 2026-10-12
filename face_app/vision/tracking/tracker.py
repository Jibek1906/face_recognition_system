"""Tracker implementation.

Moved from legacy `face_system/tracker.py`.
"""

from __future__ import annotations

import logging
import time
from enum import Enum

import numpy as np


logger = logging.getLogger("tracker")


class TrackState(Enum):
    TENTATIVE = 1  # Just seen, waiting for confirmation
    CONFIRMED = 2  # Stable track
    LOST = 3  # Temporarily lost (occlusion)
    DELETED = 4  # Gone for good


class Track:
    _id_counter = 0

    def __init__(self, bbox, embedding, body_embedding, frame_img, min_hits_to_confirm=2, pose_info=None, score=0.0):
        Track._id_counter += 1
        self.track_id = Track._id_counter
        self.min_hits_to_confirm = min_hits_to_confirm

        self.state = TrackState.TENTATIVE
        self.bbox = bbox  # (x, y, w, h)
        self.face_bbox = None
        self.embeddings = [embedding] if embedding is not None else []
        self.body_embeddings = [body_embedding] if body_embedding is not None else []

        # History
        self.first_seen_time = time.time()
        self.last_seen_time = time.time()
        self.age_frames = 1
        self.hits = 1
        cat = pose_info.get('angle_category') if pose_info else None
        valid_frontal_cats = ['frontal']
        self.frontal_hits = 1 if (cat in valid_frontal_cats and score >= 0.70) else 0
        self.misses = 0

        self.person_id = None  # Assigned by IdentityManager

        # Best face for recognition
        self.best_face_img = frame_img
        self.accumulated_faces = []
        if frame_img is not None:
            self.accumulated_faces.append({'img': frame_img, 'pose': pose_info, 'score': score, 'sharpness': 0.0})
        self.best_face_score = score
        self.best_face_sharpness = 0.0
        self.best_body_score = 0.0
        self.latest_pose = pose_info

        # Velocity for stream drawing
        self.dx = 0.0
        self.dy = 0.0
        self.last_update_time = time.time()
        self._last_bbox = bbox
        self._last_face_bbox = None

    def is_active(self):
        """Returns True if the track is strictly visible (present in last frame)."""

        return self.state == TrackState.CONFIRMED and self.misses < 20

    def update(self, bbox, embedding, body_embedding, score, frame_img, pose_info=None, face_bbox=None):
        self.bbox = bbox
        if face_bbox is not None:
            self.face_bbox = face_bbox
            self._last_face_bbox = self.face_bbox

        # Calculate sharpness using Laplacian variance
        sharpness = 0.0
        if frame_img is not None and frame_img.size > 0:
            try:
                import cv2
                gray = cv2.cvtColor(frame_img, cv2.COLOR_BGR2GRAY)
                sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            except Exception:
                pass

        if pose_info:
            self.latest_pose = pose_info
            cat = pose_info.get('angle_category')
            valid_frontal_cats = ['frontal']
            # Only count as frontal hit if the detection is confident
            if cat in valid_frontal_cats and score >= 0.70:
                self.frontal_hits += 1
            
        # Only append embedding if the face is reasonably sharp or if we have very few
        if embedding is not None and (sharpness > 40.0 or len(self.embeddings) < 3):
            self.embeddings.append(embedding)
            if len(self.embeddings) > 20:
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
        # Update if the face is reasonably confident AND (it's significantly sharper OR (similarly sharp and more confident))
        # This prevents a highly confident but blurry face from overwriting a clear face
        if score >= 0.45:
            # If current best is very blurry, or new image is significantly sharper
            if self.best_face_sharpness < 50 or sharpness > self.best_face_sharpness * 1.2:
                self.best_face_score = score
                self.best_face_sharpness = sharpness
                self.best_face_img = frame_img
            # If sharpness is comparable, but confidence is better
            elif sharpness > self.best_face_sharpness * 0.8 and score > self.best_face_score:
                self.best_face_score = score
                self.best_face_sharpness = sharpness
                self.best_face_img = frame_img
                
            # Accumulate quality faces (up to 5)
            # Only accumulate frontal faces with high confidence score for DB saving
            angle_cat = pose_info.get('angle_category') if pose_info else 'unknown'
            if angle_cat == 'frontal' and sharpness > 40.0 and score >= 0.70 and frame_img is not None and frame_img.size > 0:
                if len(self.accumulated_faces) < 5:
                    self.accumulated_faces.append({'img': frame_img, 'pose': pose_info, 'sharpness': sharpness, 'score': score})
                else:
                    # Find the least sharp face
                    min_s_idx = 0
                    min_s_val = float('inf')
                    for i, fd in enumerate(self.accumulated_faces):
                        s = fd.get('sharpness', 0.0)
                        if s < min_s_val:
                            min_s_val = s
                            min_s_idx = i
                    # Replace if the new one is sharper
                    if sharpness > min_s_val:
                        self.accumulated_faces[min_s_idx] = {'img': frame_img, 'pose': pose_info, 'sharpness': sharpness, 'score': score}

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
            xi2 = min(x1 + w1, x2 + w2)
            yi2 = min(y1 + h1, y2 + h2)

            inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            union_area = (w1 * h1) + (w2 * h2) - inter_area
            return inter_area / (union_area + 1e-6)

        if self.face_bbox is not None and face_bbox is not None:
            face_iou = _calc_iou(self.face_bbox, face_bbox)
            body_iou = _calc_iou(self.bbox, bbox)
            return max(face_iou, body_iou)

        return _calc_iou(self.bbox, bbox)


class Tracker:
    def __init__(self, max_age=60, iou_threshold=0.15, sim_threshold=0.6, min_hits_for_candidate=2):
        self.tracks = []
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.sim_threshold = sim_threshold
        self.min_hits = min_hits_for_candidate

        # Filtering config
        self.min_face_size = 5
        self.min_aspect_ratio = 0.3
        self.max_aspect_ratio = 3.0

    def _is_valid_detection(self, det):
        x, y, w, h = det["x"], det["y"], det["w"], det["h"]

        # Minimum size check
        # Use configurable threshold (updated by DetectionThread from settings)
        min_size = max(10, int(getattr(self, "min_face_size", 60)))
        if w < min_size or h < min_size:
            # NOTE: keep this at DEBUG; logging per-frame rejections can consume significant CPU/disk.
            logger.debug(f"Tracker rejection: face too small ({w}x{h} < {min_size})")
            return False

        ratio = w / float(h)
        if ratio < self.min_aspect_ratio or ratio > self.max_aspect_ratio:
            logger.debug(f"Tracker rejection: bad aspect ratio {ratio:.2f}")
            return False

        confidence = det.get("confidence", 0.0)
        if confidence < 0.20:
            logger.debug(f"Tracker rejection: low confidence {confidence:.2f}")
            return False

        max_dimension = max(w, h)
        if max_dimension > 1000:
            logger.debug(f"Tracker rejection: face too large ({max_dimension} > 1000)")
            return False

        if det.get("embedding") is None and det.get("body_embedding") is None:
            # ✅ FIX: Разрешаем body-only детекции (затылок, профиль, человек на входе).
            # Если есть body_crop, трекер может использовать тело для отслеживания.
            # Полностью блокируем только если НЕТ ни embedding, ни body_crop.
            if det.get("body_crop") is None and det.get("face_img") is None:
                logger.debug(f"Tracker rejection: no embeddings and no body/face image")
                return False

        return True

    def get_active_tracks_count(self):
        count = 0
        for track in self.tracks:
            if track.state == TrackState.CONFIRMED and track.misses < 15:
                count += 1
        return count

    def update(self, detections, frame_shape):
        valid_detections = [d for d in detections if self._is_valid_detection(d)]

        matched_track_indices = set()
        matched_det_indices = set()

        # First Pass: IOU
        for t_idx, track in enumerate(self.tracks):
            best_iou = 0
            best_det_idx = -1

            for d_idx, det in enumerate(valid_detections):
                if d_idx in matched_det_indices:
                    continue
                bbox = (det["x"], det["y"], det["w"], det["h"])

                iou = track.iou(bbox, face_bbox=det.get("face_bbox"))
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = d_idx

            if best_iou > self.iou_threshold:
                d = valid_detections[best_det_idx]

                # Verify similarity if both have embeddings
                is_mismatch = False
                track_emb = track.get_average_embedding()
                det_emb = d.get("embedding")

                if track_emb is not None and det_emb is not None:
                    sim = np.dot(track_emb, det_emb / (np.linalg.norm(det_emb) + 1e-6))
                    if sim < self.sim_threshold:
                        is_mismatch = True
                        logger.debug(
                            f"IOU Match Rejected: Track {track.track_id} vs Det {best_det_idx} (Sim: {sim:.2f} < {self.sim_threshold})"
                        )

                if not is_mismatch:
                    track.update(
                        (d["x"], d["y"], d["w"], d["h"]),
                        d.get("embedding"),
                        d.get("body_embedding"),
                        d.get("confidence", 0),
                        d.get("face_img"),
                        pose_info=d.get("pose_info"),
                        face_bbox=d.get("face_bbox"),
                    )
                    matched_track_indices.add(t_idx)
                    matched_det_indices.add(best_det_idx)

        # Second Pass: Embedding for LOST tracks (recovery)
        for t_idx, track in enumerate(self.tracks):
            if t_idx in matched_track_indices:
                continue

            best_sim = 0
            best_det_idx = -1
            track_avg_emb = track.get_average_embedding()

            if track_avg_emb is None:
                continue

            for d_idx, det in enumerate(valid_detections):
                if d_idx in matched_det_indices:
                    continue

                det_emb = det.get("embedding")
                if det_emb is None:
                    continue

                det_emb = det_emb / (np.linalg.norm(det_emb) + 1e-6)

                sim = np.dot(track_avg_emb, det_emb)
                if sim > best_sim:
                    best_sim = sim
                    best_det_idx = d_idx

            if best_sim > self.sim_threshold and best_det_idx != -1:
                d = valid_detections[best_det_idx]
                track.update(
                    (d["x"], d["y"], d["w"], d["h"]),
                    d.get("embedding"),
                    d.get("body_embedding"),
                    d.get("confidence", 0),
                    d.get("face_img"),
                    pose_info=d.get("pose_info"),
                    face_bbox=d.get("face_bbox"),
                )
                matched_track_indices.add(t_idx)
                matched_det_indices.add(best_det_idx)

        # Handle unassigned -> Missed
        for t_idx, track in enumerate(self.tracks):
            if t_idx not in matched_track_indices:
                track.mark_missed()

        # Unclaimed detections -> New Tracks
        for d_idx, det in enumerate(valid_detections):
            if d_idx not in matched_det_indices:
                new_track = Track(
                    (det["x"], det["y"], det["w"], det["h"]),
                    det.get("embedding"),
                    det.get("body_embedding"),
                    det.get("face_img"),
                    min_hits_to_confirm=self.min_hits,
                    pose_info=det.get("pose_info"),
                    score=det.get("confidence", 0.0),
                )
                new_track.face_bbox = det.get("face_bbox")
                new_track.best_face_score = det.get("confidence", 0.0)
                self.tracks.append(new_track)

        # Cleanup Dead Tracks
        self.tracks = [t for t in self.tracks if t.misses < self.max_age]

        # Internally Deduplicate Overlapping Tracks (Self-Healing)
        tracks_to_drop = set()
        for i, t1 in enumerate(self.tracks):
            if i in tracks_to_drop:
                continue
            for j, t2 in enumerate(self.tracks):
                if i >= j or j in tracks_to_drop:
                    continue

                overlap = t1.iou(t2.bbox, face_bbox=t2.face_bbox)
                if overlap > 0.70:
                    if t2.person_id and not t1.person_id:
                        tracks_to_drop.add(i)
                        break
                    if t1.person_id and not t2.person_id:
                        tracks_to_drop.add(j)
                    elif t2.hits > t1.hits:
                        tracks_to_drop.add(i)
                        break
                    else:
                        tracks_to_drop.add(j)

        self.tracks = [t for i, t in enumerate(self.tracks) if i not in tracks_to_drop]

        return self.tracks
