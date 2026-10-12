import cv2
import numpy as np
from typing import List, Optional
import logging
from face_app.core.config.env import SIMILARITY_THRESHOLD
from face_app.core.settings.manager import settings
import base64

logger = logging.getLogger(__name__)

# ✅ CPU FIX: By default onnxruntime spins up one intra-op thread PER CPU CORE for
# every single inference call (InsightFace detection/recognition, both ONNX-based).
# On a 4-core box that means each face detection/embedding call tries to grab all 4
# cores at once, starving the capture thread, HLS encoder and API - causing the
# stream to stutter/freeze even though nothing is actually wrong with the video.
# This caps the thread pool without touching model weights/accuracy - it only makes
# a single inference call finish using fewer parallel threads (slightly slower per
# call, but there is plenty of headroom at our frame-skip rate).
import onnxruntime as _ort
if not getattr(_ort.InferenceSession, "_cpu_capped", False):
    _original_ort_session_init = _ort.InferenceSession.__init__

    def _capped_ort_session_init(self, *args, **kwargs):
        if kwargs.get("sess_options") is None:
            so = _ort.SessionOptions()
            so.intra_op_num_threads = 2
            so.inter_op_num_threads = 1
            kwargs["sess_options"] = so
        return _original_ort_session_init(self, *args, **kwargs)

    _ort.InferenceSession.__init__ = _capped_ort_session_init
    _ort.InferenceSession._cpu_capped = True


import threading

class BodyRecognizer:
    """
    BodyRecognizer: извлекает 512-мерные эмбеддинги тела с помощью OSNet (torchreid).
    Используется для ReID по силуэту, одежде, затылку.
    Также может считать количество людей в кадре (детектор "person" через OpenCV DNN).
    """
    def __init__(self, load_embedding_model: bool = False):
        """Body recognizer.

        Historically this class loaded the OSNet (torchreid) model in __init__.
        That is very expensive (RAM + CPU startup + GPU init) and in most of the
        pipeline we only need *person boxes* via OpenCV DNN.

        We now load OSNet lazily on the first call to get_body_embedding().
        """

        # Lazy-loaded OSNet
        self._embedding_lock = threading.Lock()
        self._embedding_model_loaded = False
        self.device = None
        self.model = None
        self.transform = None

        if load_embedding_model:
            self._ensure_embedding_model()

        # Load OpenCV DNN person detector (MobileNet-SSD)
        import os
        from face_app.core.config.env import BASE_DIR

        model_dir = os.path.join(BASE_DIR, "models")
        prototxt_path = os.path.join(model_dir, "MobileNetSSD_deploy.prototxt")
        caffemodel_path = os.path.join(model_dir, "MobileNetSSD_deploy.caffemodel")

        self.person_net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
        self.person_class_id = 15  # MobileNet-SSD "person" class id

    def _ensure_embedding_model(self):
        """Load OSNet model lazily (thread-safe)."""
        if self._embedding_model_loaded:
            return
        with self._embedding_lock:
            if self._embedding_model_loaded:
                return

            import torch
            from torchvision import transforms
            from PIL import Image
            import torchreid

            self._torch = torch
            self._pil_image_cls = Image
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = torchreid.models.build_model(
                name="osnet_x1_0",
                num_classes=1000,
                pretrained=True,
            )
            self.model.eval()
            self.model.to(self.device)
            self.transform = transforms.Compose(
                [
                    transforms.Resize((256, 128)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )

            self._embedding_model_loaded = True

    def detect_bodies(self, frame: np.ndarray, conf_threshold=0.40) -> list:
        """
        Detect all people in the frame using OpenCV DNN.
        Returns: list of bounding boxes [x, y, w, h] for each detected person.
        """
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        self.person_net.setInput(blob)
        detections = self.person_net.forward()
        boxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            class_id = int(detections[0, 0, i, 1])
            if confidence > conf_threshold and class_id == self.person_class_id:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                # ✅ FILTER: Loosened to allow SITTING people (wider aspect ratio)
                bw, bh = x2 - x1, y2 - y1
                if bw < 25 or bh < 25: continue # Increased min size from 10 to 25
                
                aspect_ratio = bw / float(bh)
                # STRICTER: 3.5 max ratio (was 4.0) to avoid long horizontal "things"
                if aspect_ratio > 3.5: 
                    continue
                    
                boxes.append([x1, y1, bw, bh])
        return boxes

    def get_body_embedding(self, image: np.ndarray) -> np.ndarray:
        """
        image: np.ndarray (BGR, OpenCV)
        Returns: np.ndarray (512,) or None
        """
        try:
            self._ensure_embedding_model()
            if self.model is None or self.transform is None or self.device is None:
                return None

            # Use the lazily imported PIL.Image class
            img_pil = self._pil_image_cls.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            input_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
            with self._torch.no_grad():
                features = self.model(input_tensor)
            emb = features.cpu().numpy().flatten()
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            return emb
        except Exception as e:
            logger.error(f"Body embedding extraction error: {e}", exc_info=True)
            return None

class FaceRecognizer:
    def __init__(self):
        # NOTE: these values are treated as fallbacks; thresholds are refreshed dynamically.
        self.min_detection_confidence = float(settings.get("min_detection_confidence", 0.45))
        self.min_face_size = int(settings.get("min_face_size", 40))
        self._load_model()
   
    def _load_model(self):
        """✅ ИСПРАВЛЕНО: Оптимизированные настройки для лучшей детекции"""
        try:
            import insightface
            # OPTIMIZATION: Only load needed models (Detection + Recognition)
            # disabling genderage and 3d landmarks saves significant CPU time
            self.app = insightface.app.FaceAnalysis(
                name='buffalo_l', 
                allowed_modules=['detection', 'recognition']
            )
            
            import os
            # Use a LOW model-level threshold (0.25) to capture all face candidates.
            # The REAL user-facing threshold is applied in _filter_quality_faces() using
            # settings.json "min_detection_confidence". This prevents the model from
            # silently discarding faces the user wants to detect.
            model_det_thresh = 0.25
            self.app.prepare(
                ctx_id=-1,
                det_size=(640, 640),
                det_thresh=model_det_thresh
            )

            logger.info(f"✓ InsightFace model loaded (buffalo_l, GlintAsia)")
            logger.info(f"  - Model detection threshold: {model_det_thresh} (low, post-filter applies user threshold)")
            logger.info(f"  - User filter threshold: {self.min_detection_confidence}")
            logger.info("  - Detection size: 640x640")

        except Exception as e:
            logger.error(f"✗ Failed to load InsightFace model: {e}")
            raise
   
    def detect_faces(self, frame: np.ndarray) -> List[dict]:
        """✅ ИСПРАВЛЕНО: Ускоренная детекция без тяжелых fallback'ов"""
        try:
            faces = self._detect_direct(frame)
            
            # Removed the slow `_detect_with_enhancement` fallback which 
            # doubled processing time (lag) when fewer than 2 faces were found.
            
            faces = self._filter_quality_faces(faces)
            
            if len(faces) > 0:
                logger.debug(f"✓ Detected {len(faces)} face(s)")
            
            return faces
            
        except Exception as e:
            logger.error(f"Face detection error: {e}", exc_info=True)
            return []
   
    def _detect_direct(self, frame: np.ndarray) -> List[dict]:
        """Прямая детекция"""

        # Refresh thresholds from settings on every call so admin changes apply immediately.
        try:
            self.min_detection_confidence = float(settings.get("min_detection_confidence", self.min_detection_confidence))
        except Exception:
            pass
        try:
            self.min_face_size = int(settings.get("min_face_size", self.min_face_size))
        except Exception:
            pass

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.app.get(frame_rgb)
        return self._process_results(frame, results)
   
    def _detect_with_enhancement(self, frame: np.ndarray) -> List[dict]:
        """Детекция с улучшением изображения"""
        try:
            enhanced = cv2.convertScaleAbs(frame, alpha=1.3, beta=30)
            
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            frame_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
            results = self.app.get(frame_rgb)
            return self._process_results(frame, results)
            
        except Exception as e:
            logger.debug(f"Enhanced detection failed: {e}")
            return []
   
    def _merge_detections(self, faces1: List[dict], faces2: List[dict]) -> List[dict]:
        """Объединить детекции, убрав дубликаты"""
        if not faces2:
            return faces1
        
        merged = list(faces1)
        
        for face2 in faces2:
            is_duplicate = False
            bbox2 = (face2['x'], face2['y'], face2['w'], face2['h'])
            
            for face1 in faces1:
                bbox1 = (face1['x'], face1['y'], face1['w'], face1['h'])
                if self._bbox_iou(bbox1, bbox2) > 0.5:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                merged.append(face2)
        
        return merged
   
    def _bbox_iou(self, bbox1, bbox2):
        """Вычислить IoU между двумя bbox"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / (union_area + 1e-6)
   
    def _nms_faces(self, faces: List[dict], iou_threshold=0.4) -> List[dict]:
        """
        Non-Maximum Suppression (NMS) для лиц — убирает дублирующиеся перекрывающиеся боксы.
        """
        if not faces:
            return []
        boxes = np.array([[f['x'], f['y'], f['w'], f['h']] for f in faces])
        scores = np.array([f.get('confidence', 0.0) for f in faces])
        indices = cv2.dnn.NMSBoxes(
            bboxes=boxes.tolist(),
            scores=scores.tolist(),
            score_threshold=0.0,
            nms_threshold=iou_threshold
        )
        if len(indices) == 0:
            return []
        indices = indices.flatten() if hasattr(indices, 'flatten') else [i[0] for i in indices]
        return [faces[i] for i in indices]

    def _filter_quality_faces(self, faces: List[dict]) -> List[dict]:
        """✅ ИСПРАВЛЕНО: Фильтрация ложных срабатываний + NMS"""
        filtered = []
        
        for face in faces:
            confidence = float(face.get('confidence', 0.0) or 0.0)
            if confidence < self.min_detection_confidence:
                logger.debug(f"Rejected: low confidence={confidence:.3f}")
                continue

            w, h = int(face.get('w', 0) or 0), int(face.get('h', 0) or 0)
            if w < self.min_face_size or h < self.min_face_size:
                logger.debug(f"Rejected: too small ({w}x{h})")
                continue

            aspect_ratio = (w / h) if h > 0 else 0
            if aspect_ratio < 0.4 or aspect_ratio > 2.5:  # Relaxed slightly
                logger.debug(f"Rejected: bad aspect ratio={aspect_ratio:.2f}")
                continue

            embedding = face.get('embedding')
            if embedding is not None:
                emb_norm = float(np.linalg.norm(embedding))
                if emb_norm < 0.1 or emb_norm > 100.0:
                    logger.debug(f"Rejected: invalid embedding norm={emb_norm:.3f}")
                    continue

                emb_std = float(np.std(embedding))
                if emb_std < 0.005:
                    logger.debug(f"Rejected: uniform embedding std={emb_std:.4f}")
                    continue

            face_img = face.get('face_img')
            if face_img is not None and getattr(face_img, 'size', 0) > 0:
                gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                mean_brightness = float(np.mean(gray))
                if mean_brightness < 2 or mean_brightness > 253:
                    logger.debug(f"Rejected: extreme brightness={mean_brightness:.1f}")
                    continue

                contrast = float(np.std(gray))
                if contrast < 5:
                    logger.debug(f"Rejected: low contrast={contrast:.1f}")
                    continue

            filtered.append(face)
        
        if len(faces) > len(filtered):
            logger.debug(f"Filtered: {len(faces)} -> {len(filtered)} faces (removed {len(faces) - len(filtered)} low-quality)")
        
        # Применяем NMS для удаления дубликатов
        filtered_nms = self._nms_faces(filtered, iou_threshold=0.4)
        if len(filtered) > len(filtered_nms):
            logger.debug(f"NMS: {len(filtered)} -> {len(filtered_nms)} faces (removed {len(filtered) - len(filtered_nms)} overlaps)")
        return filtered_nms
   
    def _process_results(self, frame: np.ndarray, results) -> List[dict]:
        """✅ ИСПРАВЛЕНО: Обработка с более строгими условиями"""
        faces = []
        h_img, w_img = frame.shape[:2]
        crop_pad = 0.25  # 25% padding around bbox to reduce object bleed

        for person in results:
            bbox = person.bbox.astype(int)
            x, y, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

            # Add padding to bbox
            w = x2 - x
            h = y2 - y
            pad_w = int(w * crop_pad)
            pad_h = int(h * crop_pad)
            x_pad = max(0, x - pad_w)
            y_pad = max(0, y - pad_h)
            x2_pad = min(w_img, x2 + pad_w)
            y2_pad = min(h_img, y2 + pad_h)
            w_pad = x2_pad - x_pad
            h_pad = y2_pad - y_pad

            if w_pad >= self.min_face_size and h_pad >= self.min_face_size:
                face_img = frame[y_pad:y2_pad, x_pad:x2_pad]
                confidence = float(person.det_score)

                if face_img.size > 0 and confidence >= self.min_detection_confidence:
                    logger.debug(f"Accepted face: confidence={confidence:.3f}, bbox=({x_pad},{y_pad},{w_pad},{h_pad})")
                    
                    pose_info = {"angle_category": "unknown", "pitch": 0, "yaw": 0, "roll": 0}
                    if hasattr(person, 'pose') and person.pose is not None:
                        pitch = float(person.pose[0] * 180 / np.pi)
                        yaw = float(person.pose[1] * 180 / np.pi)
                        roll = float(person.pose[2] * 180 / np.pi)
                        cat = self._categorize_pose(pitch, yaw, roll)
                        pose_info = {"angle_category": cat, "pitch": round(pitch, 2), "yaw": round(yaw, 2), "roll": round(roll, 2)}
                    else:
                        pose_info = {"angle_category": "frontal", "pitch": 0, "yaw": 0, "roll": 0}

                    faces.append({
                        'x': x_pad,
                        'y': y_pad,
                        'w': w_pad,
                        'h': h_pad,
                        'face_img': face_img,
                        'confidence': confidence,
                        'embedding': getattr(person, 'embedding', None),
                        'pose_info': pose_info
                    })
                else:
                    logger.debug(f"Rejected in process: confidence={confidence:.3f}, bbox=({x_pad},{y_pad},{w_pad},{h_pad})")
            else:
                 logger.debug(f"Rejected in process: too small ({w_pad}x{h_pad})")

        return faces
   
    def get_embedding(self, face_img: np.ndarray) -> Optional[np.ndarray]:
        """Get face embedding (512-dimensional vector)"""
        try:
            if face_img is None or face_img.size == 0:
                logger.warning("Empty face image provided")
                return None
            
            # Минимальный размер для качественного embedding
            if face_img.shape[0] < 80 or face_img.shape[1] < 80:
                scale = max(80 / face_img.shape[0], 80 / face_img.shape[1])
                new_size = (int(face_img.shape[1] * scale), int(face_img.shape[0] * scale))
                face_img = cv2.resize(face_img, new_size, interpolation=cv2.INTER_CUBIC)
            
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
            faces = self.app.get(face_rgb)
            
            if len(faces) > 0:
                embedding = faces[0].embedding.astype(np.float32)
                embedding = embedding / (np.linalg.norm(embedding) + 1e-6)
                logger.debug(f"Extracted embedding with shape: {embedding.shape}")
                return embedding
            else:
                logger.warning("No face found in cropped image")
                return None
            
        except Exception as e:
            logger.error(f"Embedding extraction error: {e}", exc_info=True)
            return None
   
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            if vec1 is None or vec2 is None:
                return 0.0
            
            if vec1.shape != vec2.shape:
                logger.warning(f"Shape mismatch: {vec1.shape} vs {vec2.shape}")
                return 0.0
            
            vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-6)
            vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-6)
            
            similarity = float(np.dot(vec1_norm, vec2_norm))
            similarity = (similarity + 1) / 2
            similarity = max(0.0, min(1.0, similarity))
            
            return similarity
            
        except Exception as e:
            logger.error(f"Similarity calculation error: {e}", exc_info=True)
            return 0.0
   
    def detect_head_pose(self, face_img: np.ndarray) -> dict:
        """
        ✅ NEW: Detect head pose angles (pitch, yaw, roll) and categorize viewing angle
        
        Returns:
            dict with keys: angle_category, pitch, yaw, roll
            angle_category: 'frontal', 'left_profile', 'right_profile', 'looking_down', 
                           'looking_up', 'back', 'tilted'
        """
        try:
            if face_img is None or face_img.size == 0:
                return {"angle_category": "unknown", "pitch": 0, "yaw": 0, "roll": 0}
            
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            faces = self.app.get(face_rgb)
            
            if len(faces) == 0:
                return {"angle_category": "unknown", "pitch": 0, "yaw": 0, "roll": 0}
            
            face = faces[0]
            
            # Get pose angles from InsightFace (in radians)
            if hasattr(face, 'pose') and face.pose is not None:
                pitch = float(face.pose[0] * 180 / np.pi)
                yaw = float(face.pose[1] * 180 / np.pi)
                roll = float(face.pose[2] * 180 / np.pi)
            else:
                # Cannot determine pose -> DO NOT assume it is frontal!
                return {"angle_category": "unknown", "pitch": 0, "yaw": 0, "roll": 0}
            
            # Categorize angle based on yaw and pitch
            angle_category = self._categorize_pose(pitch, yaw, roll)
            
            return {
                "angle_category": angle_category,
                "pitch": round(pitch, 2),
                "yaw": round(yaw, 2),
                "roll": round(roll, 2)
            }
            
        except Exception as e:
            logger.error(f"Head pose detection error: {e}", exc_info=True)
            return {"angle_category": "unknown", "pitch": 0, "yaw": 0, "roll": 0}
    
    def _categorize_pose(self, pitch: float, yaw: float, roll: float) -> str:
        """Categorize face angle based on pose angles"""
        # Yaw (left-right rotation): -90 (left) to 90 (right)
        # Pitch (up-down rotation): -90 (down) to 90 (up)
        # Roll (tilt): -90 to 90
        
        abs_yaw = abs(yaw)
        abs_pitch = abs(pitch)
        abs_roll = abs(roll)
        
        # Check for extreme roll (tilted head)
        if abs_roll > 20:  # Stricter: was 30
            return "tilted"
        
        # Check for looking up/down
        if pitch > 15:     # Stricter: was 20
            return "looking_up"
        elif pitch < -15:  # Stricter: was 20
            return "looking_down"
        
        # Check for profile views
        if abs_yaw > 60:
            return "back"  # Almost back view
        elif yaw > 15:     # Stricter: was 30
            return "right_profile"
        elif yaw < -15:    # Stricter: was 30
            return "left_profile"
        else:
            return "frontal"  # Front-facing
    
    def face_to_base64(self, face_img: np.ndarray) -> str:
        """Convert face image to base64 string"""
        try:
            if face_img is None or face_img.size == 0:
                return ""
            
            max_size = 200
            h, w = face_img.shape[:2]
            if h > max_size or w > max_size:
                scale = min(max_size / h, max_size / w)
                new_size = (int(w * scale), int(h * scale))
                face_img = cv2.resize(face_img, new_size, interpolation=cv2.INTER_AREA)
            
            success, buffer = cv2.imencode('.jpg', face_img,
                                          [cv2.IMWRITE_JPEG_QUALITY, 85])
            if success:
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                return img_base64
            return ""
            
        except Exception as e:
            logger.error(f"Base64 encoding error: {e}", exc_info=True)
            return ""
   
    def base64_to_image(self, img_base64: str) -> Optional[np.ndarray]:
        """Convert base64 string to numpy array"""
        try:
            if not img_base64:
                return None
            
            img_data = base64.b64decode(img_base64)
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            logger.error(f"Base64 decoding error: {e}", exc_info=True)
            return None
    
    def get_embedding_with_mbd(self, face_img: np.ndarray) -> Optional[np.ndarray]:
        """✅ NEW: Get embedding using MBD (Multi-task Cascaded Networks) for back/profile views
        
        MBD is better at detecting faces from non-frontal angles like:
        - Back of head (затылок)
        - Side profiles (профили)
        - Tilted heads
        
        This method tries to detect and extract embedding even when standard detection fails.
        """
        try:
            if face_img is None or face_img.size == 0:
                logger.warning("Empty face image provided to MBD")
                return None
            
            # Try multiple scales for better detection
            scales = [1.0, 0.8, 1.2]
            best_embedding = None
            best_confidence = 0.0
            
            for scale in scales:
                if scale != 1.0:
                    h, w = face_img.shape[:2]
                    new_size = (int(w * scale), int(h * scale))
                    scaled_img = cv2.resize(face_img, new_size, interpolation=cv2.INTER_CUBIC)
                else:
                    scaled_img = face_img
                
                try:
                    face_rgb = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2RGB)
                    faces = self.app.get(face_rgb)
                    
                    if len(faces) > 0:
                        face = faces[0]
                        confidence = float(face.det_score)
                        
                        if confidence > best_confidence:
                            best_confidence = confidence
                            embedding = face.embedding.astype(np.float32)
                            embedding = embedding / (np.linalg.norm(embedding) + 1e-6)
                            best_embedding = embedding
                            
                            logger.debug(f"MBD: Found face at scale {scale} with confidence {confidence:.3f}")
                
                except Exception as e:
                    logger.debug(f"MBD scale {scale} failed: {e}")
                    continue
            
            if best_embedding is not None:
                logger.info(f"✅ MBD embedding extracted successfully (confidence: {best_confidence:.3f})")
                return best_embedding
            else:
                logger.debug("MBD: No face detected at any scale")
                return None
                
        except Exception as e:
            logger.error(f"MBD embedding extraction error: {e}", exc_info=True)
            return None
    
    def get_embedding_enhanced(self, face_img: np.ndarray) -> Optional[np.ndarray]:
        """✅ NEW: Get embedding with image enhancement for difficult angles
        
        Applies preprocessing to improve detection:
        - Histogram equalization
        - Contrast enhancement
        - Brightness adjustment
        """
        try:
            if face_img is None or face_img.size == 0:
                logger.warning("Empty face image provided to enhanced detection")
                return None
            
            # Try different enhancement techniques
            enhancements = [
                ("original", face_img),
                ("clahe", self._apply_clahe(face_img)),
                ("histogram", self._apply_histogram_eq(face_img)),
                ("contrast", self._apply_contrast_enhancement(face_img)),
            ]
            
            best_embedding = None
            best_confidence = 0.0
            best_method = "none"
            
            for method_name, enhanced_img in enhancements:
                if enhanced_img is None:
                    continue
                
                try:
                    face_rgb = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)
                    faces = self.app.get(face_rgb)
                    
                    if len(faces) > 0:
                        face = faces[0]
                        confidence = float(face.det_score)
                        
                        if confidence > best_confidence:
                            best_confidence = confidence
                            embedding = face.embedding.astype(np.float32)
                            embedding = embedding / (np.linalg.norm(embedding) + 1e-6)
                            best_embedding = embedding
                            best_method = method_name
                            
                            logger.debug(f"Enhanced: Found face with {method_name} (confidence: {confidence:.3f})")
                
                except Exception as e:
                    logger.debug(f"Enhanced method {method_name} failed: {e}")
                    continue
            
            if best_embedding is not None:
                logger.info(f"✅ Enhanced embedding extracted with {best_method} (confidence: {best_confidence:.3f})")
                return best_embedding
            else:
                logger.debug("Enhanced: No face detected with any enhancement")
                return None
                
        except Exception as e:
            logger.error(f"Enhanced embedding extraction error: {e}", exc_info=True)
            return None
    
    def _apply_clahe(self, img: np.ndarray) -> Optional[np.ndarray]:
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
        try:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            return enhanced
        except Exception as e:
            logger.debug(f"CLAHE failed: {e}")
            return None
    
    def _apply_histogram_eq(self, img: np.ndarray) -> Optional[np.ndarray]:
        """Apply histogram equalization"""
        try:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            v = cv2.equalizeHist(v)
            enhanced = cv2.merge([h, s, v])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_HSV2BGR)
            return enhanced
        except Exception as e:
            logger.debug(f"Histogram equalization failed: {e}")
            return None
    
    def _apply_contrast_enhancement(self, img: np.ndarray) -> Optional[np.ndarray]:
        """Apply contrast enhancement"""
        try:
            # Increase contrast using CLAHE on grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_gray = clahe.apply(gray)
            
            # Convert back to BGR
            enhanced = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
            return enhanced
        except Exception as e:
            logger.debug(f"Contrast enhancement failed: {e}")
            return None