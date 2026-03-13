import cv2
import numpy as np
from typing import List, Optional
import logging
from config import SIMILARITY_THRESHOLD
import base64


logger = logging.getLogger(__name__)


import torch
from torchvision import transforms
from PIL import Image
import torchreid


class BodyRecognizer:
    """
    BodyRecognizer: извлекает 512-мерные эмбеддинги тела с помощью OSNet (torchreid).
    Используется для ReID по силуэту, одежде, затылку.
    Также может считать количество людей в кадре (детектор "person" через OpenCV DNN).
    """
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torchreid.models.build_model(
            name='osnet_x1_0',
            num_classes=1000,
            pretrained=True
        )
        self.model.eval()
        self.model.to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # 2. Person Detection Model (YOLOv8)
        try:
            from ultralytics import YOLO
            # Using Nano model for speed on CPU/Jetson
            self.yolo_model = YOLO('yolov8n.pt') 
            logger.info("✓ YOLOv8n (Person Detector) loaded in Utils")
        except Exception as e:
            logger.error(f"Failed to load YOLOv8: {e}")
            raise

    def detect_bodies(self, frame: np.ndarray, conf_threshold=0.35) -> list:
        """
        Detect all people in the frame using YOLOv8.
        Returns: list of bounding boxes [x, y, w, h] for each detected person.
        """
        boxes = []
        try:
            # Run inference
            results = self.yolo_model(frame, classes=0, conf=conf_threshold, verbose=False)
            
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    
                    w = x2 - x1
                    h = y2 - y1
                    
                    if w < 20 or h < 50: continue 
                    
                    aspect_ratio = w / float(h)
                    if aspect_ratio > 2.0: 
                         continue

                    boxes.append([x1, y1, w, h])
            return boxes
            
        except Exception as e:
            logger.error(f"YOLO detection error: {e}")
            return []

    def get_body_embedding(self, image: np.ndarray) -> np.ndarray:
        """
        image: np.ndarray (BGR, OpenCV)
        Returns: np.ndarray (512,) or None
        """
        try:
            img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            input_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
            with torch.no_grad():
                features = self.model(input_tensor)
            emb = features.cpu().numpy().flatten()
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            return emb
        except Exception as e:
            logger.error(f"Body embedding extraction error: {e}", exc_info=True)
            return None


class FaceRecognizer:
    def __init__(self):
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
            # RESTORED to 1280x1280 for maximum accuracy (User reported missed detections)
            det_thresh = float(os.getenv("DETECTION_THRESHOLD", "0.45"))
            self.app.prepare(
                ctx_id=-1,
                det_size=(1280, 1280), # High Accuracy
            det_thresh=0.35 # BALANCED: 0.35 middle ground
            )


            logger.info(f"✓ InsightFace model loaded (buffalo_l, GlintAsia)")
            logger.info(f"  - Detection threshold: {det_thresh}")
            logger.info("  - Detection size: 1280x1280 (High Accuracy)")


        except Exception as e:
            logger.error(f"✗ Failed to load InsightFace model: {e}")
            raise
  
    def detect_faces(self, frame: np.ndarray) -> List[dict]:
        """✅ ИСПРАВЛЕНО: Улучшенная детекция с меньшей фильтрацией"""
        try:
            faces = self._detect_direct(frame)
           
            if len(faces) < 2:
                faces_enhanced = self._detect_with_enhancement(frame)
                faces = self._merge_detections(faces, faces_enhanced)
           
            faces = self._filter_quality_faces(faces)
           
            if len(faces) > 0:
                logger.debug(f"✓ Detected {len(faces)} face(s)")
           
            return faces
           
        except Exception as e:
            logger.error(f"Face detection error: {e}", exc_info=True)
            return []
  
    def _detect_direct(self, frame: np.ndarray) -> List[dict]:
        """Прямая детекция"""
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
  
    def _filter_quality_faces(self, faces: List[dict]) -> List[dict]:
        """✅ ИСПРАВЛЕНО: Фильтрация ложных срабатываний"""
        filtered = []
       
        for face in faces:
            confidence = face.get('confidence', 0.0)
            if confidence < 0.30:  # BALANCED: 0.40 -> 0.30
                logger.debug(f"Rejected: low confidence={confidence:.3f}")
                continue


            w, h = face['w'], face['h']
            if w < 25 or h < 25:  # BALANCED: 35 -> 25 (capture smaller faces)
                logger.debug(f"Rejected: too small ({w}x{h})")
                continue
           
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio < 0.35 or aspect_ratio > 3.0:  # BALANCED: 0.4-2.5 -> 0.35-3.0
                logger.debug(f"Rejected: bad aspect ratio={aspect_ratio:.2f}")
                continue
           
            embedding = face.get('embedding')
            if embedding is not None:
                emb_norm = np.linalg.norm(embedding)
                if emb_norm < 0.1 or emb_norm > 100.0:
                    logger.debug(f"Rejected: invalid embedding norm={emb_norm:.3f}")
                    continue
               
                emb_std = np.std(embedding)
                if emb_std < 0.01:  # RAISED from 0.005 to 0.01 for stricter filtering
                    logger.debug(f"Rejected: uniform embedding std={emb_std:.4f}")
                    continue
           
            face_img = face.get('face_img')
            if face_img is not None and face_img.size > 0:
                gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                mean_brightness = np.mean(gray)
                if mean_brightness < 2 or mean_brightness > 253:
                    logger.debug(f"Rejected: extreme brightness={mean_brightness:.1f}")
                    continue


                contrast = np.std(gray)
                if contrast < 8:  # RAISED from 5 to 8 for stricter quality
                    logger.debug(f"Rejected: low contrast={contrast:.1f}")
                    continue
           
            filtered.append(face)
       
        if len(faces) > len(filtered):
            logger.debug(f"Filtered: {len(faces)} -> {len(filtered)} faces (removed {len(faces) - len(filtered)} low-quality)")
       
        return filtered
  
    def _process_results(self, frame: np.ndarray, results) -> List[dict]:
        """✅ ИСПРАВЛЕНО: Обработка с более строгими условиями"""
        faces = []
        h_img, w_img = frame.shape[:2]
        crop_pad = 0.5  # 50% padding around bbox
        
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


            if w_pad > 20 and h_pad > 20:  # REDUCED from 30
                face_img = frame[y_pad:y2_pad, x_pad:x2_pad]
                confidence = float(person.det_score)


                # BALANCED for multi-person detection
                if face_img.size > 0 and confidence >= 0.30: # BALANCED: 0.40 -> 0.30
                    logger.debug(f"Accepted face: confidence={confidence:.3f}, bbox=({x_pad},{y_pad},{w_pad},{h_pad})")
                    faces.append({
                        'x': x_pad,
                        'y': y_pad,
                        'w': w_pad,
                        'h': h_pad,
                        'face_img': face_img,
                        'confidence': confidence,
                        'embedding': person.embedding
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
            if hasattr(face, 'pose'):
                pitch = float(face.pose[0] * 180 / np.pi) if face.pose is not None else 0
                yaw = float(face.pose[1] * 180 / np.pi) if face.pose is not None else 0
                roll = float(face.pose[2] * 180 / np.pi) if face.pose is not None else 0
            else:
                # Fallback: estimate from landmark positions if available
                pitch, yaw, roll = 0, 0, 0
           
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
        if abs_roll > 30:
            return "tilted"
       
        # Check for looking up/down
        if pitch > 20:
            return "looking_up"
        elif pitch < -20:
            return "looking_down"
       
        # Check for profile views
        if abs_yaw > 60:
            return "back"  # Almost back view
        elif yaw > 30:
            return "right_profile"
        elif yaw < -30:
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


# ✅ NEW: Helper for unified face creation logic
def create_or_get_face_by_embedding(
    db,
    embedding: list,
    photo_base64: str,
    name: str = None,
    angle: str = 'frontal',
    body_embedding: list = None,
    pose_info: dict = None,
    similarity_threshold: float = 0.97,
    body_similarity_threshold: float = 0.45
):
    """
    Checks if a face with a similar embedding exists.
    If yes -> returns existing face.
    If no -> creates a new face and PersonPhoto.
    
    Returns:
        (Face, PersonPhoto, is_duplicate, message)
    """
    from models import Face, PersonPhoto
    import json
    import numpy as np

    if not embedding:
        return None, None, False, "No embedding provided"

    # 1. Search for duplicates
    # Getting all faces (optimized: could use pgvector in future)
    try:
        all_faces = db.query(Face).filter(Face.status != 'deleted').all()
    except Exception as e:
        logger.error(f"DB Query failed: {e}")
        return None, None, False, "Database error"
    
    best_match = None
    best_score = 0.0
    
    target_emb = np.array(embedding)
    target_norm = np.linalg.norm(target_emb)
    
    for face in all_faces:
        if not face.embedding:
            continue
            
        try:
            curr_emb = np.array(json.loads(face.embedding))
            if curr_emb.shape != target_emb.shape:
                continue
                
            dot_product = np.dot(target_emb, curr_emb)
            norm_b = np.linalg.norm(curr_emb)
            
            sim = dot_product / ((target_norm * norm_b) + 1e-6)
            sim = (sim + 1) / 2 
            
            if sim > best_score:
                best_score = sim
                best_match = face
                
        except Exception as e:
            continue

    # 2. Check thresholds
    if best_match and best_score >= similarity_threshold:
        return best_match, None, True, f"Face exists (id={best_match.id}, score={best_score:.3f})"

    # 3. Create New Face
    try:
        new_face = Face(
            name=name,
            embedding=json.dumps(embedding),
            body_embedding=json.dumps(body_embedding) if body_embedding else None,
            photo_base64=photo_base64,
            status='new'
        )
        db.add(new_face)
        db.flush() # Get ID
        
        # 4. Create PersonPhoto record
        new_photo = PersonPhoto(
            face_id=new_face.id,
            photo_base64=photo_base64,
            embedding=json.dumps(embedding),
            body_embedding=json.dumps(body_embedding) if body_embedding else None,
            angle=angle,
            pose_pitch=pose_info.get('pitch', 0) if pose_info else 0,
            pose_yaw=pose_info.get('yaw', 0) if pose_info else 0,
            pose_roll=pose_info.get('roll', 0) if pose_info else 0
        )
        db.add(new_photo)
        db.commit()
        
        return new_face, new_photo, False, "Created new face"
        
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to create face: {e}")
        return None, None, False, str(e)
