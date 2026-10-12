import logging

logger = logging.getLogger(__name__)


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
    from face_app.db.models import Face, PersonPhoto
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
