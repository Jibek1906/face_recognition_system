import numpy as np
from face_app.db.models import Face
from face_app.domain.identities.duplicate_utils import find_duplicate_face_by_embedding

def assign_person_id_from_db(db_session, embedding, similarity_threshold=0.4):
    """
    Assigns a person_id based on full-DB similarity check.
    If similarity >= threshold to any, returns existing person_id.
    If not, creates a new person_id.
    """
    # Check for invalid embedding
    if embedding is None or (isinstance(embedding, np.ndarray) and np.linalg.norm(embedding) < 1e-6):
        return None

    # Full-DB duplicate check
    duplicate_face = find_duplicate_face_by_embedding(
        db_session, embedding,
        similarity_threshold=similarity_threshold
    )
    if duplicate_face:
        return duplicate_face.person_id

    # Create new person_id (max+1)
    max_person_id = db_session.query(Face.person_id).order_by(Face.person_id.desc()).first()
    new_person_id = (max_person_id[0] + 1) if max_person_id and max_person_id[0] else 1
    return new_person_id
