import numpy as np
from models import Face


def find_duplicate_face_by_embedding(db_session, embedding, angle=None, body_embedding=None, similarity_threshold=0.65, body_similarity_threshold=0.35):
   """
   Improved: Find a Face in the database with an embedding similar to the given one,
   using all face and body embeddings (from Face and PersonPhoto).
   For non-frontal, allow match by face OR body embedding (not strictly both).
   Returns the Face object if found, else None.
   """
   from models import PersonPhoto
   import json
   import logging
   logger = logging.getLogger(__name__)


   def normalize_angle(a):
       if not a:
           return "unknown"
       a = a.lower()
       if a in ("затылок", "back", "back_view", "rear"):
           return "back"
       if a in ("frontal", "front"):
           return "frontal"
       if a in ("left_profile", "left"):
           return "left_profile"
       if a in ("right_profile", "right"):
           return "right_profile"
       if a in ("tilted", "tilt"):
           return "tilted"
       if a in ("looking_up", "up"):
           return "looking_up"
       if a in ("looking_down", "down"):
           return "looking_down"
       return a


   current_angle = normalize_angle(angle)
   all_faces = db_session.query(Face).all()
  
   emb_new = None
   if embedding is not None and len(embedding) > 0:
       emb_new = np.array(embedding, dtype=np.float32)
       norm_new = np.linalg.norm(emb_new)
       if norm_new > 0:
           emb_new = emb_new / norm_new
  
   body_emb_new = None
   if body_embedding is not None and len(body_embedding) > 0:
       body_emb_new = np.array(body_embedding, dtype=np.float32)
       norm_body = np.linalg.norm(body_emb_new)
       if norm_body > 0:
           body_emb_new = body_emb_new / norm_body


   best_match = None
   best_score = 0.0


   for f in all_faces:
       try:
           # Aggregate all embeddings for this person
           person_embeddings = []
          
           # 1. Main Face Embedding
           if f.embedding:
               try:
                   emb_existing = np.array(Face.embedding_to_list(f.embedding), dtype=np.float32)
                   norm_existing = np.linalg.norm(emb_existing)
                   if norm_existing > 0:
                       emb_existing = emb_existing / norm_existing
                   person_embeddings.append({
                       "embedding": emb_existing,
                       "angle": "frontal", # Assume main profile is frontal
                       "body_embedding": None
                   })
               except Exception:
                   pass


           # 2. Main Body Embedding (Backfill if available in main table)
           if hasattr(f, "body_embedding") and f.body_embedding:
               try:
                   main_body_emb = np.array(json.loads(f.body_embedding), dtype=np.float32)
                   norm_main_body = np.linalg.norm(main_body_emb)
                   if norm_main_body > 0:
                       main_body_emb = main_body_emb / norm_main_body
                   # Attach to the first embedding if exists, or create new entry
                   if person_embeddings:
                       person_embeddings[0]["body_embedding"] = main_body_emb
                   else:
                       person_embeddings.append({
                           "embedding": None,
                           "angle": "unknown",
                           "body_embedding": main_body_emb
                       })
               except Exception:
                   pass


           # 3. Additional Photos
           additional_photos = getattr(f, "additional_photos", [])
           for photo in additional_photos:
               try:
                   emb_photo = None
                   if photo.embedding:
                        emb_photo_raw = np.array(json.loads(photo.embedding), dtype=np.float32)
                        norm_photo = np.linalg.norm(emb_photo_raw)
                        if norm_photo > 0:
                            emb_photo = emb_photo_raw / norm_photo
                  
                   photo_angle = normalize_angle(getattr(photo, "angle", None) or "unknown")
                  
                   body_emb_photo = None
                   if hasattr(photo, "body_embedding") and photo.body_embedding:
                       body_emb_list = json.loads(photo.body_embedding)
                       body_emb_photo_raw = np.array(body_emb_list, dtype=np.float32)
                       norm_body_photo = np.linalg.norm(body_emb_photo_raw)
                       if norm_body_photo > 0:
                           body_emb_photo = body_emb_photo_raw / norm_body_photo
                          
                   person_embeddings.append({
                       "embedding": emb_photo,
                       "angle": photo_angle,
                       "body_embedding": body_emb_photo
                   })
               except Exception:
                   continue


           # Check matches
           for emb_info in person_embeddings:
               emb = emb_info["embedding"]
               emb_body = emb_info["body_embedding"]
              
               # Face Similarity
               sim_face = 0.0
               if emb is not None and emb_new is not None:
                   sim_face = float(np.dot(emb, emb_new))
                   sim_face = (sim_face + 1) / 2 # Normalize -1..1 to 0..1 for consitency
              
               # Body Similarity
               sim_body = 0.0
               if emb_body is not None and body_emb_new is not None:
                   sim_body = float(np.dot(emb_body, body_emb_new))
                   sim_body = (sim_body + 1) / 2


               # --- MATCHING LOGIC ---
              
               # 1. Frontal/Standard Match (High Confidence Face)
               if sim_face > similarity_threshold:
                   if sim_face > best_score:
                       best_score = sim_face
                       best_match = f
                       logger.debug(f"Match found (FACE): {f.name} (id={f.id}), score={sim_face:.3f}")


               # 2. Back/Rear View Match (Body Only or Weak Face + Body)
               elif current_angle == "back":
                   # Strong body match required for back view
                   if sim_body > 0.65: # High confidence for pure body match
                        if sim_body > best_score:
                           best_score = sim_body
                           best_match = f
                           logger.debug(f"Match found (BODY-BACK): {f.name} (id={f.id}), score={sim_body:.3f}")
              
               # 3. Hybrid Match (Weak Face + Moderate Body)
               # Useful for profiles or blurry faces where face score is 0.4-0.6
               if sim_face > 0.45 and sim_body > 0.60:
                    avg_score = (sim_face + sim_body) / 2
                    if avg_score > best_score and avg_score > similarity_threshold:
                       best_score = avg_score
                       best_match = f
                       logger.debug(f"Match found (HYBRID): {f.name} (id={f.id}), score={avg_score:.3f}")


       except Exception as e:
           logger.error(f"Error checking duplicate for face {f.id}: {e}")
           continue

   return best_match