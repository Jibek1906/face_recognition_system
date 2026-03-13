"""
Скрипт для объединения дубликатов лиц в базе данных
Использует те же алгоритмы сравнения что и основная система
"""

import numpy as np
import logging
from db import get_session, init_db
from models import Face
from config import SIMILARITY_THRESHOLD

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Вычислить косинусное сходство"""
    try:
        vec1 = vec1 / (np.linalg.norm(vec1) + 1e-8)
        vec2 = vec2 / (np.linalg.norm(vec2) + 1e-8)
        return float(np.dot(vec1, vec2))
    except:
        return 0.0


def find_duplicates(threshold=0.55):
    """
    Найти все дубликаты в базе
    Возвращает список групп дубликатов
    """
    db_session = get_session()
    
    try:
        all_faces = db_session.query(Face).order_by(Face.first_seen).all()
        
        if len(all_faces) < 2:
            logger.info("Not enough faces to check for duplicates")
            return []
        
        logger.info(f"Checking {len(all_faces)} faces for duplicates...")
        
        embeddings = []
        for face in all_faces:
            emb = np.array(Face.embedding_to_list(face.embedding), dtype=np.float32)
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            embeddings.append({
                'face_id': face.id,
                'person_id': face.person_id,
                'embedding': emb,
                'first_seen': face.first_seen
            })
        
        duplicate_groups = []
        processed = set()
        
        for i in range(len(embeddings)):
            if embeddings[i]['face_id'] in processed:
                continue
            
            group = [embeddings[i]]
            processed.add(embeddings[i]['face_id'])
            
            for j in range(i + 1, len(embeddings)):
                if embeddings[j]['face_id'] in processed:
                    continue
                
                similarities = []
                for member in group:
                    sim = cosine_similarity(embeddings[j]['embedding'], member['embedding'])
                    similarities.append(sim)
                
                avg_sim = np.mean(similarities)
                max_sim = max(similarities)
                
                combined_sim = 0.6 * max_sim + 0.4 * avg_sim
                
                if combined_sim >= threshold:
                    group.append(embeddings[j])
                    processed.add(embeddings[j]['face_id'])
                    logger.info(f"  Found similar: Face #{embeddings[j]['face_id']} ~ #{embeddings[i]['face_id']} (sim: {combined_sim:.3f})")
            
            if len(group) > 1:
                duplicate_groups.append(group)
        
        db_session.close()
        return duplicate_groups
        
    except Exception as e:
        logger.error(f"Error finding duplicates: {e}")
        db_session.close()
        return []


def merge_group(group, dry_run=True):
    """
    Объединить группу дубликатов в одну запись
    Оставляет самую старую запись, остальные удаляет
    """
    db_session = get_session()
    
    try:
        group_sorted = sorted(group, key=lambda x: x['first_seen'])
        
        primary = group_sorted[0]
        duplicates = group_sorted[1:]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Merging group of {len(group)} faces:")
        logger.info(f"  PRIMARY: Face #{primary['face_id']}, Person #{primary['person_id']}")
        
        if dry_run:
            logger.info(f"  [DRY RUN] Would delete:")
            for dup in duplicates:
                logger.info(f"    - Face #{dup['face_id']}, Person #{dup['person_id']}")
            return
        
        for dup in duplicates:
            face_to_delete = db_session.query(Face).filter(Face.id == dup['face_id']).first()
            if face_to_delete:
                logger.info(f"  Deleting Face #{dup['face_id']}, Person #{dup['person_id']}")
                db_session.delete(face_to_delete)
        
        db_session.commit()
        logger.info(f"✓ Group merged successfully")
        
    except Exception as e:
        logger.error(f"Error merging group: {e}")
        db_session.rollback()
    finally:
        db_session.close()


def main():
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║              Duplicate Face Merger                           ║
    ║    Finds and merges duplicate person records in database     ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    init_db()
    
    logger.info("\n[Step 1] Scanning for duplicates...")
    duplicate_groups = find_duplicates(threshold=0.55)
    
    if not duplicate_groups:
        logger.info("✓ No duplicates found!")
        return
    
    logger.info(f"\n[Step 2] Found {len(duplicate_groups)} groups of duplicates")
    
    total_duplicates = sum(len(group) - 1 for group in duplicate_groups)
    logger.info(f"Total faces to merge: {total_duplicates}")
    
    print("\n" + "="*60)
    print("PREVIEW OF MERGES (DRY RUN):")
    print("="*60)
    for group in duplicate_groups:
        merge_group(group, dry_run=True)
    
    print("\n" + "="*60)
    response = input("\nProceed with merging? This will DELETE duplicate records! (yes/no): ")
    
    if response.lower() != 'yes':
        logger.info("Merge cancelled by user")
        return
    
    logger.info("\n[Step 3] Merging duplicates...")
    for i, group in enumerate(duplicate_groups, 1):
        logger.info(f"\nProcessing group {i}/{len(duplicate_groups)}...")
        merge_group(group, dry_run=False)
    
    logger.info("\n✓ All duplicates merged successfully!")
    logger.info(f"✓ Removed {total_duplicates} duplicate records")


if __name__ == "__main__":
    main()