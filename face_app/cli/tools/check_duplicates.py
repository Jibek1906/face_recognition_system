"""
✅ СКРИПТ: Проверка дубликатов без объединения


Показывает какие лица похожи друг на друга, но не объединяет их.
Полезно для анализа перед запуском merge_duplicates_auto.py
"""

import numpy as np
import logging
from face_app.db.session import get_session, init_db
from face_app.db.models import Face
import json
from tabulate import tabulate

logging.basicConfig(
   level=logging.INFO,
   format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_embedding_from_face(face):
   """Получить embedding из Face"""
   try:
       embedding_list = json.loads(face.embedding)
       embedding = np.array(embedding_list, dtype=np.float32)
       norm = np.linalg.norm(embedding)
       if norm > 0:
           embedding = embedding / norm
       return embedding
   except Exception as e:
       logger.error(f"Error parsing embedding for face {face.id}: {e}")
       return None

def calculate_similarity(emb1, emb2):
   """Рассчитать косинусное сходство"""
   if emb1 is None or emb2 is None:
       return 0.0
   return float(np.dot(emb1, emb2))




def check_duplicates(similarity_threshold=0.85):
   """
   ✅ НОВОЕ: Проверить дубликаты без объединения
   """
   db_session = get_session()
   try:
       all_faces = db_session.query(Face).all()
      
       logger.info(f"Checking {len(all_faces)} faces for duplicates...")
      
       # Получаем embeddings для всех лиц
       embeddings = {}
       face_info = {}
      
       for face in all_faces:
           emb = get_embedding_from_face(face)
           if emb is not None:
               embeddings[face.id] = emb
               face_info[face.id] = {
                   'name': face.name or f"Client {face.person_id}",
                   'status': face.status,
                   'person_id': face.person_id,
                   'visits': len(face.visits)
               }
      
       logger.info(f"Loaded {len(embeddings)} embeddings")
      
       # Находим похожие пары
       duplicates = []
       checked = set()
      
       for face_id1, emb1 in embeddings.items():
           for face_id2, emb2 in embeddings.items():
               if face_id1 >= face_id2 or (face_id1, face_id2) in checked:
                   continue
              
               checked.add((face_id1, face_id2))
              
               similarity = calculate_similarity(emb1, emb2)
              
               if similarity > similarity_threshold:
                   duplicates.append({
                       'face_id_1': face_id1,
                       'face_id_2': face_id2,
                       'name_1': face_info[face_id1]['name'],
                       'name_2': face_info[face_id2]['name'],
                       'similarity': similarity,
                       'visits_1': face_info[face_id1]['visits'],
                       'visits_2': face_info[face_id2]['visits'],
                       'status_1': face_info[face_id1]['status'],
                       'status_2': face_info[face_id2]['status']
                   })
      
       if not duplicates:
           logger.info("✅ No duplicates found!")
           return
      
       # Сортируем по сходству (от большего к меньшему)
       duplicates.sort(key=lambda x: x['similarity'], reverse=True)
      
       # Выводим таблицу
       print("\n" + "=" * 120)
       print(f"Found {len(duplicates)} duplicate pairs (similarity > {similarity_threshold})")
       print("=" * 120 + "\n")
      
       table_data = []
       for dup in duplicates:
           table_data.append([
               f"Face {dup['face_id_1']}",
               dup['name_1'],
               dup['status_1'],
               dup['visits_1'],
               f"Face {dup['face_id_2']}",
               dup['name_2'],
               dup['status_2'],
               dup['visits_2'],
               f"{dup['similarity']:.3f}"
           ])
      
       headers = [
           "Face 1",
           "Name 1",
           "Status 1",
           "Visits 1",
           "Face 2",
           "Name 2",
           "Status 2",
           "Visits 2",
           "Similarity"
       ]
      
       print(tabulate(table_data, headers=headers, tablefmt="grid"))
       print()
      
       # Статистика
       print("=" * 120)
       print("STATISTICS:")
       print("=" * 120)
      
       # Группируем дубликаты
       groups = []
       used = set()
      
       for dup in duplicates:
           if dup['face_id_1'] in used or dup['face_id_2'] in used:
               continue
          
           group = [dup['face_id_1'], dup['face_id_2']]
           used.add(dup['face_id_1'])
           used.add(dup['face_id_2'])
          
           # Ищем еще похожие в этой группе
           for dup2 in duplicates:
               if dup2['face_id_1'] in group and dup2['face_id_2'] not in used:
                   group.append(dup2['face_id_2'])
                   used.add(dup2['face_id_2'])
               elif dup2['face_id_2'] in group and dup2['face_id_1'] not in used:
                   group.append(dup2['face_id_1'])
                   used.add(dup2['face_id_1'])
          
           groups.append(group)
      
       print(f"Total duplicate pairs: {len(duplicates)}")
       print(f"Total duplicate groups: {len(groups)}")
       print(f"Total faces involved: {len(used)}")
       print()
      
       # Показываем группы
       print("DUPLICATE GROUPS:")
       print("-" * 120)
      
       for i, group in enumerate(groups, 1):
           print(f"\nGroup {i}: {len(group)} faces")
          
           group_data = []
           total_visits = 0
          
           for face_id in group:
               info = face_info[face_id]
               total_visits += info['visits']
               group_data.append([
                   f"Face {face_id}",
                   info['name'],
                   info['status'],
                   info['visits']
               ])
          
           print(tabulate(group_data, headers=["Face", "Name", "Status", "Visits"], tablefmt="simple"))
           print(f"  Total visits in group: {total_visits}")
      
       print("\n" + "=" * 120)
       print("To merge duplicates, run:")
       print(f"  python -m face_app.cli.tools.merge_duplicates {similarity_threshold}")
       print("=" * 120 + "\n")
  
   finally:
       db_session.close()

if __name__ == "__main__":
   import sys
  
   init_db()
  
   similarity_threshold = 0.85
   if len(sys.argv) > 1:
       try:
           similarity_threshold = float(sys.argv[1])
       except ValueError:
           logger.error(f"Invalid threshold: {sys.argv[1]}")
           sys.exit(1)
  
   check_duplicates(similarity_threshold)