from fastapi import FastAPI, Depends, HTTPException, WebSocket, WebSocketDisconnect
from typing import List, Dict, Optional
from fastapi.responses import FileResponse, StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import os

# Ensure HLS directory exists before mounting
hls_dir = "/tmp/hls"
if not os.path.exists(hls_dir):
    os.makedirs(hls_dir, exist_ok=True)
import cv2
from imou_camera import ImouCamera
import threading
import time
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import desc, func, and_
from datetime import datetime, timedelta
import logging
import os
from db import get_db, init_db
from models import Face, Visit, group_visits_to_sessions, PersonPhoto
from config import API_HOST, API_PORT
from pydantic import BaseModel
import asyncio
import json
import base64
import traceback




logger = logging.getLogger(__name__)


app = FastAPI(title="Face Recognition API", version="1.0.0")

# Serve HLS directory as static files
app.mount("/hls", StaticFiles(directory="/tmp/hls"), name="hls")

# Serve HLS files directly — avoids 302 round-trips which caused latency spikes
from fastapi.responses import Response

@app.get("/stream.m3u8")
async def serve_stream_m3u8():
    path = "/tmp/hls/stream.m3u8"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Stream not yet available")
    with open(path, "rb") as f:
        content = f.read()
    return Response(
        content=content,
        media_type="application/vnd.apple.mpegurl",
        headers={"Cache-Control": "no-cache, no-store", "Access-Control-Allow-Origin": "*"}
    )

@app.get("/stream{segment}.ts")
async def serve_stream_ts(segment: str):
    path = f"/tmp/hls/stream{segment}.ts"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Segment not found")
    return FileResponse(
        path,
        media_type="video/mp2t",
        headers={"Cache-Control": "no-cache", "Access-Control-Allow-Origin": "*"}
    )


app.add_middleware(
   CORSMiddleware,
   allow_origins=["*"],
   allow_credentials=True,
   allow_methods=["*"],
   allow_headers=["*"],
)




class FaceResponse(BaseModel):
   id: int
   status: str
   total_time_seconds: float
   detection_count: int
   first_seen: str
   last_seen: str




class FaceStatsResponse(BaseModel):
   id: int
   status: str
   last_seen: str
   first_seen: str
   visits_today: int
   visits_week: int
   visits_month: int




class FacesListResponse(BaseModel):
   total_count: int
   new_faces: int
   known_faces: int


class FacesListResponse(BaseModel):
   total_count: int
   new_faces: int
   known_faces: int
   faces: list




class CameraSwitchRequest(BaseModel):
    source: str  # 'imou', 'reolink', 'local', 'webrtc'
    quality: str = "HD" # 'HD', 'SD'

class WebRTCOffer(BaseModel):
    sdp: str
    type: str






@app.on_event("startup")
async def startup():
   init_db()
   logger.info("API startup complete")




@app.get("/health", tags=["Health"])
async def health_check():
   return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}




@app.get("/", tags=["Admin"])
async def admin_panel():
   """Вернуть админ панель"""
   admin_html_path = os.path.join(os.path.dirname(__file__), 'admin.html')
 
   if not os.path.exists(admin_html_path):
       raise HTTPException(
           status_code=404,
           detail="⚠️ Admin Dashboard Not Found\nСоздайте файл admin.html в корне проекта.\n\nAPI документация: /docs"
       )
 
   return FileResponse(admin_html_path)

@app.get("/visual_debug.js")
async def serve_visual_debug():
    file_path = os.path.join(os.path.dirname(__file__), 'visual_debug.js')
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="visual_debug.js not found")
    return FileResponse(file_path)




@app.get("/faces", response_model=FacesListResponse, tags=["Faces"])
async def get_faces(db: Session = Depends(get_db)):
   """✅ ИСПРАВЛЕНО: Правильный подсчет визитов через сессии"""
   try:
       db.expire_all()
       faces = db.query(Face).order_by(desc(Face.last_seen)).all()
       new_count = len([f for f in faces if f.status == 'new'])
       known_count = len([f for f in faces if f.status == 'known'])


       now = datetime.utcnow()
       today_start = datetime(now.year, now.month, now.day)
       week_ago = now - timedelta(days=7)
       month_ago = now - timedelta(days=30)


       faces_data = []
       for f in faces:
           all_visits = db.query(Visit).filter(Visit.face_id == f.id).order_by(Visit.timestamp).all()
          
           visits_today_list = [v for v in all_visits if v.timestamp >= today_start]
           visits_week_list = [v for v in all_visits if v.timestamp >= week_ago]
           visits_month_list = [v for v in all_visits if v.timestamp >= month_ago]
          
           sessions_today, seconds_today = group_visits_to_sessions(visits_today_list)
           sessions_week, seconds_week = group_visits_to_sessions(visits_week_list)
           sessions_month, seconds_month = group_visits_to_sessions(visits_month_list)
           sessions_all, seconds_all = group_visits_to_sessions(all_visits)
          
           faces_data.append({
               "id": f.id,
               "person_id": f.person_id,
               "name": f.name if f.name else f"Client {f.person_id}",  # ✅ FIX: Add name field
               "status": f.status,
               "first_seen": f.first_seen.isoformat(),
               "last_seen": f.last_seen.isoformat(),
               "photo_base64": f.photo_base64 if f.photo_base64 else "",
               "name": f.name if f.name else f"Client {f.person_id}",  # ✅ FIXED: Missing name field
               "total_time_seconds": float(getattr(f, "total_time_seconds", 0.0)),
              
               "visits_today": len(sessions_today),
               "visits_week": len(sessions_week),
               "visits_month": len(sessions_month),
               "visits_all": len(sessions_all),
               "seconds_today": seconds_today,
               "seconds_week": seconds_week,
               "seconds_month": seconds_month,
               "seconds_all": seconds_all,
               "sessions_today": len(sessions_today),
               "sessions_week": len(sessions_week),
               "sessions_month": len(sessions_month),
          
               "visits": [v.timestamp.isoformat() for v in all_visits]
           })


       return FacesListResponse(
           total_count=len(faces),
           new_faces=new_count,
           known_faces=known_count,
           faces=faces_data
       )
   except Exception as e:
       logger.error(f"Error fetching faces: {e}")
       raise HTTPException(status_code=500, detail=str(e))




@app.get("/analytics/realtime_total")
async def get_realtime_total():
   """Returns the total number of people currently tracked (identified + unidentified)."""
   import video_state
   if video_state.global_face_system and video_state.global_face_system.tracker:
       total = video_state.global_face_system.tracker.get_active_tracks_count()
       return {"total": total}
   return {"total": 0}


def __bbox_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
    yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0: return 0.0
    return interArea / float((boxA[2] * boxA[3]) + (boxB[2] * boxB[3]) - interArea + 1e-6)

@app.get("/tracks")
async def get_tracks():
   """
   ✅ ENHANCED: Returns all active tracks for frontend overlay with complete visual data.
   Includes bounding boxes, names, confidence, pose info, and color coding.
   """
   import video_state
   from tracker import TrackState
   tracks_data = []
  
   system = video_state.global_face_system
   if system and system.tracker:
       current_time = time.time()
      
       # Get frame dimensions for normalization (assumed 1280x720, adjust if needed)
       # You can also pass this from frontend or store in system
       frame_width = DISPLAY_WIDTH if 'DISPLAY_WIDTH' in globals() else 1280
       frame_height = DISPLAY_HEIGHT if 'DISPLAY_HEIGHT' in globals() else 720
      
       for track in system.tracker.tracks:
           # Only send CONFIRMED tracks that are active (not too stale)
           # ✅ STABILIZED: Drop tracks from UI immediately if missed for >2 frames
           # Prevents freezing/ghosting UI boxes
           if track.state == TrackState.CONFIRMED and track.misses > 2:
               continue   
           # Resolving identity info
           display_name = "Analyzing..."
           is_employee = False
           recognized = False
          
           if track.person_id:
               info = system.identity_manager.get_identity_info(track.person_id)
               if info:
                   display_name = info['name']
                   is_employee = info['is_employee']
                   recognized = True


           # Bounding box (absolute pixels) - prioritize face_bbox
           display_bbox = track.face_bbox if hasattr(track, 'face_bbox') and track.face_bbox else track.bbox
           x, y, w, h = display_bbox
          
           # ✅ Normalized coordinates for responsive frontend (0-1 range)
           norm_x = x / frame_width
           norm_y = y / frame_height
           norm_w = w / frame_width
           norm_h = h / frame_height
          
           # ✅ Color coding based on status
           if is_employee:
               box_color = "gold"  # Employee
           elif recognized:
               box_color = "green"  # Known person
           else:
               box_color = "red"   # Analyzing/Unknown
          
           # ✅ Processing progress (for "Analyzing" state)
           progress_percent = min(100, int((track.hits / system.identity_manager.min_frames_to_create) * 100))
          
           tracks_data.append({
               "id": track.track_id,
               "bbox": [x, y, w, h],  # Absolute pixels
               "bbox_norm": [norm_x, norm_y, norm_w, norm_h],  # Normalized 0-1
               "person_id": track.person_id,
               "name": display_name,
               "is_employee": is_employee,
               "recognized": recognized,
               "confidence": float(track.best_face_score),
               "hits": track.hits,
               "progress_percent": progress_percent,
               "box_color": box_color,
               "state": track.state.name,
               "pose": getattr(track, 'latest_pose', {}).get('angle_category', 'unknown') if hasattr(track, 'latest_pose') else 'unknown',
               "last_seen_seconds_ago": current_time - track.last_seen_time
           })

   # ✅ Spatial NMS to hide "Analyzing..." boxes that overlap with recognized people
   known_boxes = [t['bbox'] for t in tracks_data if t['recognized'] or t['is_employee']]
   final_tracks_data = []
   for t in tracks_data:
       should_keep = True
       if not t['recognized'] and not t['is_employee']:
           # Require at least 10 hits before showing "Analyzing..." to avoid instant false positives
           if t['hits'] < 10:
               should_keep = False
           else:
               for k_box in known_boxes:
                   if __bbox_iou(t['bbox'], k_box) > 0.35:
                       should_keep = False
                       break
       if should_keep:
           final_tracks_data.append(t)
   
   return final_tracks_data


@app.get("/analytics/total", tags=["Analytics"])
async def get_total_analytics(db: Session = Depends(get_db)):
   """✅ ИСПРАВЛЕНО: Общая статистика за всё время"""
   try:
       total_unique_visitors = db.query(Face).filter(
           Face.status != 'employee'
       ).count()
       employee_ids = db.query(Face.id).filter(Face.status == 'employee').all()
       employee_ids = [eid[0] for eid in employee_ids]
       visitor_faces = db.query(Face).filter(Face.status != 'employee').all()
      
       total_sessions = 0
       for face in visitor_faces:
           visits = db.query(Visit).filter(Visit.face_id == face.id).order_by(Visit.timestamp).all()
           sessions, _ = group_visits_to_sessions(visits)
           total_sessions += len(sessions)
     
       return {
           "total_unique_visitors": total_unique_visitors,
           "total_visits": total_sessions
       }
   except Exception as e:
       logger.error(f"Error fetching total analytics: {e}")
       raise HTTPException(status_code=500, detail=str(e))




@app.get("/faces", response_model=FacesListResponse, tags=["Faces"])
async def get_faces(db: Session = Depends(get_db), days: int = 90):
  """✅ ОПТИМИЗИРОВАНО: Быстрая загрузка с кешированием"""
  try:
      from functools import lru_cache
      import time
     
      # ✅ Кеширование: перестраивается каждые 30 секунд
      cache_key = f"faces_{int(time.time() // 30)}"
     
      db.expire_all()
      faces = db.query(Face).order_by(desc(Face.last_seen)).all()
      new_count = len([f for f in faces if f.status == 'new'])
      known_count = len([f for f in faces if f.status == 'known'])


      now = datetime.utcnow()
      today_start = datetime(now.year, now.month, now.day)
      week_ago = now - timedelta(days=7)
      month_ago = now - timedelta(days=30)


      # ✅ ОПТИМИЗИРОВАНО: Получаем только нужные визиты из БД
      cutoff_date = now - timedelta(days=days)


      faces_data = []
      for f in faces:
          # ✅ Получаем визиты через SQL запрос с фильтром
          all_visits = db.query(Visit).filter(
              Visit.face_id == f.id,
              Visit.timestamp >= cutoff_date
          ).order_by(Visit.timestamp).all()
         
          visits_today_list = [v for v in all_visits if v.timestamp >= today_start]
          visits_week_list = [v for v in all_visits if v.timestamp >= week_ago]
          visits_month_list = [v for v in all_visits if v.timestamp >= month_ago]
         
          sessions_today, seconds_today = group_visits_to_sessions(visits_today_list)
          sessions_week, seconds_week = group_visits_to_sessions(visits_week_list)
          sessions_month, seconds_month = group_visits_to_sessions(visits_month_list)
          sessions_all, seconds_all = group_visits_to_sessions(all_visits)
         
          faces_data.append({
              "id": f.id,
              "person_id": f.person_id,
              "status": f.status,
              "first_seen": f.first_seen.isoformat(),
              "last_seen": f.last_seen.isoformat(),
              "photo_base64": f.photo_base64 if f.photo_base64 else "",
              "total_time_seconds": float(getattr(f, "total_time_seconds", 0.0)),
            
              "visits_today": len(sessions_today),
              "visits_week": len(sessions_week),
              "visits_month": len(sessions_month),
              "visits_all": len(sessions_all),
              "seconds_today": seconds_today,
              "seconds_week": seconds_week,
              "seconds_month": seconds_month,
              "seconds_all": seconds_all,
              "sessions_today": len(sessions_today),
              "sessions_week": len(sessions_week),
              "sessions_month": len(sessions_month),
            
              # ✅ Не отправляем весь массив visits - это убивает скорость!
              "visits": [v.timestamp.isoformat() for v in all_visits[-50:]]  # Только последние 50
          })


      return FacesListResponse(
          total_count=len(faces),
          new_faces=new_count,
          known_faces=known_count,
          faces=faces_data
      )
  except Exception as e:
      logger.error(f"Error fetching faces: {e}")
      raise HTTPException(status_code=500, detail=str(e))




@app.post("/reset", tags=["Admin"])
async def reset_database(db: Session = Depends(get_db)):
   try:
       visits_count = db.query(Visit).delete()
       faces_count = db.query(Face).delete()
       db.commit()
       logger.warning(f"Database reset: {faces_count} faces and {visits_count} visits deleted")
     
       await notify_update()
     
       return {
           "message": f"Deleted {faces_count} faces and {visits_count} visits",
           "timestamp": datetime.utcnow().isoformat()
       }
   except Exception as e:
       db.rollback()
       logger.error(f"Error resetting database: {e}")
       raise HTTPException(status_code=500, detail=str(e))




class ConnectionManager:
   def __init__(self):
       self.active_connections: list[WebSocket] = []


   async def connect(self, websocket: WebSocket):
       await websocket.accept()
       self.active_connections.append(websocket)
       logger.info(f"WebSocket client connected. Total: {len(self.active_connections)}")


   def disconnect(self, websocket: WebSocket):
       if websocket in self.active_connections:
           self.active_connections.remove(websocket)
           logger.info(f"WebSocket client disconnected. Total: {len(self.active_connections)}")


   async def broadcast(self, message: str):
       disconnected = []
       for connection in self.active_connections:
           try:
               await connection.send_text(message)
           except Exception as e:
               logger.error(f"Error broadcasting to client: {e}")
               disconnected.append(connection)
     
       for conn in disconnected:
           self.disconnect(conn)




manager = ConnectionManager()




@app.websocket("/ws/updates")
async def websocket_endpoint(websocket: WebSocket):
   await manager.connect(websocket)
   try:
       while True:
           await websocket.receive_text()
   except WebSocketDisconnect:
       manager.disconnect(websocket)
   except Exception as e:
       logger.error(f"WebSocket error: {e}")
       manager.disconnect(websocket)




async def notify_update():
   await manager.broadcast("update")
   logger.info("📡 Broadcasted update to all WebSocket clients")




@app.websocket("/ws/tracks")
async def websocket_tracks(websocket: WebSocket):
   """
   ✅ Real-time track streaming WebSocket — sends delta updates on change, full state every 2s.
   """
   await websocket.accept()
   logger.info("Track WebSocket client connected")
  
   try:
       import video_state
       from tracker import TrackState
       import asyncio
      
       # ✅ Delay queue to synchronize WebSocket updates with HLS stream (~4.0s latency)
       delay_queue = asyncio.Queue()
       
       async def sender_task():
           try:
               while True:
                   update = await delay_queue.get()
                   wait_time = update['send_time'] - time.time()
                   if wait_time > 0.05:
                       await asyncio.sleep(wait_time)
                   await websocket.send_json({
                       "type": "tracks_update",
                       "tracks": update['tracks'],
                       "removed": update['removed']
                   })
           except asyncio.CancelledError:
               pass
           except Exception as e:
               pass
               
       sender_task_handle = asyncio.create_task(sender_task())
      
       last_track_state = {}
       last_full_push = 0.0
      
       while True:
           try:
               system = video_state.global_face_system
               current_tracks = {}
               tracks_update = []
               now = time.time()
               force_full = (now - last_full_push) >= 2.0  # full state every 2s
               
               if system and system.tracker:
                   for track in system.tracker.tracks:
                       if track.state == TrackState.CONFIRMED and track.misses > 240:
                           continue
                       if track.state == TrackState.CONFIRMED:
                           track_key = track.track_id
                           display_bbox = track.face_bbox if hasattr(track, 'face_bbox') and track.face_bbox else track.bbox
                           track_snapshot = (
                               track.person_id,
                               tuple(display_bbox),
                               track.state.name,
                               track.hits
                           )
                           current_tracks[track_key] = track_snapshot
                           
                           if force_full or track_key not in last_track_state or last_track_state[track_key] != track_snapshot:
                               display_name = "Analyzing..."
                               is_employee = False
                               recognized = False
                               if track.person_id:
                                   info = system.identity_manager.get_identity_info(track.person_id)
                                   if info:
                                       is_employee = info.get('is_employee', False)
                                       if is_employee:
                                           display_name = info.get('name', f"Employee {track.person_id}")
                                       else:
                                           time_str = system.shared_state.get_today_time_str(track.person_id)
                                           display_name = f"Client {time_str}"
                                       recognized = True
                               
                               tracks_update.append({
                                   "id": int(track.track_id),
                                   "bbox": [int(b) for b in display_bbox],
                                   "person_id": int(track.person_id) if track.person_id is not None else None,
                                   "name": display_name,
                                   "is_employee": is_employee,
                                   "recognized": recognized,
                                   "confidence": float(track.best_face_score),
                                   "hits": int(track.hits),
                                   "state": track.state.name
                               })
               
               # Deduplication by person_id (keep highest confidence)
               seen_pids = {}
               unique_tracks = []
               for t in tracks_update:
                   pid = t['person_id']
                   if pid is not None:
                       if pid in seen_pids:
                           if t['confidence'] > unique_tracks[seen_pids[pid]]['confidence']:
                               unique_tracks[seen_pids[pid]] = t
                       else:
                           seen_pids[pid] = len(unique_tracks)
                           unique_tracks.append(t)
                   else:
                       unique_tracks.append(t)
               
               # Filter: only send recognized/employee tracks — no orange "Analyzing..." boxes
               final_tracks = [t for t in unique_tracks if t['recognized'] or t['is_employee']]

               removed_ids = [tid for tid in last_track_state if tid not in current_tracks]

               if final_tracks or removed_ids or force_full:
                   await delay_queue.put({
                       "send_time": time.time() + 4.0, # Sync 4.0s HLS delay
                       "tracks": final_tracks,
                       "removed": removed_ids
                   })
                   if force_full:
                       last_full_push = now

               last_track_state = current_tracks

           except WebSocketDisconnect:
               raise
           except Exception as inner_err:
               err_str = str(inner_err)
               if "1001" in err_str or "1011" in err_str or "closed" in err_str.lower() or "keepalive" in err_str.lower() or "close frame" in err_str.lower():
                   raise WebSocketDisconnect()
               logger.error(f"Track WS inner error: {inner_err}")
               raise WebSocketDisconnect() # Safety break to avoid infinite loop
            
           # 200ms poll — gives asyncio enough time to handle WS pings
           await asyncio.sleep(0.2)
          
   except WebSocketDisconnect:
       logger.info("Track WebSocket client disconnected")
   except Exception as e:
       if "1001" in str(e) or "closed" in str(e).lower():
           logger.info("Track WebSocket client disconnected")
       else:
           logger.error(f"Track WebSocket error: {e}")
   finally:
       if 'sender_task_handle' in locals():
           sender_task_handle.cancel()




@app.post("/faces/{face_id}/status", tags=["Faces"])
async def update_face_status(face_id: int, status: str, db: Session = Depends(get_db)):
   """Обновить статус лица"""
   try:
       if status not in ['new', 'known', 'employee']:
           raise HTTPException(status_code=400, detail="Status must be 'new', 'known', or 'employee'")
     
       face = db.query(Face).filter(Face.id == face_id).first()
       if not face:
           raise HTTPException(status_code=404, detail="Face not found")
     
       old_status = face.status
       face.status = status


       # Если статус меняется на employee, обновляем имя на employee_N
       if status == 'employee':
           # Найти максимальный номер сотрудника
           max_num = 0
           employees = db.query(Face).filter(Face.status == 'employee').all()
           for emp in employees:
               if emp.name and emp.name.startswith("employee_"):
                   try:
                       num = int(emp.name.split("_")[1])
                       if num > max_num:
                           max_num = num
                   except Exception:
                       continue
           face.name = f"employee_{max_num + 1}"


       db.commit()
     
       logger.info(f"✅ Face {face_id} status updated: {old_status} -> {status}")
     
       await notify_update()
     
       try:
           with open("RELOAD_EMBEDDINGS.flag", "w") as f:
               f.write("1")
           logger.info("🔄 Reload flag set for main process")
       except Exception as e:
           logger.error(f"Error setting reload flag: {e}")
     
       return {"id": face.id, "status": face.status, "message": "Status updated successfully"}
     
   except HTTPException:
       raise
   except Exception as e:
       db.rollback()
       logger.error(f"Error updating face status: {e}")
       raise HTTPException(status_code=500, detail=str(e))




@app.delete("/faces/{face_id}", tags=["Faces"])
async def delete_face(face_id: int, db: Session = Depends(get_db)):
   """Удалить запись лица"""
   try:
       face = db.query(Face).filter(Face.id == face_id).first()
       if not face:
           raise HTTPException(status_code=404, detail="Face not found")
     
       person_id = face.person_id


       # --- NEW: Delete all PersonPhoto records for this face ---
       from models import PersonPhoto
       db.query(PersonPhoto).filter(PersonPhoto.face_id == face_id).delete()
     
       db.query(Visit).filter(Visit.face_id == face_id).delete()
       
       # ✅ CLEANUP MEMORY IMMEDIATELY
       import video_state
       system = getattr(video_state, "global_face_system", None)
       if system and face.person_id:
            system.identity_manager.delete_person(face.person_id)

       db.delete(face)
       db.commit()
     
       logger.warning(f"🗑️ Face {face_id} (Person {person_id}) and all visits/photos deleted")
       
       # ✅ FORCE RESET ACTIVE TRACKS
       # Iterate through all current tracks. If any track has this person_id,
       # reset it to None so it gets re-evaluated (as a new person).
       if video_state.global_face_system and video_state.global_face_system.tracker:
           active_reset_count = 0
           for track in video_state.global_face_system.tracker.tracks:
               if track.person_id == person_id:
                   track.person_id = None
                   track.state = TrackState.TENTATIVE # Demote to tentative to force re-check
                   # track.hits = 0 # Optional: reset hits? No, keep history but new ID.
                   active_reset_count += 1
           if active_reset_count > 0:
               logger.info(f"🔄 Reset {active_reset_count} active tracks for deleted Person {person_id}")
     
       await notify_update()
     
       try:
           with open("RELOAD_EMBEDDINGS.flag", "w") as f:
               f.write("1")
       except Exception as e:
           logger.error(f"Error setting reload flag: {e}")
     
       return {"message": "Face deleted successfully", "face_id": face_id}
     
   except HTTPException:
       raise
   except Exception as e:
       db.rollback()
       logger.error(f"Error deleting face: {e}")
       raise HTTPException(status_code=500, detail=str(e))




# ✅ ИСПРАВЛЕННЫЙ эндпоинт для загрузки дополнительных фото


class ManualCreateFaceRequest(BaseModel):
   name: str = None
   photo_base64: str
   angle: str = None  # Optional: frontal, left_profile, right_profile, back, etc.


class AddPhotoRequest(BaseModel):
   photo_base64: str
   angle: str = None  # Optional: frontal, left_profile, right_profile, back, etc.




@app.post("/faces/manual_create", tags=["Faces"])
async def manual_create_face(request: ManualCreateFaceRequest, db: Session = Depends(get_db)):
   """
   ✅ NEW: Ручное добавление клиента с фото (и опционально именем и ракурсом)
   """
   try:
       # Обрабатываем фото
       from face_recognition import FaceRecognizer
       recognizer = FaceRecognizer()
       img = recognizer.base64_to_image(request.photo_base64)
       if img is None:
           raise HTTPException(status_code=400, detail="Invalid image data")


       # Получаем embedding
       embedding = recognizer.get_embedding(img)
       if embedding is None:
           embedding = recognizer.get_embedding_with_mbd(img)
       if embedding is None:
           embedding = recognizer.get_embedding_enhanced(img)
       if embedding is None:
           embedding = np.zeros(512, dtype=np.float32)
           pose_info = {"angle_category": "unknown", "pitch": 0, "yaw": 0, "roll": 0}
       else:
           pose_info = recognizer.detect_head_pose(img)


       # Normalize angle: map "затылок" and other variants to "back"
       angle_raw = request.angle if request.angle else pose_info['angle_category']
       if angle_raw and angle_raw.lower() in ("затылок", "back", "back_view", "rear"):
           final_angle = "back"
       else:
           final_angle = angle_raw


       # Use unified face creation logic with duplicate check
       from face_creation_utils import create_or_get_face_by_embedding
       from face_recognition import BodyRecognizer
       body_recognizer = BodyRecognizer()
       body_emb = body_recognizer.get_body_embedding(img)


       face_obj, new_photo, is_duplicate, message = create_or_get_face_by_embedding(
           db=db,
           embedding=embedding,
           photo_base64=request.photo_base64,
           name=request.name,
           angle=final_angle,
           body_embedding=body_emb,
           pose_info=pose_info,
           similarity_threshold=0.97,
           body_similarity_threshold=0.45
       )


       if face_obj is None or new_photo is None:
           logger.error(f"Manual face creation failed: {message}")
           raise HTTPException(status_code=400, detail=message)


       try:
           with open("RELOAD_EMBEDDINGS.flag", "w") as f:
               f.write("1")
       except Exception as e:
           logger.error(f"Error setting reload flag: {e}")


       await notify_update()


       return {
           "id": face_obj.id,
           "person_id": face_obj.person_id,
           "name": face_obj.name if face_obj.name else f"Client {face_obj.person_id}",
           "angle": final_angle,
           "photo_id": new_photo.id,
           "photo_base64": request.photo_base64,
           "message": message
       }
   except HTTPException:
       raise
   except Exception as e:
       db.rollback()
       logger.error(f"Error in manual_create_face: {e}", exc_info=True)
       raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/faces/{face_id}/photos", tags=["Faces"])
async def add_face_photo(face_id: int, request: AddPhotoRequest, db: Session = Depends(get_db)):
   """✅ ИСПРАВЛЕНО: Добавить дополнительное фото для улучшенного распознавания"""
   try:
       face = db.query(Face).filter(Face.id == face_id).first()
       if not face:
           raise HTTPException(status_code=404, detail="Face not found")
      
       # Декодируем и обрабатываем фото
       from face_recognition import FaceRecognizer
       recognizer = FaceRecognizer()
      
       # ✅ ФИХ 1: Проверяем формат base64
       photo_data = request.photo_base64
       if not photo_data or len(photo_data) < 100:
           raise HTTPException(status_code=400, detail="Invalid photo data: too short")
      
       # Декодируем изображение
       img = recognizer.base64_to_image(photo_data)
       if img is None:
           raise HTTPException(status_code=400, detail="Failed to decode image")
      
       logger.info(f"📸 Processing photo for Face {face_id}, image shape: {img.shape}")
      
       # ✅ ФИХ 2: Пробуем получить embedding несколькими способами
       embedding = None
       detection_method = "none"
      
       # Метод 1: Стандартная детекция
       embedding = recognizer.get_embedding(img)
       if embedding is not None:
           detection_method = "standard"
           logger.info(f"✓ Standard detection successful")
      
       # Метод 2: Улучшенная детекция (если стандартная не сработала)
       if embedding is None:
           logger.info(f"Standard detection failed, trying enhanced...")
           embedding = recognizer.get_embedding_enhanced(img)
           if embedding is not None:
               detection_method = "enhanced"
               logger.info(f"✓ Enhanced detection successful")
      
       # Метод 3: MBD для затылков и профилей
       if embedding is None:
           logger.info(f"Enhanced detection failed, trying MBD...")
           embedding = recognizer.get_embedding_with_mbd(img)
           if embedding is not None:
               detection_method = "mbd"
               logger.info(f"✓ MBD detection successful")
      
       # ✅ ФИХ 3: Если все методы не сработали - всё равно сохраняем фото с нулевым embedding
       if embedding is None:
           logger.warning(f"⚠ No face detected with any method, saving photo with zero embedding")
           embedding = np.zeros(512, dtype=np.float32)
           detection_method = "failed"
           pose_info = {
               'angle_category': 'unknown',
               'pitch': 0,
               'yaw': 0,
               'roll': 0
           }
       else:
           # Определяем ракурс головы
           pose_info = recognizer.detect_head_pose(img)
           logger.info(f"✓ Pose detected: {pose_info['angle_category']}")
      
       # ✅ ФИХ 4: Если угол был указан вручную - используем его
       final_angle = request.angle if request.angle else pose_info['angle_category']
      
       # Создаём запись в БД
       from face_recognition import BodyRecognizer
       body_recognizer = BodyRecognizer()
       body_emb = body_recognizer.get_body_embedding(img)
       body_emb_json = json.dumps(body_emb.tolist()) if body_emb is not None else None


       new_photo = PersonPhoto(
           face_id=face_id,
           photo_base64=photo_data,
           embedding=json.dumps(embedding.tolist()),
           body_embedding=body_emb_json,
           angle=final_angle,
           pose_pitch=pose_info['pitch'],
           pose_yaw=pose_info['yaw'],
           pose_roll=pose_info['roll']
       )
      
       db.add(new_photo)
       db.commit()
       db.refresh(new_photo)
      
       logger.info(f"✅ Photo saved: id={new_photo.id}, angle={final_angle}, method={detection_method}")
      
       # Перезагружаем эмбеддинги
       try:
           with open("RELOAD_EMBEDDINGS.flag", "w") as f:
               f.write("1")
           logger.info("🔄 Reload flag set")
       except Exception as e:
           logger.error(f"Error setting reload flag: {e}")
      
       await notify_update()
      
       return {
           "id": new_photo.id,
           "face_id": face_id,
           "angle": final_angle,
           "pose_pitch": pose_info['pitch'],
           "pose_yaw": pose_info['yaw'],
           "pose_roll": pose_info['roll'],
           "detection_method": detection_method,
           "message": "Photo added successfully"
       }
      
   except HTTPException:
       raise
   except Exception as e:
       db.rollback()
       logger.error(f"Error adding face photo: {e}", exc_info=True)
       raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")




class UpdateNameRequest(BaseModel):
   name: str


@app.put("/faces/{face_id}/name", tags=["Faces"])
async def update_face_name(face_id: int, request: UpdateNameRequest, db: Session = Depends(get_db)):
   """✅ NEW: Update person name"""
   try:
       face = db.query(Face).filter(Face.id == face_id).first()
       if not face:
           raise HTTPException(status_code=404, detail="Face not found")
      
       # Валидация имени
       new_name = request.name.strip() if request.name else ""
       if not new_name or new_name == "??":
           raise HTTPException(status_code=400, detail="Недопустимое имя: пустое или '??'")


       old_name = face.name
       face.name = new_name
       db.commit()
      
       logger.info(f"✅ Face {face_id} name updated: '{old_name}' -> '{new_name}'")
       await notify_update()
      
       return {"id": face.id, "name": face.name, "message": "Name updated successfully"}
   except HTTPException:
       raise
   except Exception as e:
       db.rollback()
       logger.error(f"Error updating face name: {e}")
       raise HTTPException(status_code=500, detail=str(e))




@app.get("/faces/{face_id}/photos", tags=["Faces"])
async def get_face_photos(face_id: int, db: Session = Depends(get_db)):
   """✅ NEW: Get all photos for a person"""
   try:
       face = db.query(Face).filter(Face.id == face_id).first()
       if not face:
           raise HTTPException(status_code=404, detail="Face not found")
      
       photos = db.query(PersonPhoto).filter(PersonPhoto.face_id == face_id).all()
      
       result = {
           "face_id": face.id,
           "name": face.name,
           "primary_photo": face.photo_base64 if face.photo_base64 else "",
           "additional_photos": [
               {
                   "id": photo.id,
                   "photo_base64": photo.photo_base64,
                   "angle": photo.angle,
                   "pose_pitch": photo.pose_pitch,
                   "pose_yaw": photo.pose_yaw,
                   "pose_roll": photo.pose_roll,
                   "created_at": photo.created_at.isoformat() if photo.created_at else None
               }
               for photo in photos
           ]
       }
      
       return result
   except HTTPException:
       raise
   except Exception as e:
       logger.error(f"Error fetching face photos: {e}")
       raise HTTPException(status_code=500, detail=str(e))




class AddPhotoRequest(BaseModel):
   photo_base64: str
   angle: str = None  # ✅ NEW: Manual angle selection by user (frontal, left_profile, right_profile, back, looking_up, looking_down, tilted)


@app.post("/faces/{face_id}/photos", tags=["Faces"])
async def add_face_photo(face_id: int, request: AddPhotoRequest, db: Session = Depends(get_db)):
   """✅ NEW: Add additional photo for a person with automatic angle detection
  
   ✅ УЛУЧШЕНО: Теперь загружает фото даже если лицо не видно четко (затылки, профили)
   Использует MBD для детекции затылков и других ракурсов
   """
   try:
       face = db.query(Face).filter(Face.id == face_id).first()
       if not face:
           raise HTTPException(status_code=404, detail="Face not found")
      
       # Decode and process the photo
       from face_recognition import FaceRecognizer
       recognizer = FaceRecognizer()
      
       img = recognizer.base64_to_image(request.photo_base64)
       if img is None:
           raise HTTPException(status_code=400, detail="Invalid image data")
      
       # ✅ УЛУЧШЕНО: Пытаемся получить embedding несколькими способами
       embedding = recognizer.get_embedding(img)
      
       # ��� НОВОЕ: Если обычная детекция не сработала, пробуем MBD для затылков
       if embedding is None:
           logger.info(f"Standard face detection failed for photo, trying MBD for back/profile views...")
           embedding = recognizer.get_embedding_with_mbd(img)
      
       # ✅ НОВОЕ: Если и MBD не сработал, используем детекцию с улучшением
       if embedding is None:
           logger.info(f"MBD detection failed, trying enhanced detection...")
           embedding = recognizer.get_embedding_enhanced(img)
      
       # ✅ НОВОЕ: Если все методы не сработали, но фото валидное - сохраняем как есть
       if embedding is None:
           logger.warning(f"No face detected in image with any method, but saving photo anyway")
           # Генерируем нулевой embedding для фото без лица
           embedding = np.zeros(512, dtype=np.float32)
           pose_info = {"angle_category": "unknown", "pitch": 0, "yaw": 0, "roll": 0}
       else:
           pose_info = recognizer.detect_head_pose(img)
      
       # Create new photo record
       from face_recognition import BodyRecognizer
       body_recognizer = BodyRecognizer()
       body_emb = body_recognizer.get_body_embedding(img)
       body_emb_json = json.dumps(body_emb.tolist()) if body_emb is not None else None


       # Normalize angle: map "затылок" and other variants to "back"
       angle_raw = request.angle if request.angle else pose_info['angle_category']
       if angle_raw and angle_raw.lower() in ("затылок", "back", "back_view", "rear"):
           final_angle = "back"
       else:
           final_angle = angle_raw


       new_photo = PersonPhoto(
           face_id=face_id,
           photo_base64=request.photo_base64,
           embedding=json.dumps(embedding.tolist()),
           body_embedding=body_emb_json,
           angle=final_angle,
           pose_pitch=pose_info['pitch'],
           pose_yaw=pose_info['yaw'],
           pose_roll=pose_info['roll']
       )
      
       db.add(new_photo)
       db.commit()
       db.refresh(new_photo)
      
       logger.info(f"✅ Added photo for Face {face_id} with angle: {pose_info['angle_category']}")
      
       # Reload embeddings to include new photo
       try:
           with open("RELOAD_EMBEDDINGS.flag", "w") as f:
               f.write("1")
       except Exception as e:
           logger.error(f"Error setting reload flag: {e}")
      
       await notify_update()
      
       return {
           "id": new_photo.id,
           "face_id": face_id,
           "angle": pose_info['angle_category'],
           "pose_pitch": pose_info['pitch'],
           "pose_yaw": pose_info['yaw'],
           "pose_roll": pose_info['roll'],
           "message": "Photo added successfully"
       }
   except HTTPException:
       raise
   except Exception as e:
       db.rollback()
       logger.error(f"Error adding face photo: {e}", exc_info=True)
       raise HTTPException(status_code=500, detail=str(e))




@app.put("/faces/{face_id}/photos/{photo_id}/angle", tags=["Faces"])
async def update_photo_angle(face_id: int, photo_id: int, angle: str, db: Session = Depends(get_db)):
   """✅ NEW: Update photo angle manually"""
   try:
       valid_angles = ['frontal', 'left_profile', 'right_profile', 'looking_up', 'looking_down', 'back', 'tilted', 'unknown']
      
       if angle not in valid_angles:
           raise HTTPException(status_code=400, detail=f"Invalid angle. Must be one of: {', '.join(valid_angles)}")
      
       photo = db.query(PersonPhoto).filter(
           PersonPhoto.id == photo_id,
           PersonPhoto.face_id == face_id
       ).first()
      
       if not photo:
           raise HTTPException(status_code=404, detail="Photo not found")
      
       photo.angle_manual = angle
       db.commit()
      
       logger.info(f"✅ Updated photo {photo_id} angle to: {angle}")
      
       # Reload embeddings
       try:
           with open("RELOAD_EMBEDDINGS.flag", "w") as f:
               f.write("1")
       except Exception as e:
           logger.error(f"Error setting reload flag: {e}")
      
       await notify_update()
      
       return {"id": photo.id, "angle": angle, "message": "Angle updated successfully"}
   except HTTPException:
       raise
   except Exception as e:
       db.rollback()
       logger.error(f"Error updating photo angle: {e}")
       raise HTTPException(status_code=500, detail=str(e))




@app.delete("/faces/{face_id}/photos/{photo_id}", tags=["Faces"])
async def delete_face_photo(face_id: int, photo_id: int, db: Session = Depends(get_db)):
   """✅ NEW: Delete an additional photo"""
   try:
       photo = db.query(PersonPhoto).filter(
           PersonPhoto.id == photo_id,
           PersonPhoto.face_id == face_id
       ).first()
      
       if not photo:
           raise HTTPException(status_code=404, detail="Photo not found")
      
       db.delete(photo)
       db.commit()
      
       logger.info(f"🗑️ Deleted photo {photo_id} for Face {face_id}")
      
       # Reload embeddings
       try:
           with open("RELOAD_EMBEDDINGS.flag", "w") as f:
               f.write("1")
       except Exception as e:
           logger.error(f"Error setting reload flag: {e}")
      
       await notify_update()
      
       return {"message": "Photo deleted successfully", "photo_id": photo_id}
   except HTTPException:
       raise
   except Exception as e:
       db.rollback()
       logger.error(f"Error deleting photo: {e}")
       raise HTTPException(status_code=500, detail=str(e))




@app.get("/camera_hls_url")
def camera_hls_url(quality: str = "SD"):
    """
    Возвращает HLS ссылку на поток с backend-аннотациями (независимо от источника камеры).
    """
    # Всегда отдаём наш HLS-поток с аннотациями
    return {"success": True, "url": "/hls/stream.m3u8"}


@app.get("/video_feed")
def video_feed():
   """Clean MJPEG video stream (no overlays, for frontend overlays)"""
   import video_state


   def mjpeg_stream():
       last_frame_id = None
       failed_reads = 0
       max_failed_reads = 30


       while True:
           try:
               system = getattr(video_state, "global_face_system", None)


               if not system:
                   logger.warning("System not ready")
                   time.sleep(0.5)
                   continue


               with system.frame_lock:
                   raw_frame = getattr(system, "latest_frame", None)  
                   if raw_frame is not None:
                       frame = raw_frame.copy() # ✅ FIX: Must copy to avoid tearing with CaptureThread
                   else:
                       frame = None

               if frame is None:
                   failed_reads += 1
                   if failed_reads > max_failed_reads:
                       logger.error("No frames for 6 seconds, stream timeout")
                       break
                   time.sleep(0.2)
                   continue


               failed_reads = 0


               current_frame_id = id(frame)
               if current_frame_id == last_frame_id:
                   time.sleep(0.02)
                   continue


               last_frame_id = current_frame_id


               # RESIZE for performance (2K -> 720p)
               # User reported massive lag. Sending 2560x1440 MJPEG is too heavy.
               h, w = frame.shape[:2]
               if w > 1280:
                   new_w = 1280
                   new_h = int(h * (1280 / w))
                   frame = cv2.resize(frame, (new_w, new_h))

               # Рисуем квадратик вокруг лица на backend
               if hasattr(system, "draw_overlay"):
                   system.draw_overlay(frame)

               ret, jpeg = cv2.imencode('.jpg', frame, [
                   cv2.IMWRITE_JPEG_QUALITY, 85,  # Lowered slightly for speed
                   cv2.IMWRITE_JPEG_OPTIMIZE, 1
               ])


               if ret:
                   yield (b"--frame\r\n"
                          b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
               else:
                   logger.warning("Failed to encode frame")

               # ✅ FIX: Removed time.sleep(0.15) which artificially capped stream at 6 FPS!
               time.sleep(0.01) # Small 10ms yield to prevent CPU pinning


           except Exception as e:
               logger.error(f"Stream error: {e}")
               time.sleep(0.5)
               break


   return StreamingResponse(
       mjpeg_stream(),
       media_type="multipart/x-mixed-replace; boundary=frame",
       headers={
           "Cache-Control": "no-cache, no-store, must-revalidate",
           "Pragma": "no-cache",
           "Expires": "0",
           "Connection": "close"
       }
   )


@app.get("/video_feed_raw")
def video_feed_raw():
   """✅ Новый endpoint: MJPEG поток raw-кадров (без разметки)"""
   import video_state


   def mjpeg_stream_raw():
       last_frame_id = None
       failed_reads = 0
       max_failed_reads = 30


       while True:
           try:
               system = getattr(video_state, "global_face_system", None)


               if not system:
                   logger.warning("System not ready")
                   time.sleep(0.5)
                   continue


               with system.frame_lock:
                   raw_frame = getattr(system, "latest_frame", None)
                   if raw_frame is not None:
                       frame = raw_frame.copy()
                   else:
                       frame = None

               if frame is None:
                   failed_reads += 1
                   if failed_reads > max_failed_reads:
                       logger.error("No raw frames for 6 seconds, stream timeout")
                       break
                   time.sleep(0.2)
                   continue


               failed_reads = 0


               current_frame_id = id(frame)
               if current_frame_id == last_frame_id:
                   time.sleep(0.02)
                   continue


               last_frame_id = current_frame_id


               ret, jpeg = cv2.imencode('.jpg', frame, [
                   cv2.IMWRITE_JPEG_QUALITY, 95,
                   cv2.IMWRITE_JPEG_OPTIMIZE, 1
               ])


               if ret:
                   yield (b"--frame\r\n"
                          b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
               else:
                   logger.warning("Failed to encode raw frame")

               # ✅ FIX: Removed time.sleep(0.15) which artificially capped stream at 6 FPS!
               time.sleep(0.01)


           except Exception as e:
               logger.error(f"Raw stream error: {e}")
               time.sleep(0.5)
               break


   return StreamingResponse(
       mjpeg_stream_raw(),
       media_type="multipart/x-mixed-replace; boundary=frame",
       headers={
           "Cache-Control": "no-cache, no-store, must-revalidate",
           "Pragma": "no-cache",
           "Expires": "0",
           "Connection": "close"
       }
   )


@app.get("/currently_visible", tags=["Camera"])
async def get_currently_visible():
   """
   ✅ IMPROVED: Uses pre-computed cache from run.py for better performance and consistency.
   The cache is updated in real-time with proper grace periods to prevent flickering.
   """
   import video_state
  
   system = getattr(video_state, "global_face_system", None)
   if not system:
       return {"currently_visible": [], "count": 0}
  
   # ✅ USE PREPARED CACHE (updated every frame in run.py)
   with system.cache_lock:
       active_people = list(system.currently_visible_cache)
  
   # ✅ Add photo_base64 for each person from database
   if active_people:
       from db import get_session
       from models import Face
      
       try:
           session = get_session()
           for person in active_people:
               if person.get('person_id'):
                   try:
                       face = session.query(Face).filter(Face.person_id == person['person_id']).first()
                       if face and face.photo_base64:
                           person['photo_base64'] = face.photo_base64
                       else:
                           person['photo_base64'] = None
                   except Exception as e:
                       logger.error(f"Error fetching photo for person {person['person_id']}: {e}")
                       person['photo_base64'] = None
               else:
                   person['photo_base64'] = None
           session.close()
       except Exception as e:
           logger.error(f"Database error in currently_visible: {e}")
  
   return {
       "currently_visible": active_people,
       "count": len(active_people)
   }






@app.get("/calendar_stats", tags=["Calendar"])
async def calendar_stats(year: int, month: int, db: Session = Depends(get_db)):
  """✅ ИСПРАВЛЕНО: Календарь теперь работает правильно"""
  from collections import defaultdict
  from calendar import monthrange


  try:
      days_in_month = monthrange(year, month)[1]
      start_date = datetime(year, month, 1)
      end_date = datetime(year, month, days_in_month, 23, 59, 59)


      faces = db.query(Face).filter(Face.status != 'employee').all()
 
      day_stats = {}
      for day in range(1, days_in_month + 1):
          day_date = datetime(year, month, day)
          day_stats[day] = {
              "date": day_date.strftime("%Y-%m-%d"),
              "unique_visitors": 0,
              "total_sessions": 0,
              "total_seconds": 0
          }


      for face in faces:
          visits = db.query(Visit).filter(
              and_(
                  Visit.face_id == face.id,
                  Visit.timestamp >= start_date,
                  Visit.timestamp <= end_date
              )
          ).order_by(Visit.timestamp).all()
       
          if not visits:
              continue
       
          visits_by_day = defaultdict(list)
          for v in visits:
              visits_by_day[v.timestamp.day].append(v)
       
          for day, vlist in visits_by_day.items():
              sessions, seconds = group_visits_to_sessions(vlist)
              day_stats[day]["unique_visitors"] += 1
              day_stats[day]["total_sessions"] += len(sessions)
              day_stats[day]["total_seconds"] += seconds


      return {
          "year": year,
          "month": month,
          "days": day_stats
      }
 
  except Exception as e:
      logger.error(f"Calendar error: {e}", exc_info=True)
      raise HTTPException(status_code=500, detail=str(e))




@app.get("/camera_stats", tags=["Camera"])
async def camera_stats():
   """
   Returns the people count for each camera (for now, just the main system).
   Extendable for multi-camera setups.
   """
   import video_state
   import time
   systems = []
   # For now, only one system is registered
   system = getattr(video_state, "global_face_system", None)
   if not system:
       return {"cameras": []}
   now = time.time()
   # If multi-camera: iterate over all systems
   # For now, just one
   cameras = [{
       "camera_id": 1,
       "name": "Main Camera",
       "people_count": len(getattr(system, "active_people", {})),
       "last_update": now
   }]
   return {"cameras": cameras}


from pydantic import BaseModel


class SaveFaceRectRequest(BaseModel):
   x: int = 0
   y: int = 0
   w: int = 0
   h: int = 0
   name: Optional[str] = None
   image_base64: Optional[str] = None # ✅ NEW: Image captured on frontend


@app.post("/save_face_rect", tags=["Faces"])
async def save_face_rect(request: SaveFaceRectRequest, db: Session = Depends(get_db)):
   """
   NEW: Сохранить лицо/фото по выделенному прямоугольнику на последнем кадре с камеры.
   """
   try:
       import video_state
       from face_recognition import FaceRecognizer, BodyRecognizer
       from face_creation_utils import create_or_get_face_by_embedding
       from tracker import TrackState # ✅ ADDED for linking
       import json


       system = getattr(video_state, "global_face_system", None)
       if not system:
           raise HTTPException(status_code=500, detail="Система не инициализирована")


       # Принудительно обновляем latest_frame перед обработкой (если возможно)
       try:
           if hasattr(system, "run_video_capture") and callable(getattr(system, "run_video_capture", None)):
               # Попытка захватить новый кадр (если реализовано)
               pass  # Здесь можно добавить логику захвата нового кадра, если поддерживается
       except Exception as e:
           logger.warning(f"Не удалось обновить кадр камеры: {e}")


       crop = None
       photo_base64 = None


       if request.image_base64:
           # ✅ OPTION A: Use image provided by frontend (No coordinate errors!)
           try:
               # Remove header if present (data:image/jpeg;base64,...)
               header_data = request.image_base64
               if "," in header_data:
                   header_data = header_data.split(",")[1]
              
               # ✅ CLEAN BASE64 STRING
               clean_b64 = header_data.strip().replace(" ", "+")
              
               img_bytes = base64.b64decode(clean_b64)
               nparr = np.frombuffer(img_bytes, np.uint8)
               crop = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
              
               if crop is not None:
                   photo_base64 = clean_b64 # Keep cleaned for DB
                   logger.info("✅ Используется изображение, переданное с фронтенда")
               else:
                   logger.error("cv2.imdecode returned None for frontend image")
                   raise ValueError("Failed to decode image as CV2 object")
                  
           except Exception as e:
               logger.error(f"Error decoding frontend image: {e}")
               raise HTTPException(status_code=400, detail=f"Не удалось декодировать изображение: {e}")
       else:
           # ❌ OPTION B: Fallback to backend cropping (often has coordinate/delay issues)
           with system.frame_lock:
               frame = getattr(system, "latest_frame", None)
          
           if frame is None:
               raise HTTPException(status_code=500, detail="Нет доступного кадра на сервере")


           x, y, w, h = request.x, request.y, request.w, request.h
           h_img, w_img = frame.shape[:2]
           x = max(0, min(x, w_img - 1))
           y = max(0, min(y, h_img - 1))
           w = max(1, min(w, w_img - x))
           h = max(1, min(h, h_img - y))
           crop = frame[y:y+h, x:x+w]
          
           # Encode for DB visibility
           _, buffer = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
           photo_base64 = base64.b64encode(buffer).decode('utf-8')


       if crop is None or crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
           raise HTTPException(status_code=400, detail="Изображение пустое или слишком маленькое")


       # ✅ NEW: Try to match with an active track to use its STABLE embedding
       track_match = None
       if request.w > 0 and request.h > 0 and system.tracker:
           req_box = (request.x, request.y, request.w, request.h)
           best_iou = 0
           for track in system.tracker.tracks:
               # Calculate IoU with request box
               iou = track.iou(req_box)
               if iou > 0.35 and iou > best_iou: # Lower threshold for manual selection overlap
                   best_iou = iou
                   track_match = track
          
           if track_match:
               logger.info(f"✅ Manual add matched to Track {track_match.track_id} (IoU={best_iou:.2f})")


       # ✅ EMBEDDING EXTRACTION
       recognizer = system.recognizer
       embedding = None
       pose_info = None
      
       # Priority 1: Use Track Embedding (Time-averaged, highly stable)
       if track_match:
           avg_emb = track_match.get_average_embedding()
           if avg_emb is not None:
               embedding = avg_emb
               logger.info("Using TRACK average embedding")
          
           if track_match.latest_pose:
               pose_info = track_match.latest_pose


       # Priority 2: Extract from crop (Single-shot)
       if embedding is None:
           embedding = recognizer.get_embedding(crop)
           if embedding is None:
               embedding = recognizer.get_embedding_with_mbd(crop)
      
       if embedding is None:
           # If still None, it might not be a "face", but we allow it as a person photo
           embedding = np.zeros(512, dtype=np.float32)
           pose_info = {"angle_category": "unknown"} if not pose_info else pose_info
       else:
           if not pose_info:
               pose_info = recognizer.detect_head_pose(crop)


       angle_raw = pose_info.get('angle_category', 'unknown') if pose_info else 'unknown'
       if angle_raw and angle_raw.lower() in ("затылок", "back", "back_view", "rear"):
           final_angle = "back"
       else:
           final_angle = angle_raw


       # ✅ FIXED: Use system's body recognizer
       body_recognizer = getattr(system, "body_recognizer", None)
       body_emb = None
      
       # Priority 1: Use Track Body Embedding (Full body from tracking)
       if track_match:
           avg_body = track_match.get_average_body_embedding()
           if avg_body is not None:
               body_emb = avg_body
               logger.info("Using TRACK average body embedding")


       # Priority 2: Extract from crop (Likely HEAD only, but better than nothing)
       if body_emb is None and body_recognizer:
           body_emb = body_recognizer.get_body_embedding(crop)
      
       # Сохраняем в БД
       from face_creation_utils import create_new_face_with_photo
      
       face_obj, new_photo, message = create_new_face_with_photo(
           db=db,
           photo_base64=photo_base64,
           embedding=embedding,
           name=request.name if request.name else None, # ✅ Ensure null handling
           angle=final_angle,
           body_embedding=body_emb,
           pose_info=pose_info
       )


       # ✅ LINK TRACK TO NEW PERSON IMMEDIATELY
       if track_match and face_obj:
           track_match.person_id = face_obj.person_id
           track_match.state = TrackState.CONFIRMED # Force confirm
           logger.info(f"⚡ Linked Track {track_match.track_id} to new Person {face_obj.person_id}")


       if face_obj is None:
           logger.error(f"Failed to create face: {message}")
           raise HTTPException(status_code=400, detail=f"Не удалось создать запись: {message}")


       # ✅ FIXED: Missing commit!
       try:
           db.commit()
           logger.info(f"Successfully saved face_id={face_obj.id}, person_id={face_obj.person_id}, duplicate={new_photo is None}")
       except Exception as e:
           db.rollback()
           logger.error(f"Database error during commit: {e}")
           raise HTTPException(status_code=500, detail=f"Ошибка БД при сохранении: {str(e)}")


       # Signal background tasks to reload embeddings if needed
       try:
           with open("RELOAD_EMBEDDINGS.flag", "w") as f:
               f.write("1")
       except Exception as e:
           logger.error(f"Error setting reload flag: {e}")


       await notify_update()


       return {
           "success": True,
           "id": face_obj.id,
           "person_id": face_obj.person_id,
           "angle": final_angle,
           "photo_id": new_photo.id if new_photo else None, # ✅ FIXED: Might be None on duplicate
           "is_duplicate": (new_photo is None),
           "message": message
       }
   except HTTPException:
       raise
   except Exception as e:
       db.rollback()
       logger.error(f"Error in save_face_rect: {e}", exc_info=True)
       raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


   except Exception as e:
       db.rollback()
       logger.error(f"Error in save_face_rect: {e}", exc_info=True)
       raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/camera/switch", tags=["Camera"])
def switch_camera(request: CameraSwitchRequest):
   """Switch camera source or quality (Sync to avoid blocking event loop)"""
   import video_state
   system = video_state.global_face_system
  
   if not system:
       raise HTTPException(status_code=503, detail="System not initialized")
  
   try:
       # This blocks, so using 'def' makes FastAPI run it in a threadpool
       success = system.switch_camera(request.source, request.quality)
       if success:
           return {"status": "success", "source": request.source, "quality": request.quality}
       else:
           raise HTTPException(status_code=500, detail="Failed to switch camera")
   except Exception as e:
       logger.error(f"Camera switch error: {e}")
       raise HTTPException(status_code=500, detail=str(e))


@app.get("/camera/status", tags=["Camera"])
async def get_camera_status():
   """Get current camera status"""
   import video_state
   system = video_state.global_face_system
  
   if not system:
       return {"source": "unknown", "quality": "unknown"}
      
   return {
       "source": getattr(system, "current_camera_source", "local"),
       "quality": getattr(system, "current_quality", "HD")
   }

@app.post("/webrtc/offer", tags=["Camera"])
async def webrtc_offer(offer: WebRTCOffer):
    """
    WebRTC Signaling Endpoint: Receive browser's SDP offer and return server's answer.
    This enables browser camera streaming to the backend for face recognition.
    """
    try:
        from webrtc_manager import GlobalWebRTCManager
        
        logger.info("📡 Received WebRTC offer from client")
        
        # Handle the offer and get answer
        answer = await GlobalWebRTCManager.handle_offer(offer.sdp, offer.type)
        
        logger.info("✅ WebRTC answer generated successfully")
        return answer
        
    except Exception as e:
        logger.error(f"WebRTC offer handling error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"WebRTC error: {str(e)}")


def run_api():
    import uvicorn
    import os
    
    # ✅ SSL Certificate paths
    cert_file = os.path.join(os.path.dirname(__file__), 'cert.pem')
    key_file = os.path.join(os.path.dirname(__file__), 'key.pem')
    
    # Check if SSL certificates exist
    ssl_enabled = os.path.exists(cert_file) and os.path.exists(key_file)
    
    if ssl_enabled:
        logger.info(f"🔒 Starting API server with HTTPS on {API_HOST}:{API_PORT}")
        config = uvicorn.Config(
            app, 
            host=API_HOST, 
            port=API_PORT, 
            log_level="info",
            ssl_certfile=cert_file,
            ssl_keyfile=key_file
        )
    else:
        logger.warning(f"⚠️ SSL certificates not found. Starting with HTTP only.")
        logger.warning(f"   WebRTC will NOT work without HTTPS!")
        config = uvicorn.Config(app, host=API_HOST, port=API_PORT, log_level="info")
    
    server = uvicorn.Server(config=config)
    
    # Monkey-patch install_signal_handlers to do nothing
    # This ensures uvicorn doesn't steal SIGINT/SIGTERM from our main run.py
    server.install_signal_handlers = lambda: None
    
    server.run()
