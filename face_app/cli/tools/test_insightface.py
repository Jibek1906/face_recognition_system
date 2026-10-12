import cv2
from insightface.app import FaceAnalysis

app = FaceAnalysis(allowed_modules=['detection', 'recognition'], providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

img = cv2.imread('/var/www/face_recognition/face_system/static/placeholder.jpg') # Or any image
if img is not None:
    faces = app.get(img)
    if faces:
        print(dir(faces[0]))
        if hasattr(faces[0], 'kps'):
            print(faces[0].kps)
