import threading
import asyncio
import queue
import cv2
import time

class FramePacket:
    def __init__(self, frame, ts):
        self.frame = frame
        self.ts = ts

class AsyncVideoPipeline:
    def __init__(self, camera_index=0, queue_size=8):
        self.frame_queue = queue.Queue(maxsize=queue_size)
        self.result_queue = queue.Queue(maxsize=queue_size)
        self.stop_event = threading.Event()
        self.camera_index = camera_index

    def start(self, detect_fn, track_fn, recognize_fn):
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.detect_thread = threading.Thread(target=self._detect_loop, args=(detect_fn,), daemon=True)
        self.track_thread = threading.Thread(target=self._track_loop, args=(track_fn,), daemon=True)
        self.recognize_thread = threading.Thread(target=self._recognize_loop, args=(recognize_fn,), daemon=True)
        self.capture_thread.start()
        self.detect_thread.start()
        self.track_thread.start()
        self.recognize_thread.start()

    def _capture_loop(self):
        cap = cv2.VideoCapture(self.camera_index)
        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue
            pkt = FramePacket(frame, time.time())
            try:
                # Если очередь переполнена — сбрасываем старый кадр, чтобы не было задержки
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put(pkt, timeout=0.1)
            except queue.Full:
                continue
        cap.release()

    def _detect_loop(self, detect_fn):
        while not self.stop_event.is_set():
            try:
                pkt = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            detections = detect_fn(pkt.frame)
            self.result_queue.put((pkt, detections))

    def _track_loop(self, track_fn):
        while not self.stop_event.is_set():
            try:
                pkt, detections = self.result_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            tracks = track_fn(detections)
            # Pass to next stage or store as needed

    def _recognize_loop(self, recognize_fn):
        while not self.stop_event.is_set():
            # Implement as needed for async recognition
            time.sleep(0.01)

    def stop(self):
        self.stop_event.set()
