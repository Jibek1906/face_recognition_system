import cv2
import subprocess
import threading
import time
import numpy as np
import video_state

TARGET_FPS = 15  # Raised from 10 → 15 for smoother playback

# Max display resolution fed into ffmpeg (pre-scaled in Python to cut pipe bandwidth)
STREAM_WIDTH = 1280
STREAM_HEIGHT = 720

def get_ffmpeg_cmd():
    return [
        "ffmpeg",
        "-y",
        "-f", "image2pipe",
        "-vcodec", "mjpeg",
        "-r", str(TARGET_FPS),
        "-i", "-",
        # Input is already 1280x720 from Python; just ensure even dimensions
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-tune", "zerolatency",
        "-pix_fmt", "yuv420p",
        "-crf", "20",                # 🔥 High quality, no forced bitrate drop (fixes pulsing)
        "-r", str(TARGET_FPS),
        "-g", str(TARGET_FPS * 2),   # keyframe every 2 seconds
        # 🔥 Removed -maxrate and -bufsize which caused the dark/bright pulsing artifact
        "-f", "hls",
        "-hls_time", "2",            # 2s segments (balances latency and stability)
        "-hls_list_size", "4",       # 4 segments = fixes 5-10s streaming freezes
        "-hls_flags", "delete_segments+independent_segments+omit_endlist",
        "/tmp/hls/stream.m3u8"
    ]

def get_annotated_frame():
    """
    ✅ STREAM PERFORMANCE FIX: DetectionThread now draws the overlay and pushes
    a pre-rendered annotated frame via shared_state.set_annotated_frame().
    We simply read that snapshot here — zero drawing, zero lock contention.
    """
    system = getattr(video_state, "global_face_system", None)
    if not system:
        return None
    # get_annotated_frame() is a single lock acquire + memcopy; no drawing at all.
    return system.shared_state.get_annotated_frame()

def resize_for_stream(frame):
    """Pre-scale frame to STREAM_WIDTH×STREAM_HEIGHT in Python before piping to ffmpeg.
    This cuts pipe bandwidth by ~4× compared to sending raw 2K frames."""
    h, w = frame.shape[:2]
    if w == STREAM_WIDTH and h == STREAM_HEIGHT:
        return frame
    # Maintain aspect ratio: fit within bounds
    scale = min(STREAM_WIDTH / w, STREAM_HEIGHT / h)
    new_w = int(w * scale) & ~1   # ensure even
    new_h = int(h * scale) & ~1
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

def hls_streamer_loop():
    print("[HLS] Waiting for first valid frame...")
    while True:
        frame = get_annotated_frame()
        if frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
            break
        time.sleep(0.1)
    print(f"[HLS] Got first frame: shape={frame.shape}, starting ffmpeg...")

    frame_interval = 1.0 / TARGET_FPS

    while True:  # Outer loop: restart ffmpeg if it dies
        proc = None
        try:
            import os
            hls_dir = "/tmp/hls"
            os.makedirs(hls_dir, exist_ok=True)
            # Clean stale files so ffmpeg starts fresh
            for f in os.listdir(hls_dir):
                try:
                    os.remove(os.path.join(hls_dir, f))
                except Exception:
                    pass

            proc = subprocess.Popen(
                get_ffmpeg_cmd(),
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            print("[HLS] ffmpeg started.")

            while True:
                t0 = time.time()
                frame = get_annotated_frame()
                if frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
                    # Pre-resize in Python — 4× less data through the pipe
                    frame = resize_for_stream(frame)
                    ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    if ret:
                        try:
                            proc.stdin.write(jpeg.tobytes())
                            proc.stdin.flush() # CRITICAL: Push immediately to prevent stream stalls
                        except Exception as e:
                            print("[HLS] FFmpeg pipe error:", e)
                            break
                # Precise sleep to maintain target FPS
                elapsed = time.time() - t0
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
        except Exception as e:
            print(f"[HLS] ffmpeg error: {e}")
        finally:
            if proc:
                try:
                    proc.stdin.close()
                except Exception:
                    pass
                proc.wait()
        print("[HLS] ffmpeg exited, restarting in 2s...")
        time.sleep(2)

def start_hls_streamer():
    t = threading.Thread(target=hls_streamer_loop, daemon=True)
    t.start()