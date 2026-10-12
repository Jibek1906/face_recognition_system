import cv2
import subprocess
import threading
import time
import os
import select
import numpy as np
from face_app.state import video_state
from face_app.core.config.env import STREAM_FPS

TARGET_FPS = STREAM_FPS or 20  # fallback for legacy configs
ANNOTATION_MAX_AGE = 0.6  # seconds — if annotations older than this, redraw on the fly

# HLS latency tuning (override via env without code changes)
# NOTE: latency ~= hls_time * (player_buffer_segments). Safari/native HLS typically buffers a few segments.
try:
    HLS_TIME = float(os.getenv("HLS_TIME", "1"))
except Exception:
    HLS_TIME = 1.0
try:
    HLS_LIST_SIZE = int(os.getenv("HLS_LIST_SIZE", "3"))
except Exception:
    HLS_LIST_SIZE = 3
try:
    HLS_CRF = int(os.getenv("HLS_CRF", "28"))
except Exception:
    HLS_CRF = 28

# Max display resolution fed into ffmpeg (pre-scaled in Python to cut pipe bandwidth)
STREAM_WIDTH = 1280
STREAM_HEIGHT = 720

# Pipe write timeout: if ffmpeg stdin is blocked longer than this, we drop the frame
# and restart ffmpeg. This prevents the streamer loop from freezing indefinitely.
PIPE_WRITE_TIMEOUT = 1.0  # seconds

def get_ffmpeg_cmd():
    return [
        "/usr/bin/ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{STREAM_WIDTH}x{STREAM_HEIGHT}",
        "-r", str(TARGET_FPS),
        "-i", "-",
        # Input is already STREAM_WIDTH x STREAM_HEIGHT from Python; just ensure even dimensions
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-tune", "zerolatency",
        "-pix_fmt", "yuv420p",
        "-crf", str(HLS_CRF),        # Higher CRF = smaller frames = pipe never overflows
        "-r", str(TARGET_FPS),
        "-g", str(max(1, int(round(TARGET_FPS * HLS_TIME)))),  # keyframe every segment (best for low-latency seeking)
        "-sc_threshold", "0",       # STRICT: Do not insert I-frames at scene changes (prevents uneven chunks)
        "-an",                      # EXPLICITLY disable audio to prevent A/V desync in HLS players
        "-f", "hls",
        "-hls_time", str(HLS_TIME),  # Smaller = lower latency but more filesystem churn
        "-hls_list_size", str(HLS_LIST_SIZE),  # smaller list => lower latency
        "-hls_flags", "delete_segments+independent_segments+omit_endlist",
        "/tmp/hls/stream.m3u8"
    ]

def acquire_stream_frame():
    """
    Returns a frame suitable for streaming.
    Always uses the freshest raw frame and renders the overlay on the fly
    to ensure the video stream is completely smooth and real-time.
    """
    system = getattr(video_state, "global_face_system", None)
    if not system:
        return None

    shared = system.shared_state

    # If annotated frame is too old (detector lag), fallback to latest raw frame.
    # This prevents the video from "freezing" on the last annotated frame.
    annotated_is_fresh = shared.is_annotation_fresh(ANNOTATION_MAX_AGE)
    frame = shared.get_annotated_frame(max_age=ANNOTATION_MAX_AGE, fallback_to_latest=True)
    if frame is None:
        return None

    try:
        # If we fell back to raw frame, we need to draw overlays on-the-fly.
        # If annotation is fresh, the frame already includes overlays.
        if not annotated_is_fresh:
            system.draw_overlay(frame)
    except Exception:
        # Drawing is best-effort; don't break the streaming loop.
        pass

    return frame

def resize_for_stream(frame):
    """Pre-scale frame to STREAM_WIDTH×STREAM_HEIGHT in Python before piping to ffmpeg.
    We always output a fixed size so rawvideo frames are consistent for ffmpeg."""
    h, w = frame.shape[:2]
    if w == STREAM_WIDTH and h == STREAM_HEIGHT:
        return frame
    return cv2.resize(frame, (STREAM_WIDTH, STREAM_HEIGHT), interpolation=cv2.INTER_LINEAR)

def _write_to_pipe_nonblocking(pipe, data):
    """
    Write `data` to `pipe` without blocking indefinitely.

    Uses select() to check if the pipe fd is ready for writing before each write chunk.
    If the pipe is not ready within PIPE_WRITE_TIMEOUT seconds, returns early (caller
    should restart ffmpeg).

    Returns a tuple (ok, partial):
    - ok=True: the full frame was written successfully.
    - ok=False, partial=False: nothing was written yet (safe to just drop this frame).
    - ok=False, partial=True: SOME bytes of this frame already reached ffmpeg's stdin.
      ffmpeg's rawvideo reader expects an exact byte count per frame, so a partial
      write permanently desyncs the frame boundary for every frame after it (ffmpeg
      keeps decoding "tail of this frame + head of the next" forever) until ffmpeg is
      restarted. The caller MUST restart ffmpeg immediately in this case, not just
      drop-and-continue.
    """
    fd = pipe.fileno()
    total = len(data)
    sent = 0
    deadline = time.time() + PIPE_WRITE_TIMEOUT

    while sent < total:
        remaining_time = deadline - time.time()
        if remaining_time <= 0:
            # Pipe is stuck — ffmpeg is not consuming data fast enough
            return False, sent > 0
        try:
            # select() tells us when fd is writable (non-blocking check)
            _, writable, _ = select.select([], [fd], [], remaining_time)
        except (ValueError, OSError):
            return False, sent > 0

        if not writable:
            # Timed out waiting for pipe to be writable
            return False, sent > 0

        try:
            n = os.write(fd, data[sent:sent + 65536])  # write up to 64KB at a time
            if n == 0:
                return False, sent > 0
            sent += n
        except BlockingIOError:
            # The pipe became full between select() and write(), wait a bit and retry
            time.sleep(0.001)
            continue
        except (BrokenPipeError, OSError):
            return False, sent > 0

    return True, False

def hls_streamer_loop():
    print("[HLS] Waiting for first valid frame...")
    while True:
        frame = acquire_stream_frame()
        if frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
            break
        time.sleep(0.1)
    print(f"[HLS] Got first frame: shape={frame.shape}, starting ffmpeg...")

    frame_interval = 1.0 / TARGET_FPS

    # If we stop receiving raw frames for too long, the HLS stream will freeze
    # (it may keep re-encoding the last frame). We treat this as a critical error
    # and restart ffmpeg to keep the pipeline self-healing.
    MAX_RAW_FRAME_AGE_SECONDS = float(os.getenv("HLS_MAX_RAW_FRAME_AGE_SECONDS", "5.0"))

    while True:  # Outer loop: restart ffmpeg if it dies
        proc = None
        try:
            hls_dir = "/tmp/hls"
            os.makedirs(hls_dir, exist_ok=True)
            # NOTE:
            # Do NOT wipe /tmp/hls on each restart.
            #
            # When we delete existing segments, clients that already downloaded the
            # playlist may still request the listed `.ts` files and get 404.
            # This results in a black screen / "loading timeout" in the UI.
            #
            # We rely on ffmpeg's `-hls_flags delete_segments` + `-hls_list_size`
            # to handle cleanup safely.

            proc = subprocess.Popen(
                get_ffmpeg_cmd(),
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                # Large OS-level pipe buffer: gives ffmpeg room to breathe between keyframes
                # without blocking the Python loop.
                bufsize=10 * 1024 * 1024  # 10 MB
            )
            print("[HLS] ffmpeg started.")

            # Set the stdin fd to non-blocking mode so writes return EAGAIN
            # instead of blocking indefinitely. We handle EAGAIN in _write_to_pipe_nonblocking.
            import fcntl
            fl = fcntl.fcntl(proc.stdin.fileno(), fcntl.F_GETFL)
            fcntl.fcntl(proc.stdin.fileno(), fcntl.F_SETFL, fl | os.O_NONBLOCK)

            consecutive_failures = 0
            last_frame = None
            
            # Use absolute timestamps for strict pacing without drift over time
            next_frame_time = time.time()

            while True:
                frame = acquire_stream_frame()
                if frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
                    last_frame = frame
                elif last_frame is None:
                    # Still no frames; wait briefly and retry
                    time.sleep(0.01)
                    continue

                # No new frame available — reuse the last frame to keep ffmpeg fed
                frame = last_frame if last_frame is not None else frame

                # Watchdog: if raw frames stop updating, break and restart ffmpeg.
                try:
                    system = getattr(video_state, "global_face_system", None)
                    shared = getattr(system, "shared_state", None) if system else None
                    if shared is not None:
                        raw_age = float(shared.get_latest_frame_age_seconds())
                        if raw_age > MAX_RAW_FRAME_AGE_SECONDS:
                            print(f"[HLS] Raw frames stale for {raw_age:.1f}s (> {MAX_RAW_FRAME_AGE_SECONDS:.1f}s) -> restarting ffmpeg")
                            break
                except Exception:
                    pass

                # Pre-resize in Python — fixed size for rawvideo input
                frame = resize_for_stream(frame)
                if frame is not None:
                    ok, partial = _write_to_pipe_nonblocking(proc.stdin, frame.tobytes())
                    if partial:
                        # Some bytes of this frame already reached ffmpeg's stdin - the
                        # rawvideo frame boundary is now permanently desynced. Every frame
                        # after this would decode as "torn" (half old + half new content)
                        # until ffmpeg is restarted, so restart immediately instead of
                        # tolerating it for several frames.
                        print("[HLS] Partial pipe write detected (frame boundary desynced), restarting ffmpeg immediately...")
                        break
                    if not ok:
                        consecutive_failures += 1
                        print(f"[HLS] Pipe write failed/timeout (#{consecutive_failures}), dropping frame")
                        if consecutive_failures >= 5:
                            # ffmpeg is stuck; restart it
                            print("[HLS] Too many pipe failures, restarting ffmpeg...")
                            break
                    else:
                        consecutive_failures = 0

                # Strict sleep tracking absolute time to eliminate average drift
                now = time.time()
                next_frame_time += frame_interval
                sleep_time = next_frame_time - now
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    # If we fall behind by more than 2 frames, snap clock so we don't burst excessively
                    if sleep_time < - (frame_interval * 2):
                        next_frame_time = now

        except Exception as e:
            print(f"[HLS] ffmpeg error: {e}")
        finally:
            if proc:
                try:
                    proc.stdin.close()
                except Exception:
                    pass
                try:
                    proc.terminate()
                    proc.wait(timeout=3)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
        print("[HLS] ffmpeg exited, restarting in 1s...")
        time.sleep(1)

def start_hls_streamer():
    t = threading.Thread(target=hls_streamer_loop, daemon=True)
    t.start()