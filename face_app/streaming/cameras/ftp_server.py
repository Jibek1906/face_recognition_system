import os
import sys
import logging
import threading
import queue
import time
from pathlib import Path
from datetime import datetime

from face_app.core.config.env import BASE_DIR

# Сначала импортируем библиотеку, чтобы классы были доступны
try:
    from pyftpdlib.handlers import FTPHandler
    from pyftpdlib.authorizers import DummyAuthorizer
    from pyftpdlib.servers import FTPServer
    from pyftpdlib.filesystems import UnixFilesystem
except ImportError:
    print("Ошибка: Библиотека pyftpdlib не установлена. Запустите: pip install pyftpdlib")
    sys.exit(1)

# ──────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s [FTP] %(levelname)s %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("ftp_server")

# Also log to a file so you can inspect incoming connections without journalctl.
# Can be overridden by env FTP_LOG_FILE.
try:
    _default_log_file = os.path.join(BASE_DIR, "ftp_server.log")
    log_file = os.getenv("FTP_LOG_FILE", _default_log_file)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(fh)
except Exception:
    pass

# ──────────────────────────────────────────────────────────
# Config (can be overridden via env)
# ──────────────────────────────────────────────────────────
FTP_HOST     = os.getenv("FTP_HOST", "0.0.0.0")

# Many cameras only support the default FTP port (21) or hide the port field.
# For development (non-root) we keep a high port by default.
_default_ftp_port = "21" if (hasattr(os, "geteuid") and os.geteuid() == 0) else "2121"
FTP_PORT     = int(os.getenv("FTP_PORT", _default_ftp_port))
FTP_USER     = os.getenv("FTP_USER", "ftpuser")
FTP_PASSWORD = os.getenv("FTP_PASSWORD", "123456")
FTP_ROOT     = os.getenv("FTP_ROOT", os.path.join(BASE_DIR, "ftp_uploads"))

# Debug aid: accept any username/password (do NOT enable on public servers unless you
# understand the risk). This is helpful because some cameras hardcode username or
# send an unexpected one.
FTP_TRUST_ANY_USER = str(os.getenv("FTP_TRUST_ANY_USER", "0")).strip() == "1"

PASSIVE_PORTS_START = int(os.getenv("FTP_PASSIVE_PORTS_START", "60000"))
PASSIVE_PORTS_END = int(os.getenv("FTP_PASSIVE_PORTS_END", "60099"))
PASSIVE_PORTS = range(PASSIVE_PORTS_START, PASSIVE_PORTS_END + 1)

# If server is behind NAT or has multiple interfaces, set a public IP here.
# Example: FTP_MASQUERADE_ADDRESS=213.199.xx.xx
FTP_MASQUERADE_ADDRESS = os.getenv("FTP_MASQUERADE_ADDRESS", "")

new_file_queue: queue.Queue = queue.Queue()
os.makedirs(FTP_ROOT, exist_ok=True)

# ──────────────────────────────────────────────────────────
# Custom Classes
# ──────────────────────────────────────────────────────────

class TrustingAuthorizer(DummyAuthorizer):
    """Authorizer with extra diagnostics.

    Historically we allowed `ftpuser` with any password for debugging.
    This implementation:
    - logs every auth attempt
    - supports FTP_TRUST_ANY_USER=1 to accept any username/password
    - otherwise accepts FTP_USER with any password (to avoid camera-side password quirks)
    """

    def validate_authentication(self, username, password, handler):
        try:
            remote = getattr(handler, "remote_ip", "?")
        except Exception:
            remote = "?"

        logger.info(
            "🔑 Auth attempt: user='%s' pass_len=%s remote=%s trust_any=%s",
            username,
            len(password or ""),
            remote,
            int(FTP_TRUST_ANY_USER),
        )

        if FTP_TRUST_ANY_USER:
            # DummyAuthorizer expects the user to exist in user_table, otherwise
            # later steps (e.g. resolving home dir after PASS) will crash.
            # So we provision the user on-the-fly.
            try:
                if username not in getattr(self, "user_table", {}):
                    self.add_user(username, password or "", FTP_ROOT, perm="elradfmw")
                    logger.info("➕ Provisioned dynamic FTP user '%s' -> %s", username, FTP_ROOT)
            except Exception as e:
                logger.warning("⚠️ Could not provision dynamic user '%s': %s", username, e)

            logger.info("✅ FTP_TRUST_ANY_USER=1 -> accepting any credentials")
            return True

        # For camera integrations we accept the configured username with any password.
        # Many cameras are buggy with special chars or use fixed password.
        if username == FTP_USER:
            logger.info("✅ Password ignored for configured FTP_USER='%s' (debug mode)", FTP_USER)
            return True

        return super().validate_authentication(username, password, handler)

class CameraFTPHandler(FTPHandler):
    def on_connect(self):
        try:
            logger.info("🔌 Connect: remote=%s:%s", getattr(self, "remote_ip", "?"), getattr(self, "remote_port", "?"))
        except Exception:
            logger.info("🔌 Connect")

    def on_disconnect(self):
        try:
            logger.info("🔌 Disconnect: remote=%s:%s", getattr(self, "remote_ip", "?"), getattr(self, "remote_port", "?"))
        except Exception:
            logger.info("🔌 Disconnect")

    def on_login(self, username):
        try:
            logger.info(
                "✅ Login: user='%s' remote=%s:%s cwd=%s",
                username,
                getattr(self, "remote_ip", "?"),
                getattr(self, "remote_port", "?"),
                getattr(self, "cwd", "?"),
            )
        except Exception:
            logger.info("✅ Login: user='%s'", username)

    def on_login_failed(self, username, password):
        try:
            logger.warning(
                "❌ Login FAILED: user='%s' pass_len=%s remote=%s:%s",
                username,
                len(password or ""),
                getattr(self, "remote_ip", "?"),
                getattr(self, "remote_port", "?"),
            )
        except Exception:
            logger.warning("❌ Login FAILED: user='%s'", username)

    def on_file_received(self, file_path: str):
        logger.info(f"📷 Файл получен: {file_path}")

        # Trigger event-gating in the main pipeline (if running in same process).
        # This helps keep detection responsive right after an alarm snapshot arrives.
        try:
            from face_app.core.settings.manager import settings
            from face_app.state import video_state

            fs = getattr(video_state, 'global_face_system', None)
            if fs is not None and getattr(fs, 'shared_state', None) is not None:
                active_s = float(settings.get('event_gating_active_after_event_seconds', 12.0))
                fs.shared_state.trigger_event(active_s)
        except Exception:
            pass

        new_file_queue.put(file_path)

    def on_incomplete_file_received(self, file_path: str):
        logger.warning(f"⚠️ Файл не докачан: {file_path}")
        try: os.remove(file_path)
        except: pass

# ──────────────────────────────────────────────────────────
# Functions
# ──────────────────────────────────────────────────────────

def run_ftp_server():
    authorizer = TrustingAuthorizer()
    authorizer.add_user(FTP_USER, FTP_PASSWORD, FTP_ROOT, perm="elradfmw")
    authorizer.add_anonymous(FTP_ROOT, perm="elradfmw")

    handler = CameraFTPHandler
    handler.authorizer = authorizer
    handler.passive_ports = PASSIVE_PORTS
    if FTP_MASQUERADE_ADDRESS:
        handler.masquerade_address = FTP_MASQUERADE_ADDRESS
    handler.banner = "Face Recognition FTP Server ready."

    server = FTPServer((FTP_HOST, FTP_PORT), handler)
    logger.info(f"🚀 FTP сервер запущен на {FTP_HOST}:{FTP_PORT}")
    if FTP_MASQUERADE_ADDRESS:
        logger.info(f"🌐 FTP masquerade address: {FTP_MASQUERADE_ADDRESS}")
    server.serve_forever()

def process_uploaded_file(file_path: str):
    """Обрабатывает фото и СРАЗУ удаляет его, чтобы не забивать диск."""
    import cv2
    import requests
    import subprocess
    import tempfile

    def _convert_to_jpeg_if_needed(path: str) -> tuple[str, bool]:
        """Returns (path_to_jpeg, is_temp).

        Many cameras upload "snapshots" as raw H.265/HEVC elementary streams
        (e.g. *.dat). OpenCV can't read them directly, so we extract the first
        frame via ffmpeg.
        """
        ext = os.path.splitext(path)[1].lower()
        if ext in {".jpg", ".jpeg", ".png"}:
            return path, False

        # Fast signature checks: JPEG (FF D8), PNG (89 50 4E 47)
        try:
            with open(path, "rb") as f:
                sig = f.read(8)
            if sig.startswith(b"\xFF\xD8") or sig.startswith(b"\x89PNG"):
                return path, False
        except Exception:
            pass

        tmp = tempfile.NamedTemporaryFile(prefix="ftp_snapshot_", suffix=".jpg", delete=False)
        tmp.close()

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            # Most common for these uploads
            "-f",
            "hevc",
            "-i",
            path,
            "-frames:v",
            "1",
            "-q:v",
            "2",
            tmp.name,
        ]

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            err = (e.stderr or b"").decode("utf-8", errors="ignore")
            logger.warning(f"Не удалось конвертировать в JPEG через ffmpeg: {path} | {err.strip()}")
            try:
                os.unlink(tmp.name)
            except Exception:
                pass
            return path, False

        return tmp.name, True
    try:
        try:
            if os.path.getsize(file_path) == 0:
                logger.warning(f"Пустой файл от камеры (0 байт), пропускаю: {file_path}")
                return
        except Exception:
            pass

        jpeg_path, is_tmp = _convert_to_jpeg_if_needed(file_path)

        img = cv2.imread(jpeg_path)
        if img is None:
            logger.warning(f"Не удалось прочитать файл (даже после конвертации): {file_path}")
            return

        _, buf = cv2.imencode(".jpg", img)
        img_bytes = buf.tobytes()

        # Optional: save decoded snapshot for debugging/verification
        # Enable by setting: FTP_SAVE_JPEG=1
        try:
            if str(os.getenv("FTP_SAVE_JPEG", "0")).strip() == "1":
                debug_dir = os.getenv("FTP_DEBUG_SAVE_DIR")
                if not debug_dir:
                    debug_dir = os.path.join(FTP_ROOT, "_decoded")
                os.makedirs(debug_dir, exist_ok=True)
                out_path = os.path.join(debug_dir, f"{Path(file_path).stem}.jpg")
                with open(out_path, "wb") as f:
                    f.write(img_bytes)
                try:
                    # readable for user even if service runs as root
                    os.chmod(out_path, 0o644)
                except Exception:
                    pass
                logger.info(f"💾 Decoded snapshot saved: {out_path}")

                # Keep only last N debug files if configured
                try:
                    keep_last = int(os.getenv("FTP_DEBUG_KEEP_LAST", "0"))
                except Exception:
                    keep_last = 0
                _enforce_keep_last(debug_dir, keep_last)
        except Exception as e:
            logger.debug(f"Не удалось сохранить debug JPEG: {e}")

        api_port = int(os.getenv("API_PORT", 9200))

        # API is started in HTTPS mode when cert/key exist.
        # For localhost calls we can use https + verify=False.
        cert_file = os.path.join(BASE_DIR, 'cert.pem')
        key_file = os.path.join(BASE_DIR, 'key.pem')
        ssl_enabled = os.path.exists(cert_file) and os.path.exists(key_file)
        scheme = "https" if ssl_enabled else "http"
        api_url = f"{scheme}://127.0.0.1:{api_port}/ftp/process_image"

        resp = requests.post(
            api_url,
            files={"file": ("snapshot.jpg", img_bytes, "image/jpeg")},
            data={"source_path": file_path},
            timeout=10,
            verify=False,
        )
        result = resp.json() if resp.ok else {}
        faces = result.get("processed_faces", 0)
        logger.info(f"✅ Распознано лиц: {faces} | файл: {os.path.basename(file_path)}")

    except requests.exceptions.ConnectionError:
        logger.warning("⚠️  run.py не запущен — фото сохранено, но не распознано.")
    except Exception as e:
        logger.error(f"Ошибка обработки: {e}")
    finally:
        # If we created a temp converted JPEG, delete it.
        try:
            if 'is_tmp' in locals() and is_tmp and 'jpeg_path' in locals():
                os.remove(jpeg_path)
        except Exception:
            pass

        # 🗑️ Удаляем файл СРАЗУ после обработки — диск не забивается
        try:
            os.remove(file_path)
            logger.info(f"🗑️  Файл удалён: {os.path.basename(file_path)}")
            # Удаляем пустые папки
            folder = os.path.dirname(file_path)
            for _ in range(4):  # максимум 4 уровня вверх
                if folder == FTP_ROOT:
                    break
                if os.path.isdir(folder) and not os.listdir(folder):
                    os.rmdir(folder)
                    folder = os.path.dirname(folder)
        except Exception as del_err:
            logger.debug(f"Ошибка удаления файла: {del_err}")


def _enforce_keep_last(directory: str, keep_last: int):
    """Delete oldest files in directory leaving only newest keep_last (best-effort)."""
    try:
        if keep_last <= 0:
            return
        if not os.path.isdir(directory):
            return
        entries = []
        for name in os.listdir(directory):
            path = os.path.join(directory, name)
            if os.path.isfile(path):
                try:
                    entries.append((os.path.getmtime(path), path))
                except Exception:
                    pass
        entries.sort(reverse=True)  # newest first
        for _, path in entries[keep_last:]:
            try:
                os.remove(path)
            except Exception:
                pass
    except Exception:
        pass


def cleanup_old_files():
    """Страховочная очистка: удаляет файлы старше 1 часа (на случай сбоев)."""
    while True:
        time.sleep(3600)  # каждый час
        try:
            cutoff = time.time() - 3600
            for root, dirs, files in os.walk(FTP_ROOT):
                for fname in files:
                    fpath = os.path.join(root, fname)
                    if os.path.getmtime(fpath) < cutoff:
                        os.remove(fpath)
                        logger.info(f"🗑️  Авто-удалён старый файл: {fname}")
        except Exception as e:
            logger.error(f"Ошибка очистки: {e}")

def file_processor_worker():
    while True:
        path = None
        try:
            path = new_file_queue.get()
            logger.info("⚙️  Start processing uploaded file: %s", path)
            process_uploaded_file(path)
        except Exception as e:
            logger.exception("❌ FTP worker failed for file %s: %s", path, e)
        finally:
            try:
                if path is not None:
                    new_file_queue.task_done()
            except Exception:
                pass

if __name__ == "__main__":
    threading.Thread(target=file_processor_worker, daemon=True).start()
    run_ftp_server()
