import subprocess
from face_app.streaming.hls_streamer import get_ffmpeg_cmd

cmd = get_ffmpeg_cmd()
print(" ".join(cmd))
proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
out, err = proc.communicate(b"")
print(err.decode())
