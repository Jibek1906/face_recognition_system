"""Legacy global state bridge.

Historically, `video_state.py` held `global_face_system` to share the running
pipeline instance between `run.py` and `api.py`.
"""

global_face_system = None
