"""Entry point launched by face_system.service (ExecStart=... run.py).

The real implementation lives in face_app/cli/run.py - kept as a thin
launcher here, same pattern as e.g. Django's manage.py, so the systemd
unit doesn't need to change.
"""

from face_app.cli.run import main

if __name__ == "__main__":
    main()
