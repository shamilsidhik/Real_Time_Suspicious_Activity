from __future__ import annotations

import sys

MESSAGE = """
camera_server.py is deprecated.

The camera is now started inside Django by surveillance.apps.SurveillanceConfig
using the embedded in-process camera service.

Start only Django / Waitress.
Do not run camera_server.py in a second terminal anymore.
""".strip()


if __name__ == "__main__":
    print(MESSAGE)
    sys.exit(0)
