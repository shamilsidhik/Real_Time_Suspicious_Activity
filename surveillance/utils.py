import urllib.request
from datetime import datetime

from django.core.files.base import ContentFile


CAMERA_SNAPSHOT_URL = "http://127.0.0.1:8765/snapshot.jpg"


def attach_detection_image(log_entry):
    """Attach the current camera frame to a DetectionLog."""

    try:
        request = urllib.request.Request(
            CAMERA_SNAPSHOT_URL,
            headers={
                "User-Agent": "RealTimeSecurity",
                "Cache-Control": "no-cache",
            },
        )

        with urllib.request.urlopen(request, timeout=3) as response:
            image_data = response.read()

        if len(image_data) < 1000:
            print("Detection snapshot was empty.")
            return False

        filename = (
            f"detection_{log_entry.pk}_"
            f"{datetime.now():%Y%m%d_%H%M%S_%f}.jpg"
        )

        log_entry.image.save(
            filename,
            ContentFile(image_data),
            save=True,
        )

        return True

    except Exception as error:
        print(f"Could not save detection image: {error}")
        return False