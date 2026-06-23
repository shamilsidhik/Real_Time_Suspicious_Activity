# surveillance/views.py

import logging
import re
import threading
import time
from pathlib import Path
from uuid import uuid4

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.contrib.admin.views.decorators import staff_member_required
from django.core.files.base import ContentFile
from django.core.paginator import Paginator
from django.http import (
    HttpResponse,
    JsonResponse,
    StreamingHttpResponse,
)
from django.shortcuts import redirect, render
from django.contrib import messages
from django.utils import timezone
from PIL import Image, UnidentifiedImageError


logger = logging.getLogger(__name__)


try:
    from surveillance.models import DetectionLog, UploadedVideo

    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Camera server settings
# ---------------------------------------------------------------------------

CAM = getattr(
    settings,
    "CAMERA_SERVER_URL",
    "http://localhost:8765",
)

CAM_FALLBACKS = tuple(
    dict.fromkeys(
        [
            CAM,
            CAM.replace("localhost", "127.0.0.1"),
        ]
    )
)

# Prevent the same continuous detection from being logged many times.
LIVE_LOG_COOLDOWN_SEC = 2.0

KNOWN_FACES_DIR = Path(
    getattr(
        settings,
        "KNOWN_FACES_DIR",
        Path(settings.BASE_DIR) / "known_faces",
    )
)
KNOWN_PERSON_MAX_FILE_SIZE = 10 * 1024 * 1024
KNOWN_PERSON_ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}

# Activity labels treated as suspicious.
FIGHT_LABELS = {
    "fight",
    "fighting",
    "violence",
    "violent",
    "attack",
    "assault",
    "punch",
    "punching",
    "kick",
    "kicking",
}

SUSPICIOUS_ACTIVITY_LABELS = {
    *FIGHT_LABELS,
    "suspicious",
    "suspicious activity",
    "theft",
    "stealing",
    "robbery",
    "fall",
    "falling",
    "chasing",
    "vandalism",
}


# ---------------------------------------------------------------------------
# Shared HTTP session
# ---------------------------------------------------------------------------

_session = requests.Session()

_adapter = HTTPAdapter(
    max_retries=Retry(total=0)
)

_session.mount("http://", _adapter)
_session.mount("https://", _adapter)


# ---------------------------------------------------------------------------
# Live event state
# ---------------------------------------------------------------------------

_last_live_logs = {}
_live_event_states = {}
_live_event_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------

def _cam_status():
    """Return camera-server status data."""

    last_error = ""

    for cam_url in CAM_FALLBACKS:
        try:
            response = _session.get(
                f"{cam_url}/status",
                timeout=(1, 2),
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.ConnectionError:
            last_error = (
                f"Camera server not running on {cam_url}"
            )

        except requests.exceptions.Timeout:
            last_error = (
                f"Camera server timed out on {cam_url}"
            )

        except Exception as error:
            last_error = str(error)

    return {
        "error": last_error
        or f"Camera server not running on {CAM}",
        "camera_ok": False,
    }


def _cam_alive():
    """Check whether the camera server is available."""

    for cam_url in CAM_FALLBACKS:
        try:
            response = _session.get(
                f"{cam_url}/healthz",
                timeout=(0.5, 1),
            )

            if response.status_code == 200:
                return True

        except Exception:
            pass

    return False


def _cam_stream_url():
    """Return the first available camera-server URL."""

    for cam_url in CAM_FALLBACKS:
        try:
            response = _session.get(
                f"{cam_url}/healthz",
                timeout=(0.5, 1),
            )

            if response.status_code == 200:
                return cam_url

        except Exception:
            pass

    return CAM


def _fetch_detection_snapshot():
    """Download the current camera frame."""

    for cam_url in CAM_FALLBACKS:
        try:
            response = _session.get(
                f"{cam_url}/snapshot.jpg",
                timeout=(1, 3),
                headers={
                    "Cache-Control": "no-cache",
                    "User-Agent": "RealTimeSecurity",
                },
            )

            response.raise_for_status()

            content_type = response.headers.get(
                "Content-Type",
                "",
            ).lower()

            if (
                "image" in content_type
                and len(response.content) > 1000
            ):
                return response.content

        except Exception as error:
            logger.debug(
                "Could not download snapshot from %s: %s",
                cam_url,
                error,
            )

    return None


def _attach_detection_image(log_entry):
    """Attach a camera snapshot to a DetectionLog."""

    if not hasattr(log_entry, "image"):
        logger.warning(
            "DetectionLog has no image field. "
            "Run migrations after updating models.py."
        )
        return False

    image_data = _fetch_detection_snapshot()

    if not image_data:
        logger.warning(
            "No camera snapshot was available for log %s.",
            log_entry.pk,
        )
        return False

    filename = (
        f"{log_entry.activity_type}_"
        f"{log_entry.pk}_"
        f"{timezone.now():%Y%m%d_%H%M%S_%f}.jpg"
    )

    try:
        log_entry.image.save(
            filename,
            ContentFile(image_data),
            save=True,
        )
        return True

    except Exception as error:
        logger.exception(
            "Could not attach image to detection log %s: %s",
            log_entry.pk,
            error,
        )
        return False


# ---------------------------------------------------------------------------
# Camera response normalisation
# ---------------------------------------------------------------------------

def _normalise(cam):
    """Convert camera-server fields into dashboard fields."""

    cam_models = cam.get("models", {})

    # Permanent dashboard totals.
    permanent_id_count = int(
        cam.get(
            "id_cards",
            cam.get("id_event_count", 0),
        )
        or 0
    )

    permanent_weapon_count = int(
        cam.get(
            "weapons",
            cam.get("weapon_event_count", 0),
        )
        or 0
    )

    # Objects visible in the current camera frame.
    live_id_count = int(
        cam.get(
            "id_cards_live",
            cam.get("id_cards_detected_now", 0),
        )
        or 0
    )

    live_weapon_count = int(
        cam.get(
            "weapons_live",
            cam.get("weapons_detected_now", 0),
        )
        or 0
    )

    # Face and anti-spoof values.
    anti_spoof_status = str(
        cam.get("anti_spoof", "unknown")
    ).strip().lower()

    anti_spoof_confidence = float(
        cam.get("anti_spoof_conf", 0.0) or 0.0
    )

    spoof_detected = bool(
        cam.get("spoof_detected", False)
    ) or anti_spoof_status == "spoof"

    known_face_count = int(
        cam.get(
            "known_faces",
            cam.get("faces", 0),
        )
        or 0
    )

    unknown_face_count = int(
        cam.get(
            "unknown_faces",
            cam.get("unknown", 0),
        )
        or 0
    )

    total_face_count = int(
        cam.get(
            "faces_total",
            known_face_count + unknown_face_count,
        )
        or 0
    )

    live_face_count = int(
        cam.get("faces_live", 0) or 0
    )

    recognized_names = cam.get("recognized_names", [])
    if not isinstance(recognized_names, list):
        recognized_names = []

    return {
        "ok": cam.get("camera_ok", False),
        "camera_open": cam.get("camera_ok", False),
        "capture_fps": cam.get("fps", 0),
        "status": (
            "online"
            if cam.get("camera_ok")
            else "reconnecting"
        ),
        "last_error": cam.get("error", ""),

        "overlay": {
            "activity_label": cam.get(
                "activity",
                "unknown",
            ),
            "activity_conf": float(
                cam.get("activity_conf", 0.0) or 0.0
            ),

            # Permanent values displayed on dashboard cards.
            "id_count": permanent_id_count,
            "weapon_count": permanent_weapon_count,

            # Current-frame values used for creating logs.
            "id_live_count": live_id_count,
            "weapon_live_count": live_weapon_count,

            "id_event_count": int(
                cam.get(
                    "id_event_count",
                    permanent_id_count,
                )
                or 0
            ),
            "weapon_event_count": int(
                cam.get(
                    "weapon_event_count",
                    permanent_weapon_count,
                )
                or 0
            ),

            # Face and anti-spoof values.
            # Existing dashboard cards: Faces = recognised people,
            # Unknown = unrecognised live people.
            "faces": known_face_count,
            "known_faces": known_face_count,
            "unknown": unknown_face_count,
            "unknown_faces": unknown_face_count,
            "faces_total": total_face_count,
            "faces_live": live_face_count,
            "recognized_names": recognized_names,
            "face_recognition_status": cam.get(
                "face_recognition_status",
                "unknown",
            ),
            "known_gallery_size": int(
                cam.get("known_gallery_size", 0) or 0
            ),
            "anti_spoof": anti_spoof_status,
            "anti_spoof_conf": anti_spoof_confidence,
            "spoof_detected": spoof_detected,
        },

        "models": {
            "activity": {
                "available": bool(
                    cam_models.get("activity", False)
                )
            },
            "id_card": {
                "available": bool(
                    cam_models.get("id_card", False)
                )
            },
            "weapon": {
                "available": bool(
                    cam_models.get("weapon", False)
                )
            },
            "face": {
                "available": bool(
                    cam_models.get("face", False)
                )
            },
            "face_recognition": {
                "available": bool(
                    cam_models.get("face_recognition", False)
                )
            },
            "anti_spoof": {
                "available": bool(
                    cam_models.get("anti_spoof", False)
                )
            },
        },
    }


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _dashboard_totals():
    totals = {
        "total_uploaded_videos": 0,
        "total_logs": 0,
        "suspicious_count": 0,
        "id_card_count": 0,
    }

    if not MODELS_AVAILABLE:
        return totals

    try:
        totals["total_uploaded_videos"] = (
            UploadedVideo.objects.count()
        )

        totals["total_logs"] = (
            DetectionLog.objects.count()
        )

        totals["suspicious_count"] = (
            DetectionLog.objects.filter(
                activity_type__in=[
                    "suspicious",
                    "weapon_detected",
                    "fight_detected",
                    "spoof_detected",
                ]
            ).count()
        )

        totals["id_card_count"] = (
            DetectionLog.objects.filter(
                activity_type="id_detected"
            ).count()
        )

    except Exception as error:
        logger.exception(
            "Could not calculate dashboard totals: %s",
            error,
        )

    return totals


def _recent_logs():
    if not MODELS_AVAILABLE:
        return []

    try:
        return DetectionLog.objects.order_by(
            "-detected_at"
        )[:20]

    except Exception as error:
        logger.exception(
            "Could not load recent logs: %s",
            error,
        )
        return []


def _normalise_activity_label(label):
    value = str(label or "").strip().lower()

    for character in ("_", "-", "/", "\\"):
        value = value.replace(character, " ")

    return " ".join(value.split())


def _contains_activity_label(activity, labels):
    if activity in labels:
        return True

    return any(
        label in activity
        for label in labels
    )


def _is_new_live_event(
    user_id,
    event_type,
    active,
    signature,
):
    """
    Return True only when a new event begins.

    A continuously visible object is logged once. After it disappears,
    the next appearance can create another log.
    """

    state_key = (user_id, event_type)
    cooldown_key = (
        user_id,
        event_type,
        signature,
    )

    now = time.monotonic()

    with _live_event_lock:
        was_active = _live_event_states.get(
            state_key,
            False,
        )

        if not active:
            _live_event_states[state_key] = False
            return False

        _live_event_states[state_key] = True

        # Event is still continuously active.
        if was_active:
            return False

        last_log_time = _last_live_logs.get(
            cooldown_key,
            0.0,
        )

        if now - last_log_time < LIVE_LOG_COOLDOWN_SEC:
            return False

        _last_live_logs[cooldown_key] = now
        return True


def _create_detection_log(
    *,
    user,
    activity_type,
    confidence_score,
    message,
):
    """Create one log and attach the current camera image."""

    log_entry = DetectionLog.objects.create(
        user=user,
        activity_type=activity_type,
        confidence_score=max(
            0.0,
            min(float(confidence_score or 0.0), 1.0),
        ),
        message=message,
    )

    _attach_detection_image(log_entry)

    return log_entry


def _record_live_events(request, data):
    """Create logs only when a new live detection event begins."""

    if (
        not MODELS_AVAILABLE
        or not request.user.is_authenticated
    ):
        return

    overlay = data.get("overlay", {})

    activity = _normalise_activity_label(
        overlay.get(
            "activity_label",
            "unknown",
        )
    )

    activity_confidence = float(
        overlay.get(
            "activity_conf",
            0.0,
        )
        or 0.0
    )

    # Use only live counts when deciding whether a new event is active.
    id_live_count = int(
        overlay.get(
            "id_live_count",
            0,
        )
        or 0
    )

    weapon_live_count = int(
        overlay.get(
            "weapon_live_count",
            0,
        )
        or 0
    )

    # Permanent totals are used only in the log message.
    id_total = int(
        overlay.get(
            "id_count",
            0,
        )
        or 0
    )

    weapon_total = int(
        overlay.get(
            "weapon_count",
            0,
        )
        or 0
    )

    anti_spoof_status = str(
        overlay.get(
            "anti_spoof",
            "unknown",
        )
    ).strip().lower()

    anti_spoof_confidence = float(
        overlay.get(
            "anti_spoof_conf",
            0.0,
        )
        or 0.0
    )

    spoof_active = bool(
        overlay.get(
            "spoof_detected",
            False,
        )
    ) or anti_spoof_status == "spoof"

    face_count = int(
        overlay.get(
            "faces_total",
            overlay.get("faces", 0),
        )
        or 0
    )

    user_id = request.user.id

    weapon_active = weapon_live_count > 0
    id_active = id_live_count > 0

    fight_active = _contains_activity_label(
        activity,
        FIGHT_LABELS,
    )

    suspicious_activity_active = (
        _contains_activity_label(
            activity,
            SUSPICIOUS_ACTIVITY_LABELS,
        )
        and not fight_active
        and activity != "weapon"
    )

    try:
        # ---------------------------------------------------------------
        # Weapon event
        # ---------------------------------------------------------------
        if _is_new_live_event(
            user_id,
            "weapon_detected",
            weapon_active,
            "weapon",
        ):
            _create_detection_log(
                user=request.user,
                activity_type="weapon_detected",
                confidence_score=activity_confidence,
                message=(
                    "Live weapon detected. "
                    f"Visible now: {weapon_live_count}. "
                    f"Saved event count: {weapon_total}."
                ),
            )

        # ---------------------------------------------------------------
        # ID-card event
        # ---------------------------------------------------------------
        if _is_new_live_event(
            user_id,
            "id_detected",
            id_active,
            "id_card",
        ):
            _create_detection_log(
                user=request.user,
                activity_type="id_detected",
                confidence_score=1.0,
                message=(
                    "Live ID-card detected. "
                    f"Visible now: {id_live_count}. "
                    f"Saved event count: {id_total}."
                ),
            )

        # ---------------------------------------------------------------
        # Spoof-face event
        # ---------------------------------------------------------------
        if _is_new_live_event(
            user_id,
            "spoof_detected",
            spoof_active,
            "spoof_face",
        ):
            _create_detection_log(
                user=request.user,
                activity_type="spoof_detected",
                confidence_score=anti_spoof_confidence,
                message=(
                    "Spoof face detected. "
                    f"Faces visible: {face_count}. "
                    f"Anti-spoof status: {anti_spoof_status}. "
                    f"Confidence: {anti_spoof_confidence:.0%}."
                ),
            )

        # ---------------------------------------------------------------
        # Fight event
        # ---------------------------------------------------------------
        if _is_new_live_event(
            user_id,
            "fight_detected",
            fight_active,
            activity,
        ):
            _create_detection_log(
                user=request.user,
                activity_type="fight_detected",
                confidence_score=activity_confidence,
                message=(
                    f"Live fight activity detected: "
                    f"{activity}."
                ),
            )

        # ---------------------------------------------------------------
        # Other suspicious activity event
        # ---------------------------------------------------------------
        if _is_new_live_event(
            user_id,
            "suspicious",
            suspicious_activity_active,
            activity,
        ):
            _create_detection_log(
                user=request.user,
                activity_type="suspicious",
                confidence_score=activity_confidence,
                message=(
                    f"Live suspicious activity detected: "
                    f"{activity}."
                ),
            )

    except Exception as error:
        logger.exception(
            "Could not record live detection event: %s",
            error,
        )




# ---------------------------------------------------------------------------
# Known-person enrolment
# ---------------------------------------------------------------------------

def _safe_person_folder_name(person_name):
    """Return a safe folder name while keeping the person's readable name."""

    cleaned = re.sub(
        r"[^A-Za-z0-9 _-]+",
        "",
        str(person_name or ""),
    )
    cleaned = " ".join(cleaned.split()).strip()

    if not cleaned:
        return ""

    return cleaned.replace(" ", "_")[:80]


def _known_people():
    """Return enrolled people and their saved image counts."""

    KNOWN_FACES_DIR.mkdir(parents=True, exist_ok=True)
    people = []

    for folder in sorted(KNOWN_FACES_DIR.iterdir()):
        if not folder.is_dir():
            continue

        images = [
            path
            for path in folder.iterdir()
            if (
                path.is_file()
                and path.suffix.lower()
                in KNOWN_PERSON_ALLOWED_EXTENSIONS
            )
        ]

        people.append(
            {
                "name": folder.name.replace("_", " ").replace("-", " ").title(),
                "image_count": len(images),
            }
        )

    return people


def _reload_camera_face_gallery():
    """Ask the local camera server to reload enrolled face embeddings."""

    last_error = ""

    for cam_url in CAM_FALLBACKS:
        try:
            response = _session.post(
                f"{cam_url}/reload-faces",
                timeout=(2, 120),
            )

            try:
                payload = response.json()
            except ValueError:
                payload = {}

            if response.ok and payload.get("ok"):
                return True, payload

            last_error = (
                payload.get("error")
                or f"Camera server returned HTTP {response.status_code}"
            )

        except Exception as error:
            last_error = str(error)

    return False, {
        "error": last_error or "Camera server is unavailable",
    }


def _save_known_person_image(uploaded_file, destination):
    """Validate and convert an uploaded image to a safe JPEG file."""

    if uploaded_file.size > KNOWN_PERSON_MAX_FILE_SIZE:
        raise ValueError(
            f"{uploaded_file.name} is larger than 10 MB."
        )

    extension = Path(uploaded_file.name).suffix.lower()

    if extension not in KNOWN_PERSON_ALLOWED_EXTENSIONS:
        raise ValueError(
            f"{uploaded_file.name} is not a supported image."
        )

    try:
        uploaded_file.seek(0)
        with Image.open(uploaded_file) as image:
            image.load()
            image = image.convert("RGB")

            # Very large phone photos are resized to reduce recognition time.
            image.thumbnail((1600, 1600))

            image.save(
                destination,
                format="JPEG",
                quality=92,
                optimize=True,
            )

    except (UnidentifiedImageError, OSError) as error:
        raise ValueError(
            f"{uploaded_file.name} is not a valid image."
        ) from error


@staff_member_required
def known_person_upload(request):
    """Upload photos used to recognise a known person."""

    KNOWN_FACES_DIR.mkdir(parents=True, exist_ok=True)

    if request.method == "POST":
        person_name = str(
            request.POST.get("person_name", "")
        ).strip()
        folder_name = _safe_person_folder_name(person_name)
        uploaded_images = request.FILES.getlist("images")

        if not folder_name:
            messages.error(
                request,
                "Enter a valid person name.",
            )

        elif not uploaded_images:
            messages.error(
                request,
                "Select at least one clear face image.",
            )

        else:
            person_folder = KNOWN_FACES_DIR / folder_name
            person_folder.mkdir(
                parents=True,
                exist_ok=True,
            )

            saved_files = []

            try:
                for uploaded_file in uploaded_images:
                    filename = f"{uuid4().hex}.jpg"
                    destination = person_folder / filename

                    _save_known_person_image(
                        uploaded_file,
                        destination,
                    )
                    saved_files.append(destination)

            except ValueError as error:
                for saved_file in saved_files:
                    try:
                        saved_file.unlink()
                    except OSError:
                        pass

                messages.error(request, str(error))

            except Exception as error:
                logger.exception(
                    "Known-person upload failed: %s",
                    error,
                )
                messages.error(
                    request,
                    "The person images could not be saved.",
                )

            else:
                reloaded, reload_result = (
                    _reload_camera_face_gallery()
                )

                if reloaded:
                    messages.success(
                        request,
                        (
                            f"{len(saved_files)} image(s) saved for "
                            f"{person_name}. Face gallery reloaded "
                            f"with {reload_result.get('known_gallery_size', 0)} "
                            "image(s)."
                        ),
                    )
                else:
                    messages.warning(
                        request,
                        (
                            f"{len(saved_files)} image(s) saved for "
                            f"{person_name}, but the camera gallery could "
                            "not reload. Restart start.ps1. "
                            f"{reload_result.get('error', '')}"
                        ),
                    )

                return redirect("known_person_upload")

    return render(
        request,
        "surveillance/known_person_upload.html",
        {
            "known_people": _known_people(),
        },
    )


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

@login_required
def dashboard(request):
    data = _normalise(_cam_status())
    totals = _dashboard_totals()

    context = {
        "camera": {
            "capture_fps": data["capture_fps"],
            "overlay": data["overlay"],
            "status": data["status"],
            "camera_open": data["camera_open"],
            "models": data["models"],
        },
        **totals,
        "recent_logs": _recent_logs(),
    }

    return render(
        request,
        "surveillance/dashboard.html",
        context,
    )


# ---------------------------------------------------------------------------
# Detection status API
# ---------------------------------------------------------------------------

@login_required
def detection_api(request):
    data = _normalise(_cam_status())

    _record_live_events(
        request,
        data,
    )

    data["totals"] = _dashboard_totals()

    return JsonResponse(data)


# ---------------------------------------------------------------------------
# MJPEG stream proxy
# ---------------------------------------------------------------------------

def _mjpeg_generator(cam_url):
    try:
        with requests.get(
            cam_url,
            stream=True,
            timeout=(2, None),
        ) as upstream:

            upstream.raise_for_status()

            for chunk in upstream.iter_content(
                chunk_size=8192
            ):
                if chunk:
                    yield chunk

    except Exception as error:
        logger.debug(
            "Camera stream stopped: %s",
            error,
        )
        return


@login_required
def stream_proxy(request):
    if not _cam_alive():
        return HttpResponse(
            (
                "Camera server is not running. "
                "Start it with start.ps1"
            ),
            status=503,
            content_type="text/plain",
        )

    response = StreamingHttpResponse(
        _mjpeg_generator(
            f"{_cam_stream_url()}/stream.mjpg"
        ),
        content_type=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        ),
    )

    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"

    return response


# ---------------------------------------------------------------------------
# Live stream page
# ---------------------------------------------------------------------------

@login_required
def stream_page(request):
    return render(
        request,
        "surveillance/stream.html",
        {
            "stream_url": "/stream/",
            "status_api_url": "/api/status/",
        },
    )


# ---------------------------------------------------------------------------
# Uploaded-video page
# ---------------------------------------------------------------------------

@login_required
def upload(request):
    form = None
    uploads = []
    error = None

    if MODELS_AVAILABLE:
        try:
            from surveillance.forms import UploadedVideoForm

            if request.method == "POST":
                form = UploadedVideoForm(
                    request.POST,
                    request.FILES,
                )

                if form.is_valid():
                    uploaded_video = form.save(
                        commit=False
                    )

                    uploaded_video.user = request.user
                    uploaded_video.save()

                    return redirect("upload")

            else:
                form = UploadedVideoForm()

            uploads = (
                UploadedVideo.objects.filter(
                    user=request.user
                )
                .order_by("-uploaded_at")[:20]
            )

        except Exception as exception:
            form = None
            error = str(exception)

    return render(
        request,
        "surveillance/upload.html",
        {
            "form": form,
            "uploads": uploads,
            "error": error,
        },
    )


# ---------------------------------------------------------------------------
# Detection logs
# ---------------------------------------------------------------------------

@login_required
def logs(request):
    if MODELS_AVAILABLE:
        try:
            all_logs = DetectionLog.objects.order_by(
                "-detected_at"
            )
        except Exception as error:
            logger.exception(
                "Could not load detection logs: %s",
                error,
            )
            all_logs = DetectionLog.objects.none()
    else:
        all_logs = []

    paginator = Paginator(
        all_logs,
        25,
    )

    logs_page = paginator.get_page(
        request.GET.get("page")
    )

    return render(
        request,
        "surveillance/logs.html",
        {
            "logs_page": logs_page,
        },
    )

