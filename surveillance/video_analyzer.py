# surveillance/video_analyzer.py

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import cv2
from django.conf import settings
from django.core.files.base import ContentFile
from django.db import close_old_connections
from ultralytics import YOLO

from surveillance.models import DetectionLog, UploadedVideo
from surveillance.email_alerts import queue_detection_alert


logger = logging.getLogger(__name__)

_activity_model: YOLO | None = None
_weapon_model: YOLO | None = None

_model_lock = threading.Lock()
_analysis_lock = threading.Lock()


SUSPICIOUS_ACTIVITY_LABELS = {
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
    "theft",
    "stealing",
    "robbery",
    "suspicious",
    "fall",
    "falling",
    "chasing",
    "vandalism",
}

SAFE_ACTIVITY_PHRASES = {
    "normal",
    "safe",
    "non violence",
    "no violence",
    "nonviolent",
    "no fight",
}

WEAPON_LABEL_WORDS = {
    "weapon",
    "gun",
    "pistol",
    "rifle",
    "knife",
    "blade",
    "firearm",
    "revolver",
}


def _normalise_label(label: str) -> str:
    value = str(label or "").strip().lower()

    for character in ("_", "-", "/", "\\"):
        value = value.replace(character, " ")

    return " ".join(value.split())


def _contains_any(value: str, words: set[str]) -> bool:
    normalised = _normalise_label(value)
    return any(word in normalised for word in words)


def _is_suspicious_activity(label: str) -> bool:
    value = _normalise_label(label)

    if not value or value in SAFE_ACTIVITY_PHRASES:
        return False

    if any(phrase in value for phrase in SAFE_ACTIVITY_PHRASES):
        return False

    return _contains_any(
        value,
        SUSPICIOUS_ACTIVITY_LABELS,
    )


def _resolve_path(
    setting_name: str,
    default_relative: str,
) -> Path:
    configured = getattr(
        settings,
        setting_name,
        Path(settings.BASE_DIR) / default_relative,
    )

    path = Path(configured)

    if not path.is_absolute():
        path = Path(settings.BASE_DIR) / path

    return path.resolve()


def _get_activity_model() -> YOLO:
    global _activity_model

    if _activity_model is not None:
        return _activity_model

    with _model_lock:
        if _activity_model is not None:
            return _activity_model

        model_path = _resolve_path(
            "ACTIVITY_MODEL_PATH",
            "ml/models/activity_yolov8/best.pt",
        )

        if not model_path.exists():
            raise FileNotFoundError(
                f"Activity model was not found: {model_path}"
            )

        logger.info(
            "Loading uploaded-media activity model: %s",
            model_path,
        )

        _activity_model = YOLO(str(model_path))
        return _activity_model


def _get_optional_weapon_model() -> YOLO | None:
    global _weapon_model

    if _weapon_model is not None:
        return _weapon_model

    with _model_lock:
        if _weapon_model is not None:
            return _weapon_model

        model_path = _resolve_path(
            "WEAPON_MODEL_PATH",
            "ml/models/weapons_yolov8/best.pt",
        )

        if not model_path.exists():
            logger.info(
                "Weapon model not found; activity-only analysis: %s",
                model_path,
            )
            return None

        logger.info(
            "Loading uploaded-media weapon model: %s",
            model_path,
        )

        _weapon_model = YOLO(str(model_path))
        return _weapon_model


def _tensor_values(value: Any) -> list:
    """Convert a tensor/list-like value to a normal Python list."""

    if value is None:
        return []

    try:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "tolist"):
            value = value.tolist()
    except Exception:
        pass

    if isinstance(value, (list, tuple)):
        return list(value)

    return [value]


def _predict_activity(
    model: YOLO,
    frame: Any,
    *,
    threshold: float | None = None,
    image_size: int | None = None,
) -> tuple[str, float]:
    """
    Predict activity from one frame.

    For classification models with fight/normal/violence classes, all class
    probabilities are inspected. Fight and violence are combined so a fighting
    image is not incorrectly marked normal just because the suspicious
    probability is split between both suspicious classes.
    """

    if threshold is None:
        threshold = float(
            getattr(settings, "VIDEO_ACTIVITY_CONF", 0.70)
        )

    if image_size is None:
        image_size = int(
            getattr(settings, "VIDEO_ACTIVITY_IMGSZ", 416)
        )

    result = model.predict(
        source=frame,
        imgsz=int(image_size),
        verbose=False,
    )[0]

    probabilities = getattr(result, "probs", None)

    # YOLO classification model.
    if probabilities is not None:
        raw_values = getattr(probabilities, "data", None)

        if raw_values is not None:
            try:
                values = raw_values.detach().cpu().tolist()
            except Exception:
                values = raw_values.tolist()

            class_scores = {
                _normalise_label(str(result.names[index])): float(score)
                for index, score in enumerate(values)
            }

            fight_score = float(
                class_scores.get("fight", 0.0)
            )
            violence_score = float(
                class_scores.get("violence", 0.0)
            )
            normal_score = float(
                class_scores.get("normal", 0.0)
            )

            suspicious_score = fight_score + violence_score

            image_suspicious_min = float(
                getattr(
                    settings,
                    "IMAGE_SUSPICIOUS_CLASS_CONF",
                    0.20,
                )
            )
            image_suspicious_sum = float(
                getattr(
                    settings,
                    "IMAGE_SUSPICIOUS_COMBINED_CONF",
                    0.38,
                )
            )

            best_suspicious_label = (
                "fight"
                if fight_score >= violence_score
                else "violence"
            )
            best_suspicious_score = max(
                fight_score,
                violence_score,
            )

            if (
                best_suspicious_score >= image_suspicious_min
                or suspicious_score >= image_suspicious_sum
            ):
                return (
                    best_suspicious_label,
                    best_suspicious_score,
                )

            best_index = max(
                range(len(values)),
                key=lambda index: values[index],
            )

            return (
                str(result.names[best_index]),
                float(values[best_index]),
            )

        class_id = int(probabilities.top1)
        confidence = float(probabilities.top1conf)
        return str(result.names[class_id]), confidence

    boxes = getattr(result, "boxes", None)

    # YOLO detection model: inspect every result and prioritise suspicious labels.
    candidates: list[tuple[str, float]] = []

    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            try:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                label = str(result.names[class_id])
            except (KeyError, TypeError, ValueError, IndexError):
                continue

            if confidence >= float(threshold):
                candidates.append((label, confidence))

    if not candidates:
        return "normal", 0.0

    suspicious_candidates = [
        (label, confidence)
        for label, confidence in candidates
        if _is_suspicious_activity(label)
    ]

    if suspicious_candidates:
        return max(
            suspicious_candidates,
            key=lambda item: item[1],
        )

    return max(
        candidates,
        key=lambda item: item[1],
    )


def _predict_weapon(
    model: YOLO | None,
    frame: Any,
) -> tuple[bool, str, float]:
    if model is None:
        return False, "", 0.0

    threshold = float(
        getattr(settings, "VIDEO_WEAPON_CONF", 0.55)
    )

    image_size = int(
        getattr(settings, "VIDEO_WEAPON_IMGSZ", 512)
    )

    result = model.predict(
        source=frame,
        imgsz=image_size,
        conf=threshold,
        verbose=False,
    )[0]

    boxes = getattr(result, "boxes", None)

    if boxes is None or len(boxes) == 0:
        return False, "", 0.0

    candidates: list[tuple[float, str]] = []

    for box in boxes:
        class_id = int(box.cls[0])
        label = str(result.names[class_id])
        confidence = float(box.conf[0])

        if _contains_any(label, WEAPON_LABEL_WORDS):
            candidates.append((confidence, label))

    if not candidates:
        return False, "", 0.0

    confidence, label = max(candidates)
    return True, label, confidence


def _analyse_frame(
    frame: Any,
    activity_model: YOLO,
    weapon_model: YOLO | None,
    *,
    activity_threshold: float | None = None,
    activity_image_size: int | None = None,
) -> dict[str, Any]:
    activity_label, activity_confidence = _predict_activity(
        activity_model,
        frame,
        threshold=activity_threshold,
        image_size=activity_image_size,
    )

    weapon_found, weapon_label, weapon_confidence = (
        _predict_weapon(
            weapon_model,
            frame,
        )
    )

    activity_suspicious = _is_suspicious_activity(
        activity_label
    )

    suspicious = activity_suspicious or weapon_found

    if weapon_found and weapon_confidence >= activity_confidence:
        best_label = weapon_label
        best_confidence = weapon_confidence
        best_reason = "weapon"
    else:
        best_label = activity_label
        best_confidence = activity_confidence
        best_reason = "activity"

    return {
        "suspicious": suspicious,
        "best_label": best_label,
        "best_confidence": best_confidence,
        "best_reason": best_reason,
    }


def _save_log_image(
    log_entry: DetectionLog,
    frame: Any,
) -> None:
    if frame is None or not hasattr(log_entry, "image"):
        return

    success, encoded = cv2.imencode(
        ".jpg",
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, 90],
    )

    if not success:
        return

    filename = (
        f"uploaded_media_{log_entry.video_id}_"
        f"log_{log_entry.pk}.jpg"
    )

    log_entry.image.save(
        filename,
        ContentFile(encoded.tobytes()),
        save=True,
    )


def _create_analysis_log(
    uploaded_media: UploadedVideo,
    *,
    is_suspicious: bool,
    confidence: float,
    message: str,
    frame: Any,
) -> None:
    log_entry = DetectionLog.objects.create(
        user=uploaded_media.user,
        video=uploaded_media,
        activity_type=(
            "suspicious"
            if is_suspicious
            else "normal"
        ),
        confidence_score=max(
            0.0,
            min(float(confidence or 0.0), 1.0),
        ),
        message=message,
    )

    _save_log_image(log_entry, frame)

    if is_suspicious:
        queue_detection_alert(log_entry.pk)


def _analyse_uploaded_image(
    uploaded_media: UploadedVideo,
    activity_model: YOLO,
    weapon_model: YOLO | None,
) -> None:
    """
    Analyse a still image using more sensitive image-specific settings.

    The original image and a horizontally flipped copy are checked. A
    suspicious prediction from either pass is preferred.
    """

    image_path = Path(uploaded_media.image.path)
    frame = cv2.imread(str(image_path))

    if frame is None:
        raise ValueError(
            "The uploaded image could not be read."
        )

    image_threshold = float(
        getattr(settings, "IMAGE_ACTIVITY_CONF", 0.35)
    )
    image_size = int(
        getattr(settings, "IMAGE_ACTIVITY_IMGSZ", 640)
    )

    variants = [
        ("original", frame),
        ("flipped", cv2.flip(frame, 1)),
    ]

    results: list[tuple[str, dict[str, Any]]] = []

    for variant_name, variant_frame in variants:
        result = _analyse_frame(
            variant_frame,
            activity_model,
            weapon_model,
            activity_threshold=image_threshold,
            activity_image_size=image_size,
        )
        results.append((variant_name, result))

    suspicious_results = [
        (variant_name, result)
        for variant_name, result in results
        if bool(result.get("suspicious"))
    ]

    if suspicious_results:
        selected_variant, result = max(
            suspicious_results,
            key=lambda item: float(
                item[1].get("best_confidence", 0.0)
            ),
        )
    else:
        selected_variant, result = max(
            results,
            key=lambda item: float(
                item[1].get("best_confidence", 0.0)
            ),
        )

    is_suspicious = bool(result["suspicious"])

    final_status = (
        "Suspicious"
        if is_suspicious
        else "Normal"
    )

    UploadedVideo.objects.filter(
        pk=uploaded_media.pk
    ).update(status=final_status)

    message = (
        f"Uploaded image analysis completed: {final_status}. "
        f"Best detection: {result['best_label']} "
        f"({result['best_reason']}) at "
        f"{result['best_confidence']:.0%}. "
        f"Image pass: {selected_variant}. "
        f"Activity threshold: {image_threshold:.0%}."
    )

    _create_analysis_log(
        uploaded_media,
        is_suspicious=is_suspicious,
        confidence=float(result["best_confidence"]),
        message=message,
        frame=frame,
    )


def _analyse_uploaded_video(
    uploaded_media: UploadedVideo,
    activity_model: YOLO,
    weapon_model: YOLO | None,
) -> None:
    video_path = Path(uploaded_media.video.path)
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise ValueError(
            "The uploaded file is not a readable video."
        )

    try:
        source_fps = float(
            capture.get(cv2.CAP_PROP_FPS) or 25.0
        )

        analysis_fps = float(
            getattr(settings, "VIDEO_ANALYSIS_FPS", 2.0)
        )

        frame_step = max(
            1,
            round(
                source_fps
                / max(analysis_fps, 0.5)
            ),
        )

        required_hits = int(
            getattr(settings, "VIDEO_CONFIRM_HITS", 3)
        )

        ratio_threshold = float(
            getattr(
                settings,
                "VIDEO_SUSPICIOUS_RATIO",
                0.10,
            )
        )

        frame_number = 0
        analysed_frames = 0
        suspicious_frames = 0
        consecutive_suspicious = 0
        maximum_consecutive = 0

        best_label = "normal"
        best_confidence = 0.0
        best_reason = "activity"

        first_frame = None
        suspicious_frame = None

        while True:
            success, frame = capture.read()

            if not success:
                break

            frame_number += 1

            if frame_number % frame_step != 0:
                continue

            analysed_frames += 1

            if first_frame is None:
                first_frame = frame.copy()

            result = _analyse_frame(
                frame,
                activity_model,
                weapon_model,
            )

            if float(result["best_confidence"]) > best_confidence:
                best_label = str(result["best_label"])
                best_confidence = float(
                    result["best_confidence"]
                )
                best_reason = str(result["best_reason"])

            if result["suspicious"]:
                suspicious_frames += 1
                consecutive_suspicious += 1

                if suspicious_frame is None:
                    suspicious_frame = frame.copy()
            else:
                consecutive_suspicious = 0

            maximum_consecutive = max(
                maximum_consecutive,
                consecutive_suspicious,
            )

    finally:
        capture.release()

    if analysed_frames == 0:
        raise ValueError(
            "No readable video frames were found."
        )

    suspicious_ratio = (
        suspicious_frames / analysed_frames
    )

    is_suspicious = (
        maximum_consecutive >= required_hits
        or (
            suspicious_frames >= required_hits
            and suspicious_ratio >= ratio_threshold
        )
    )

    final_status = (
        "Suspicious"
        if is_suspicious
        else "Normal"
    )

    UploadedVideo.objects.filter(
        pk=uploaded_media.pk
    ).update(status=final_status)

    message = (
        f"Uploaded video analysis completed: {final_status}. "
        f"Best detection: {best_label} "
        f"({best_reason}) at {best_confidence:.0%}. "
        f"Suspicious frames: "
        f"{suspicious_frames}/{analysed_frames}."
    )

    representative_frame = (
        suspicious_frame
        if is_suspicious
        else first_frame
    )

    _create_analysis_log(
        uploaded_media,
        is_suspicious=is_suspicious,
        confidence=best_confidence,
        message=message,
        frame=representative_frame,
    )


def analyze_uploaded_media(upload_id: int) -> None:
    """
    Analyse an uploaded image or video.

    Status flow:
        Processing -> Normal / Suspicious / Failed
    """

    close_old_connections()

    try:
        uploaded_media = (
            UploadedVideo.objects
            .select_related("user")
            .get(pk=upload_id)
        )

        UploadedVideo.objects.filter(
            pk=upload_id
        ).update(status="Processing")

        activity_model = _get_activity_model()
        weapon_model = _get_optional_weapon_model()

        with _analysis_lock:
            if uploaded_media.image:
                _analyse_uploaded_image(
                    uploaded_media,
                    activity_model,
                    weapon_model,
                )

            elif uploaded_media.video:
                _analyse_uploaded_video(
                    uploaded_media,
                    activity_model,
                    weapon_model,
                )

            else:
                raise ValueError(
                    "This upload contains no image or video."
                )

        logger.info(
            "Uploaded %s %s analysis completed",
            uploaded_media.media_type,
            upload_id,
        )

    except Exception as error:
        logger.exception(
            "Uploaded-media analysis failed for %s: %s",
            upload_id,
            error,
        )

        try:
            UploadedVideo.objects.filter(
                pk=upload_id
            ).update(status="Failed")
        except Exception:
            logger.exception(
                "Could not mark upload %s as failed",
                upload_id,
            )

    finally:
        close_old_connections()


def queue_media_analysis(upload_id: int) -> None:
    """Start image/video analysis without blocking the upload request."""

    worker = threading.Thread(
        target=analyze_uploaded_media,
        args=(upload_id,),
        daemon=True,
        name=f"media-analysis-{upload_id}",
    )

    worker.start()


# Backward-compatible names used by earlier commands.
def queue_video_analysis(upload_id: int) -> None:
    queue_media_analysis(upload_id)


def analyze_uploaded_video(upload_id: int) -> None:
    analyze_uploaded_media(upload_id)
