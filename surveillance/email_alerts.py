# surveillance/email_alerts.py

from __future__ import annotations

import logging
import mimetypes
import threading
from typing import Iterable

from django.conf import settings
from django.core.mail import EmailMessage
from django.db import close_old_connections
from django.utils import timezone

from surveillance.models import DetectionLog


logger = logging.getLogger(__name__)


ALERT_ACTIVITY_TYPES = {
    "suspicious",
    "weapon_detected",
    "fight_detected",
    "spoof_detected",
}


def _normalise_email_list(value) -> list[str]:
    """Convert a string/list setting into a clean recipient list."""

    if not value:
        return []

    if isinstance(value, str):
        values: Iterable[str] = value.split(",")
    else:
        values = value

    recipients = []

    for item in values:
        email = str(item or "").strip()

        if email and "@" in email:
            recipients.append(email)

    # Preserve order while removing duplicates.
    return list(dict.fromkeys(recipients))


def _alert_recipients(log_entry: DetectionLog) -> list[str]:
    """
    Get configured recipients and optionally include the event owner's email.
    """

    recipients = _normalise_email_list(
        getattr(settings, "SECURITY_ALERT_EMAILS", [])
    )

    include_user = bool(
        getattr(
            settings,
            "SECURITY_ALERT_INCLUDE_USER",
            True,
        )
    )

    if (
        include_user
        and log_entry.user_id
        and getattr(log_entry.user, "email", "")
    ):
        recipients.append(log_entry.user.email.strip())

    return list(
        dict.fromkeys(
            email
            for email in recipients
            if email
        )
    )


def _attach_detection_image(
    message: EmailMessage,
    log_entry: DetectionLog,
) -> bool:
    """Attach the saved DetectionLog image using its Django storage backend."""

    image_field = getattr(log_entry, "image", None)

    if not image_field or not image_field.name:
        return False

    try:
        image_field.open("rb")
        image_bytes = image_field.read()
        image_field.close()

        filename = image_field.name.rsplit("/", 1)[-1]
        content_type = (
            mimetypes.guess_type(filename)[0]
            or "image/jpeg"
        )

        message.attach(
            filename,
            image_bytes,
            content_type,
        )
        return True

    except Exception as error:
        logger.exception(
            "Could not attach detection image for log %s: %s",
            log_entry.pk,
            error,
        )
        return False


def send_detection_alert(log_id: int) -> bool:
    """Send one suspicious-event email with the detected image attached."""

    close_old_connections()

    try:
        log_entry = (
            DetectionLog.objects
            .select_related("user", "video")
            .get(pk=log_id)
        )

        if log_entry.activity_type not in ALERT_ACTIVITY_TYPES:
            return False

        recipients = _alert_recipients(log_entry)

        if not recipients:
            logger.warning(
                "Email alert skipped for log %s: no recipients configured.",
                log_entry.pk,
            )
            return False

        detected_time = timezone.localtime(
            log_entry.detected_at
        )

        event_name = log_entry.get_activity_type_display()
        confidence = max(
            0.0,
            min(float(log_entry.confidence_score or 0.0), 1.0),
        )

        source_name = "Live camera"

        if log_entry.video_id:
            uploaded_item = log_entry.video
            source_name = getattr(
                uploaded_item,
                "file_name",
                "",
            ) or str(uploaded_item)

        subject_prefix = str(
            getattr(
                settings,
                "SECURITY_ALERT_SUBJECT_PREFIX",
                "[RealTimeSecurity ALERT]",
            )
        ).strip()

        subject = f"{subject_prefix} {event_name}"

        body = (
            "Suspicious activity has been detected.\n\n"
            f"Event: {event_name}\n"
            f"Confidence: {confidence:.0%}\n"
            f"Detected at: {detected_time:%Y-%m-%d %H:%M:%S %Z}\n"
            f"User: {log_entry.user.username}\n"
            f"Source: {source_name}\n"
            f"Message: {log_entry.message or '-'}\n\n"
            "The detected image is attached when available.\n"
            "Please review the surveillance dashboard and detection logs."
        )

        from_email = getattr(
            settings,
            "DEFAULT_FROM_EMAIL",
            None,
        )

        email_message = EmailMessage(
            subject=subject,
            body=body,
            from_email=from_email,
            to=recipients,
        )

        attached = _attach_detection_image(
            email_message,
            log_entry,
        )

        sent_count = email_message.send(
            fail_silently=False
        )

        logger.info(
            "Security alert email sent for log %s to %s "
            "(attached_image=%s, sent_count=%s)",
            log_entry.pk,
            recipients,
            attached,
            sent_count,
        )

        return sent_count > 0

    except DetectionLog.DoesNotExist:
        logger.warning(
            "Email alert skipped: DetectionLog %s no longer exists.",
            log_id,
        )
        return False

    except Exception as error:
        logger.exception(
            "Security alert email failed for log %s: %s",
            log_id,
            error,
        )
        return False

    finally:
        close_old_connections()


def queue_detection_alert(log_id: int) -> None:
    """
    Send the email in a background thread so camera polling stays responsive.
    """

    if not bool(
        getattr(settings, "SECURITY_EMAIL_ALERTS_ENABLED", True)
    ):
        return

    worker = threading.Thread(
        target=send_detection_alert,
        args=(log_id,),
        daemon=True,
        name=f"security-email-alert-{log_id}",
    )

    worker.start()
