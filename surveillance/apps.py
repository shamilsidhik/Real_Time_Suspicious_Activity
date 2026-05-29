from __future__ import annotations

import logging
import os
import sys

from django.apps import AppConfig

logger = logging.getLogger(__name__)


class SurveillanceConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "surveillance"
    verbose_name = "Surveillance"

    _bootstrapped = False

    def ready(self) -> None:
        """
        Start the embedded camera service exactly once.

        Django documents that ready() runs when the app registry is fully
        populated, but it may also run during management commands and may run
        more than once in corner cases, so this method is intentionally
        idempotent.
        """
        if SurveillanceConfig._bootstrapped:
            return

        if os.environ.get("SURVEILLANCE_DISABLE_CAMERA", "").strip() == "1":
            logger.info("Embedded camera service disabled by SURVEILLANCE_DISABLE_CAMERA=1")
            return

        # Avoid starting the camera for commands that do not need it.
        skip_commands = {
            "collectstatic",
            "createsuperuser",
            "dbshell",
            "makemigrations",
            "migrate",
            "shell",
            "showmigrations",
            "test",
        }
        argv = set(sys.argv[1:])

        if argv & skip_commands:
            logger.debug("Skipping camera bootstrap for management command: %s", " ".join(sys.argv))
            return

        # Avoid duplicate startup in Django's autoreloader parent process.
        if "runserver" in argv and os.environ.get("RUN_MAIN") != "true":
            logger.debug("Skipping camera bootstrap in Django autoreload parent process")
            return

        SurveillanceConfig._bootstrapped = True

        try:
            from .camera import camera_service

            camera_service.start()
            logger.info("Embedded surveillance camera service started")
        except Exception:
            logger.exception("Failed to start embedded surveillance camera service")
