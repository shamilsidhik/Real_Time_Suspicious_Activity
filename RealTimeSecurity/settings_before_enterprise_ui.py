import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from:
# Real_Time_Suspicious_Activity/.env
load_dotenv(BASE_DIR / ".env")


def env_bool(name: str, default: bool = False) -> bool:
    """Read a boolean environment variable safely."""

    value = os.getenv(name)

    if value is None:
        return default

    return value.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


SECRET_KEY = os.getenv(
    "DJANGO_SECRET_KEY",
    "django-insecure-change-me-in-production-abc123xyz",
)

DEBUG = env_bool(
    "DJANGO_DEBUG",
    True,
)

ALLOWED_HOSTS = [
    host.strip()
    for host in os.getenv(
        "DJANGO_ALLOWED_HOSTS",
        "*",
    ).split(",")
    if host.strip()
]


INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "surveillance.apps.SurveillanceConfig",
    "users",
]


MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]


ROOT_URLCONF = "RealTimeSecurity.urls"


TEMPLATES = [
    {
        "BACKEND": (
            "django.template.backends.django.DjangoTemplates"
        ),
        "DIRS": [
            BASE_DIR / "templates",
        ],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                (
                    "django.template.context_processors."
                    "request"
                ),
                (
                    "django.contrib.auth.context_processors."
                    "auth"
                ),
                (
                    "django.contrib.messages.context_processors."
                    "messages"
                ),
            ],
        },
    },
]


WSGI_APPLICATION = "RealTimeSecurity.wsgi.application"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"


DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}


AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": (
            "django.contrib.auth.password_validation."
            "UserAttributeSimilarityValidator"
        ),
    },
    {
        "NAME": (
            "django.contrib.auth.password_validation."
            "MinimumLengthValidator"
        ),
    },
]


LANGUAGE_CODE = "en-us"

TIME_ZONE = "Asia/Kolkata"

USE_I18N = True
USE_TZ = True


# ---------------------------------------------------------------------------
# Static files
# ---------------------------------------------------------------------------

STATIC_URL = "/static/"

STATICFILES_DIRS = [
    BASE_DIR / "static",
]

STATIC_ROOT = BASE_DIR / "staticfiles"

STATICFILES_STORAGE = (
    "whitenoise.storage."
    "CompressedManifestStaticFilesStorage"
)


# ---------------------------------------------------------------------------
# Uploaded media and detection snapshots
# ---------------------------------------------------------------------------

MEDIA_URL = "/media/"

MEDIA_ROOT = BASE_DIR / "media"


# ---------------------------------------------------------------------------
# Camera server
# ---------------------------------------------------------------------------

CAMERA_SERVER_URL = os.getenv(
    "CAMERA_SERVER_URL",
    "http://localhost:8765",
)


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

LOGIN_URL = "/accounts/login/"

# After login, open the surveillance dashboard.
LOGIN_REDIRECT_URL = "/dashboard/"

# After logout, return to the public SecureVision home page.
LOGOUT_REDIRECT_URL = "/"


# ---------------------------------------------------------------------------
# Known-person face recognition
# ---------------------------------------------------------------------------

KNOWN_FACES_DIR = BASE_DIR / "known_faces"


# ---------------------------------------------------------------------------
# Activity and weapon model paths
# ---------------------------------------------------------------------------

ACTIVITY_MODEL_PATH = (
    BASE_DIR
    / "ml"
    / "models"
    / "activity_yolov8"
    / "best.pt"
)

WEAPON_MODEL_PATH = (
    BASE_DIR
    / "ml"
    / "models"
    / "weapons_yolov8"
    / "best.pt"
)


# ---------------------------------------------------------------------------
# Uploaded-video analysis settings
# ---------------------------------------------------------------------------

VIDEO_ACTIVITY_CONF = float(
    os.getenv(
        "VIDEO_ACTIVITY_CONF",
        "0.70",
    )
)

VIDEO_ACTIVITY_IMGSZ = int(
    os.getenv(
        "VIDEO_ACTIVITY_IMGSZ",
        "416",
    )
)

VIDEO_WEAPON_CONF = float(
    os.getenv(
        "VIDEO_WEAPON_CONF",
        "0.55",
    )
)

VIDEO_WEAPON_IMGSZ = int(
    os.getenv(
        "VIDEO_WEAPON_IMGSZ",
        "512",
    )
)

VIDEO_ANALYSIS_FPS = float(
    os.getenv(
        "VIDEO_ANALYSIS_FPS",
        "2.0",
    )
)

VIDEO_CONFIRM_HITS = int(
    os.getenv(
        "VIDEO_CONFIRM_HITS",
        "3",
    )
)

VIDEO_SUSPICIOUS_RATIO = float(
    os.getenv(
        "VIDEO_SUSPICIOUS_RATIO",
        "0.10",
    )
)


# ---------------------------------------------------------------------------
# Uploaded-image analysis settings
# ---------------------------------------------------------------------------

IMAGE_ACTIVITY_CONF = float(
    os.getenv(
        "IMAGE_ACTIVITY_CONF",
        "0.10",
    )
)

IMAGE_ACTIVITY_IMGSZ = int(
    os.getenv(
        "IMAGE_ACTIVITY_IMGSZ",
        "640",
    )
)

IMAGE_SUSPICIOUS_CLASS_CONF = float(
    os.getenv(
        "IMAGE_SUSPICIOUS_CLASS_CONF",
        "0.20",
    )
)

IMAGE_SUSPICIOUS_COMBINED_CONF = float(
    os.getenv(
        "IMAGE_SUSPICIOUS_COMBINED_CONF",
        "0.38",
    )
)


# ---------------------------------------------------------------------------
# Email alerts
# ---------------------------------------------------------------------------

EMAIL_BACKEND = (
    "django.core.mail.backends.smtp.EmailBackend"
)

EMAIL_HOST = os.getenv(
    "EMAIL_HOST",
    "smtp.gmail.com",
)

EMAIL_PORT = int(
    os.getenv(
        "EMAIL_PORT",
        "587",
    )
)

EMAIL_USE_TLS = env_bool(
    "EMAIL_USE_TLS",
    True,
)

EMAIL_USE_SSL = env_bool(
    "EMAIL_USE_SSL",
    False,
)

EMAIL_HOST_USER = os.getenv(
    "EMAIL_HOST_USER",
    "",
).strip()

EMAIL_HOST_PASSWORD = os.getenv(
    "EMAIL_HOST_PASSWORD",
    "",
).strip()

EMAIL_TIMEOUT = int(
    os.getenv(
        "EMAIL_TIMEOUT",
        "20",
    )
)

DEFAULT_FROM_EMAIL = os.getenv(
    "DEFAULT_FROM_EMAIL",
    EMAIL_HOST_USER
    or "RealTimeSecurity <no-reply@localhost>",
)

SERVER_EMAIL = DEFAULT_FROM_EMAIL


SECURITY_EMAIL_ALERTS_ENABLED = env_bool(
    "SECURITY_EMAIL_ALERTS_ENABLED",
    True,
)

SECURITY_ALERT_EMAILS = [
    email.strip()
    for email in os.getenv(
        "SECURITY_ALERT_EMAILS",
        "",
    ).split(",")
    if email.strip()
]

SECURITY_ALERT_INCLUDE_USER = env_bool(
    "SECURITY_ALERT_INCLUDE_USER",
    True,
)

SECURITY_ALERT_SUBJECT_PREFIX = os.getenv(
    "SECURITY_ALERT_SUBJECT_PREFIX",
    "[RealTimeSecurity ALERT]",
).strip()
