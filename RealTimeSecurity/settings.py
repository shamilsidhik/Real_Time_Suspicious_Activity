import os
from pathlib import Path

from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Base paths and environment
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")


def env_bool(name: str, default: bool = False) -> bool:
    """Return a boolean value from an environment variable."""

    value = os.getenv(name)

    if value is None:
        return default

    return value.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def env_list(name: str, default: str = "") -> list[str]:
    """Return a comma-separated environment variable as a clean list."""

    return [
        item.strip()
        for item in os.getenv(name, default).split(",")
        if item.strip()
    ]


# ---------------------------------------------------------------------------
# Core Django settings
# ---------------------------------------------------------------------------

SECRET_KEY = os.getenv(
    "DJANGO_SECRET_KEY",
    "django-insecure-change-me-in-production-abc123xyz",
)

DEBUG = env_bool(
    "DJANGO_DEBUG",
    True,
)

ALLOWED_HOSTS = env_list(
    "DJANGO_ALLOWED_HOSTS",
    "*",
)

CSRF_TRUSTED_ORIGINS = env_list(
    "DJANGO_CSRF_TRUSTED_ORIGINS",
)


# ---------------------------------------------------------------------------
# Installed applications
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",

    # Keep WhiteNoise immediately after SecurityMiddleware.
    "whitenoise.middleware.WhiteNoiseMiddleware",

    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]


ROOT_URLCONF = "RealTimeSecurity.urls"

WSGI_APPLICATION = "RealTimeSecurity.wsgi.application"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}


# ---------------------------------------------------------------------------
# Password validation
# ---------------------------------------------------------------------------

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
    {
        "NAME": (
            "django.contrib.auth.password_validation."
            "CommonPasswordValidator"
        ),
    },
    {
        "NAME": (
            "django.contrib.auth.password_validation."
            "NumericPasswordValidator"
        ),
    },
]


# ---------------------------------------------------------------------------
# Language and time
# ---------------------------------------------------------------------------

LANGUAGE_CODE = "en-us"

TIME_ZONE = "Asia/Kolkata"

USE_I18N = True
USE_TZ = True


# ---------------------------------------------------------------------------
# Static files and professional UI assets
# ---------------------------------------------------------------------------

STATIC_URL = "/static/"

STATIC_ROOT = BASE_DIR / "staticfiles"

PROJECT_STATIC_DIR = BASE_DIR / "static"

STATICFILES_DIRS = (
    [PROJECT_STATIC_DIR]
    if PROJECT_STATIC_DIR.exists()
    else []
)

# Modern Django storage configuration.
STORAGES = {
    "default": {
        "BACKEND": (
            "django.core.files.storage.FileSystemStorage"
        ),
    },
    "staticfiles": {
        "BACKEND": (
            "whitenoise.storage."
            "CompressedManifestStaticFilesStorage"
        ),
    },
}

WHITENOISE_MAX_AGE = (
    0
    if DEBUG
    else 31_536_000
)


# ---------------------------------------------------------------------------
# Uploaded media and detection snapshots
# ---------------------------------------------------------------------------

MEDIA_URL = "/media/"

MEDIA_ROOT = BASE_DIR / "media"


# ---------------------------------------------------------------------------
# Authentication redirects
# ---------------------------------------------------------------------------

LOGIN_URL = "/accounts/login/"

# Login opens the enterprise surveillance dashboard.
LOGIN_REDIRECT_URL = "/dashboard/"

# Logout returns to the public SecureVision front page.
LOGOUT_REDIRECT_URL = "/"


# ---------------------------------------------------------------------------
# Camera server
# ---------------------------------------------------------------------------

CAMERA_SERVER_URL = os.getenv(
    "CAMERA_SERVER_URL",
    "http://localhost:8765",
).rstrip("/")


# ---------------------------------------------------------------------------
# Known-person face recognition
# ---------------------------------------------------------------------------

KNOWN_FACES_DIR = BASE_DIR / "known_faces"


# ---------------------------------------------------------------------------
# Activity and weapon model paths
# ---------------------------------------------------------------------------

ACTIVITY_MODEL_PATH = Path(
    os.getenv(
        "ACTIVITY_MODEL_PATH",
        str(
            BASE_DIR
            / "ml"
            / "models"
            / "activity_yolov8"
            / "best.pt"
        ),
    )
)

WEAPON_MODEL_PATH = Path(
    os.getenv(
        "WEAPON_MODEL_PATH",
        str(
            BASE_DIR
            / "ml"
            / "models"
            / "weapons_yolov8"
            / "best.pt"
        ),
    )
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

# TLS and SSL must not both be enabled.
if EMAIL_USE_TLS and EMAIL_USE_SSL:
    raise ValueError(
        "EMAIL_USE_TLS and EMAIL_USE_SSL cannot both be enabled."
    )

EMAIL_HOST_USER = os.getenv(
    "EMAIL_HOST_USER",
    "",
).strip()

EMAIL_HOST_PASSWORD = os.getenv(
    "EMAIL_HOST_PASSWORD",
    "",
).replace(" ", "").strip()

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
).strip()

SERVER_EMAIL = DEFAULT_FROM_EMAIL


# ---------------------------------------------------------------------------
# Suspicious-event email configuration
# ---------------------------------------------------------------------------

SECURITY_EMAIL_ALERTS_ENABLED = env_bool(
    "SECURITY_EMAIL_ALERTS_ENABLED",
    True,
)

SECURITY_ALERT_EMAILS = env_list(
    "SECURITY_ALERT_EMAILS",
)

SECURITY_ALERT_INCLUDE_USER = env_bool(
    "SECURITY_ALERT_INCLUDE_USER",
    True,
)

SECURITY_ALERT_SUBJECT_PREFIX = os.getenv(
    "SECURITY_ALERT_SUBJECT_PREFIX",
    "[RealTimeSecurity ALERT]",
).strip()


# ---------------------------------------------------------------------------
# Security settings
# ---------------------------------------------------------------------------

SESSION_COOKIE_HTTPONLY = True

CSRF_COOKIE_HTTPONLY = False

X_FRAME_OPTIONS = "DENY"

SECURE_CONTENT_TYPE_NOSNIFF = True

# Enable secure cookies only when running behind HTTPS in production.
SESSION_COOKIE_SECURE = env_bool(
    "DJANGO_SESSION_COOKIE_SECURE",
    not DEBUG,
)

CSRF_COOKIE_SECURE = env_bool(
    "DJANGO_CSRF_COOKIE_SECURE",
    not DEBUG,
)

SECURE_SSL_REDIRECT = env_bool(
    "DJANGO_SECURE_SSL_REDIRECT",
    False,
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_DIR = BASE_DIR / "logs"

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "detailed": {
            "format": (
                "{asctime} [{levelname}] "
                "{name}: {message}"
            ),
            "style": "{",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "detailed",
        },
        "django_file": {
            "class": (
                "logging.handlers."
                "RotatingFileHandler"
            ),
            "filename": LOG_DIR / "django.log",
            "maxBytes": 5 * 1024 * 1024,
            "backupCount": 3,
            "formatter": "detailed",
        },
    },
    "root": {
        "handlers": [
            "console",
            "django_file",
        ],
        "level": os.getenv(
            "DJANGO_LOG_LEVEL",
            "INFO",
        ),
    },
}
