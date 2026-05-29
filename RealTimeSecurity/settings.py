import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = 'your-secret-key'  # Replace with a secure random key (keep secret!)【52†L23-L27】
DEBUG = True  # Turn off (False) in production for security
ALLOWED_HOSTS = []  # Add your domain(s) here for production【51†L139-L148】

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',

    'surveillance.apps.SurveillanceConfig',
    'users',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',

    'django.contrib.sessions.middleware.SessionMiddleware',

    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',

    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',

    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'RealTimeSecurity.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',

        'DIRS': [BASE_DIR / 'templates'],

        'APP_DIRS': True,

        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.request',

                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]
WSGI_APPLICATION = 'RealTimeSecurity.wsgi.application'

# Use BigAutoField by default to avoid warnings about auto-created PKs
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# Database: default is SQLite
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# Password validation (default)
AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    # ...
]

# Internationalization
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

# Static files (CSS, JavaScript, images)
STATIC_URL = '/static/'
STATICFILES_DIRS = [BASE_DIR / 'static']     # place your CSS/JS here
STATIC_ROOT = BASE_DIR / 'staticfiles'       # collectstatic target
# For example:
#   STATICFILES_DIRS = [BASE_DIR / "static",]
#   STATIC_ROOT = "/var/www/RealTimeSecurity/static/"

# Media (user-uploaded files)
MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

# Email (SMTP) configuration - placeholders
EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
EMAIL_HOST = 'smtp.example.com'  # e.g., smtp.gmail.com
EMAIL_PORT = 587
EMAIL_HOST_USER = 'user@example.com'
EMAIL_HOST_PASSWORD = 'your-email-password'
EMAIL_USE_TLS = True

# Login redirect (after login)
LOGIN_REDIRECT_URL = 'dashboard'
LOGOUT_REDIRECT_URL = 'login'
