# run_server.py - replaces manage.py runserver
import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "RealTimeSecurity.settings")

import django
django.setup()

from waitress import serve
from django.core.wsgi import get_wsgi_application

print("Starting production server on http://0.0.0.0:8000")
serve(get_wsgi_application(), host="0.0.0.0", port=8000, threads=8)
