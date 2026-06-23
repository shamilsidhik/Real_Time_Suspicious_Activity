"""
run_server.py
Serves Django with whitenoise (handles static files) via waitress.
Install once:  pip install waitress whitenoise
"""
import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "RealTimeSecurity.settings")

import django
django.setup()

from waitress import serve
from django.core.wsgi import get_wsgi_application
from whitenoise import WhiteNoise

app = get_wsgi_application()
app = WhiteNoise(app, root=os.path.join(os.path.dirname(__file__), "staticfiles"),
                 prefix="static")

print("Starting server on http://localhost:8000")
print("Open in browser: http://localhost:8000")
serve(app, host="0.0.0.0", port=8000, threads=8)