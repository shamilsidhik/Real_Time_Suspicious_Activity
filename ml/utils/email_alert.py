"""
Throttled email alerting.
Usage:
    alerter = AlertManager()
    alerter.send_if_needed("Suspicious activity detected in camera feed", frame_jpg_bytes)
"""
import os, logging, smtplib, threading
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

logger = logging.getLogger(__name__)

# Min time between alerts to prevent spam
MIN_INTERVAL_SECONDS = 60


class AlertManager:
    def __init__(self):
        self._last_sent: datetime | None = None
        self._lock = threading.Lock()

    def _cooldown_ok(self) -> bool:
        if self._last_sent is None:
            return True
        return datetime.now() - self._last_sent > timedelta(seconds=MIN_INTERVAL_SECONDS)

    def send_if_needed(self, message: str, image_bytes: bytes | None = None):
        with self._lock:
            if not self._cooldown_ok():
                logger.debug("Alert suppressed (cooldown active)")
                return False

            success = self._send(message, image_bytes)
            if success:
                self._last_sent = datetime.now()
            return success

    def _send(self, message: str, image_bytes: bytes | None) -> bool:
        host  = os.environ.get("EMAIL_HOST", "smtp.gmail.com")
        port  = int(os.environ.get("EMAIL_PORT", 587))
        user  = os.environ.get("EMAIL_HOST_USER", "")
        pwd   = os.environ.get("EMAIL_HOST_PASSWORD", "")
        dest  = os.environ.get("ALERT_RECEIVER_EMAIL", user)

        if not user or not pwd:
            logger.warning("Email credentials not configured. Skipping alert.")
            return False

        try:
            msg = MIMEMultipart()
            msg["From"] = user
            msg["To"] = dest
            msg["Subject"] = f"[ALERT] Suspicious Activity — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            msg.attach(MIMEText(message, "plain"))

            if image_bytes:
                img = MIMEImage(image_bytes, name="alert_frame.jpg")
                msg.attach(img)

            with smtplib.SMTP(host, port) as server:
                server.starttls()
                server.login(user, pwd)
                server.sendmail(user, dest, msg.as_string())

            logger.info(f"Alert email sent to {dest}")
            return True
        except Exception as e:
            logger.error(f"Failed to send alert email: {e}")
            return False


_alerter = None

def get_alerter() -> AlertManager:
    global _alerter
    if _alerter is None:
        _alerter = AlertManager()
    return _alerter