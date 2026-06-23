# surveillance/models.py

from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
from django.db import models


class UploadedVideo(models.Model):
    """
    Stores either one uploaded video or one uploaded image.

    The model name is kept as UploadedVideo so existing database records,
    foreign keys, templates, and migrations continue to work.
    """

    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
    )

    video = models.FileField(
        upload_to="uploaded_videos/",
        blank=True,
        null=True,
    )

    image = models.ImageField(
        upload_to="uploaded_images/",
        blank=True,
        null=True,
    )

    uploaded_at = models.DateTimeField(
        auto_now_add=True,
    )

    status = models.CharField(
        max_length=50,
        default="Pending",
    )

    def clean(self):
        super().clean()

        has_video = bool(self.video)
        has_image = bool(self.image)

        if has_video == has_image:
            raise ValidationError(
                "Upload exactly one file: either a video or an image."
            )

    @property
    def media_type(self):
        if self.image:
            return "Image"
        if self.video:
            return "Video"
        return "Unknown"

    @property
    def file_name(self):
        if self.image:
            return self.image.name
        if self.video:
            return self.video.name
        return ""

    @property
    def file_url(self):
        if self.image:
            return self.image.url
        if self.video:
            return self.video.url
        return ""

    def __str__(self):
        return (
            f"{self.user.username} - "
            f"{self.media_type} - "
            f"{self.file_name}"
        )


class DetectionLog(models.Model):
    ACTIVITY_CHOICES = [
        ("normal", "Normal"),
        ("suspicious", "Suspicious"),
        ("weapon_detected", "Weapon Detected"),
        ("fight_detected", "Fight Detected"),
        ("id_detected", "ID Detected"),
        ("spoof_detected", "Spoof Detected"),
    ]

    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
    )

    video = models.ForeignKey(
        UploadedVideo,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
    )

    activity_type = models.CharField(
        max_length=50,
        choices=ACTIVITY_CHOICES,
    )

    confidence_score = models.FloatField(
        default=0.0,
    )

    message = models.TextField(
        blank=True,
        null=True,
    )

    image = models.ImageField(
        upload_to="detection_logs/%Y/%m/%d/",
        blank=True,
        null=True,
    )

    detected_at = models.DateTimeField(
        auto_now_add=True,
    )

    class Meta:
        ordering = ["-detected_at"]

    def __str__(self):
        return (
            f"{self.get_activity_type_display()} - "
            f"{self.confidence_score:.2f}"
        )
