# surveillance/forms.py

from pathlib import Path

from django import forms
from django.core.exceptions import ValidationError

from .models import UploadedVideo


class UploadedVideoForm(forms.ModelForm):
    """Accept either one supported video or one supported image."""

    VIDEO_EXTENSIONS = {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".webm",
        ".m4v",
    }

    IMAGE_EXTENSIONS = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".bmp",
    }

    MAX_VIDEO_SIZE = 500 * 1024 * 1024
    MAX_IMAGE_SIZE = 20 * 1024 * 1024

    class Meta:
        model = UploadedVideo
        fields = ["video", "image"]

        widgets = {
            "video": forms.ClearableFileInput(
                attrs={
                    "accept": (
                        ".mp4,.avi,.mov,.mkv,.webm,.m4v,"
                        "video/mp4,video/quicktime,"
                        "video/x-msvideo,video/x-matroska,video/webm"
                    ),
                }
            ),
            "image": forms.ClearableFileInput(
                attrs={
                    "accept": (
                        ".jpg,.jpeg,.png,.webp,.bmp,image/*"
                    ),
                }
            ),
        }

    def clean_video(self):
        uploaded_file = self.cleaned_data.get("video")

        if not uploaded_file:
            return uploaded_file

        extension = Path(uploaded_file.name).suffix.lower()

        if extension not in self.VIDEO_EXTENSIONS:
            raise ValidationError(
                "Select a valid video: MP4, AVI, MOV, MKV, WEBM or M4V."
            )

        if uploaded_file.size <= 0:
            raise ValidationError("The selected video is empty.")

        if uploaded_file.size > self.MAX_VIDEO_SIZE:
            raise ValidationError(
                "The video must be smaller than 500 MB."
            )

        content_type = str(
            getattr(uploaded_file, "content_type", "") or ""
        ).lower()

        if content_type and not (
            content_type.startswith("video/")
            or content_type == "application/octet-stream"
        ):
            raise ValidationError(
                "The selected file is not recognised as a video."
            )

        return uploaded_file

    def clean_image(self):
        uploaded_file = self.cleaned_data.get("image")

        if not uploaded_file:
            return uploaded_file

        extension = Path(uploaded_file.name).suffix.lower()

        if extension not in self.IMAGE_EXTENSIONS:
            raise ValidationError(
                "Select a valid image: JPG, JPEG, PNG, WEBP or BMP."
            )

        if uploaded_file.size <= 0:
            raise ValidationError("The selected image is empty.")

        if uploaded_file.size > self.MAX_IMAGE_SIZE:
            raise ValidationError(
                "The image must be smaller than 20 MB."
            )

        return uploaded_file

    def clean(self):
        cleaned_data = super().clean()

        video = cleaned_data.get("video")
        image = cleaned_data.get("image")

        if not video and not image:
            raise ValidationError(
                "Choose either one video or one image."
            )

        if video and image:
            raise ValidationError(
                "Choose only one file at a time, not both."
            )

        return cleaned_data
