from __future__ import annotations

import os

import requests
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from django.core.paginator import EmptyPage, PageNotAnInteger, Paginator
from django.http import HttpResponseRedirect, JsonResponse
from django.shortcuts import render
from django.urls import reverse, reverse_lazy
from django.views.generic import FormView, TemplateView

from .forms import UploadForm
from .models import DetectionLog, UploadedVideo


CAMERA_SERVER_URL = os.environ.get("CAMERA_SERVER_URL", "http://127.0.0.1:8765").rstrip("/")


def _default_camera_status() -> dict:
    return {
        "ok": False,
        "status": "camera_server_unavailable",
        "camera_open": False,
        "capture_fps": 0.0,
        "last_error": "Camera server is not reachable",
        "stream_url": f"{CAMERA_SERVER_URL}/stream.mjpg",
        "snapshot_url": f"{CAMERA_SERVER_URL}/snapshot.jpg",
        "models": {
            "activity": {"available": False, "backend": "unknown", "path": "", "last_error": ""},
            "id_card": {"available": False, "backend": "unknown", "path": "", "last_error": ""},
            "weapon": {"available": False, "backend": "unknown", "path": "", "last_error": ""},
            "anti_spoof": {"available": False, "backend": "disabled_live_mode", "path": "", "last_error": ""},
        },
        "overlay": {"activity_label": "unknown", "activity_confidence": 0.0, "id_count": 0, "weapon_count": 0},
    }


def get_camera_status() -> dict:
    try:
        response = requests.get(f"{CAMERA_SERVER_URL}/status", timeout=0.7)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        data = _default_camera_status()
        data["last_error"] = str(exc)
        return data


class DashboardView(LoginRequiredMixin, TemplateView):
    template_name = "dashboard.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        user = self.request.user
        logs_qs = DetectionLog.objects.filter(user=user).order_by("-detected_at")
        uploads_qs = UploadedVideo.objects.filter(user=user).order_by("-uploaded_at")
        camera = get_camera_status()

        context.update(
            {
                "recent_logs": logs_qs[:8],
                "logs": logs_qs[:8],
                "total_logs": logs_qs.count(),
                "suspicious_count": logs_qs.filter(activity_type="suspicious").count(),
                "id_card_count": logs_qs.filter(activity_type="id_detected").count(),
                "total_uploaded_videos": uploads_qs.count(),
                "upload_form": UploadForm(),
                "camera": camera,
                "stream_url": camera.get("stream_url", f"{CAMERA_SERVER_URL}/stream.mjpg"),
            }
        )
        return context


class UploadView(LoginRequiredMixin, FormView):
    form_class = UploadForm
    template_name = "upload.html"
    success_url = reverse_lazy("upload")

    def form_valid(self, form):
        upload = form.save(commit=False)
        upload.user = self.request.user
        upload.status = "Processing"
        upload.save()
        messages.success(self.request, "Video uploaded successfully.")
        return super().form_valid(form)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["uploads"] = UploadedVideo.objects.filter(user=self.request.user).order_by("-uploaded_at")
        return context


@login_required
def video_feed(request):
    return HttpResponseRedirect(f"{CAMERA_SERVER_URL}/stream.mjpg")


@login_required
def detection_api(request):
    return JsonResponse(get_camera_status())


class LogsView(LoginRequiredMixin, TemplateView):
    template_name = "logs.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        qs = DetectionLog.objects.filter(user=self.request.user).order_by("-detected_at")
        paginator = Paginator(qs, 25)
        page = self.request.GET.get("page", 1)
        try:
            context["logs_page"] = paginator.page(page)
        except PageNotAnInteger:
            context["logs_page"] = paginator.page(1)
        except EmptyPage:
            context["logs_page"] = paginator.page(paginator.num_pages)
        return context


@login_required
def live_detection_view(request):
    camera = get_camera_status()
    return render(
        request,
        "live_detection.html",
        {
            "camera": camera,
            "stream_url": camera.get("stream_url", f"{CAMERA_SERVER_URL}/stream.mjpg"),
            "status_api_url": reverse("detection_api"),
        },
    )
