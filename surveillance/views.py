from __future__ import annotations

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from django.core.paginator import EmptyPage, PageNotAnInteger, Paginator
from django.http import JsonResponse, StreamingHttpResponse
from django.shortcuts import render
from django.urls import reverse_lazy
from django.views import View
from django.views.generic import FormView, TemplateView

from .camera import camera_service
from .forms import UploadForm
from .models import DetectionLog, UploadedVideo


class DashboardView(LoginRequiredMixin, TemplateView):
    template_name = "dashboard.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        user = self.request.user

        try:
            logs_qs = DetectionLog.objects.filter(user=user).order_by("-detected_at")
            context["logs"] = logs_qs[:10]
            context["recent_logs"] = logs_qs[:10]
            context["total_logs"] = logs_qs.count()
            context["suspicious_count"] = logs_qs.filter(activity_type="suspicious").count()
            context["id_card_count"] = logs_qs.filter(activity_type="id_detected").count()
        except Exception:
            context["logs"] = []
            context["recent_logs"] = []
            context["total_logs"] = 0
            context["suspicious_count"] = 0
            context["id_card_count"] = 0

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


def _mjpeg_stream():
    last_frame_id = -1

    try:
        while True:
            frame_id, frame_bytes = camera_service.get_jpeg(last_frame_id=last_frame_id, timeout=2.0)
            last_frame_id = frame_id

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Cache-Control: no-cache, no-store, must-revalidate\r\n\r\n"
                + frame_bytes
                + b"\r\n"
            )
    except GeneratorExit:
        # Client disconnected.
        return


class VideoStreamView(LoginRequiredMixin, View):
    def get(self, request, *args, **kwargs):
        response = StreamingHttpResponse(
            _mjpeg_stream(),
            content_type="multipart/x-mixed-replace; boundary=frame",
        )
        response["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response["Pragma"] = "no-cache"
        response["Expires"] = "0"
        response["X-Accel-Buffering"] = "no"
        return response


@login_required
def video_feed(request):
    response = StreamingHttpResponse(
        _mjpeg_stream(),
        content_type="multipart/x-mixed-replace; boundary=frame",
    )
    response["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response["Pragma"] = "no-cache"
    response["Expires"] = "0"
    response["X-Accel-Buffering"] = "no"
    return response


@login_required
def detection_api(request):
    return JsonResponse(camera_service.get_status())


class LogsView(LoginRequiredMixin, TemplateView):
    template_name = "logs.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        try:
            qs = DetectionLog.objects.filter(user=self.request.user).order_by("-detected_at")
            paginator = Paginator(qs, 25)
            page = self.request.GET.get("page", 1)

            try:
                logs_page = paginator.page(page)
            except PageNotAnInteger:
                logs_page = paginator.page(1)
            except EmptyPage:
                logs_page = paginator.page(paginator.num_pages)

            context["logs_page"] = logs_page
        except Exception:
            context["logs_page"] = []

        return context


@login_required
def live_detection_view(request):
    status = camera_service.get_status()
    context = {
        "activity_model_available": status["models"]["activity_available"],
        "id_card_model_available": status["models"]["id_available"],
        "weapon_model_available": status["models"]["weapon_available"],
        "anti_spoof_mode": status["models"]["anti_spoof_mode"],
        "camera_status": status["status"],
        "camera_fps": status["capture_fps"],
    }
    return render(request, "live_detection.html", context)
