from django.shortcuts import render
from django.urls import reverse_lazy
from django.views import View
from django.views.generic import TemplateView, FormView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import JsonResponse, StreamingHttpResponse
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.contrib import messages

import os
import time
import cv2
import numpy as np

from .models import DetectionLog, UploadedVideo
from .forms import UploadForm


class DashboardView(LoginRequiredMixin, TemplateView):
    template_name = 'dashboard.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        user = self.request.user

        try:
            logs_qs = DetectionLog.objects.filter(user=user).order_by('-detected_at')

            context['logs'] = logs_qs[:10]
            context['recent_logs'] = logs_qs[:10]
            context['total_logs'] = logs_qs.count()
            context['suspicious_count'] = logs_qs.filter(activity_type='suspicious').count()
            context['id_card_count'] = logs_qs.filter(activity_type='id_detected').count()

        except Exception:
            context['logs'] = []
            context['recent_logs'] = []
            context['total_logs'] = 0
            context['suspicious_count'] = 0
            context['id_card_count'] = 0

        return context


class UploadView(LoginRequiredMixin, FormView):
    form_class = UploadForm
    template_name = 'upload.html'
    success_url = reverse_lazy('upload')

    def form_valid(self, form):
        upload = form.save(commit=False)
        upload.user = self.request.user
        upload.status = "Processing"
        upload.save()

        messages.success(self.request, "Video uploaded successfully.")
        return super().form_valid(form)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        context['uploads'] = UploadedVideo.objects.filter(
            user=self.request.user
        ).order_by('-uploaded_at')

        return context


def generate_unavailable_frame(message="Camera unavailable"):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    cv2.putText(
        frame,
        message,
        (50, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    ret, buffer = cv2.imencode(".jpg", frame)

    if not ret:
        return None

    return buffer.tobytes()


def generate_camera_frames(request_user):

    from ml.inference.id_detector import IDDetector
    from ml.inference.activity_predictor import ActivityPredictor
    from ml.inference.anti_spoof import (
        frame_difference_score,
        repeated_frame_ratio
    )

    id_det = IDDetector()
    act = ActivityPredictor()

    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not camera.isOpened():
        camera = cv2.VideoCapture(0)

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    time.sleep(1)

    frame_gray_buffer = []

    SEQ_LEN = 30

    try:
        while True:

            success, frame = camera.read()

            if not success or frame is None:

                frame_bytes = generate_unavailable_frame(
                    "Failed to read camera"
                )

                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' +
                    frame_bytes +
                    b'\r\n'
                )

                time.sleep(1)
                continue

            frame = cv2.resize(frame, (640, 480))

            display = frame.copy()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            frame_gray_buffer.append(gray)

            if len(frame_gray_buffer) > SEQ_LEN:
                frame_gray_buffer.pop(0)

            text_lines = ["Live Detection Running"]

            spoof_flag = False

            if len(frame_gray_buffer) == SEQ_LEN:

                diff_score = frame_difference_score(frame_gray_buffer)

                repeat_ratio = repeated_frame_ratio(
                    frame_gray_buffer
                )

                spoof_flag = (
                    diff_score < 2.0 or repeat_ratio > 0.6
                )

                if spoof_flag:
                    text_lines.append("Anti-spoof: SPOOF")
                else:
                    text_lines.append("Anti-spoof: OK")

            bbox_dets = []

            if id_det.is_model_available():

                id_out = id_det.detect_image(frame)

                if (
                    isinstance(id_out, dict)
                    and id_out.get('status') == 'ok'
                ):

                    for d in id_out.get('detections', []):

                        x1, y1, x2, y2 = d['bbox']

                        conf = d.get('conf', 0.0)

                        bbox_dets.append(
                            (
                                int(x1),
                                int(y1),
                                int(x2),
                                int(y2),
                                float(conf)
                            )
                        )

                    for (
                        x1,
                        y1,
                        x2,
                        y2,
                        conf
                    ) in bbox_dets:

                        cv2.rectangle(
                            display,
                            (x1, y1),
                            (x2, y2),
                            (0, 255, 255),
                            2
                        )

                        cv2.putText(
                            display,
                            f"ID {conf:.2f}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 255),
                            2
                        )

            if (
                len(frame_gray_buffer) == SEQ_LEN
                and act.is_model_available()
                and not spoof_flag
            ):

                seq_rgb = [
                    cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
                    for f in frame_gray_buffer
                ]

                seq_rgb = [
                    cv2.resize(f, (224, 224))
                    for f in seq_rgb
                ]

                activity_result = act.predict_sequence(
                    np.array(seq_rgb)
                )

                if (
                    isinstance(activity_result, dict)
                    and activity_result.get('status') == 'ok'
                ):

                    lbl = activity_result.get(
                        'label',
                        'unknown'
                    )

                    conf = float(
                        activity_result.get(
                            'confidence',
                            0.0
                        )
                    )

                    text_lines.append(
                        f"Activity: {lbl} ({conf:.2f})"
                    )

            y = 35

            for line in text_lines:

                cv2.putText(
                    display,
                    line,
                    (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                y += 30

            ret, buffer = cv2.imencode('.jpg', display)

            if not ret:
                continue

            frame_bytes = buffer.tobytes()

            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' +
                frame_bytes +
                b'\r\n'
            )

            time.sleep(0.03)

    finally:
        camera.release()


class VideoStreamView(LoginRequiredMixin, View):

    def get(self, request, *args, **kwargs):

        return StreamingHttpResponse(
            generate_camera_frames(request.user),
            content_type='multipart/x-mixed-replace; boundary=frame'
        )


@login_required
def video_feed(request):

    return StreamingHttpResponse(
        generate_camera_frames(request.user),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )


def detection_api(request):

    return JsonResponse({
        'status': 'success',
        'message': 'Detection API working'
    })


class LogsView(LoginRequiredMixin, TemplateView):

    template_name = 'logs.html'

    def get_context_data(self, **kwargs):

        context = super().get_context_data(**kwargs)

        try:

            qs = DetectionLog.objects.filter(
                user=self.request.user
            ).order_by('-detected_at')

            paginator = Paginator(qs, 25)

            page = self.request.GET.get('page', 1)

            try:
                logs_page = paginator.page(page)

            except PageNotAnInteger:
                logs_page = paginator.page(1)

            except EmptyPage:
                logs_page = paginator.page(
                    paginator.num_pages
                )

            context['logs_page'] = logs_page

        except Exception:
            context['logs_page'] = []

        return context


@login_required
def live_detection_view(request):

    context = {
        'activity_model_available': True,
        'id_card_model_available': True,
    }

    return render(
        request,
        'live_detection.html',
        context
    )