from django.urls import path

from .views import DashboardView, LogsView, UploadView, detection_api, live_detection_view, video_feed


urlpatterns = [
    path("", DashboardView.as_view(), name="dashboard"),
    path("upload/", UploadView.as_view(), name="upload"),
    path("stream/", live_detection_view, name="stream_page"),
    path("video-feed/", video_feed, name="video_feed"),
    path("logs/", LogsView.as_view(), name="logs"),
    path("api/detect/", detection_api, name="detection_api"),
]
