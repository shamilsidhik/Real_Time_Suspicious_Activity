from django.urls import path

from surveillance import views


urlpatterns = [
    # Public SecureVision front page
    path(
        "",
        views.home,
        name="home",
    ),

    # Authenticated surveillance dashboard
    path(
        "dashboard/",
        views.dashboard,
        name="dashboard",
    ),

    # Camera stream
    path(
        "stream/",
        views.stream_proxy,
        name="stream_proxy",
    ),
    path(
        "live/",
        views.stream_page,
        name="stream_page",
    ),

    # Image and video upload
    path(
        "upload/",
        views.upload,
        name="upload",
    ),

    # Detection logs
    path(
        "logs/",
        views.logs,
        name="logs",
    ),

    # Live detection API
    path(
        "api/status/",
        views.detection_api,
        name="detection_api",
    ),

    # Known-person registration
    path(
        "known-person/add/",
        views.known_person_upload,
        name="known_person_upload",
    ),

    # Start or stop AI detection while keeping the camera active
    path(
        "api/detection/control/",
        views.detection_control,
        name="detection_control",
    ),
]
