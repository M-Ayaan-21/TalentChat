from django.urls import path
from rest_framework.routers import SimpleRouter

from .views import (
    ChatAPIViewSet,
    LanguageViewSet,
    get_candidates_for_jd,
    generate_interview_questions,
    index,
    health,
)

router = SimpleRouter()
router.register(r"chat/service", ChatAPIViewSet, basename="chat-service")
router.register(r"language", LanguageViewSet, basename="language")

urlpatterns = [
    # UI & health
    path("", index, name="index"),
    path("health/", health, name="api_health"),

    # Candidate retrieval & questions
    path(
        "chat/get_candidates_for_jd/",
        get_candidates_for_jd,
        name="get_candidates_for_jd",
    ),
    path(
        "chat/generate_interview_questions/",
        generate_interview_questions,
        name="generate_interview_questions",
    ),
]

urlpatterns += router.urls
