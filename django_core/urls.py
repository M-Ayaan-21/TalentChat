from django.contrib import admin
from django.urls import path, include, re_path

from api.views import index, health
from rest_framework import permissions

# Swagger / Redoc (drf_yasg)
try:
    from drf_yasg.views import get_schema_view
    from drf_yasg import openapi

    schema_view = get_schema_view(
        openapi.Info(
            title="TalentChat API",
            default_version="v1",
            description="Endpoints for Talent Acquisition, Chat, and Language services",
        ),
        public=True,
        permission_classes=[permissions.AllowAny],
    )
except Exception:
    schema_view = None


urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/", include("api.urls")),
    path("health/", health, name="health"),
    path("", index, name="root_index"),
]

# Optional Swagger endpoints (skip if drf_yasg not installed)
if schema_view:
    urlpatterns += [
        re_path(r"^swagger(?P<format>\.json|\.yaml)$", schema_view.without_ui(cache_timeout=0),
                name="schema-json"),
        path("swagger/", schema_view.with_ui("swagger", cache_timeout=0),
             name="schema-swagger-ui"),
        path("redoc/", schema_view.with_ui("redoc", cache_timeout=0),
             name="schema-redoc"),
    ]
