from django.urls import path

from django_app.classifier.views import dashboard, health, predict

urlpatterns = [
    path("", dashboard),
    path("health/", health),
    path("predict/", predict),
]
