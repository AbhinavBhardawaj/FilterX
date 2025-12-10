# backend/filterx/urls.py

from django.urls import path
from .views import ClassifyView, HealthCheckView, StatusView, ModelFilesView

urlpatterns = [
    path('classify/', ClassifyView.as_view(), name='classify'),
    path('health/', HealthCheckView.as_view(), name='health'),
    path('status/', StatusView.as_view(), name='status'),
    path('model-files/', ModelFilesView.as_view(), name='model-files'),
]
