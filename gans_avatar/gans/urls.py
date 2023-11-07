from django.urls import path
from .views import ImageViewSet, serve_static_file, CombineViewSet, AudioViewSet
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('Avatarandimage/', ImageViewSet.as_view(), name='image-list'),
    path('static/<path:path>', serve_static_file),
    path('combine/', CombineViewSet.as_view(), name='combine-list'),
    path('audioinput/', AudioViewSet.as_view(), name='audio-list'),
]+ static(settings.STATIC_URL, document_root=settings.STATIC_ROOT) + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)