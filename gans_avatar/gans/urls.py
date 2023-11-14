from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('imageavatar/', views.image_avatar, name='imageavatar'),
    path('getimageavatar/', views.get_image_avatar, name='getimageavatar'),
    path('bgimage/', views.bg_image, name='bgimage'),
    path('getbgimage/', views.get_bg_image, name='getbgimage'),
    path('postcombinedimage/', views.combinedimage, name='postcombinedimage'),
    path('getcombinedimage/', views.getcombinedimage, name='getcombinedimage'),
    
    # path('postaudiogen/', views.postaudiogen, name='postaudiogen'),
    path('getaudiogen/', views.getaudiogen, name='getaudiogen'),
    path('postonlyclick/', views.postonlyclick, name='postonlyclick'),
    
    
    path('finalcombinedimage/', views.post_combined_image, name='finalcombinedimage'),
    path('getcombinedimage/', views.get_combined_image, name='getcombinedimage'),
    
    
    
]+ static(settings.STATIC_URL, document_root=settings.STATIC_ROOT) + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)