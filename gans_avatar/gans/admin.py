from django.contrib import admin
from . models import Avatar,Backgroundandavatarcombine,Audio_input,onlyClick,Background,Combine_Image

# Register your models here.
# class AvatarAdmin(admin.ModelAdmin):
#     list_display = ('avatar_images', 'background_images')

# class BackgroundandavatarcombineAdmin(admin.ModelAdmin):
#     list_display = ('avatar')

# class Audio_inputsAdmin(admin.ModelAdmin):
#     list_display = ('prompt', 'language', 'voice')

admin.site.register(Avatar)
admin.site.register(Backgroundandavatarcombine)
admin.site.register(Audio_input)
admin.site.register(onlyClick)
admin.site.register(Background)
admin.site.register(Combine_Image)