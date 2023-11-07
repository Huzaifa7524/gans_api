from django.contrib import admin
from . models import Avatarandbackground,Backgroundandavatarcombine,Audio_inputs

# Register your models here.
class AvatarandbackgroundAdmin(admin.ModelAdmin):
    list_display = ('avatar_images', 'background_images')

class BackgroundandavatarcombineAdmin(admin.ModelAdmin):
    list_display = ('avatar')

class Audio_inputsAdmin(admin.ModelAdmin):
    list_display = ('prompt', 'language', 'voice')

admin.site.register(Avatarandbackground)
admin.site.register(Backgroundandavatarcombine)
admin.site.register(Audio_inputs)

