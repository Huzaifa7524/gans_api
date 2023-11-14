from rest_framework import serializers
from .models import Avatar,onlyClick,Background, Backgroundandavatarcombine, Audio_input,Combine_Image

class AvatarSerializers(serializers.ModelSerializer):
    class Meta:
        model = Avatar
        fields = ['id','avatar_images', 'image_name']

class BackgroundSerializer(serializers.ModelSerializer):
    class Meta:
        model = Background
        fields = ['id','background_image', 'image_name']


class CombineSerializer(serializers.ModelSerializer):
    class Meta:
        model = Backgroundandavatarcombine
        fields = ['id','backgroun_final', 'image_name']

class AudioinputSerializer(serializers.ModelSerializer):
    class Meta:
        model = Audio_input
        # fields = ['id','prompt', 'language','voice', 'gender', 'video_image']
        fields = ['id','prompt', 'language','voice', 'gender', 'video_image']

class onlyClickSerializer(serializers.ModelSerializer):
    class Meta:
        model = onlyClick
        fields = ['id','click']

class Combine_ImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = Combine_Image
        fields = ['id','combined_image','combined_image_name']