from rest_framework import serializers
from .models import Avatarandbackground, Backgroundandavatarcombine, Audio_inputs

class ImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = Avatarandbackground
        fields = '__all__'

class CombineSerializer(serializers.ModelSerializer):
    class Meta:
        model = Backgroundandavatarcombine
        fields = '__all__'

class AudioSerializer(serializers.ModelSerializer):
    class Meta:
        model = Audio_inputs
        fields = '__all__'