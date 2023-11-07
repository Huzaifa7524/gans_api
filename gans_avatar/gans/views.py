# views.py

from rest_framework.response import Response
from rest_framework import generics
from .models import Avatarandbackground, Backgroundandavatarcombine,Audio_inputs
from .serializers import ImageSerializer,CombineSerializer,AudioSerializer
from django.http import FileResponse
from django.contrib.staticfiles import finders
from django.http import HttpResponse, JsonResponse



class ImageViewSet(generics.ListAPIView):
    queryset = Avatarandbackground.objects.all()
    serializer_class = ImageSerializer


class CombineViewSet(generics.ListCreateAPIView):
    queryset = Backgroundandavatarcombine.objects.all()
    serializer_class = CombineSerializer

class AudioViewSet(generics.ListCreateAPIView):
    queryset = Audio_inputs.objects.all()
    serializer_class = AudioSerializer


def serve_static_file(request, path):
    file_path = finders.find(path)
    if file_path:
        file_response = FileResponse(open(file_path, 'rb'), content_type='image/jpeg')  # Change content_type as needed
        return file_response
    return HttpResponse("File not found", status=404)


def audio_generation(request):
    # get data from database and get 'prompt', 'language', 'voice'
    data = Audio_inputs.objects.all()
    prompt = data.values_list('prompt', flat=True)
    language = data.values_list('language', flat=True)
    voice = data.values_list('voice', flat=True)
    # generate audio file if these are not empty
    
    if prompt and language and voice is not None:
        lang = []
        if language == "Engilsh":
            male_voice = "v2/en_speaker_6"
            female = "v2/en_speaker_9"
            if voice == "Male":
                lang = male_voice
                lang.append(male_voice)
            else:
                lang = female
                lang.append(female)
        elif language == "Spanish":
            male_voice = "v2/es_speaker_6"
            female_voice = "v2/es_speaker_9"
            if voice == "Male":
                lang = male_voice
                lang.append(male_voice)
            else:
                lang = female_voice
                lang.append(female_voice)

        
        # generate audio file
        # SPEAKER = lang
        # # Generate audio
        # audio_array = generate_audio(prompt, history_prompt=SPEAKER)

        # # Set the output file name
        # output_filename = "./examples/driven_audio/output_audio.wav"

        # # Save the audio to a WAV file
        # wavfile.write(output_filename, SAMPLE_RATE, audio_array)

        # print(f"Audio saved to {output_filename}")
        return JsonResponse({'message': 'Audio file generated successfully'})
        
    
    



