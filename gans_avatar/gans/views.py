import os
from rest_framework.response import Response
from rest_framework.decorators import api_view
from .models import Avatar,Background,onlyClick, Backgroundandavatarcombine, Audio_input,Combine_Image
from .serializers import AvatarSerializers,onlyClickSerializer,BackgroundSerializer, CombineSerializer, AudioinputSerializer,Combine_ImageSerializer
from django.http import FileResponse
from django.contrib.staticfiles import finders
from django.http import HttpResponse, JsonResponse
from rest_framework import status
from datetime import datetime
from PIL import Image
import cv2
import numpy as np
from django.conf import settings
from io import BytesIO
from django.core.files.base import ContentFile
import shutil
from urllib.parse import urljoin
import torch
from transformers import BarkModel
from transformers import AutoProcessor
from transformers import set_seed
from optimum.bettertransformer import BetterTransformer
import scipy
import scipy.io.wavfile as wavfile
from scipy.io.wavfile import write


model = BarkModel.from_pretrained("suno/bark-small")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = model.to(device)

processor = AutoProcessor.from_pretrained("suno/bark-small")

# --------------------avatar image API-------------------------
@api_view(['POST'])
def image_avatar(request):
    avatar_data = request.data.get('avatar_images')
    current_time = datetime.now()
    seconds = current_time.second

    # Create a new file name by appending the seconds to the original file name
    file_name, file_extension = avatar_data.name.split('.')
    new_file_name = f'{file_name}{seconds}.{file_extension}'

    # Create a dictionary to hold the data, including the file with the new name
    data = {'avatar_images': avatar_data}
    serializer = AvatarSerializers(data=data)
    if serializer.is_valid():
        # Save the file with the new name
        serializer.validated_data['avatar_images'].name = new_file_name
        serializer.save()
        return Response(serializer.data, status=status.HTTP_201_CREATED)
    else:
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(['GET'])
def get_image_avatar(request):
    try:
        all_data = Avatar.objects.all()
        serializer = AvatarSerializers(all_data, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)
    except Exception as e:
        return Response({"response": f"Something went wrong: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# --------------------background image API-------------------------

@api_view(['POST'])
def bg_image(request):
    background_image = request.data.get('background_image')
    current_time = datetime.now()
    seconds = current_time.second

    # Create a new file name by appending the seconds to the original file name
    file_name, file_extension = background_image.name.split('.')
    new_file_name = f'{file_name}{seconds}.{file_extension}'

    # Create a dictionary to hold the data, including the file with the new name
    data = {'background_image': background_image}
    serializer = BackgroundSerializer(data=data)
    if serializer.is_valid():
        # Save the file with the new name
        serializer.validated_data['background_image'].name = new_file_name
        serializer.save()
        return Response(serializer.data, status=status.HTTP_201_CREATED)
    else:
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    

@api_view(['GET'])
def get_bg_image(request):
    try:
        all_data = Background.objects.all()
        serializer = BackgroundSerializer(all_data, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)
    except Exception as e:
        return Response({"response": f"Something went wrong: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# --------------------combined image API-------------------------

@api_view(['POST'])
def combinedimage(request):
    backgroun_final = request.data.get('backgroun_final')
    current_time = datetime.now()
    seconds = current_time.second

    # Create a new file name by appending the seconds to the original file name
    file_name, file_extension = backgroun_final.name.split('.')
    new_file_name = f'{file_name}{seconds}.{file_extension}'

    # Create a dictionary to hold the data, including the file with the new name
    data = {'backgroun_final': backgroun_final}
    serializer = CombineSerializer(data=data)
    if serializer.is_valid():
        # Save the file with the new name
        serializer.validated_data['backgroun_final'].name = new_file_name
        serializer.save()
        return Response(serializer.data, status=status.HTTP_201_CREATED)
    else:
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    

@api_view(['GET'])
def getcombinedimage(request):
    try:
        all_data = Backgroundandavatarcombine.objects.all()
        serializer = CombineSerializer(all_data, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)
    except Exception as e:
        return Response({"response": f"Something went wrong: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# --------------------audio generate API-------------------------
# @api_view(['POST'])
# def postaudiogen(request):
#     all_data = request.data
#     serializer = AudioinputSerializer(data=all_data)
#     if serializer.is_valid():
#         serializer.save()
#         return Response(serializer.data, status=status.HTTP_201_CREATED)
#     else:
#         return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# Pepare an utility function for latency and GPU memory footprint measurement
def measure_latency_and_memory_use(model, inputs, nb_loops = 5):

    # define Events that measure start and end of the generate pass
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # reset cuda memory stats and empty cache
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # get the start time
    start_event.record()

    # actually generate
    for _ in range(nb_loops):
            # set seed for reproductibility
            set_seed(0)
            output = model.generate(**inputs, do_sample = True, fine_temperature = 0.4, coarse_temperature = 0.8)

    # get the end time
    end_event.record()
    torch.cuda.synchronize()

    # measure memory footprint and elapsed time
    max_memory = torch.cuda.max_memory_allocated(device)
    elapsed_time = start_event.elapsed_time(end_event) * 1.0e-3

    print('Execution time:', elapsed_time/nb_loops, 'seconds')
    print('Max memory footprint', max_memory*1e-9, ' GB')

    return output

@api_view(['POST'])
def postaudiogen(request):
    
    # deleting existing files in the results folder
    
    file_result_rem = os.path.join(settings.BASE_DIR, settings.MEDIA_ROOT, 'output_audio/')

    # Remove existing files in the folder
    for file_name in os.listdir(file_result_rem):
        file_path = os.path.join(file_result_rem, file_name)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")
    
    
    
    language = request.data.get('language')
    # lang = []
    gen = request.data.get('gender')
    
    if language == "English":
        male_voice = "v2/en_speaker_6"
        female = "v2/en_speaker_9"
        if gen == "Male":
            # lang = male_voice
            voice_preset = male_voice
        else:
            # lang = female
            voice_preset = female

    elif language == "Spanish":
        male_voice = "v2/es_speaker_6"
        female_voice = "v2/es_speaker_9"
        if gen == "Male":
            # lang = male_voice
            voice_preset = male_voice
        else:
            # lang = female_voice
            voice_preset = female_voice

    else:
        voice_preset = "unvalid....!"

    # voice_preset = lang
    text_prompt = request.data.get('prompt')
    
    inputs = processor(text_prompt,voice_preset = voice_preset).to(device)
    
    # Use bettertransform for flash attention
    model = BetterTransformer.transform(model, keep_original_model=False)
    
    with torch.inference_mode():
        speech_output = measure_latency_and_memory_use(model, inputs, nb_loops = 5)
        
    # save_path = f'path/to/save/{serializer.data["id"]}_speech_output.wav'
    save_path = os.path.join(settings.BASE_DIR, settings.MEDIA_ROOT, 'audio_output/speech_output.wav')
    
    # Determine the sampling rate
    sampling_rate = model.generation_config.sample_rate

    # Convert the Tensor to a NumPy array
    audio_data = speech_output[0].cpu().numpy()

    # Save the audio data as a WAV file
    write(save_path, sampling_rate, audio_data)
    return Response({"response": f"audio file saved"}, status=status.HTTP_201_CREATED)

    # all_data = {
    #     'language' : language,
    #     'gender' : gen,
    #     'voice' : voice_preset,
    #     'prompt' : text_prompt,
    # }
    
    # serializer = AudioinputSerializer(data=all_data)
    # if serializer.is_valid():
    #     serializer.save()
    #     return Response(serializer.data, status=status.HTTP_201_CREATED)
    # else:
    #     return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['GET'])
def getaudiogen(request):
    try:
        all_data = Audio_input.objects.all()
        serializer = AudioinputSerializer(all_data, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)
    except Exception as e:
        return Response({"response": f"Something went wrong: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# --------------------only click API-------------------------
@api_view(['POST'])
def postonlyclick(request):
    
    
    # deleting existing files in the results folder
    
    file_result_rem = os.path.join(settings.BASE_DIR, settings.MEDIA_ROOT, 'results/')

    # Remove existing files in the folder
    for file_name in os.listdir(file_result_rem):
        file_path = os.path.join(file_result_rem, file_name)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")
    
    # return JsonResponse('deleted files', status=status.HTTP_200_OK,safe=False) 
    
    
    queryset = Combine_Image.objects.filter(combined_image__isnull=False)
    combined_image_url = str(queryset.first().combined_image) if queryset.exists() else ""
    # print(combined_image_url)
    
    driven_audio = os.path.join(settings.MEDIA_ROOT, 'audio_output/speech_output.wav')
    source_image = combined_image_url
    result_dir = os.path.join(settings.MEDIA_ROOT, 'results')

    command = f"python inference.py --driven_audio {driven_audio} --source_image {source_image} --result_dir {result_dir} --still --preprocess full --enhancer gfpgan"

    # Execute the command in the terminal
    os.system(command)
    
    print("Inference completed. Check the terminal for details.")
    
    # Display the video result
    result_video_path = os.path.join(result_dir, "result.mp4")
    if os.path.exists(result_video_path):
        final_result_video = request.build_absolute_uri(result_video_path)
        print(final_result_video)
        return JsonResponse({'final_result_video': final_result_video}, status=status.HTTP_200_OK)
    else:
        # st.warning("No video result found.")
        return JsonResponse({'error': 'No video result found.'}, status=status.HTTP_404_NOT_FOUND) 
    
    
    
    # ------------------
    # deleting the file testing
    # Define the folder where images are saved
    # combined_images_folder = os.path.join(settings.BASE_DIR, settings.MEDIA_ROOT, 'results/')

    # # Remove existing files in the folder
    # for file_name in os.listdir(combined_images_folder):
    #     file_path = os.path.join(combined_images_folder, file_name)
    #     try:
    #         if os.path.isfile(file_path):
    #             os.remove(file_path)
    #     except Exception as e:
    #         print(f"Error deleting file {file_path}: {e}")
    
    # return JsonResponse('deleted files', status=status.HTTP_200_OK,safe=False) 
    
    
    
    
    
    # -----------
    # # testing
    # queryset = Combine_Image.objects.filter(combined_image__isnull=False)
    # first_image_url = str(queryset.first().combined_image) if queryset.exists() else ""
    # print(first_image_url)
    # return JsonResponse(first_image_url, status=status.HTTP_200_OK,safe=False)
    # -----------
    # serializer = onlyClickSerializer(data=data)
    # if serializer.is_valid():
    #     return Response(serializer.data, status=status.HTTP_201_CREATED)
    # else:
    #     return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)





# --------------------combined image with avatar and background image API-------------------------


def combine_images(avatar_path, background_path,resulting_image, avatar_width=None, avatar_height=None, background_width=None, background_height=None):
    avatar = cv2.imread(avatar_path, cv2.IMREAD_UNCHANGED)
    background = cv2.imread(background_path)
    # Check if images are loaded successfully
    if avatar is None or background is None:
        print("Error: Failed to load one or both of the images.")
        # Handle the error or return an appropriate response
        return Response({"error": "Failed to load images"}, status=status.HTTP_400_BAD_REQUEST)

    # Print dimensions of loaded images
    print("Avatar dimensions:", avatar.shape)
    print("Background dimensions:", background.shape)
    
    
    if avatar_width and avatar_height:
        avatar = cv2.resize(avatar, (avatar_width, avatar_height))
    elif avatar_width:
        avatar = cv2.resize(avatar, (avatar_width, int(avatar.shape[0] * avatar_width / avatar.shape[1])))
    elif avatar_height:
        avatar = cv2.resize(avatar, (int(avatar.shape[1] * avatar_height / avatar.shape[0]), avatar_height))
    if background_width and background_height:
        background = cv2.resize(background, (background_width, background_height))
    elif background_width:
        background = cv2.resize(background, (background_width, int(background.shape[0] * background_width / background.shape[1])))
    elif background_height:
        background = cv2.resize(background, (int(background.shape[1] * background_height / background.shape[0]), background_height))
    y_offset = background.shape[0] - avatar.shape[0]
    x_offset = (background.shape[1] - avatar.shape[1]) // 2
    alpha_channel = avatar[:, :, 3] / 255.0
    alpha_channel = np.stack([alpha_channel] * 3, axis=-1)

    blended_image = (1.0 - alpha_channel) * background[y_offset:y_offset+avatar.shape[0], x_offset:x_offset+avatar.shape[1]] + \
                    alpha_channel * avatar[:, :, :3]
    background[y_offset:y_offset+avatar.shape[0], x_offset:x_offset+avatar.shape[1]] = blended_image
    cv2.imwrite(resulting_image, background)
    
    print('hello')
    return background


# @api_view(['POST'])
# def post_combined_image(request):
#     avatar_image = request.data.get('avatar_image')
#     background_image = request.data.get('background_image')
    
#     avatar_path = f'{settings.BASE_DIR}{settings.STATIC_URL}images/avatar/{avatar_image}'
#     background_path = f'{settings.BASE_DIR}{settings.STATIC_URL}images/background_images/{background_image}'

#     avatar_width = 1200
#     avatar_height = 1200
#     background_width = 2500
#     background_height = 1700

#     # Save combined image
#     resulting_image= 'final_image.png'
#     combined_image=combine_images(avatar_path, background_path,resulting_image, avatar_width, avatar_height, background_width, background_height)
    
#     current_time = datetime.now()
#     seconds = current_time.second
#     file_name, file_extension = os.path.splitext(resulting_image)
#     new_file_name = f'{file_name}{seconds}{file_extension}'

#     # Save combined image directly to the database
#     combined_image_instance = Combine_Image(
#         combined_image_name=new_file_name,
#         combined_image=ContentFile(cv2.imencode('.png', combined_image)[1].tobytes(), name=new_file_name)
#     )
#     combined_image_instance.save()

#     return Response({"message": "Combined image saved successfully"}, status=status.HTTP_201_CREATED)
    

@api_view(['POST'])
def post_combined_image(request):
    avatar_image = request.data.get('avatar_image')
    background_image = request.data.get('background_image')
    
    avatar_path = f'{settings.BASE_DIR}{settings.STATIC_URL}images/avatar/{avatar_image}'
    background_path = f'{settings.BASE_DIR}{settings.STATIC_URL}images/background_images/{background_image}'

    avatar_width = 1200
    avatar_height = 1200
    background_width = 2500
    background_height = 1700

    # Save combined image
    resulting_image= 'final_image.png'
    combined_image=combine_images(avatar_path, background_path,resulting_image, avatar_width, avatar_height, background_width, background_height)
    
    current_time = datetime.now()
    seconds = current_time.second
    file_name, file_extension = os.path.splitext(resulting_image)
    new_file_name = f'{file_name}{seconds}{file_extension}'


    Combine_Image.objects.all().delete()

    # Define the folder where images are saved
    combined_images_folder = os.path.join(settings.BASE_DIR, settings.MEDIA_ROOT, 'images/combine/')

    # Remove existing files in the folder
    for file_name in os.listdir(combined_images_folder):
        file_path = os.path.join(combined_images_folder, file_name)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")

    # Construct the base URL dynamically
    base_url = request.build_absolute_uri('/')

    # Construct the complete image URL
    # image_url = f"{base_url}{settings.MEDIA_URL}images/combine/{new_file_name}"
    image_url = urljoin(base_url, f"{settings.MEDIA_URL}images/combine/{new_file_name}")

    # Save combined image directly to the database
    combined_image_instance = Combine_Image(
        combined_image_name=new_file_name,
        combined_image=ContentFile(cv2.imencode('.png', combined_image)[1].tobytes(), name=new_file_name)
    )
    combined_image_instance.save()

    return Response({"message": "Combined image saved successfully", "image_url": image_url}, status=status.HTTP_201_CREATED)


@api_view(['GET'])
def get_combined_image(request):
    try:
        all_data = Combine_Image.objects.all()
        serializer = Combine_ImageSerializer(all_data, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)
    except Exception as e:
        return Response({"response": f"Something went wrong: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    