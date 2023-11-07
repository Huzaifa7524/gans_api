# models.py
from django.db import models

class Avatarandbackground(models.Model):
    avatar_images = models.ImageField(upload_to='images/avatar/')
    background_images = models.ImageField(upload_to='images/bg/')

    def __str__(self):
        return self.avatar_images.url 


class Backgroundandavatarcombine(models.Model):
    backgroun_final= models.ImageField(upload_to='images/combine/')

    def __str__(self):
        return self.backgroun_final.url 

class Audio_inputs(models.Model):
    prompt = models.CharField(max_length=10000)
    language = models.CharField(max_length=100)
    voice = models.CharField(max_length=100)
    

