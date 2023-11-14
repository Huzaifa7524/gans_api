# models.py
from django.db import models
from django.conf import settings
class Avatar(models.Model):
    avatar_images = models.ImageField(upload_to='images/avatar/')
    image_name = models.CharField(null=True,blank=True, max_length=500)
    
    def save(self, *args, **kwargs):
        if self.avatar_images:
            self.image_name = self.avatar_images.name.split('/')[-1]
        super(Avatar, self).save(*args, **kwargs)

    def __str__(self):
        return self.avatar_images.url
        # return self.avatar_images.url 
    
class Background(models.Model):
    background_image = models.ImageField(upload_to='images/bg/')
    image_name = models.CharField(null=True,blank=True, max_length=500)
    
    def save(self, *args, **kwargs):
        if self.background_image:
            self.image_name = self.background_image.name.split('/')[-1]
        super(Background, self).save(*args, **kwargs)

    def __str__(self):
        return self.background_image.url

class Backgroundandavatarcombine(models.Model):
    backgroun_final= models.ImageField(upload_to='images/combine/')
    image_name = models.CharField(null=True,blank=True, max_length=500)
    
    def save(self, *args, **kwargs):
        if self.backgroun_final:
            self.image_name = self.backgroun_final.name.split('/')[-1]
        super(Backgroundandavatarcombine, self).save(*args, **kwargs)

    def __str__(self):
        return self.backgroun_final.url

class Audio_input(models.Model):
    prompt = models.CharField(max_length=10000, null=True, blank=True)
    language = models.CharField(max_length=1000, null=True, blank=True)
    gender = models.CharField(max_length=250, null=True, blank=True)
    video_image = models.CharField(max_length=1000, null=True, blank=True)
    voice = models.CharField(max_length=1000, null=True, blank=True)
    # output_audio = models.FileField(upload_to='output_audio/')
    def __str__(self):
        return self.gender

class onlyClick(models.Model):
    click = models.BooleanField(null=True, blank=True)
    def __str__(self):
        return self.click
    
    
    
    
class Combine_Image(models.Model):
    combined_image= models.ImageField(upload_to= 'images/combine/')
    combined_image_name = models.CharField(null=True,blank=True, max_length=500)
    
    def save(self, *args, **kwargs):
        if self.combined_image:
            self.combined_image_name = self.combined_image.name.split('/')[-1]
        super(Combine_Image, self).save(*args, **kwargs)

    def __str__(self):
        return self.combined_image_name









