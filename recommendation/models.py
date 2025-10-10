# crop_selector/recommendation/models.py

from django.db import models
from django.contrib.auth.models import User

class CropQuery(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    soil_texture = models.CharField(max_length=20)
    soil_ph = models.FloatField()
    organic_matter = models.CharField(max_length=20)
    drainage_status = models.CharField(max_length=20)
    rainfall_mm = models.FloatField()
    avg_temperature = models.FloatField()
    season = models.CharField(max_length=20, choices=[
        ('kharif', 'Kharif'),
        ('rabi', 'Rabi'),
        ('zaid', 'Zaid'),
        ('any', 'Any'),
    ])
    previous_crop = models.CharField(max_length=50, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - Query {self.id}"


class CropMaster(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField()
    soil_texture = models.CharField(max_length=20)
    soil_ph_min = models.FloatField()
    soil_ph_max = models.FloatField()
    organic_matter = models.CharField(max_length=20)
    drainage_status = models.CharField(max_length=20)
    rainfall_min = models.FloatField()
    rainfall_max = models.FloatField()
    temperature_min = models.FloatField()
    temperature_max = models.FloatField()
    season = models.CharField(max_length=20, choices=[
        ('kharif', 'Kharif'),
        ('rabi', 'Rabi'),
        ('zaid', 'Zaid'),
        ('any', 'Any'),
    ])
    previous_crop = models.CharField(max_length=50, blank=True, null=True)
    image = models.ImageField(upload_to='crop_images/', blank=True, null=True)
    def __str__(self):
        return self.name


class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    role = models.CharField(max_length=20, choices=[('farmer', 'Farmer'), ('admin', 'Admin')])

    def __str__(self):
        return f"{self.user.username} - {self.role}"