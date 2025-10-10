# crop_selector/recommendation/admin.py
from django.contrib import admin
from .models import CropMaster, CropQuery, UserProfile

@admin.register(CropMaster)
class CropMasterAdmin(admin.ModelAdmin):
    list_display = ('name', 'soil_texture', 'soil_ph_min', 'soil_ph_max', 'organic_matter', 'drainage_status', 'season')
    search_fields = ('name',)
    list_filter = ('soil_texture', 'organic_matter', 'drainage_status', 'season')

@admin.register(CropQuery)
class CropQueryAdmin(admin.ModelAdmin):
    list_display = ('user', 'soil_texture', 'rainfall_mm', 'avg_temperature', 'season', 'created_at')
    list_filter = ('season', 'soil_texture')
    search_fields = ('user__username',)

@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'role')
    list_filter = ('role',)
    search_fields = ('user__username',)
