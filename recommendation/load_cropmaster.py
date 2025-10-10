# recommendation/load_cropmaster.py

import os
import shutil
from django.conf import settings
from django.core.files import File
from .models import CropMaster
import csv

def run():
    """
    Load crop data and move images from static to media folder.
    Source: recommendation/static/images/
    Destination: media/crop_images/
    """
    CropMaster.objects.all().delete()
    
    csv_path = os.path.join(settings.BASE_DIR, 'recommendation', 'data', 'crop_data.csv')
    source_images = os.path.join(settings.BASE_DIR, 'recommendation', 'static', 'images')
    media_crop_images = os.path.join(settings.MEDIA_ROOT, 'crop_images')
    
    # Create media directory if it doesn't exist
    os.makedirs(media_crop_images, exist_ok=True)

    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            crop_name_normalized = row['name'].strip().lower().replace(" ", "_").replace("(", "").replace(")", "")
            image_filename = f'{crop_name_normalized}.jpg'
            
            source_image_path = os.path.join(source_images, image_filename)
            dest_image_path = os.path.join(media_crop_images, image_filename)
            
            image_field_value = None
            
            if os.path.exists(source_image_path):
                # Copy image to media folder
                shutil.copy2(source_image_path, dest_image_path)
                image_field_value = f'crop_images/{image_filename}'
                print(f"✓ Copied {image_filename} to media folder")
            else:
                print(f"⚠️  Image not found: {image_filename}")

            CropMaster.objects.create(
                name=row['name'],
                description=row['description'],
                soil_texture=row['soil_texture'],
                soil_ph_min=row['soil_ph_min'],
                soil_ph_max=row['soil_ph_max'],
                organic_matter=row['organic_matter'],
                drainage_status=row['drainage_status'],
                rainfall_min=row['rainfall_min'],
                rainfall_max=row['rainfall_max'],
                temperature_min=row['temperature_min'],
                temperature_max=row['temperature_max'],
                season=row['season'],
                previous_crop=row.get('previous_crop', ''),
                image=image_field_value
            )
    
    print(f"\n✅ Loaded {CropMaster.objects.count()} crops!")