# recommendation/load_cropmaster.py

import os
from django.conf import settings
from .models import CropMaster
import csv

def run():
    CropMaster.objects.all().delete()

    with open('recommendation/data/crop_data.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            crop_name = row['name'].strip().lower().replace(" ", "_")
            image_path = os.path.join(settings.MEDIA_ROOT, 'crop_images', f'{crop_name}.jpg')

            # Use image only if file exists
            image_field = f'crop_images/{crop_name}.jpg' if os.path.exists(image_path) else None

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
                image=image_field  # only set if image exists
            )
