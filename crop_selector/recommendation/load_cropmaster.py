import os
import csv
import django

# --- START: DJANGO SETUP ---
# This is the crucial part that makes the script work standalone.
# It tells Python where your Django project's settings are.
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'crop_selector.settings')

# This call configures Django.
django.setup()
# --- END: DJANGO SETUP ---

# Now that Django is set up, you can safely import your models
from django.conf import settings
from recommendation.models import CropMaster

def run():
    """
    This function contains the logic to load data from the CSV
    into the database.
    """
    print("Deleting old CropMaster data...")
    CropMaster.objects.all().delete()

    # Construct the full path to the CSV file to avoid path issues
    csv_file_path = os.path.join(settings.BASE_DIR, 'recommendation', 'data', 'crop_data.csv')
    print(f"Loading data from {csv_file_path}...")

    with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            crop_name = row['name'].strip().lower().replace(" ", "_")
            # Construct the full system path to the image to check if it exists
            image_path = os.path.join(settings.MEDIA_ROOT, 'crop_images', f'{crop_name}.jpg')

            # Use the relative path for the ImageField only if the file exists
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
                image=image_field
            )
            print(f"Created crop: {row['name']}")

    print("Data loading complete!")

# This standard Python construct ensures that the `run()` function is called
# only when the script is executed directly (not when imported as a module).

if __name__ == '__main__':
    run()
