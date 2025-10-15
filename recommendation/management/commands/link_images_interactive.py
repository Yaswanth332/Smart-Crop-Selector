from django.core.management.base import BaseCommand
from django.conf import settings
from django.core.files import File
from recommendation.models import CropMaster
import os
import shutil

class Command(BaseCommand):
    help = 'Interactively link crop images'

    def handle(self, *args, **options):
        static_images_dir = os.path.join(settings.BASE_DIR, 'recommendation', 'static', 'images')
        media_crop_images_dir = os.path.join(settings.MEDIA_ROOT, 'crop_images')
        
        os.makedirs(media_crop_images_dir, exist_ok=True)
        
        image_files = sorted([f for f in os.listdir(static_images_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
        
        crops_without_images = CropMaster.objects.filter(image='')
        
        self.stdout.write(f'\n🌾 Found {crops_without_images.count()} crops without images\n')
        
        for i, crop in enumerate(crops_without_images, 1):
            self.stdout.write(f'\n[{i}/{crops_without_images.count()}] Crop: {crop.name}')
            self.stdout.write('\nAvailable images:')
            
            for j, img in enumerate(image_files[:10], 1):
                self.stdout.write(f'  {j}. {img}')
            
            if len(image_files) > 10:
                self.stdout.write(f'  ... {len(image_files) - 10} more (type number or filename)')
            
            choice = input('\nEnter image number/filename (or "s" to skip, "q" to quit): ').strip()
            
            if choice.lower() == 'q':
                break
            elif choice.lower() == 's':
                continue
            else:
                # Handle number or filename
                selected_image = None
                
                if choice.isdigit():
                    idx = int(choice) - 1
                    if 0 <= idx < len(image_files):
                        selected_image = image_files[idx]
                else:
                    if choice in image_files:
                        selected_image = choice
                
                if selected_image:
                    try:
                        source_path = os.path.join(static_images_dir, selected_image)
                        crop_name_normalized = crop.name.lower().replace(' ', '_').replace('(', '').replace(')', '')
                        dest_filename = f'{crop_name_normalized}.jpg'
                        dest_path = os.path.join(media_crop_images_dir, dest_filename)
                        
                        shutil.copy2(source_path, dest_path)
                        crop.image = f'crop_images/{dest_filename}'
                        crop.save()
                        
                        self.stdout.write(self.style.SUCCESS(f'✅ Linked to {selected_image}'))
                        
                        # Remove from available list
                        image_files.remove(selected_image)
                        
                    except Exception as e:
                        self.stdout.write(self.style.ERROR(f'❌ Error: {str(e)}'))
                else:
                    self.stdout.write(self.style.WARNING('⚠️  Invalid choice'))
        
        self.stdout.write(self.style.SUCCESS('\n✅ Interactive linking completed!'))