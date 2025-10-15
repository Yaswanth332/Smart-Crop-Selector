# recommendation/management/commands/migrate_crop_images.py
from django.core.management.base import BaseCommand
from django.conf import settings
from django.core.files import File
from recommendation.models import CropMaster
import os
import shutil
from pathlib import Path

class Command(BaseCommand):
    help = 'Migrate crop images from static to media folder and link to database'

    def handle(self, *args, **options):
        # Paths
        static_images_dir = os.path.join(settings.BASE_DIR, 'recommendation', 'static', 'images')
        media_crop_images_dir = os.path.join(settings.MEDIA_ROOT, 'crop_images')
        
        # Create media directory if it doesn't exist
        os.makedirs(media_crop_images_dir, exist_ok=True)
        
        self.stdout.write(self.style.SUCCESS('=' * 70))
        self.stdout.write(self.style.SUCCESS('🌾 CROP IMAGE MIGRATION TOOL'))
        self.stdout.write(self.style.SUCCESS('=' * 70))
        
        if not os.path.exists(static_images_dir):
            self.stdout.write(self.style.ERROR(f'❌ Static images directory not found: {static_images_dir}'))
            return
        
        # Get all image files from static folder
        image_files = [f for f in os.listdir(static_images_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        
        self.stdout.write(f'\n📁 Found {len(image_files)} images in static folder')
        self.stdout.write(f'📂 Source: {static_images_dir}')
        self.stdout.write(f'📂 Destination: {media_crop_images_dir}\n')
        
        # Get all crops
        crops = CropMaster.objects.all()
        
        success_count = 0
        failed_count = 0
        already_linked = 0
        
        for crop in crops:
            # Check if crop already has an image
            if crop.image:
                already_linked += 1
                self.stdout.write(f'✓ {crop.name} - Already has image')
                continue
            
            # Try to find matching image
            matched_image = None
            crop_name_normalized = crop.name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
            
            # Try different naming patterns
            possible_names = [
                f'{crop_name_normalized}.jpg',
                f'{crop_name_normalized}.jpeg',
                f'{crop_name_normalized}.png',
                f'{crop_name_normalized}.webp',
                f'{crop.name.lower().replace(" ", "_")}.jpg',
                f'{crop.name.split()[0].lower()}.jpg',  # First word only
            ]
            
            for possible_name in possible_names:
                if possible_name in image_files:
                    matched_image = possible_name
                    break
            
            if matched_image:
                source_path = os.path.join(static_images_dir, matched_image)
                dest_filename = f'{crop_name_normalized}.jpg'
                dest_path = os.path.join(media_crop_images_dir, dest_filename)
                
                try:
                    # Copy image to media folder
                    shutil.copy2(source_path, dest_path)
                    
                    # Update database with relative path
                    crop.image = f'crop_images/{dest_filename}'
                    crop.save()
                    
                    success_count += 1
                    self.stdout.write(self.style.SUCCESS(f'✅ {crop.name} - Linked to {matched_image}'))
                    
                except Exception as e:
                    failed_count += 1
                    self.stdout.write(self.style.ERROR(f'❌ {crop.name} - Error: {str(e)}'))
            else:
                failed_count += 1
                self.stdout.write(self.style.WARNING(f'⚠️  {crop.name} - No matching image found'))
                self.stdout.write(self.style.WARNING(f'   Tried: {", ".join(possible_names[:3])}'))
        
        # Summary
        self.stdout.write('\n' + '=' * 70)
        self.stdout.write(self.style.SUCCESS('📊 MIGRATION SUMMARY'))
        self.stdout.write('=' * 70)
        self.stdout.write(f'✅ Successfully linked: {success_count}')
        self.stdout.write(f'✓ Already had images: {already_linked}')
        self.stdout.write(f'⚠️  Failed/Not found: {failed_count}')
        self.stdout.write(f'📁 Total crops: {crops.count()}')
        self.stdout.write('=' * 70)
        
        # List unmatched images
        if failed_count > 0:
            self.stdout.write(self.style.WARNING('\n💡 TIP: Check these crop names in your database:'))
            for crop in CropMaster.objects.filter(image=''):
                self.stdout.write(f'   - {crop.name}')
            
            self.stdout.write(self.style.WARNING('\n💡 Available image files in static folder:'))
            for img in sorted(image_files)[:20]:  # Show first 20
                self.stdout.write(f'   - {img}')
            if len(image_files) > 20:
                self.stdout.write(f'   ... and {len(image_files) - 20} more')
        
        self.stdout.write(self.style.SUCCESS('\n✅ Migration completed!'))


class Command2(BaseCommand):
    """Alternative version with interactive matching"""
    help = 'Interactively match crop images'

    def add_arguments(self, parser):
        parser.add_argument(
            '--auto',
            action='store_true',
            help='Auto-match without confirmation',
        )

    def handle(self, *args, **options):
        static_images_dir = os.path.join(settings.BASE_DIR, 'recommendation', 'static', 'images')
        media_crop_images_dir = os.path.join(settings.MEDIA_ROOT, 'crop_images')
        
        os.makedirs(media_crop_images_dir, exist_ok=True)
        
        # Get available images
        image_files = [f for f in os.listdir(static_images_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        
        crops_without_images = CropMaster.objects.filter(image='')
        
        self.stdout.write(f'\n🔍 Found {crops_without_images.count()} crops without images')
        self.stdout.write(f'📁 Available images: {len(image_files)}\n')
        
        for crop in crops_without_images:
            crop_name_normalized = crop.name.lower().replace(' ', '_').replace('(', '').replace(')', '')
            
            # Find best match
            best_matches = []
            for img in image_files:
                img_name = img.lower().replace('.jpg', '').replace('.jpeg', '').replace('.png', '')
                
                # Check if crop name words are in image filename
                crop_words = crop_name_normalized.split('_')
                matches = sum(1 for word in crop_words if word in img_name)
                
                if matches > 0:
                    best_matches.append((img, matches))
            
            if best_matches:
                # Sort by number of matching words
                best_matches.sort(key=lambda x: x[1], reverse=True)
                suggested_image = best_matches[0][0]
                
                if options['auto']:
                    # Auto-assign
                    self._assign_image(crop, suggested_image, static_images_dir, media_crop_images_dir)
                else:
                    # Ask for confirmation
                    self.stdout.write(f'\n🌱 Crop: {self.style.SUCCESS(crop.name)}')
                    self.stdout.write(f'📷 Suggested image: {suggested_image}')
                    self.stdout.write('Other matches:')
                    for match, score in best_matches[:3]:
                        self.stdout.write(f'   - {match} (score: {score})')
                    
                    confirm = input('Use suggested image? (y/n/skip): ').lower()
                    
                    if confirm == 'y':
                        self._assign_image(crop, suggested_image, static_images_dir, media_crop_images_dir)
                    elif confirm == 'skip':
                        continue
                    else:
                        manual_input = input('Enter image filename or "skip": ')
                        if manual_input != 'skip' and manual_input in image_files:
                            self._assign_image(crop, manual_input, static_images_dir, media_crop_images_dir)
    
    def _assign_image(self, crop, image_filename, source_dir, dest_dir):
        """Assign image to crop"""
        try:
            source_path = os.path.join(source_dir, image_filename)
            crop_name_normalized = crop.name.lower().replace(' ', '_').replace('(', '').replace(')', '')
            dest_filename = f'{crop_name_normalized}.jpg'
            dest_path = os.path.join(dest_dir, dest_filename)
            
            shutil.copy2(source_path, dest_path)
            crop.image = f'crop_images/{dest_filename}'
            crop.save()
            
            self.stdout.write(self.style.SUCCESS(f'✅ Linked {crop.name} to {image_filename}'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'❌ Error: {str(e)}'))