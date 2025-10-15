# recommendation/management/commands/check_crop_images.py
from django.core.management.base import BaseCommand
from django.conf import settings
from recommendation.models import CropMaster
import os
from difflib import get_close_matches

class Command(BaseCommand):
    help = 'Check and verify crop images matching'

    def handle(self, *args, **options):
        static_images_dir = os.path.join(settings.BASE_DIR, 'recommendation', 'static', 'images')
        
        self.stdout.write(self.style.SUCCESS('=' * 70))
        self.stdout.write(self.style.SUCCESS('🔍 CROP IMAGE VERIFICATION TOOL'))
        self.stdout.write(self.style.SUCCESS('=' * 70))
        
        if not os.path.exists(static_images_dir):
            self.stdout.write(self.style.ERROR(f'❌ Directory not found: {static_images_dir}'))
            return
        
        # Get all image files
        image_files = [f for f in os.listdir(static_images_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        
        # Get all crop names from database
        crops = CropMaster.objects.all()
        crop_names = [crop.name for crop in crops]
        
        self.stdout.write(f'\n📊 Statistics:')
        self.stdout.write(f'   Total crops in database: {len(crop_names)}')
        self.stdout.write(f'   Total images in folder: {len(image_files)}')
        
        # Generate expected filenames
        expected_filenames = {}
        for crop in crops:
            normalized = crop.name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
            expected_filenames[crop.name] = f'{normalized}.jpg'
        
        # Check matches
        self.stdout.write(f'\n✅ MATCHED CROPS:')
        matched = []
        for crop_name, expected_file in expected_filenames.items():
            # Try various patterns
            patterns = [
                expected_file,
                expected_file.replace('.jpg', '.jpeg'),
                expected_file.replace('.jpg', '.png'),
                crop_name.lower().replace(' ', '_') + '.jpg',
            ]
            
            found = False
            for pattern in patterns:
                if pattern in image_files:
                    matched.append(crop_name)
                    self.stdout.write(f'   ✓ {crop_name} → {pattern}')
                    found = True
                    break
            
            if not found and len(matched) < 10:  # Show first 10 unmatched
                pass  # We'll show these later
        
        # Show unmatched crops
        unmatched_crops = [name for name in crop_names if name not in matched]
        
        if unmatched_crops:
            self.stdout.write(f'\n⚠️  CROPS WITHOUT MATCHING IMAGES ({len(unmatched_crops)}):')
            
            for crop_name in unmatched_crops[:15]:  # Show first 15
                expected = expected_filenames[crop_name]
                
                # Find close matches
                close_matches = get_close_matches(
                    expected.replace('.jpg', ''),
                    [img.replace('.jpg', '').replace('.jpeg', '').replace('.png', '') for img in image_files],
                    n=3,
                    cutoff=0.4
                )
                
                self.stdout.write(f'\n   ❌ {crop_name}')
                self.stdout.write(f'      Expected: {expected}')
                
                if close_matches:
                    self.stdout.write(f'      Suggestions:')
                    for match in close_matches:
                        # Find full filename
                        for img in image_files:
                            if match in img:
                                self.stdout.write(f'         - {img}')
                                break
            
            if len(unmatched_crops) > 15:
                self.stdout.write(f'\n   ... and {len(unmatched_crops) - 15} more')
        
        # Show unused images
        used_images = set()
        for img in image_files:
            for crop_name in matched:
                normalized = crop_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
                if normalized in img.lower():
                    used_images.add(img)
                    break
        
        unused_images = set(image_files) - used_images
        
        if unused_images:
            self.stdout.write(f'\n📷 UNUSED IMAGES ({len(unused_images)}):')
            for img in sorted(list(unused_images))[:15]:
                self.stdout.write(f'   - {img}')
            if len(unused_images) > 15:
                self.stdout.write(f'   ... and {len(unused_images) - 15} more')
        
        # Generate rename script
        self.stdout.write(f'\n' + '=' * 70)
        self.stdout.write(self.style.SUCCESS('💡 RECOMMENDATIONS:'))
        self.stdout.write('=' * 70)
        
        self.stdout.write(f'\n✅ Matched: {len(matched)}/{len(crop_names)} crops')
        self.stdout.write(f'⚠️  Need attention: {len(unmatched_crops)} crops')
        
        if unmatched_crops:
            self.stdout.write(f'\n📝 To fix unmatched crops, you can:')
            self.stdout.write(f'   1. Rename image files to match expected names')
            self.stdout.write(f'   2. Run: python manage.py migrate_crop_images')
            self.stdout.write(f'   3. Or use interactive mode: python manage.py migrate_crop_images --interactive')
        
        # Create a rename suggestions file
        rename_script_path = os.path.join(settings.BASE_DIR, 'rename_images.txt')
        with open(rename_script_path, 'w', encoding='utf-8') as f:
            f.write('# Image Rename Suggestions\n')
            f.write('# Copy these commands to rename your images\n\n')
            
            for crop_name in unmatched_crops[:20]:
                expected = expected_filenames[crop_name]
                close_matches = get_close_matches(
                    expected.replace('.jpg', ''),
                    [img.replace('.jpg', '').replace('.jpeg', '').replace('.png', '') for img in image_files],
                    n=1,
                    cutoff=0.4
                )
                
                if close_matches:
                    for img in image_files:
                        if close_matches[0] in img:
                            f.write(f'# {crop_name}\n')
                            f.write(f'# ren "{img}" "{expected}"\n\n')
                            break
        
        self.stdout.write(f'\n📄 Rename suggestions saved to: {rename_script_path}')
        self.stdout.write(self.style.SUCCESS('\n✅ Verification completed!'))