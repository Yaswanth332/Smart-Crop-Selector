# recommendation/management/commands/fix_image_names.py
from django.core.management.base import BaseCommand
from django.conf import settings
import os

class Command(BaseCommand):
    help = 'Fix image filenames by replacing spaces with underscores and lowercasing'

    def add_arguments(self, parser):
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be renamed without actually renaming',
        )

    def handle(self, *args, **options):
        static_images_dir = os.path.join(settings.BASE_DIR, 'recommendation', 'static', 'images')
        
        self.stdout.write(self.style.SUCCESS('=' * 70))
        self.stdout.write(self.style.SUCCESS('🔧 IMAGE FILENAME FIX TOOL'))
        self.stdout.write(self.style.SUCCESS('=' * 70))
        
        if not os.path.exists(static_images_dir):
            self.stdout.write(self.style.ERROR(f'❌ Directory not found: {static_images_dir}'))
            return
        
        # Get all image files
        image_files = [f for f in os.listdir(static_images_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.jfif'))]
        
        dry_run = options['dry_run']
        renamed_count = 0
        skipped_count = 0
        error_count = 0
        
        if dry_run:
            self.stdout.write(self.style.WARNING('\n⚠️  DRY RUN MODE - No files will be renamed\n'))
        
        for filename in image_files:
            # Generate new filename
            new_filename = filename
            
            # Remove spaces and replace with underscores
            new_filename = new_filename.replace(' ', '_')
            
            # Lowercase the name (keep extension case)
            name_part, ext = os.path.splitext(new_filename)
            new_filename = name_part.lower() + ext.lower()
            
            # Remove parentheses and special chars
            new_filename = new_filename.replace('(', '').replace(')', '')
            new_filename = new_filename.replace('[', '').replace(']', '')
            
            # Normalize extensions
            if new_filename.endswith('.jpeg'):
                new_filename = new_filename.replace('.jpeg', '.jpg')
            if new_filename.endswith('.jfif'):
                new_filename = new_filename.replace('.jfif', '.jpg')
            
            # Check if rename is needed
            if filename != new_filename:
                old_path = os.path.join(static_images_dir, filename)
                new_path = os.path.join(static_images_dir, new_filename)
                
                # Check if target already exists
                if os.path.exists(new_path) and not dry_run:
                    self.stdout.write(self.style.WARNING(f'⚠️  Skip (target exists): {filename} → {new_filename}'))
                    skipped_count += 1
                    continue
                
                if dry_run:
                    self.stdout.write(f'   Would rename: {filename}')
                    self.stdout.write(f'             → {new_filename}')
                else:
                    try:
                        os.rename(old_path, new_path)
                        self.stdout.write(self.style.SUCCESS(f'✅ Renamed: {filename}'))
                        self.stdout.write(f'        → {new_filename}')
                        renamed_count += 1
                    except Exception as e:
                        self.stdout.write(self.style.ERROR(f'❌ Error renaming {filename}: {str(e)}'))
                        error_count += 1
            else:
                # No rename needed
                pass
        
        # Summary
        self.stdout.write('\n' + '=' * 70)
        self.stdout.write(self.style.SUCCESS('📊 SUMMARY'))
        self.stdout.write('=' * 70)
        
        if dry_run:
            changes_needed = sum(1 for f in image_files if self._needs_rename(f))
            self.stdout.write(f'Files that need renaming: {changes_needed}')
            self.stdout.write(f'\nTo apply changes, run: python manage.py fix_image_names')
        else:
            self.stdout.write(f'✅ Successfully renamed: {renamed_count}')
            self.stdout.write(f'⚠️  Skipped (already exists): {skipped_count}')
            if error_count > 0:
                self.stdout.write(f'❌ Errors: {error_count}')
            
            if renamed_count > 0:
                self.stdout.write(self.style.SUCCESS('\n✅ Renaming completed!'))
                self.stdout.write(self.style.SUCCESS('Next step: python manage.py migrate_crop_images'))
        
        self.stdout.write('=' * 70)
    
    def _needs_rename(self, filename):
        """Check if filename needs to be renamed"""
        new_filename = filename.replace(' ', '_')
        name_part, ext = os.path.splitext(new_filename)
        new_filename = name_part.lower() + ext.lower()
        new_filename = new_filename.replace('(', '').replace(')', '')
        new_filename = new_filename.replace('[', '').replace(']', '')
        if new_filename.endswith('.jpeg'):
            new_filename = new_filename.replace('.jpeg', '.jpg')
        if new_filename.endswith('.jfif'):
            new_filename = new_filename.replace('.jfif', '.jpg')
        return filename != new_filename