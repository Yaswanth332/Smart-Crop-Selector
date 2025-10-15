# recommendation/management/commands/load_crop_images.py
from django.core.management.base import BaseCommand
from django.core.files import File
from recommendation.models import CropMaster
import os
import requests
from io import BytesIO
from PIL import Image

class Command(BaseCommand):
    help = 'Download and assign images to crops'

    def handle(self, *args, **options):
        # Free stock image APIs
        UNSPLASH_ACCESS_KEY = 'your_unsplash_key'  # Get from unsplash.com/developers
        
        # Fallback image mapping for common crops
        crop_image_urls = {
            'Rice': 'https://images.unsplash.com/photo-1586201375761-83865001e31c?w=500',
            'Wheat': 'https://images.unsplash.com/photo-1574323347407-f5e1ad6d020b?w=500',
            'Maize': 'https://images.unsplash.com/photo-1551754655-cd27e38d2076?w=500',
            'Cotton': 'https://images.unsplash.com/photo-1560472354-91b1c6e3a096?w=500',
            'Sugarcane': 'https://images.unsplash.com/photo-1483871788521-4f224a86e166?w=500',
            'Tomato': 'https://images.unsplash.com/photo-1518977956812-cd3dbadaaf31?w=500',
            'Potato': 'https://images.unsplash.com/photo-1518977956812-cd3dbadaaf31?w=500',
            'Onion': 'https://images.unsplash.com/photo-1518977956812-cd3dbadaaf31?w=500',
            'Chickpea': 'https://images.unsplash.com/photo-1592159899792-8e6b3286b04a?w=500',
            'Soybean': 'https://images.unsplash.com/photo-1551754655-cd27e38d2076?w=500',
            'Sunflower': 'https://images.unsplash.com/photo-1470509037663-253afd7f0f51?w=500',
            'Peanut': 'https://images.unsplash.com/photo-1566738780263-121935c8c4cb?w=500',
            'Mango': 'https://images.unsplash.com/photo-1553279768-865429ffd8cc?w=500',
            'Banana': 'https://images.unsplash.com/photo-1571771894821-ce9b6c11b08e?w=500',
            'Grapes': 'https://images.unsplash.com/photo-1537640538966-79f369143f8f?w=500',
        }
        
        crops = CropMaster.objects.all()
        
        for crop in crops:
            if crop.image:
                self.stdout.write(f'✓ {crop.name} already has an image')
                continue
                
            # Try to get URL from mapping
            image_url = None
            for key, url in crop_image_urls.items():
                if key.lower() in crop.name.lower():
                    image_url = url
                    break
            
            # If no mapping, use generic farming image
            if not image_url:
                image_url = 'https://images.unsplash.com/photo-1500382017468-9049fed747ef?w=500'
            
            try:
                # Download image
                response = requests.get(image_url, timeout=10)
                if response.status_code == 200:
                    # Process image
                    img = Image.open(BytesIO(response.content))
                    img = img.convert('RGB')
                    img.thumbnail((800, 600), Image.Resampling.LANCZOS)
                    
                    # Save to BytesIO
                    img_io = BytesIO()
                    img.save(img_io, format='JPEG', quality=85)
                    img_io.seek(0)
                    
                    # Save to model
                    filename = f"{crop.name.lower().replace(' ', '_')}.jpg"
                    crop.image.save(filename, File(img_io), save=True)
                    
                    self.stdout.write(self.style.SUCCESS(f'✓ Downloaded image for {crop.name}'))
                else:
                    self.stdout.write(self.style.WARNING(f'⚠ Failed to download image for {crop.name}'))
                    
            except Exception as e:
                self.stdout.write(self.style.ERROR(f'✗ Error for {crop.name}: {str(e)}'))
        
        self.stdout.write(self.style.SUCCESS('\n✅ Image loading complete!'))