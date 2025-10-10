# recommendation/forms.py

from django import forms
from .models import CropQuery

class CropQueryForm(forms.ModelForm):
    class Meta:
        model = CropQuery
        exclude = ['user', 'timestamp']  # timestamp is auto, user is added in the view

    # Override fields to add custom labels, choices, and widgets
    soil_texture = forms.ChoiceField(
        label="Soil Texture",
        choices=[
            ('sandy', 'Sandy (Gritty feel, drains quickly)'),
            ('clay', 'Clay (Sticky feel, holds water)'),
            ('loamy', 'Loamy (Balanced feel, best for most crops)'),
        ],
        widget=forms.Select(attrs={'class': 'form-select'})
    )

    soil_ph = forms.FloatField(
        label="Soil pH",
        min_value=0,
        max_value=14,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )

    organic_matter = forms.ChoiceField(
        label="Organic Matter Level",
        choices=[
            ('low', 'Low (Light-colored soil, low fertility)'),
            ('medium', 'Medium (Moderate fertility)'),
            ('high', 'High (Dark, rich, fertile soil)'),
        ],
        widget=forms.Select(attrs={'class': 'form-select'})
    )

    drainage_status = forms.ChoiceField(
        label="Drainage Status",
        choices=[
            ('well_drained', 'Well Drained (No standing water after rain)'),
            ('moderate', 'Moderate (Water drains in 4–6 hrs)'),
            ('poor', 'Poor (Water stays >24 hrs after rain)'),
        ],
        widget=forms.Select(attrs={'class': 'form-select'})
    )

    rainfall_mm = forms.FloatField(
        label="Rainfall (mm)",
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )

    avg_temperature = forms.FloatField(
        label="Temperature (°C)",
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )

    season = forms.ChoiceField(
        label="Current Season",
        choices=[
            ('kharif', 'Kharif (Monsoon crops: June–Oct)'),
            ('rabi', 'Rabi (Winter crops: Oct–Mar)'),
            ('zaid', 'Zaid (Summer crops: Mar–June)'),
            ('any', 'Any Season'),
        ],
        widget=forms.Select(attrs={'class': 'form-select'})
    )

    previous_crop = forms.ChoiceField(
        label="Previous Crop (optional)",
        required=False,
        choices=[
            ('', '--- Select Previous Crop ---'),
            ('rice', 'Rice'),
            ('wheat', 'Wheat'),
            ('maize', 'Maize'),
            ('cotton', 'Cotton'),
            ('pulses', 'Pulses'),
            ('sugarcane', 'Sugarcane'),
            ('others', 'Other / Not Listed'),
        ],
        widget=forms.Select(attrs={'class': 'form-select'})
    )
