# recommendation/views.py

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.db.models import Q
from .models import CropQuery, CropMaster
from .forms import CropQueryForm

def home(request):
    return render(request, 'recommendation/home.html')

def register_view(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('login')
    else:
        form = UserCreationForm()
    return render(request, 'recommendation/register.html', {'form': form})

def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            login(request, form.get_user())
            return redirect('recommend')
    else:
        form = AuthenticationForm()
    return render(request, 'recommendation/login.html', {'form': form})

def logout_view(request):
    logout(request)
    return redirect('login')

def calculate_crop_match(crop, query):
    """Calculate how well a crop matches the user's query"""
    score = 0
    max_score = 8
    
    # Exact matches (higher weight)
    if crop.soil_texture == query.soil_texture:
        score += 1
    if crop.organic_matter == query.organic_matter:
        score += 1
    if crop.drainage_status == query.drainage_status:
        score += 1
    
    # Range matches
    if crop.soil_ph_min <= query.soil_ph <= crop.soil_ph_max:
        score += 1
    if crop.rainfall_min <= query.rainfall_mm <= crop.rainfall_max:
        score += 1
    if crop.temperature_min <= query.avg_temperature <= crop.temperature_max:
        score += 1
    
    # Season match
    if crop.season == query.season or crop.season == 'any':
        score += 1
    
    # Previous crop match (if provided)
    if query.previous_crop and crop.previous_crop:
        if crop.previous_crop.lower() == query.previous_crop.lower():
            score += 1
    
    return score

@login_required
def recommend_crop(request):
    if request.method == 'POST':
        form = CropQueryForm(request.POST)
        if form.is_valid():
            query = form.save(commit=False)
            query.user = request.user
            query.save()

            # Calculate matches for all crops
            matched_crops = []
            for crop in CropMaster.objects.all():
                score = calculate_crop_match(crop, query)
                matched_crops.append((crop, score))

            # Sort by score (highest first) and get top 12
            matched_crops.sort(key=lambda x: x[1], reverse=True)
            top_crops = []
            
            for crop, score in matched_crops[:12]:
                crop.score = score  # Add score to crop object for template
                top_crops.append(crop)

            return render(request, 'recommendation/results.html', {
                'crops': top_crops,
                'query': query,
                'total_found': len(matched_crops)
            })
    else:
        form = CropQueryForm()
    return render(request, 'recommendation/recommend.html', {'form': form})

@login_required
def past_recommendations(request):
    """View to show user's past crop queries and recommendations"""
    queries = CropQuery.objects.filter(user=request.user).order_by('-created_at')
    
    # Pagination
    paginator = Paginator(queries, 5)  # Show 5 queries per page
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    # For each query, get the top 3 recommended crops
    queries_with_crops = []
    for query in page_obj:
        matched_crops = []
        for crop in CropMaster.objects.all():
            score = calculate_crop_match(crop, query)
            if score > 0:  # Only include crops with some match
                matched_crops.append((crop, score))
        
        # Sort and get top 3
        matched_crops.sort(key=lambda x: x[1], reverse=True)
        top_3_crops = [crop for crop, score in matched_crops[:3]]
        
        queries_with_crops.append({
            'query': query,
            'crops': top_3_crops,
            'total_matches': len(matched_crops)
        })
    
    return render(request, 'recommendation/past_recommendations.html', {
        'queries_with_crops': queries_with_crops,
        'page_obj': page_obj
    })

@login_required
def view_past_recommendation(request, query_id):
    """View detailed results for a past query"""
    query = get_object_or_404(CropQuery, id=query_id, user=request.user)
    
    # Recalculate recommendations for this query
    matched_crops = []
    for crop in CropMaster.objects.all():
        score = calculate_crop_match(crop, query)
        matched_crops.append((crop, score))

    # Sort by score and get top crops
    matched_crops.sort(key=lambda x: x[1], reverse=True)
    top_crops = []
    
    for crop, score in matched_crops[:12]:
        crop.score = score
        top_crops.append(crop)

    return render(request, 'recommendation/results.html', {
        'crops': top_crops,
        'query': query,
        'is_past_query': True,
        'total_found': len(matched_crops)
    })

@login_required
def recommendation_results(request, query_id):
    query = get_object_or_404(CropQuery, id=query_id, user=request.user)
    return redirect('recommend')

def crop_detail(request, pk):
    crop = get_object_or_404(CropMaster, id=pk)
    return render(request, 'recommendation/crop_detail.html', {'crop': crop})