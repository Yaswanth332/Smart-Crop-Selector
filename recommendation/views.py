# recommendation/views.py (UPDATED VERSION)

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.db.models import Q
from .models import CropQuery, CropMaster
from .forms import CropQueryForm
import os

# Import ML predictor
try:
    from .ml_utils.inference import get_crop_predictor
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("⚠️  ML module not available. Using rule-based fallback.")


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
    """Fallback rule-based matching (kept for compatibility)"""
    score = 0
    max_score = 8
    
    if crop.soil_texture == query.soil_texture:
        score += 1
    if crop.organic_matter == query.organic_matter:
        score += 1
    if crop.drainage_status == query.drainage_status:
        score += 1
    if crop.soil_ph_min <= query.soil_ph <= crop.soil_ph_max:
        score += 1
    if crop.rainfall_min <= query.rainfall_mm <= crop.rainfall_max:
        score += 1
    if crop.temperature_min <= query.avg_temperature <= crop.temperature_max:
        score += 1
    if crop.season == query.season or crop.season == 'any':
        score += 1
    if query.previous_crop and crop.previous_crop:
        if crop.previous_crop.lower() == query.previous_crop.lower():
            score += 1
    
    return score


@login_required
def recommend_crop(request):
    """Main recommendation view with ML integration"""
    
    if request.method == 'POST':
        form = CropQueryForm(request.POST)
        if form.is_valid():
            # Save query
            query = form.save(commit=False)
            query.user = request.user
            query.save()

            # Prepare query data for ML model
            query_data = {
                'soil_texture': query.soil_texture,
                'soil_ph': query.soil_ph,
                'organic_matter': query.organic_matter,
                'drainage_status': query.drainage_status,
                'rainfall_mm': query.rainfall_mm,
                'avg_temperature': query.avg_temperature,
                'season': query.season
            }

            # Try ML prediction first
            use_ml = ML_AVAILABLE and os.path.exists('recommendation/ml_models/crop_recommendation_model.pkl')
            
            if use_ml:
                try:
                    predictor = get_crop_predictor()
                    ml_predictions = predictor.predict_top_crops(query_data, top_n=12)
                    
                    # Get CropMaster objects for predicted crops
                    top_crops = []
                    for pred in ml_predictions:
                        try:
                            crop = CropMaster.objects.get(name=pred['crop_name'])
                            crop.ml_confidence = pred['confidence_percentage']
                            crop.score = int(pred['confidence'] * 8)  # Convert to 0-8 scale
                            top_crops.append(crop)
                        except CropMaster.DoesNotExist:
                            continue
                    
                    # Get prediction explanation
                    explanation = predictor.explain_prediction(query_data, ml_predictions)
                    
                    return render(request, 'recommendation/results.html', {
                        'crops': top_crops,
                        'query': query,
                        'total_found': len(ml_predictions),
                        'ml_enabled': True,
                        'explanation': explanation
                    })
                    
                except Exception as e:
                    print(f"❌ ML prediction failed: {e}. Falling back to rule-based.")
                    use_ml = False

            # Fallback to rule-based matching
            if not use_ml:
                matched_crops = []
                for crop in CropMaster.objects.all():
                    score = calculate_crop_match(crop, query)
                    if score > 0:
                        matched_crops.append((crop, score))

                matched_crops.sort(key=lambda x: x[1], reverse=True)
                top_crops = []
                
                for crop, score in matched_crops[:12]:
                    crop.score = score
                    top_crops.append(crop)

                return render(request, 'recommendation/results.html', {
                    'crops': top_crops,
                    'query': query,
                    'total_found': len(matched_crops),
                    'ml_enabled': False
                })
    else:
        form = CropQueryForm()
    
    return render(request, 'recommendation/recommend.html', {'form': form})


@login_required
def past_recommendations(request):
    """View past queries with pagination"""
    queries = CropQuery.objects.filter(user=request.user).order_by('-created_at')
    
    paginator = Paginator(queries, 5)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    queries_with_crops = []
    for query in page_obj:
        # Use ML if available
        use_ml = ML_AVAILABLE and os.path.exists('recommendation/ml_models/crop_recommendation_model.pkl')
        
        if use_ml:
            try:
                predictor = get_crop_predictor()
                query_data = {
                    'soil_texture': query.soil_texture,
                    'soil_ph': query.soil_ph,
                    'organic_matter': query.organic_matter,
                    'drainage_status': query.drainage_status,
                    'rainfall_mm': query.rainfall_mm,
                    'avg_temperature': query.avg_temperature,
                    'season': query.season
                }
                ml_predictions = predictor.predict_top_crops(query_data, top_n=3)
                
                top_3_crops = []
                for pred in ml_predictions:
                    try:
                        crop = CropMaster.objects.get(name=pred['crop_name'])
                        top_3_crops.append(crop)
                    except CropMaster.DoesNotExist:
                        continue
                
                queries_with_crops.append({
                    'query': query,
                    'crops': top_3_crops,
                    'total_matches': len(ml_predictions)
                })
                continue
            except:
                pass
        
        # Fallback to rule-based
        matched_crops = []
        for crop in CropMaster.objects.all():
            score = calculate_crop_match(crop, query)
            if score > 0:
                matched_crops.append((crop, score))
        
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
    
    # Use ML if available
    use_ml = ML_AVAILABLE and os.path.exists('recommendation/ml_models/crop_recommendation_model.pkl')
    
    if use_ml:
        try:
            predictor = get_crop_predictor()
            query_data = {
                'soil_texture': query.soil_texture,
                'soil_ph': query.soil_ph,
                'organic_matter': query.organic_matter,
                'drainage_status': query.drainage_status,
                'rainfall_mm': query.rainfall_mm,
                'avg_temperature': query.avg_temperature,
                'season': query.season
            }
            ml_predictions = predictor.predict_top_crops(query_data, top_n=12)
            
            top_crops = []
            for pred in ml_predictions:
                try:
                    crop = CropMaster.objects.get(name=pred['crop_name'])
                    crop.ml_confidence = pred['confidence_percentage']
                    crop.score = int(pred['confidence'] * 8)
                    top_crops.append(crop)
                except CropMaster.DoesNotExist:
                    continue
            
            return render(request, 'recommendation/results.html', {
                'crops': top_crops,
                'query': query,
                'is_past_query': True,
                'total_found': len(ml_predictions),
                'ml_enabled': True
            })
        except:
            pass
    
    # Fallback to rule-based
    matched_crops = []
    for crop in CropMaster.objects.all():
        score = calculate_crop_match(crop, query)
        matched_crops.append((crop, score))

    matched_crops.sort(key=lambda x: x[1], reverse=True)
    top_crops = []
    
    for crop, score in matched_crops[:12]:
        crop.score = score
        top_crops.append(crop)

    return render(request, 'recommendation/results.html', {
        'crops': top_crops,
        'query': query,
        'is_past_query': True,
        'total_found': len(matched_crops),
        'ml_enabled': False
    })


@login_required
def recommendation_results(request, query_id):
    query = get_object_or_404(CropQuery, id=query_id, user=request.user)
    return redirect('recommend')


def crop_detail(request, pk):
    crop = get_object_or_404(CropMaster, id=pk)
    return render(request, 'recommendation/crop_detail.html', {'crop': crop})