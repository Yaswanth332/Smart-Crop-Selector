# recommendation/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('register/', views.register_view, name='register'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('recommend/', views.recommend_crop, name='recommend'),
    path('results/', views.recommendation_results, name='results'),
    path('crop/<int:pk>/', views.crop_detail, name='crop_detail'),
    path('history/', views.past_recommendations, name='past_recommendations'),
    path('history/<int:query_id>/', views.view_past_recommendation, name='view_past_recommendation'),
]