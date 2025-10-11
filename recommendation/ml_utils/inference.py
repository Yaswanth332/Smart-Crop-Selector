# recommendation/ml_utils/inference.py

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from django.conf import settings

class CropPredictor:
    """
    ML-based crop prediction for Django integration
    """
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.target_encoder = None
        self.label_encoders = {}
        self.model_loaded = False
        
        # Try to load models on initialization
        self.load_models()
    
    def load_models(self):
        """
        Load pre-trained models and preprocessors
        """
        try:
            model_dir = Path(settings.BASE_DIR) / 'recommendation' / 'ml_models'
            
            # Load main model
            model_path = model_dir / 'crop_recommendation_model.pkl'
            if model_path.exists():
                self.model = joblib.load(model_path)
                print("✅ ML model loaded successfully")
            
            # Load scaler
            scaler_path = model_dir / 'scaler.pkl'
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
            
            # Load target encoder
            target_encoder_path = model_dir / 'target_encoder.pkl'
            if target_encoder_path.exists():
                self.target_encoder = joblib.load(target_encoder_path)
            
            # Load label encoders
            encoder_files = {
                'soil_texture': 'soil_texture_encoder.pkl',
                'organic_matter': 'organic_matter_encoder.pkl',
                'drainage_status': 'drainage_status_encoder.pkl',
                'season': 'season_encoder.pkl'
            }
            
            for name, filename in encoder_files.items():
                encoder_path = model_dir / filename
                if encoder_path.exists():
                    self.label_encoders[name] = joblib.load(encoder_path)
            
            self.model_loaded = True
            
        except Exception as e:
            print(f"⚠️  Warning: Could not load ML models: {e}")
            self.model_loaded = False
    
    def preprocess_query(self, query_data):
        """
        Preprocess user query for model prediction
        
        Args:
            query_data (dict): User query with keys:
                - soil_texture
                - soil_ph
                - organic_matter
                - drainage_status
                - rainfall_mm
                - avg_temperature
                - season
        
        Returns:
            numpy.ndarray: Preprocessed features
        """
        # Create DataFrame with query
        df = pd.DataFrame([query_data])
        
        # Encode categorical features
        categorical_features = ['soil_texture', 'organic_matter', 'drainage_status', 'season']
        
        for feature in categorical_features:
            if feature in self.label_encoders and feature in df.columns:
                try:
                    df[feature] = self.label_encoders[feature].transform(df[feature])
                except ValueError:
                    # Handle unknown categories by using most common class
                    df[feature] = 0
        
        # Scale numerical features
        numerical_features = ['soil_ph', 'rainfall_mm', 'avg_temperature']
        if self.scaler is not None:
            df[numerical_features] = self.scaler.transform(df[numerical_features])
        
        # Ensure correct feature order
        feature_order = [
            'soil_texture', 'soil_ph', 'organic_matter', 
            'drainage_status', 'rainfall_mm', 'avg_temperature', 'season'
        ]
        
        return df[feature_order].values
    
    def predict_top_crops(self, query_data, top_n=10):
        """
        Predict top N crops for given query
        
        Args:
            query_data (dict): User query
            top_n (int): Number of top predictions to return
        
        Returns:
            list: List of tuples (crop_name, confidence_score)
        """
        if not self.model_loaded:
            raise ValueError("ML model not loaded. Please train and save the model first.")
        
        # Preprocess query
        X = self.preprocess_query(query_data)
        
        # Get predictions with probabilities
        probabilities = self.model.predict_proba(X)[0]
        
        # Get top N predictions
        top_indices = np.argsort(probabilities)[-top_n:][::-1]
        top_probs = probabilities[top_indices]
        
        # Convert indices to crop names
        crop_names = self.target_encoder.inverse_transform(top_indices)
        
        # Create results
        results = [
            {
                'crop_name': crop_name,
                'confidence': float(prob),
                'confidence_percentage': float(prob * 100)
            }
            for crop_name, prob in zip(crop_names, top_probs)
        ]
        
        return results
    
    def explain_prediction(self, query_data, prediction_results):
        """
        Generate explanation for predictions
        
        Args:
            query_data (dict): Original query
            prediction_results (list): Prediction results
        
        Returns:
            dict: Explanation of predictions
        """
        explanation = {
            'query_summary': {
                'soil_type': query_data.get('soil_texture', 'Unknown'),
                'soil_ph': query_data.get('soil_ph', 'N/A'),
                'rainfall': f"{query_data.get('rainfall_mm', 'N/A')} mm",
                'temperature': f"{query_data.get('avg_temperature', 'N/A')} °C",
                'season': query_data.get('season', 'Unknown')
            },
            'top_recommendation': prediction_results[0]['crop_name'] if prediction_results else None,
            'confidence_level': 'High' if prediction_results and prediction_results[0]['confidence'] > 0.7 else 'Moderate',
            'alternative_crops': [r['crop_name'] for r in prediction_results[1:4]] if len(prediction_results) > 1 else []
        }
        
        return explanation


# Singleton instance
_predictor_instance = None

def get_crop_predictor():
    """
    Get or create singleton CropPredictor instance
    """
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = CropPredictor()
    return _predictor_instance