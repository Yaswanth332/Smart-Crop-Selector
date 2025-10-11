# recommendation/ml_utils/data_preparation.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

class CropDataPreparation:
    """
    Prepare crop data for ML model training
    """
    
    def __init__(self, csv_path='recommendation/data/crop_data.csv'):
        self.csv_path = csv_path
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def load_and_preprocess_data(self):
        """
        Load crop master data and prepare for ML training
        """
        # Load data
        df = pd.read_csv(self.csv_path)
        
        # Create feature columns
        features = [
            'soil_texture', 'soil_ph_min', 'soil_ph_max', 
            'organic_matter', 'drainage_status',
            'rainfall_min', 'rainfall_max', 
            'temperature_min', 'temperature_max',
            'season'
        ]
        
        # Target variable
        target = 'name'
        
        X = df[features].copy()
        y = df[target].copy()
        
        # Encode categorical features
        categorical_features = ['soil_texture', 'organic_matter', 'drainage_status', 'season']
        
        for feature in categorical_features:
            le = LabelEncoder()
            X[feature] = le.fit_transform(X[feature])
            self.label_encoders[feature] = le
        
        # Encode target variable
        self.target_encoder = LabelEncoder()
        y_encoded = self.target_encoder.fit_transform(y)
        
        return X, y_encoded, df
    
    def generate_synthetic_training_data(self, df, n_samples=5000):
        """
        Generate synthetic training data from crop ranges
        This simulates user queries with known successful outcomes
        """
        synthetic_data = []
        
        for _, crop in df.iterrows():
            # Generate multiple samples for each crop within its acceptable ranges
            samples_per_crop = n_samples // len(df)
            
            for _ in range(samples_per_crop):
                sample = {
                    'soil_texture': crop['soil_texture'],
                    'soil_ph': np.random.uniform(crop['soil_ph_min'], crop['soil_ph_max']),
                    'organic_matter': crop['organic_matter'],
                    'drainage_status': crop['drainage_status'],
                    'rainfall_mm': np.random.uniform(crop['rainfall_min'], crop['rainfall_max']),
                    'avg_temperature': np.random.uniform(crop['temperature_min'], crop['temperature_max']),
                    'season': crop['season'],
                    'previous_crop': crop.get('previous_crop', 'others'),
                    'recommended_crop': crop['name']
                }
                synthetic_data.append(sample)
        
        synthetic_df = pd.DataFrame(synthetic_data)
        return synthetic_df
    
    def prepare_training_data(self, synthetic_df):
        """
        Prepare features and target for model training
        """
        # Features for prediction
        feature_columns = [
            'soil_texture', 'soil_ph', 'organic_matter', 
            'drainage_status', 'rainfall_mm', 'avg_temperature', 'season'
        ]
        
        X = synthetic_df[feature_columns].copy()
        y = synthetic_df['recommended_crop'].copy()
        
        # Encode categorical features
        categorical_features = ['soil_texture', 'organic_matter', 'drainage_status', 'season']
        
        for feature in categorical_features:
            if feature not in self.label_encoders:
                le = LabelEncoder()
                X[feature] = le.fit_transform(X[feature])
                self.label_encoders[feature] = le
            else:
                X[feature] = self.label_encoders[feature].transform(X[feature])
        
        # Encode target
        y_encoded = self.target_encoder.transform(y)
        
        # Scale numerical features
        numerical_features = ['soil_ph', 'rainfall_mm', 'avg_temperature']
        X[numerical_features] = self.scaler.fit_transform(X[numerical_features])
        
        return X, y_encoded
    
    def save_preprocessors(self, save_dir='recommendation/ml_models'):
        """
        Save label encoders and scaler for inference
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save label encoders
        for name, encoder in self.label_encoders.items():
            joblib.dump(encoder, f'{save_dir}/{name}_encoder.pkl')
        
        # Save target encoder
        joblib.dump(self.target_encoder, f'{save_dir}/target_encoder.pkl')
        
        # Save scaler
        joblib.dump(self.scaler, f'{save_dir}/scaler.pkl')
        
        print(f"✅ Preprocessors saved to {save_dir}")


if __name__ == "__main__":
    # Example usage
    prep = CropDataPreparation()
    
    # Load and preprocess master data
    X, y, df = prep.load_and_preprocess_data()
    print(f"Loaded {len(df)} crops from database")
    
    # Generate synthetic training data
    synthetic_df = prep.generate_synthetic_training_data(df, n_samples=10000)
    print(f"Generated {len(synthetic_df)} synthetic training samples")
    
    # Prepare training data
    X_train, y_train = prep.prepare_training_data(synthetic_df)
    print(f"Training data shape: {X_train.shape}")
    
    # Save preprocessors
    prep.save_preprocessors()