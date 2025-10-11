# recommendation/ml_utils/train_model.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
from data_preparation import CropDataPreparation

class CropRecommendationModel:
    """
    Train and evaluate ML models for crop recommendation
    """
    
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = None
        self.feature_importance = None
        
    def create_model(self):
        """
        Initialize ML model
        """
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return self.model
    
    def train_model(self, X_train, y_train, X_val, y_val):
        """
        Train the model
        """
        print(f"\n🚀 Training {self.model_type} model...")
        
        # Create model
        self.create_model()
        
        # Train
        self.model.fit(X_train, y_train)
        
        # Evaluate on validation set
        y_pred = self.model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        
        print(f"✅ Training completed!")
        print(f"📊 Validation Accuracy: {accuracy:.4f}")
        
        # Store feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = self.model.feature_importances_
        
        return accuracy
    
    def hyperparameter_tuning(self, X_train, y_train):
        """
        Perform hyperparameter tuning using GridSearchCV
        """
        print("\n🔧 Performing hyperparameter tuning...")
        
        if self.model_type == 'random_forest':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        else:
            param_grid = {
                'n_estimators': [50, 100, 150],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        
        self.create_model()
        grid_search = GridSearchCV(
            self.model, 
            param_grid, 
            cv=5, 
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"✅ Best parameters: {grid_search.best_params_}")
        print(f"📊 Best cross-validation score: {grid_search.best_score_:.4f}")
        
        self.model = grid_search.best_estimator_
        return grid_search.best_params_
    
    def evaluate_model(self, X_test, y_test, target_encoder):
        """
        Comprehensive model evaluation
        """
        print("\n📈 Evaluating model on test set...")
        
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"🎯 Test Accuracy: {accuracy:.4f}")
        
        # Classification report
        crop_names = target_encoder.classes_
        print("\n📋 Classification Report (Top 10 crops):")
        report = classification_report(
            y_test, 
            y_pred, 
            target_names=crop_names,
            zero_division=0
        )
        print(report[:1000])  # Print first 1000 chars
        
        return accuracy
    
    def get_feature_importance(self, feature_names):
        """
        Get and display feature importance
        """
        if self.feature_importance is None:
            print("❌ Feature importance not available for this model")
            return None
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importance
        }).sort_values('importance', ascending=False)
        
        print("\n🔍 Feature Importance:")
        print(importance_df.to_string(index=False))
        
        return importance_df
    
    def save_model(self, save_path='recommendation/ml_models/crop_recommendation_model.pkl'):
        """
        Save trained model
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(self.model, save_path)
        print(f"✅ Model saved to {save_path}")
    
    def predict_with_confidence(self, X):
        """
        Predict with probability scores
        """
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        # Get top 3 predictions with confidence
        top_3_indices = np.argsort(probabilities, axis=1)[:, -3:][:, ::-1]
        top_3_probs = np.sort(probabilities, axis=1)[:, -3:][:, ::-1]
        
        return predictions, top_3_indices, top_3_probs


def full_training_pipeline():
    """
    Complete training pipeline
    """
    print("=" * 50)
    print("🌾 CROP RECOMMENDATION ML TRAINING PIPELINE")
    print("=" * 50)
    
    # Step 1: Data Preparation
    print("\n📂 Step 1: Data Preparation")
    prep = CropDataPreparation()
    X, y, df = prep.load_and_preprocess_data()
    
    # Generate synthetic data
    synthetic_df = prep.generate_synthetic_training_data(df, n_samples=15000)
    X_train_full, y_train_full = prep.prepare_training_data(synthetic_df)
    
    # Save preprocessors
    prep.save_preprocessors()
    
    # Step 2: Train-Test Split
    print("\n✂️  Step 2: Train-Test Split")
    X_train, X_test, y_train, y_test = train_test_split(
        X_train_full, y_train_full, 
        test_size=0.2, 
        random_state=42,
        stratify=y_train_full
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, 
        test_size=0.2, 
        random_state=42,
        stratify=y_train
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Step 3: Model Training
    print("\n🎓 Step 3: Model Training")
    model = CropRecommendationModel(model_type='random_forest')
    
    # Train model
    val_accuracy = model.train_model(X_train, y_train, X_val, y_val)
    
    # Optional: Hyperparameter tuning (uncomment if needed)
    # best_params = model.hyperparameter_tuning(X_train, y_train)
    
    # Step 4: Evaluation
    print("\n📊 Step 4: Model Evaluation")
    test_accuracy = model.evaluate_model(X_test, y_test, prep.target_encoder)
    
    # Feature importance
    feature_names = ['soil_texture', 'soil_ph', 'organic_matter', 
                     'drainage_status', 'rainfall_mm', 'avg_temperature', 'season']
    model.get_feature_importance(feature_names)
    
    # Step 5: Save Model
    print("\n💾 Step 5: Saving Model")
    model.save_model()
    
    print("\n" + "=" * 50)
    print("✅ TRAINING PIPELINE COMPLETED!")
    print("=" * 50)
    
    return model, prep


if __name__ == "__main__":
    model, prep = full_training_pipeline()