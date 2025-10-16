# recommendation/ml_utils/gpu_train_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, f1_score
import joblib
import os
import time
from pathlib import Path

# GPU-accelerated imports
try:
    import torch
    import xgboost as xgb
    from xgboost import XGBClassifier
    GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    GPU_AVAILABLE = False
    print("⚠️  GPU libraries not available. Install with: pip install torch xgboost")

from .data_preparation import CropDataPreparation


class GPUCropRecommendationModel:
    """
    GPU-Accelerated ML model for crop recommendation
    """
    
    def __init__(self, backend='xgboost', use_gpu=True):
        """
        Initialize GPU-accelerated model
        
        Args:
            backend: 'xgboost' or 'pytorch'
            use_gpu: Whether to use GPU acceleration
        """
        self.backend = backend
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.model = None
        self.feature_importance = None
        self.training_history = []
        
        if self.use_gpu:
            self._detect_gpu()
        else:
            print("🔧 GPU acceleration disabled - using CPU")
    
    def _detect_gpu(self):
        """Detect and display GPU information"""
        if not GPU_AVAILABLE:
            print("❌ No GPU detected")
            self.use_gpu = False
            return
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            cuda_version = torch.version.cuda
            
            print(f"✅ GPU Available: {gpu_name}")
            print(f"   CUDA Version: {cuda_version}")
            print(f"   GPU Memory: {gpu_memory:.2f} GB")
        else:
            print("❌ CUDA not available")
            self.use_gpu = False
    
    def create_model(self, n_classes):
        """
        Create GPU-accelerated model
        
        Args:
            n_classes: Number of crop classes
        """
        print("🎯 Creating Model...")
        
        if self.backend == 'xgboost':
            if self.use_gpu:
                # XGBoost GPU configuration
                tree_method = 'gpu_hist'
                device = 'cuda:0'
                print(f"🚀 Using XGBoost GPU acceleration (device={device})")
            else:
                tree_method = 'hist'
                device = 'cpu'
                print("🔧 Using XGBoost CPU mode")
            
            self.model = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                tree_method=tree_method,
                device=device,
                eval_metric='mlogloss',  # Move eval_metric here
                random_state=42,
                n_jobs=-1,
                verbosity=1
            )
        
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
        
        return self.model
    
    def train_with_cv(self, X_train, y_train, X_val, y_val, n_folds=5):
        """
        Train model with cross-validation
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            n_folds: Number of CV folds
        """
        print(f"📊 Performing {n_folds}-Fold Cross-Validation...")
        
        # Get number of classes
        n_classes = len(np.unique(y_train))
        
        # Create model
        self.create_model(n_classes)
        
        # Cross-validation
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        cv_scores = []
        
        start_time = time.time()
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
            print(f"   📂 Fold {fold}")
            
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
            
            # Train on fold
            self.model.fit(
                X_fold_train, 
                y_fold_train,
                eval_set=[(X_fold_val, y_fold_val)],
                verbose=False
            )
            
            # Evaluate fold
            y_pred = self.model.predict(X_fold_val)
            fold_accuracy = accuracy_score(y_fold_val, y_pred)
            cv_scores.append(fold_accuracy)
            
            print(f"      Accuracy: {fold_accuracy:.4f}")
        
        cv_time = time.time() - start_time
        
        print(f"\n📈 Cross-Validation Results:")
        print(f"   Mean Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        print(f"   Training Time: {cv_time:.2f} seconds")
        
        # Final training on full training set
        print("\n🎓 Training Final Model on Full Training Set...")
        
        start_time = time.time()
        
        self.model.fit(
            X_train, 
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        training_time = time.time() - start_time
        
        # Validation accuracy
        y_val_pred = self.model.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        
        print(f"✅ Training Completed!")
        print(f"   Validation Accuracy: {val_accuracy:.4f}")
        print(f"   Training Time: {training_time:.2f} seconds")
        
        # Store feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = self.model.feature_importances_
        
        # Store training history
        self.training_history.append({
            'cv_scores': cv_scores,
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'val_accuracy': val_accuracy,
            'training_time': training_time
        })
        
        return val_accuracy
    
    def evaluate_model(self, X_test, y_test, target_encoder):
        """
        Comprehensive model evaluation
        
        Args:
            X_test: Test features
            y_test: Test labels
            target_encoder: Label encoder for crop names
        """
        print("\n📈 Evaluating Model on Test Set...")
        
        start_time = time.time()
        y_pred = self.model.predict(X_test)
        inference_time = time.time() - start_time
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"🎯 Test Accuracy: {accuracy:.4f}")
        print(f"🎯 F1 Score (Weighted): {f1:.4f}")
        print(f"⚡ Inference Time: {inference_time:.4f} seconds")
        print(f"⚡ Samples/Second: {len(X_test)/inference_time:.0f}")
        
        # Top crops report (showing only crops with reasonable support)
        crop_names = target_encoder.classes_
        report = classification_report(
            y_test, 
            y_pred, 
            target_names=crop_names,
            zero_division=0,
            output_dict=True
        )
        
        # Filter and display top performing crops
        print("\n📋 Classification Report (Top Crops by Support):")
        crop_scores = []
        for crop in crop_names:
            if crop in report and report[crop]['support'] >= 10:
                crop_scores.append({
                    'crop': crop,
                    'precision': report[crop]['precision'],
                    'recall': report[crop]['recall'],
                    'f1': report[crop]['f1-score'],
                    'support': report[crop]['support']
                })
        
        # Sort by support and display top 20
        crop_scores.sort(key=lambda x: x['support'], reverse=True)
        for crop in crop_scores[:20]:
            print(f"   {crop['crop']:<25} P:{crop['precision']:.3f} R:{crop['recall']:.3f} "
                  f"F1:{crop['f1']:.3f} (n={crop['support']:.0f})")
        
        return accuracy, f1
    
    def get_feature_importance(self, feature_names):
        """
        Get and display feature importance
        
        Args:
            feature_names: List of feature names
        """
        if self.feature_importance is None:
            print("❌ Feature importance not available")
            return None
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importance
        }).sort_values('importance', ascending=False)
        
        print("\n🔍 Feature Importance:")
        for _, row in importance_df.iterrows():
            bar_length = int(row['importance'] * 50)
            bar = '█' * bar_length
            print(f"   {row['feature']:<20} {bar} {row['importance']:.4f}")
        
        return importance_df
    
    def save_model(self, save_path='recommendation/ml_models/crop_recommendation_model.pkl'):
        """
        Save trained model
        
        Args:
            save_path: Path to save the model
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save model
        joblib.dump(self.model, save_path)
        
        # Save training history
        history_path = save_path.replace('.pkl', '_history.pkl')
        joblib.dump(self.training_history, history_path)
        
        print(f"✅ Model saved to {save_path}")
        print(f"✅ Training history saved to {history_path}")
    
    def predict_with_confidence(self, X, top_k=10):
        """
        Predict with probability scores
        
        Args:
            X: Input features
            top_k: Number of top predictions to return
        """
        probabilities = self.model.predict_proba(X)
        
        # Get top K predictions for each sample
        results = []
        for probs in probabilities:
            top_indices = np.argsort(probs)[-top_k:][::-1]
            top_probs = probs[top_indices]
            results.append((top_indices, top_probs))
        
        return results


def gpu_training_pipeline(backend='xgboost', n_samples=20000, use_gpu=True):
    """
    Complete GPU-accelerated training pipeline
    
    Args:
        backend: 'xgboost' or 'pytorch'
        n_samples: Number of synthetic samples to generate
        use_gpu: Whether to use GPU acceleration
    """
    print("=" * 70)
    print("🌾 GPU-ACCELERATED CROP RECOMMENDATION TRAINING")
    print("=" * 70)
    
    # Step 1: Data Preparation
    print("\n📂 Step 1: Data Preparation")
    prep = CropDataPreparation()
    X, y, df = prep.load_and_preprocess_data()
    print(f"✅ Loaded {len(df)} crops from database")
    
    # Step 2: Generate Synthetic Data
    print(f"\n🔄 Step 2: Generating {n_samples} Training Samples")
    
    if use_gpu and GPU_AVAILABLE:
        # Display GPU info
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            cuda_version = torch.version.cuda
            print(f"✅ GPU Available: {gpu_name}")
            print(f"   CUDA Version: {cuda_version}")
            print(f"   GPU Memory: {gpu_memory:.2f} GB")
    
    synthetic_df = prep.generate_synthetic_training_data(df, n_samples=n_samples)
    print(f"✅ Generated {len(synthetic_df)} synthetic samples")
    
    X_train_full, y_train_full = prep.prepare_training_data(synthetic_df)
    
    # Save preprocessors
    prep.save_preprocessors()
    
    # Step 3: Train-Test Split
    print("\n✂️  Step 3: Train-Test Split")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_train_full, y_train_full,
        test_size=0.3,
        random_state=42,
        stratify=y_train_full
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        random_state=42,
        stratify=y_temp
    )
    
    print(f"Training: {len(X_train)} | Validation: {len(X_val)} | Test: {len(X_test)}")
    
    # Step 4: Model Training
    print("\n🎓 Step 4: Model Training")
    model = GPUCropRecommendationModel(backend=backend, use_gpu=use_gpu)
    
    # Train with cross-validation
    val_accuracy = model.train_with_cv(X_train, y_train, X_val, y_val, n_folds=5)
    
    # Step 5: Evaluation
    print("\n📊 Step 5: Model Evaluation")
    test_accuracy, f1 = model.evaluate_model(X_test, y_test, prep.target_encoder)
    
    # Feature importance
    feature_names = ['soil_texture', 'soil_ph', 'organic_matter',
                     'drainage_status', 'rainfall_mm', 'avg_temperature', 'season']
    model.get_feature_importance(feature_names)
    
    # Step 6: Save Model
    print("\n💾 Step 6: Saving Model")
    model.save_model()
    
    print("\n" + "=" * 70)
    print("✅ TRAINING PIPELINE COMPLETED!")
    print(f"   Final Test Accuracy: {test_accuracy:.4f}")
    print(f"   Final F1 Score: {f1:.4f}")
    print("=" * 70)
    
    return model, prep, test_accuracy


if __name__ == "__main__":
    # Train with GPU acceleration
    model, prep, accuracy = gpu_training_pipeline(
        backend='xgboost',
        n_samples=30000,
        use_gpu=True
    )