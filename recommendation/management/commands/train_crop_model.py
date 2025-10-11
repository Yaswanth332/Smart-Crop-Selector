# recommendation/management/commands/train_crop_model.py

from django.core.management.base import BaseCommand
from django.conf import settings
import sys
import os

class Command(BaseCommand):
    help = 'Train crop recommendation ML model'

    def add_arguments(self, parser):
        parser.add_argument(
            '--model-type',
            type=str,
            default='random_forest',
            choices=['random_forest', 'gradient_boosting'],
            help='Type of ML model to train'
        )
        parser.add_argument(
            '--samples',
            type=int,
            default=15000,
            help='Number of synthetic training samples to generate'
        )
        parser.add_argument(
            '--tune',
            action='store_true',
            help='Perform hyperparameter tuning'
        )

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('=' * 60))
        self.stdout.write(self.style.SUCCESS('🌾 CROP RECOMMENDATION ML TRAINING'))
        self.stdout.write(self.style.SUCCESS('=' * 60))
        
        # Add project root to Python path
        sys.path.append(str(settings.BASE_DIR))
        
        try:
            from recommendation.ml_utils.data_preparation import CropDataPreparation
            from recommendation.ml_utils.train_model import CropRecommendationModel
            from sklearn.model_selection import train_test_split
            
            # Step 1: Data Preparation
            self.stdout.write('\n📂 Step 1: Data Preparation')
            csv_path = os.path.join(settings.BASE_DIR, 'recommendation', 'data', 'crop_data.csv')
            
            if not os.path.exists(csv_path):
                self.stdout.write(self.style.ERROR(f'❌ Crop data file not found: {csv_path}'))
                return
            
            prep = CropDataPreparation(csv_path=csv_path)
            X, y, df = prep.load_and_preprocess_data()
            self.stdout.write(self.style.SUCCESS(f'✅ Loaded {len(df)} crops from database'))
            
            # Generate synthetic data
            self.stdout.write(f'🔄 Generating {options["samples"]} synthetic training samples...')
            synthetic_df = prep.generate_synthetic_training_data(df, n_samples=options['samples'])
            X_train_full, y_train_full = prep.prepare_training_data(synthetic_df)
            self.stdout.write(self.style.SUCCESS(f'✅ Generated {len(synthetic_df)} samples'))
            
            # Save preprocessors
            model_dir = os.path.join(settings.BASE_DIR, 'recommendation', 'ml_models')
            prep.save_preprocessors(save_dir=model_dir)
            
            # Step 2: Train-Test Split
            self.stdout.write('\n✂️  Step 2: Train-Test Split')
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
            
            self.stdout.write(f'Training: {len(X_train)} | Validation: {len(X_val)} | Test: {len(X_test)}')
            
            # Step 3: Model Training
            self.stdout.write('\n🎓 Step 3: Model Training')
            model = CropRecommendationModel(model_type=options['model_type'])
            
            if options['tune']:
                self.stdout.write('🔧 Performing hyperparameter tuning...')
                best_params = model.hyperparameter_tuning(X_train, y_train)
                self.stdout.write(self.style.SUCCESS(f'✅ Best parameters: {best_params}'))
            else:
                val_accuracy = model.train_model(X_train, y_train, X_val, y_val)
            
            # Step 4: Evaluation
            self.stdout.write('\n📊 Step 4: Model Evaluation')
            test_accuracy = model.evaluate_model(X_test, y_test, prep.target_encoder)
            
            # Feature importance
            feature_names = ['soil_texture', 'soil_ph', 'organic_matter', 
                           'drainage_status', 'rainfall_mm', 'avg_temperature', 'season']
            importance_df = model.get_feature_importance(feature_names)
            
            # Step 5: Save Model
            self.stdout.write('\n💾 Step 5: Saving Model')
            save_path = os.path.join(model_dir, 'crop_recommendation_model.pkl')
            model.save_model(save_path=save_path)
            
            self.stdout.write('\n' + '=' * 60)
            self.stdout.write(self.style.SUCCESS('✅ TRAINING COMPLETED SUCCESSFULLY!'))
            self.stdout.write(self.style.SUCCESS(f'📊 Final Test Accuracy: {test_accuracy:.2%}'))
            self.stdout.write(self.style.SUCCESS(f'💾 Model saved to: {save_path}'))
            self.stdout.write('=' * 60)
            
        except ImportError as e:
            self.stdout.write(self.style.ERROR(f'❌ Import error: {e}'))
            self.stdout.write(self.style.ERROR('Please ensure all required packages are installed:'))
            self.stdout.write('  pip install scikit-learn pandas numpy joblib')
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'❌ Training failed: {e}'))
            import traceback
            self.stdout.write(traceback.format_exc())