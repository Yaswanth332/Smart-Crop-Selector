from django.core.management.base import BaseCommand
from recommendation.ml_utils.gpu_train_model import gpu_training_pipeline

class Command(BaseCommand):
    help = 'Train crop recommendation model with GPU acceleration'

    def add_arguments(self, parser):
        parser.add_argument(
            '--gpu',
            action='store_true',
            default=True,
            help='Use GPU acceleration (default: True)'
        )
        parser.add_argument(
            '--backend',
            type=str,
            default='xgboost',
            choices=['xgboost', 'lightgbm', 'catboost'],
            help='GPU backend to use'
        )
        parser.add_argument(
            '--samples',
            type=int,
            default=30000,
            help='Number of synthetic samples to generate'
        )

    def handle(self, *args, **options):
        use_gpu = options['gpu']
        backend = options['backend']
        n_samples = options['samples']
        
        self.stdout.write(f'Training with GPU: {use_gpu}')
        self.stdout.write(f'Backend: {backend}')
        self.stdout.write(f'Samples: {n_samples}')
        
        try:
            model, prep, accuracy = gpu_training_pipeline(
                use_gpu=use_gpu,
                backend=backend,
                n_samples=n_samples
            )
            
            self.stdout.write(self.style.SUCCESS(f'\n✅ Training completed!'))
            self.stdout.write(self.style.SUCCESS(f'Final Accuracy: {accuracy:.2%}'))
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'\n❌ Training failed: {str(e)}'))
            import traceback
            self.stdout.write(traceback.format_exc())