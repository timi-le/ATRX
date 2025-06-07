#!/usr/bin/env python3
"""
Training Pipeline Demonstration for FX AI-Quant Trading System.

This demo showcases the complete training pipeline including:
- Cross-validation with time-series awareness
- Hyperparameter optimization
- Model training (LSTM, CNN, XGBoost)
- Ensemble model creation
- Retraining scheduler with drift detection
- Model versioning and deployment

Run this script to see the training pipeline in action with synthetic data.
"""

import asyncio
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import structlog
from typing import List
import argparse

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from trainers.train_model import ProductionTrainingPipeline
from trainers.retraining_scheduler import RetrainingScheduler
from trainers.cv_utils import (
    TimeSeriesKFold, WalkForwardOptimizer, PurgedKFold,
    sharpe_ratio_score, validate_cv_setup
)
from models.predictor_interface import PredictionType

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


class TrainingPipelineDemo:
    """
    Comprehensive demonstration of the training pipeline.
    
    Shows all major features and capabilities of the production-grade
    ML training system for FX trading.
    """
    
    def __init__(self):
        """Initialize the demo."""
        self.logger = logger.bind(component="TrainingPipelineDemo")
        
        # Initialize training pipeline
        self.pipeline = ProductionTrainingPipeline(
            config_path="config/model_config.yaml"
        )
        
        # Initialize retraining scheduler
        self.scheduler = RetrainingScheduler(self.pipeline)
        
        self.logger.info("Training pipeline demo initialized")
    
    def demonstrate_cross_validation(self, X: np.ndarray, y: np.ndarray):
        """Demonstrate different cross-validation methods."""
        print("\n" + "="*60)
        print("CROSS-VALIDATION DEMONSTRATION")
        print("="*60)
        
        # 1. Time Series K-Fold
        print("\n1. Time Series K-Fold Cross-Validation")
        print("-" * 40)
        
        ts_cv = TimeSeriesKFold(n_splits=5, gap=10)
        splits = list(ts_cv.split(X, y))
        
        print(f"Number of splits: {len(splits)}")
        for i, (train_idx, test_idx) in enumerate(splits):
            print(f"  Fold {i+1}: Train={len(train_idx):4d} samples, Test={len(test_idx):3d} samples")
            print(f"           Train range: [{train_idx[0]:4d}, {train_idx[-1]:4d}]")
            print(f"           Test range:  [{test_idx[0]:4d}, {test_idx[-1]:4d}]")
            print(f"           Gap: {test_idx[0] - train_idx[-1] - 1:2d} samples")
        
        # Validate setup
        is_valid = validate_cv_setup(X, y, ts_cv, min_samples_per_fold=50)
        print(f"CV setup validation: {'✓ PASSED' if is_valid else '✗ FAILED'}")
        
        # 2. Walk-Forward Optimization
        print("\n2. Walk-Forward Optimization")
        print("-" * 40)
        
        wfo = WalkForwardOptimizer(
            train_window=500,
            test_window=100,
            step_size=50,
            gap=5
        )
        
        wf_splits = list(wfo.split(X, y))
        print(f"Number of walk-forward splits: {len(wf_splits)}")
        
        for i, (train_idx, test_idx) in enumerate(wf_splits[:3]):  # Show first 3
            print(f"  Split {i+1}: Train={len(train_idx):3d} samples [{train_idx[0]:4d}:{train_idx[-1]:4d}]")
            print(f"           Test={len(test_idx):3d} samples  [{test_idx[0]:4d}:{test_idx[-1]:4d}]")
        
        if len(wf_splits) > 3:
            print(f"  ... and {len(wf_splits) - 3} more splits")
        
        # 3. Purged K-Fold
        print("\n3. Purged K-Fold Cross-Validation")
        print("-" * 40)
        
        purged_cv = PurgedKFold(n_splits=5, purge_length=20, embargo_length=10)
        purged_splits = list(purged_cv.split(X, y))
        
        print(f"Number of purged splits: {len(purged_splits)}")
        for i, (train_idx, test_idx) in enumerate(purged_splits):
            print(f"  Fold {i+1}: Train={len(train_idx):4d} samples, Test={len(test_idx):3d} samples")
        
        # 4. Scoring Functions
        print("\n4. Financial Scoring Metrics")
        print("-" * 40)
        
        # Generate sample predictions
        y_sample = y[:100]
        y_pred_sample = y_sample + np.random.randn(100) * 0.01
        
        sharpe = sharpe_ratio_score(y_sample, y_pred_sample)
        print(f"Sharpe Ratio: {sharpe:.4f}")
        
        from trainers.cv_utils import information_ratio_score, calmar_ratio_score
        ir = information_ratio_score(y_sample, y_pred_sample)
        calmar = calmar_ratio_score(y_sample, y_pred_sample)
        
        print(f"Information Ratio: {ir:.4f}")
        print(f"Calmar Ratio: {calmar:.4f}")
    
    def demonstrate_data_generation(self):
        """Demonstrate synthetic data generation with different characteristics."""
        print("\n" + "="*60)
        print("SYNTHETIC DATA GENERATION")
        print("="*60)
        
        # Generate different types of data
        scenarios = [
            {
                "name": "Low Volatility Regime",
                "params": {"n_samples": 2000, "add_regime_changes": False, "noise_level": 0.005}
            },
            {
                "name": "High Volatility with Regime Changes",
                "params": {"n_samples": 2000, "add_regime_changes": True, "noise_level": 0.02}
            },
            {
                "name": "Classification Task",
                "params": {
                    "n_samples": 2000, 
                    "prediction_type": PredictionType.RETURN_CLASSIFICATION,
                    "noise_level": 0.01
                }
            }
        ]
        
        generated_data = {}
        
        for scenario in scenarios:
            print(f"\n{scenario['name']}:")
            print("-" * len(scenario['name']) + "-")
            
            X, y = self.pipeline.generate_synthetic_data(**scenario['params'])
            generated_data[scenario['name']] = (X, y)
            
            print(f"  Data shape: X={X.shape}, y={y.shape}")
            print(f"  Y statistics: mean={np.mean(y):.6f}, std={np.std(y):.6f}")
            print(f"  Y range: [{np.min(y):.6f}, {np.max(y):.6f}]")
            
            if scenario['params'].get('prediction_type') == PredictionType.RETURN_CLASSIFICATION:
                print(f"  Class distribution: {np.bincount(y.astype(int))}")
        
        return generated_data
    
    def demonstrate_preprocessing(self, X: np.ndarray, y: np.ndarray):
        """Demonstrate data preprocessing capabilities."""
        print("\n" + "="*60)
        print("DATA PREPROCESSING")
        print("="*60)
        
        print(f"Original data: X={X.shape}, y={y.shape}")
        print(f"Original Y stats: mean={np.mean(y):.6f}, std={np.std(y):.6f}")
        
        # Preprocess data
        X_processed, y_processed = self.pipeline.preprocess_data(X, y, fit_preprocessors=True)
        
        print(f"Processed data: X={X_processed.shape}, y={y_processed.shape}")
        print(f"Processed Y stats: mean={np.mean(y_processed):.6f}, std={np.std(y_processed):.6f}")
        
        # Show scaler information
        if "feature_scaler" in self.pipeline.scalers:
            scaler = self.pipeline.scalers["feature_scaler"]
            print(f"Feature scaler: {type(scaler).__name__}")
            
            # Show scaling statistics for first few features
            if hasattr(scaler, 'mean_'):
                print("Feature scaling statistics (first 5 features):")
                for i in range(min(5, len(scaler.mean_))):
                    print(f"  Feature {i}: mean={scaler.mean_[i]:.4f}, std={scaler.scale_[i]:.4f}")
        
        return X_processed, y_processed
    
    async def demonstrate_model_training(self, X: np.ndarray, y: np.ndarray, models_to_train: List[str] = None):
        """Demonstrate training different model types."""
        print("\n" + "="*60)
        print("MODEL TRAINING DEMONSTRATION")
        print("="*60)
        
        if models_to_train is None:
            models_to_train = ["xgboost", "lstm", "cnn"]  # Train all models by default
        
        results = []
        
        for model_name in models_to_train:
            if model_name.lower() == "xgboost":
                print("\n1. Training XGBoost Model")
                print("-" * 30)
                
                start_time = datetime.now()
                xgb_result = self.pipeline.train_xgboost_model(
                    X, y, optimize_hyperparameters=False
                )
                training_time = (datetime.now() - start_time).total_seconds()
                
                print(f"✓ XGBoost training completed in {training_time:.2f}s")
                print(f"  Best score: {xgb_result.best_score:.6f}")
                print(f"  Model saved: {Path(xgb_result.model_path).name}")
                print(f"  Metrics: {', '.join(f'{k}={v:.4f}' for k, v in list(xgb_result.metrics.items())[:3])}")
                
                if xgb_result.feature_importance:
                    top_features = sorted(
                        xgb_result.feature_importance.items(), 
                        key=lambda x: x[1], 
                        reverse=True
                    )[:5]
                    print(f"  Top features: {', '.join(f'{k}={v:.3f}' for k, v in top_features)}")
                
                results.append(xgb_result)
            
            elif model_name.lower() == "lstm":
                print("\n2. Training LSTM Model")
                print("-" * 30)
                print("  Note: LSTM training may take several minutes...")
                
                start_time = datetime.now()
                lstm_result = await self.pipeline.train_lstm_model(
                    X, y, optimize_hyperparameters=False
                )
                training_time = (datetime.now() - start_time).total_seconds()
                
                print(f"✓ LSTM training completed in {training_time:.2f}s")
                print(f"  Best score: {lstm_result.best_score:.6f}")
                print(f"  Model saved: {Path(lstm_result.model_path).name}")
                print(f"  Metrics: {', '.join(f'{k}={v:.4f}' for k, v in list(lstm_result.metrics.items())[:3])}")
                if lstm_result.onnx_path:
                    print(f"  ONNX exported: {Path(lstm_result.onnx_path).name}")
                
                results.append(lstm_result)
            
            elif model_name.lower() == "cnn":
                print("\n3. Training CNN Model")
                print("-" * 30)
                print("  Note: CNN training may take several minutes...")
                
                start_time = datetime.now()
                cnn_result = await self.pipeline.train_cnn_model(
                    X, y, optimize_hyperparameters=False
                )
                training_time = (datetime.now() - start_time).total_seconds()
                
                print(f"✓ CNN training completed in {training_time:.2f}s")
                print(f"  Best score: {cnn_result.best_score:.6f}")
                print(f"  Model saved: {Path(cnn_result.model_path).name}")
                print(f"  Metrics: {', '.join(f'{k}={v:.4f}' for k, v in list(cnn_result.metrics.items())[:3])}")
                if cnn_result.onnx_path:
                    print(f"  ONNX exported: {Path(cnn_result.onnx_path).name}")
                
                results.append(cnn_result)
        
        return results
    
    def demonstrate_retraining_scheduler(self, X: np.ndarray, y: np.ndarray):
        """Demonstrate retraining scheduler and drift detection."""
        print("\n" + "="*60)
        print("RETRAINING SCHEDULER & DRIFT DETECTION")
        print("="*60)
        
        # Set reference data for drift detection
        print("\n1. Setting Reference Data")
        print("-" * 30)
        
        # Use first 80% as reference
        split_idx = int(len(X) * 0.8)
        X_ref, y_ref = X[:split_idx], y[:split_idx]
        X_current, y_current = X[split_idx:], y[split_idx:]
        
        self.scheduler.set_reference_data(X_ref, y_ref)
        print(f"✓ Reference data set: {X_ref.shape[0]} samples")
        
        # Test drift detection
        print("\n2. Drift Detection")
        print("-" * 20)
        
        # Test with similar data (no drift)
        print("Testing with similar data (no drift expected):")
        drift_result_no = self.scheduler.check_data_drift(X_current)
        print(f"  Drift detected: {drift_result_no.drift_detected}")
        print(f"  Drift score: {drift_result_no.drift_score:.6f}")
        print(f"  Threshold: {drift_result_no.threshold:.6f}")
        
        # Test with drifted data
        print("\nTesting with artificially drifted data:")
        X_drifted = X_current + np.random.randn(*X_current.shape) * 0.5
        drift_result_yes = self.scheduler.check_data_drift(X_drifted)
        print(f"  Drift detected: {drift_result_yes.drift_detected}")
        print(f"  Drift score: {drift_result_yes.drift_score:.6f}")
        print(f"  Threshold: {drift_result_yes.threshold:.6f}")
        
        # Test retraining triggers
        print("\n3. Retraining Triggers")
        print("-" * 25)
        
        # Test performance degradation
        should_retrain, triggers = self.scheduler.should_retrain(
            current_score=0.5,  # Low score to trigger retraining
            current_data=X_drifted
        )
        
        print(f"Should retrain: {should_retrain}")
        print(f"Number of triggers: {len(triggers)}")
        
        for trigger in triggers:
            print(f"  - {trigger.trigger_type}: {trigger.reason}")
            if trigger.metrics:
                print(f"    Metrics: {trigger.metrics}")
        
        # Show scheduler status
        print("\n4. Scheduler Status")
        print("-" * 20)
        
        status = self.scheduler.get_status()
        for key, value in status.items():
            if key != "recent_triggers":  # Skip detailed triggers for brevity
                print(f"  {key}: {value}")
    
    def demonstrate_model_versioning(self):
        """Demonstrate model version management."""
        print("\n" + "="*60)
        print("MODEL VERSION MANAGEMENT")
        print("="*60)
        
        version_manager = self.scheduler.version_manager
        
        # Add some dummy versions
        print("\n1. Adding Model Versions")
        print("-" * 30)
        
        versions_data = [
            {"score": 0.75, "validation": 0.72, "name": "baseline"},
            {"score": 0.82, "validation": 0.80, "name": "improved"},
            {"score": 0.78, "validation": 0.76, "name": "experimental"}
        ]
        
        for i, data in enumerate(versions_data):
            version = version_manager.add_version(
                model_path=f"/tmp/model_{data['name']}.joblib",
                performance_score=data['score'],
                validation_score=data['validation'],
                training_data_hash=f"hash_{i}",
                config_hash=f"config_{i}"
            )
            print(f"✓ Added version {version.version}: {data['name']} (score: {data['score']:.3f})")
        
        # Show version management
        print(f"\nTotal versions: {len(version_manager.versions)}")
        
        # Get best version
        best = version_manager.get_best_version("validation_score")
        if best:
            print(f"Best version: {best.version} (validation score: {best.validation_score:.3f})")
        
        # Activate best version
        if best:
            version_manager.activate_version(best.version)
            active = version_manager.get_active_version()
            print(f"Active version: {active.version if active else 'None'}")
    
    async def run_full_demo(self, models_to_train: List[str] = None):
        """Run the complete training pipeline demonstration."""
        print("🚀 FX AI-Quant Training Pipeline Demo")
        print("=" * 60)
        print("This demo showcases the production-grade ML training pipeline")
        print("with time-series cross-validation, regularization, and retraining.")
        print("=" * 60)
        
        if models_to_train is None:
            models_to_train = ["xgboost", "lstm", "cnn"]  # Train all models by default
        
        print(f"Models to train: {', '.join(models_to_train)}")
        if any(model in ["lstm", "cnn"] for model in models_to_train):
            print("⚠️  Note: LSTM and CNN training may take several minutes each")
        
        try:
            # 1. Generate synthetic data
            print("\n📊 Generating synthetic FX data...")
            generated_data = self.demonstrate_data_generation()
            
            # Use the high volatility data for main demo
            X, y = generated_data["High Volatility with Regime Changes"]
            
            # 2. Demonstrate cross-validation
            self.demonstrate_cross_validation(X, y)
            
            # 3. Demonstrate preprocessing
            X_processed, y_processed = self.demonstrate_preprocessing(X, y)
            
            # 4. Demonstrate model training
            results = await self.demonstrate_model_training(X_processed, y_processed, models_to_train)
            
            # 5. Demonstrate retraining scheduler
            self.demonstrate_retraining_scheduler(X_processed, y_processed)
            
            # 6. Demonstrate model versioning
            self.demonstrate_model_versioning()
            
            # 7. Summary
            print("\n" + "="*60)
            print("DEMO SUMMARY")
            print("="*60)
            print("✓ Cross-validation methods demonstrated")
            print("✓ Synthetic data generation with regime changes")
            print("✓ Data preprocessing and scaling")
            print(f"✓ Model training with {', '.join([r.model_name for r in results])}")
            print("✓ Drift detection and retraining triggers")
            print("✓ Model version management")
            print("\nThe training pipeline is ready for production use!")
            print("Key features:")
            print("  - Time-series aware cross-validation")
            print("  - Automatic hyperparameter optimization")
            print("  - Regularization and overfitting prevention")
            print("  - Automated retraining with drift detection")
            print("  - Model versioning and rollback capabilities")
            print("  - ONNX export for deployment")
            
            if results:
                print(f"\nModels trained: {len(results)}")
                for result in results:
                    print(f"  - {result.model_name}: Score={result.best_score:.6f}, Time={result.training_time:.2f}s")
                
                best_model = max(results, key=lambda x: x.best_score)
                print(f"\nBest model: {best_model.model_name} (Score: {best_model.best_score:.6f})")
            
        except Exception as e:
            self.logger.error("Demo failed", error=str(e), exc_info=True)
            print(f"\n❌ Demo failed: {e}")
            raise


async def main():
    """Main entry point for the demo."""
    parser = argparse.ArgumentParser(description="FX AI-Quant Training Pipeline Demo")
    parser.add_argument(
        "--models", 
        nargs="+", 
        default=["xgboost"],
        choices=["xgboost", "lstm", "cnn", "all"],
        help="Models to train in the demo (default: xgboost for speed)"
    )
    parser.add_argument(
        "--full", 
        action="store_true",
        help="Train all models (equivalent to --models all)"
    )
    
    args = parser.parse_args()
    
    # Handle model selection
    if args.full or "all" in args.models:
        models_to_train = ["xgboost", "lstm", "cnn"]
    else:
        models_to_train = args.models
    
    print(f"Running demo with models: {', '.join(models_to_train)}")
    if "lstm" in models_to_train or "cnn" in models_to_train:
        print("⚠️  Note: LSTM and CNN training may take several minutes each")
    
    demo = TrainingPipelineDemo()
    await demo.run_full_demo(models_to_train)


if __name__ == "__main__":
    asyncio.run(main())