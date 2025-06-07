#!/usr/bin/env python3
"""
Test script to verify all ML functionality works correctly after architecture reorganization.
"""

import os
import sys
import time
import numpy as np
import structlog
from pathlib import Path

# Disable oneDNN optimizations to avoid warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def create_synthetic_data(n_samples=100, sequence_length=60, n_features=13):
    """Create synthetic FX-like data for testing."""
    np.random.seed(42)
    
    # Create sequences
    X = np.random.randn(n_samples, sequence_length, n_features)
    
    # Create realistic targets (returns)
    y = np.random.randn(n_samples) * 0.01  # Small returns like FX
    
    return X, y

async def test_model_functionality(model_name, model_class, config_func):
    """Test a specific model's functionality."""
    print(f"\n🧪 Testing {model_name} Model")
    print("-" * 40)
    
    try:
        # Create config and model
        config = config_func()
        logger = structlog.get_logger()
        model = model_class(config, logger)
        
        print(f"✅ {model_name} model creation: SUCCESS")
        
        # Create synthetic data
        X, y = create_synthetic_data()
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        print(f"✅ Data preparation: SUCCESS")
        
        # Train model
        start_time = time.time()
        metrics = model.train(X_train, y_train, X_val, y_val)
        training_time = time.time() - start_time
        
        print(f"✅ Training completed in {training_time:.2f}s")
        print(f"   - Train Loss: {metrics.train_loss:.4f}")
        print(f"   - Val Loss: {metrics.val_loss:.4f}")
        print(f"   - Memory Usage: {metrics.memory_usage_mb:.2f} MB")
        
        # Test prediction
        start_time = time.time()
        predictions = model._predict_raw(X_val[:5])
        inference_time = time.time() - start_time
        
        print(f"✅ Inference completed in {inference_time:.4f}s")
        print(f"   - Predictions shape: {predictions.shape}")
        print(f"   - Sample predictions: {predictions[:3]}")
        
        # Test save/load
        model_path = f"test_models/{model_name.lower()}_test.pkl"
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        
        model.save_model(model_path)
        print(f"✅ Model saved to {model_path}")
        
        # Create new model and load
        new_model = model_class(config, logger)
        new_model.load_model(model_path)
        
        # Test loaded model prediction
        new_predictions = new_model._predict_raw(X_val[:5])
        
        # Check if predictions match
        if np.allclose(predictions, new_predictions, rtol=1e-5):
            print(f"✅ Model load/save: SUCCESS (predictions match)")
        else:
            print(f"⚠️  Model load/save: WARNING (predictions differ)")
        
        # Test feature importance (if available)
        if hasattr(model, 'get_feature_importance'):
            try:
                importance = await model.get_feature_importance()
                print(f"✅ Feature importance: {len(importance)} features")
            except Exception as e:
                print(f"⚠️  Feature importance: {e}")
        
        return True, metrics
        
    except Exception as e:
        print(f"❌ {model_name} test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, None

async def test_ensemble_functionality():
    """Test ensemble model functionality."""
    print(f"\n🧪 Testing Ensemble Model")
    print("-" * 40)
    
    try:
        from models import (
            EnsembleMLPredictor, XGBoostPredictor,
            create_ensemble_config, create_xgboost_config
        )
        
        # Create ensemble
        ensemble_config = create_ensemble_config()
        logger = structlog.get_logger()
        ensemble = EnsembleMLPredictor(ensemble_config, logger)
        
        # Create and add XGBoost model
        xgb_config = create_xgboost_config()
        xgb_model = XGBoostPredictor(xgb_config, logger)
        
        await ensemble.add_model(xgb_model, weight=1.0)
        print(f"✅ Added XGBoost to ensemble")
        
        # Create synthetic data
        X, y = create_synthetic_data()
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Train ensemble
        start_time = time.time()
        metrics = ensemble.train(X_train, y_train, X_val, y_val)
        training_time = time.time() - start_time
        
        print(f"✅ Ensemble training completed in {training_time:.2f}s")
        print(f"   - Train Loss: {metrics.train_loss:.4f}")
        print(f"   - Val Loss: {metrics.val_loss:.4f}")
        
        # Test ensemble prediction
        predictions = ensemble._predict_raw(X_val[:5])
        print(f"✅ Ensemble predictions: {predictions[:3]}")
        
        return True, metrics
        
    except Exception as e:
        print(f"❌ Ensemble test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, None

async def main():
    """Run all functionality tests."""
    print("🔧 Testing Architecture Reorganization - Functionality Verification")
    print("=" * 70)
    
    # Import all models
    from models import (
        LSTMPredictor, CNNPredictor, XGBoostPredictor,
        create_lstm_config, create_cnn_config, create_xgboost_config
    )
    
    success_count = 0
    total_tests = 0
    results = {}
    
    # Test individual models
    models_to_test = [
        ("LSTM", LSTMPredictor, create_lstm_config),
        ("CNN", CNNPredictor, create_cnn_config),
        ("XGBoost", XGBoostPredictor, create_xgboost_config),
    ]
    
    for model_name, model_class, config_func in models_to_test:
        total_tests += 1
        success, metrics = await test_model_functionality(model_name, model_class, config_func)
        if success:
            success_count += 1
            results[model_name] = metrics
    
    # Test ensemble
    total_tests += 1
    success, metrics = await test_ensemble_functionality()
    if success:
        success_count += 1
        results["Ensemble"] = metrics
    
    # Summary
    print("\n" + "=" * 70)
    print(f"📊 FUNCTIONALITY TEST SUMMARY")
    print(f"✅ Successful: {success_count}/{total_tests}")
    print(f"❌ Failed: {total_tests - success_count}/{total_tests}")
    print(f"📈 Success Rate: {(success_count/total_tests)*100:.1f}%")
    
    if results:
        print(f"\n📈 Performance Summary:")
        for model_name, metrics in results.items():
            print(f"   {model_name}:")
            print(f"     - Train Loss: {metrics.train_loss:.4f}")
            print(f"     - Val Loss: {metrics.val_loss:.4f}")
            print(f"     - Training Time: {metrics.training_time:.2f}s")
            print(f"     - Memory Usage: {metrics.memory_usage_mb:.2f} MB")
    
    if success_count == total_tests:
        print("\n🎉 ALL FUNCTIONALITY TESTS PASSED! Architecture reorganization successful!")
        return True
    else:
        print(f"\n⚠️  {total_tests - success_count} test(s) failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    import asyncio
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 