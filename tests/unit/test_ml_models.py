#!/usr/bin/env python3
"""
Test Script for ML Models in FX AI-Quant Trading System.

This script tests the ML model implementations to ensure they work correctly
with synthetic data and can train, predict, save, and load models.
"""

import os
# Disable oneDNN optimizations to avoid pooling issues
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import asyncio
import sys
import time
from pathlib import Path
import numpy as np
import structlog

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models.predictor_interface import PredictionType, ModelType
from models import (
    get_available_models, create_model,
    LSTM_AVAILABLE, CNN_AVAILABLE, ENSEMBLE_AVAILABLE
)

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
        structlog.dev.ConsoleRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


def generate_test_data(n_samples=1000, sequence_length=60, n_features=13):
    """Generate synthetic test data."""
    logger.info("Generating test data", n_samples=n_samples, sequence_length=sequence_length, n_features=n_features)
    
    np.random.seed(42)
    
    # Generate features with some autocorrelation
    features = []
    returns = []
    
    for i in range(n_samples + sequence_length):
        if i == 0:
            feature_vec = np.random.randn(n_features) * 0.1
            ret = np.random.randn() * 0.01
        else:
            # Add autocorrelation
            feature_vec = 0.8 * features[-1] + 0.2 * np.random.randn(n_features) * 0.1
            ret = 0.1 * returns[-1] + 0.9 * np.random.randn() * 0.01
        
        features.append(feature_vec)
        returns.append(ret)
    
    features = np.array(features)
    returns = np.array(returns)
    
    # Create sequences
    X_sequences = []
    y_targets = []
    
    for i in range(sequence_length, len(features)):
        X_sequences.append(features[i-sequence_length:i])
        y_targets.append(returns[i])
    
    X = np.array(X_sequences)
    y = np.array(y_targets)
    
    logger.info("Test data generated", X_shape=X.shape, y_shape=y.shape)
    return X, y


async def test_model_basic_functionality(model_type: str):
    """Test basic model functionality."""
    logger.info(f"Testing {model_type} model basic functionality")
    
    try:
        # Import the specific model
        if model_type == "lstm" and LSTM_AVAILABLE:
            from models.lstm_model import create_lstm_config
            config = create_lstm_config(
                prediction_type=PredictionType.RETURN_REGRESSION,
                sequence_length=20,  # Smaller for faster testing
                features_dim=5,
                epochs=2,  # Very few epochs for testing
                batch_size=16
            )
        elif model_type == "cnn" and CNN_AVAILABLE:
            from models.cnn_model import create_cnn_config
            config = create_cnn_config(
                prediction_type=PredictionType.RETURN_REGRESSION,
                sequence_length=20,
                features_dim=5,
                epochs=2,
                batch_size=16
            )
        elif model_type == "xgboost" and ENSEMBLE_AVAILABLE:
            from models.ensemble_model import create_xgboost_config
            config = create_xgboost_config(
                prediction_type=PredictionType.RETURN_REGRESSION,
                n_estimators=10,  # Few estimators for testing
                max_depth=3
            )
        else:
            logger.warning(f"Model type {model_type} not available, skipping")
            return False
        
        # Create model
        model = create_model(model_type, config)
        model.feature_names = [f"feature_{i}" for i in range(config.features_dim)]
        
        # Generate small test dataset
        X, y = generate_test_data(n_samples=200, sequence_length=20, n_features=5)
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        logger.info(f"Training {model_type} model")
        start_time = time.time()
        
        # Train model
        metrics = model.train(X_train, y_train)
        
        training_time = time.time() - start_time
        logger.info(f"{model_type} training completed", 
                   train_loss=metrics.train_loss, 
                   training_time=training_time)
        
        # Test prediction
        logger.info(f"Testing {model_type} prediction")
        predictions = model._predict_raw(X_test[:5])
        logger.info(f"{model_type} predictions", predictions=predictions[:3].tolist())
        
        # Test async prediction interface
        from core.interfaces.ml_interfaces import Features
        from datetime import datetime
        
        test_features = Features(
            symbol="EURUSD",
            timestamp=datetime.now(),
            features={f"feature_{i}": float(X_test[0, -1, i]) for i in range(min(config.features_dim, X_test.shape[2]))},
            feature_names=[f"feature_{i}" for i in range(min(config.features_dim, X_test.shape[2]))]
        )
        
        prediction = await model.predict(test_features)
        logger.info(f"{model_type} async prediction", 
                   prediction=prediction.prediction, 
                   confidence=prediction.confidence)
        
        # Test model info
        model_info = model.get_model_info()
        logger.info(f"{model_type} model info", model_type=model_info["model_type"])
        
        # Test feature importance
        importance = await model.get_feature_importance()
        logger.info(f"{model_type} feature importance", importance=importance)
        
        # Test save/load (basic test)
        test_dir = Path("test_models")
        test_dir.mkdir(exist_ok=True)
        
        model_path = test_dir / f"test_{model_type}_model"
        model.save_model(str(model_path))
        logger.info(f"{model_type} model saved")
        
        # Create new model instance and load
        new_model = create_model(model_type, config)
        new_model.load_model(str(model_path))
        logger.info(f"{model_type} model loaded")
        
        # Test that loaded model gives same predictions
        new_predictions = new_model._predict_raw(X_test[:5])
        prediction_diff = np.abs(predictions - new_predictions)
        max_diff = np.max(prediction_diff)
        
        if max_diff < 1e-5:
            logger.info(f"{model_type} save/load test passed", max_diff=max_diff)
        else:
            logger.warning(f"{model_type} save/load test failed", max_diff=max_diff)
        
        logger.info(f"✅ {model_type} model test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ {model_type} model test failed", error=str(e), exc_info=True)
        return False


async def test_ensemble_model():
    """Test ensemble model functionality."""
    if not ENSEMBLE_AVAILABLE:
        logger.warning("Ensemble models not available, skipping")
        return False
    
    logger.info("Testing ensemble model")
    
    try:
        from models.ensemble_model import EnsembleMLPredictor, create_ensemble_config
        from models.lstm_model import LSTMPredictor, create_lstm_config
        from models.cnn_model import CNNPredictor, create_cnn_config
        from models.ensemble_model import XGBoostPredictor, create_xgboost_config
        
        # Generate test data
        X, y = generate_test_data(n_samples=300, sequence_length=20, n_features=5)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Create ensemble
        ensemble_config = create_ensemble_config(
            prediction_type=PredictionType.RETURN_REGRESSION,
            ensemble_method='weighted_average'
        )
        ensemble = EnsembleMLPredictor(ensemble_config, logger)
        
        # Create and add individual models
        models_to_add = []
        
        if LSTM_AVAILABLE:
            lstm_config = create_lstm_config(
                prediction_type=PredictionType.RETURN_REGRESSION,
                sequence_length=20, features_dim=5, epochs=2, batch_size=16
            )
            lstm_model = LSTMPredictor(lstm_config, logger)
            lstm_model.feature_names = [f"feature_{i}" for i in range(5)]
            lstm_model.train(X_train, y_train)
            models_to_add.append(("lstm", lstm_model))
        
        if CNN_AVAILABLE:
            cnn_config = create_cnn_config(
                prediction_type=PredictionType.RETURN_REGRESSION,
                sequence_length=20, features_dim=5, epochs=2, batch_size=16
            )
            cnn_model = CNNPredictor(cnn_config, logger)
            cnn_model.feature_names = [f"feature_{i}" for i in range(5)]
            cnn_model.train(X_train, y_train)
            models_to_add.append(("cnn", cnn_model))
        
        # Add XGBoost
        xgb_config = create_xgboost_config(
            prediction_type=PredictionType.RETURN_REGRESSION,
            n_estimators=10, max_depth=3
        )
        xgb_model = XGBoostPredictor(xgb_config, logger)
        xgb_model.feature_names = [f"feature_{i}" for i in range(20 * 5)]  # Flattened
        xgb_model.train(X_train, y_train)
        models_to_add.append(("xgboost", xgb_model))
        
        # Add models to ensemble
        for name, model in models_to_add:
            await ensemble.add_model(model, weight=1.0)
            logger.info(f"Added {name} to ensemble")
        
        # Train ensemble (meta-model)
        logger.info("Training ensemble")
        ensemble_metrics = ensemble.train(X_train, y_train)
        logger.info("Ensemble training completed", train_loss=ensemble_metrics.train_loss)
        
        # Test ensemble prediction
        ensemble_pred = ensemble._predict_raw(X_test[:5])
        logger.info("Ensemble predictions", predictions=ensemble_pred[:3].tolist())
        
        # Test feature importance
        ensemble_importance = await ensemble.get_feature_importance()
        logger.info("Ensemble feature importance", importance=ensemble_importance)
        
        logger.info("✅ Ensemble model test completed successfully")
        return True
        
    except Exception as e:
        logger.error("❌ Ensemble model test failed", error=str(e), exc_info=True)
        return False


async def main():
    """Main test function."""
    logger.info("Starting ML Models Test Suite")
    logger.info("Available models", models=get_available_models())
    
    # Test individual models
    test_results = {}
    
    for model_type in ["lstm", "cnn", "xgboost"]:
        logger.info(f"\n{'='*50}")
        logger.info(f"Testing {model_type.upper()} Model")
        logger.info(f"{'='*50}")
        
        success = await test_model_basic_functionality(model_type)
        test_results[model_type] = success
    
    # Test ensemble model
    logger.info(f"\n{'='*50}")
    logger.info("Testing ENSEMBLE Model")
    logger.info(f"{'='*50}")
    
    ensemble_success = await test_ensemble_model()
    test_results["ensemble"] = ensemble_success
    
    # Print summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    for model_type, success in test_results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{model_type.upper()}: {status}")
    
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    
    logger.info(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("🎉 All tests passed! ML models are working correctly.")
    else:
        logger.warning(f"⚠️  {total_tests - passed_tests} tests failed. Check the logs above.")
    
    # Cleanup
    import shutil
    test_dir = Path("test_models")
    if test_dir.exists():
        shutil.rmtree(test_dir)
        logger.info("Cleaned up test files")


if __name__ == "__main__":
    asyncio.run(main()) 