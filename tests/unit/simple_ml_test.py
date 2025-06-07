#!/usr/bin/env python3
"""
Simple ML Model Test Script.

Tests the ML models with minimal dependencies.
"""

import os
# Disable oneDNN optimizations to avoid pooling issues
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import asyncio
import numpy as np
import structlog
from datetime import datetime

# Configure simple logging
structlog.configure(
    processors=[
        structlog.dev.ConsoleRenderer()
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


def test_xgboost():
    """Test XGBoost model."""
    logger.info("Testing XGBoost model")
    
    try:
        from models.ensemble_model import XGBoostPredictor, create_xgboost_config
        from models.predictor_interface import PredictionType
        
        # Create config
        config = create_xgboost_config(
            prediction_type=PredictionType.RETURN_REGRESSION,
            n_estimators=10,
            max_depth=3
        )
        
        # Create model
        model = XGBoostPredictor(config, logger)
        
        # Generate simple test data
        np.random.seed(42)
        X = np.random.randn(100, 20, 5)  # 100 samples, 20 timesteps, 5 features
        y = np.random.randn(100) * 0.01  # Returns
        
        # Set feature names for flattened input
        model.feature_names = [f"feature_{i}" for i in range(100)]  # 20*5=100
        
        # Train
        logger.info("Training XGBoost...")
        metrics = model.train(X, y)
        logger.info("XGBoost training completed", train_loss=metrics.train_loss)
        
        # Test prediction
        test_X = X[:5]
        predictions = model._predict_raw(test_X)
        logger.info("XGBoost predictions", predictions=predictions[:3].tolist())
        
        logger.info("✅ XGBoost test passed")
        return True
        
    except Exception as e:
        logger.error("❌ XGBoost test failed", error=str(e))
        return False


def test_lstm():
    """Test LSTM model."""
    logger.info("Testing LSTM model")
    
    try:
        from models.lstm_model import LSTMPredictor, create_lstm_config
        from models.predictor_interface import PredictionType
        
        # Create config
        config = create_lstm_config(
            prediction_type=PredictionType.RETURN_REGRESSION,
            sequence_length=20,
            features_dim=5,
            lstm_units=16,  # Small for testing
            epochs=2,
            batch_size=32
        )
        
        # Create model
        model = LSTMPredictor(config, logger)
        
        # Generate test data
        np.random.seed(42)
        X = np.random.randn(100, 20, 5)  # 100 samples, 20 timesteps, 5 features
        y = np.random.randn(100) * 0.01  # Returns
        
        # Set feature names
        model.feature_names = [f"feature_{i}" for i in range(5)]
        
        # Train
        logger.info("Training LSTM...")
        metrics = model.train(X, y)
        logger.info("LSTM training completed", train_loss=metrics.train_loss)
        
        # Test prediction
        test_X = X[:1]  # Single sample
        predictions = model._predict_raw(test_X)
        logger.info("LSTM predictions", predictions=predictions.tolist())
        
        logger.info("✅ LSTM test passed")
        return True
        
    except Exception as e:
        logger.error("❌ LSTM test failed", error=str(e))
        return False


def test_cnn():
    """Test CNN model."""
    logger.info("Testing CNN model")
    
    try:
        from models.cnn_model import CNNPredictor, create_cnn_config
        from models.predictor_interface import PredictionType
        
        # Create config
        config = create_cnn_config(
            prediction_type=PredictionType.RETURN_REGRESSION,
            sequence_length=20,
            features_dim=5,
            cnn_filters=8,  # Small for testing
            epochs=2,
            batch_size=32
        )
        
        # Create model
        model = CNNPredictor(config, logger)
        
        # Generate test data
        np.random.seed(42)
        X = np.random.randn(100, 20, 5)  # 100 samples, 20 timesteps, 5 features
        y = np.random.randn(100) * 0.01  # Returns
        
        # Set feature names
        model.feature_names = [f"feature_{i}" for i in range(5)]
        
        # Train
        logger.info("Training CNN...")
        metrics = model.train(X, y)
        logger.info("CNN training completed", train_loss=metrics.train_loss)
        
        # Test prediction
        test_X = X[:1]  # Single sample
        predictions = model._predict_raw(test_X)
        logger.info("CNN predictions", predictions=predictions.tolist())
        
        logger.info("✅ CNN test passed")
        return True
        
    except Exception as e:
        logger.error("❌ CNN test failed", error=str(e))
        return False


async def main():
    """Main test function."""
    print("Starting Simple ML Model Tests")
    logger.info("Starting Simple ML Model Tests")
    
    results = {
        "XGBoost": test_xgboost(),
        "LSTM": test_lstm(),
        "CNN": test_cnn()
    }
    
    # Summary
    passed = sum(results.values())
    total = len(results)
    
    print("\n" + "="*50)
    print("TEST RESULTS SUMMARY")
    print("="*50)
    logger.info("Test Results Summary")
    for model, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{model}: {status}")
        logger.info(f"{model}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    logger.info(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All ML models are working correctly!")
        logger.info("🎉 All ML models are working correctly!")
    else:
        print(f"⚠️ {total - passed} tests failed")
        logger.warning(f"⚠️ {total - passed} tests failed")
    
    print("="*50)


if __name__ == "__main__":
    asyncio.run(main()) 