#!/usr/bin/env python3
"""Test script to verify ML dependencies are working."""

print("Testing TensorFlow import...")
try:
    import tensorflow as tf
    try:
        version = tf.__version__
    except AttributeError:
        version = "unknown"
    print(f"✅ TensorFlow {version} imported successfully")
    
    # Test Keras import
    try:
        import keras
        try:
            keras_version = keras.__version__
        except AttributeError:
            keras_version = "unknown"
        print(f"✅ Keras {keras_version} imported successfully")
    except ImportError:
        print("❌ Keras import failed")
    
    # Test basic TensorFlow operation
    x = tf.constant([1, 2, 3])
    print(f"✅ TensorFlow operation test: {x}")
    
except ImportError as e:
    print(f"❌ TensorFlow import failed: {e}")

print("\nTesting XGBoost import...")
try:
    import xgboost as xgb
    print(f"✅ XGBoost {xgb.__version__} imported successfully")
    
    # Test basic XGBoost operation
    import numpy as np
    X = np.random.randn(100, 5)
    y = np.random.randn(100)
    model = xgb.XGBRegressor(n_estimators=10)
    model.fit(X, y)
    pred = model.predict(X[:5])
    print(f"✅ XGBoost operation test: predictions shape {pred.shape}")
    
except ImportError as e:
    print(f"❌ XGBoost import failed: {e}")

print("\nTesting ONNX imports...")
try:
    import onnxruntime as ort
    print(f"✅ ONNX Runtime {ort.__version__} imported successfully")
except ImportError as e:
    print(f"❌ ONNX Runtime import failed: {e}")

try:
    import tf2onnx
    print("✅ tf2onnx imported successfully")
except ImportError as e:
    print(f"❌ tf2onnx import failed: {e}")

print("\nAll import tests completed!") 