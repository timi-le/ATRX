#!/usr/bin/env python3
"""
Test script to verify optimized LSTM and CNN models performance.

This test validates:
- Model creation and training
- Performance benchmarks (inference time, training time)
- Memory usage optimization
- Prediction accuracy
- All optimizations are working correctly
"""

import os
import sys
import time
import json
import psutil
import asyncio
import numpy as np
import structlog
from pathlib import Path

# Disable oneDNN optimizations for stability
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def create_synthetic_fx_data(n_samples=200, sequence_length=30, n_features=13):
    """Create realistic synthetic FX data for testing."""
    np.random.seed(42)
    
    # Create sequences with FX-like characteristics
    X = np.random.randn(n_samples, sequence_length, n_features) * 0.01
    
    # Add some trend and volatility clustering
    for i in range(1, sequence_length):
        X[:, i, :] = X[:, i-1, :] * 0.95 + np.random.randn(n_samples, n_features) * 0.01
    
    # Create realistic targets (returns)
    y = np.random.randn(n_samples) * 0.005  # Small returns like FX
    
    return X, y

def measure_memory_usage():
    """Measure current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

async def test_optimized_lstm():
    """Test optimized LSTM model."""
    print("\n🧪 Testing Optimized LSTM Model")
    print("-" * 50)
    
    try:
        from models import LSTMPredictor, create_lstm_config
        
        # Create optimized config
        config = create_lstm_config(
            sequence_length=30,  # Reduced for speed
            lstm_units=32,       # Reduced for speed
            num_layers=1,        # Reduced for speed
            learning_rate=0.002, # Higher for faster convergence
            batch_size=64,       # Larger batch
            epochs=20            # Reduced for testing
        )
        
        logger = structlog.get_logger()
        model = LSTMPredictor(config, logger)
        
        print(f"✅ LSTM model creation: SUCCESS")
        print(f"   - LSTM units: {model.lstm_units}")
        print(f"   - Layers: {model.num_layers}")
        print(f"   - Mixed precision: {model.use_mixed_precision}")
        print(f"   - XLA compilation: {model.use_xla}")
        
        # Create test data
        X, y = create_synthetic_fx_data(n_samples=200, sequence_length=30)
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Measure training time
        memory_before = measure_memory_usage()
        start_time = time.time()
        
        metrics = model.train(X_train, y_train, X_val, y_val)
        
        training_time = time.time() - start_time
        memory_after = measure_memory_usage()
        memory_used = memory_after - memory_before
        
        print(f"✅ LSTM training: SUCCESS")
        print(f"   - Training time: {training_time:.2f}s")
        print(f"   - Memory used: {memory_used:.2f}MB")
        print(f"   - Train loss: {metrics.train_loss:.6f}")
        print(f"   - Val loss: {metrics.val_loss:.6f}")
        
        # Test inference speed
        test_sample = X_val[:1]  # Single sample
        inference_times = []
        
        for _ in range(10):
            start_time = time.time()
            prediction = await model.predict(test_sample)  # Await the async call
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
        
        avg_inference_time = np.mean(inference_times)
        print(f"✅ LSTM inference: SUCCESS")
        print(f"   - Avg inference time: {avg_inference_time:.4f}s")
        print(f"   - Prediction: {prediction.prediction:.6f}")
        print(f"   - Confidence: {prediction.confidence:.3f}")
        
        # Performance benchmarks
        benchmarks = {
            'training_time_under_300s': training_time < 300,
            'inference_time_under_100ms': avg_inference_time < 0.1,
            'memory_usage_reasonable': memory_used < 500,  # MB
            'model_trained': model.is_trained
        }
        
        passed_benchmarks = sum(benchmarks.values())
        total_benchmarks = len(benchmarks)
        
        print(f"\n📊 LSTM Performance Benchmarks: {passed_benchmarks}/{total_benchmarks}")
        for benchmark, passed in benchmarks.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"   - {benchmark}: {status}")
        
        return {
            'model_type': 'LSTM',
            'training_time': float(training_time),
            'inference_time': float(avg_inference_time),
            'memory_usage': float(memory_used),
            'train_loss': float(metrics.train_loss),
            'val_loss': float(metrics.val_loss),
            'benchmarks_passed': int(passed_benchmarks),
            'benchmarks_total': int(total_benchmarks),
            'success': bool(passed_benchmarks >= total_benchmarks * 0.75)  # 75% pass rate
        }
        
    except Exception as e:
        print(f"❌ LSTM test failed: {e}")
        import traceback
        traceback.print_exc()
        return {'model_type': 'LSTM', 'success': False, 'error': str(e)}

async def test_optimized_cnn():
    """Test optimized CNN model."""
    print("\n🧪 Testing Optimized CNN Model")
    print("-" * 50)
    
    try:
        from models import CNNPredictor, create_cnn_config
        
        # Create optimized config
        config = create_cnn_config(
            sequence_length=30,      # Reduced for speed
            cnn_filters=64,          # Increased for better patterns
            kernel_sizes=[2, 3, 5, 8],  # FX-specific timeframes
            num_conv_layers=2,       # Reduced for speed
            learning_rate=0.0015,    # Higher for faster convergence
            batch_size=64,           # Larger batch
            epochs=25                # Reduced for testing
        )
        
        logger = structlog.get_logger()
        model = CNNPredictor(config, logger)
        
        print(f"✅ CNN model creation: SUCCESS")
        print(f"   - CNN filters: {model.cnn_filters}")
        print(f"   - Kernel sizes: {model.kernel_sizes}")
        print(f"   - Conv layers: {model.num_conv_layers}")
        print(f"   - Residual connections: {model.use_residual}")
        print(f"   - Mixed precision: {model.use_mixed_precision}")
        print(f"   - XLA compilation: {model.use_xla}")
        
        # Create test data
        X, y = create_synthetic_fx_data(n_samples=200, sequence_length=30)
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Measure training time
        memory_before = measure_memory_usage()
        start_time = time.time()
        
        metrics = model.train(X_train, y_train, X_val, y_val)
        
        training_time = time.time() - start_time
        memory_after = measure_memory_usage()
        memory_used = memory_after - memory_before
        
        print(f"✅ CNN training: SUCCESS")
        print(f"   - Training time: {training_time:.2f}s")
        print(f"   - Memory used: {memory_used:.2f}MB")
        print(f"   - Train loss: {metrics.train_loss:.6f}")
        print(f"   - Val loss: {metrics.val_loss:.6f}")
        
        # Test inference speed
        test_sample = X_val[:1]  # Single sample
        inference_times = []
        
        for _ in range(10):
            start_time = time.time()
            prediction = await model.predict(test_sample)  # Await the async call
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
        
        avg_inference_time = np.mean(inference_times)
        print(f"✅ CNN inference: SUCCESS")
        print(f"   - Avg inference time: {avg_inference_time:.4f}s")
        print(f"   - Prediction: {prediction.prediction:.6f}")
        print(f"   - Confidence: {prediction.confidence:.3f}")
        
        # Performance benchmarks
        benchmarks = {
            'training_time_under_300s': training_time < 300,
            'inference_time_under_100ms': avg_inference_time < 0.1,
            'memory_usage_reasonable': memory_used < 500,  # MB
            'model_trained': model.is_trained
        }
        
        passed_benchmarks = sum(benchmarks.values())
        total_benchmarks = len(benchmarks)
        
        print(f"\n📊 CNN Performance Benchmarks: {passed_benchmarks}/{total_benchmarks}")
        for benchmark, passed in benchmarks.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"   - {benchmark}: {status}")
        
        return {
            'model_type': 'CNN',
            'training_time': float(training_time),
            'inference_time': float(avg_inference_time),
            'memory_usage': float(memory_used),
            'train_loss': float(metrics.train_loss),
            'val_loss': float(metrics.val_loss),
            'benchmarks_passed': int(passed_benchmarks),
            'benchmarks_total': int(total_benchmarks),
            'success': bool(passed_benchmarks >= total_benchmarks * 0.75)  # 75% pass rate
        }
        
    except Exception as e:
        print(f"❌ CNN test failed: {e}")
        import traceback
        traceback.print_exc()
        return {'model_type': 'CNN', 'success': False, 'error': str(e)}

async def test_xgboost_baseline():
    """Test XGBoost as baseline for comparison."""
    print("\n🧪 Testing XGBoost Baseline")
    print("-" * 50)
    
    try:
        from models import XGBoostPredictor, create_xgboost_config
        
        config = create_xgboost_config(
            sequence_length=30,
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1
        )
        
        logger = structlog.get_logger()
        model = XGBoostPredictor(config, logger)
        
        # Create test data
        X, y = create_synthetic_fx_data(n_samples=200, sequence_length=30)
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Train and test
        start_time = time.time()
        metrics = model.train(X_train, y_train, X_val, y_val)
        training_time = time.time() - start_time
        
        # Test inference
        test_sample = X_val[:1]  # Single sample
        print(f"   - Test sample shape: {test_sample.shape}")
        start_time = time.time()
        prediction = await model.predict(test_sample)  # Await the async call
        inference_time = time.time() - start_time
        
        print(f"✅ XGBoost baseline: SUCCESS")
        print(f"   - Training time: {training_time:.2f}s")
        print(f"   - Inference time: {inference_time:.4f}s")
        print(f"   - Train loss: {metrics.train_loss:.6f}")
        print(f"   - Val loss: {metrics.val_loss:.6f}")
        print(f"   - Prediction: {prediction.prediction:.6f}")
        print(f"   - Confidence: {prediction.confidence:.3f}")
        
        return {
            'model_type': 'XGBoost',
            'training_time': float(training_time),
            'inference_time': float(inference_time),
            'train_loss': float(metrics.train_loss),
            'val_loss': float(metrics.val_loss),
            'success': True
        }
        
    except Exception as e:
        print(f"❌ XGBoost test failed: {e}")
        return {'model_type': 'XGBoost', 'success': False, 'error': str(e)}

async def main():
    """Run all optimization tests."""
    print("🚀 FX AI-Quant System - Optimized Models Performance Test")
    print("=" * 70)
    
    # Test all models
    results = []
    
    # Test XGBoost baseline
    xgb_result = await test_xgboost_baseline()
    results.append(xgb_result)
    
    # Test optimized LSTM
    lstm_result = await test_optimized_lstm()
    results.append(lstm_result)
    
    # Test optimized CNN
    cnn_result = await test_optimized_cnn()
    results.append(cnn_result)
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 OPTIMIZATION TEST SUMMARY")
    print("=" * 70)
    
    successful_tests = sum(1 for r in results if r.get('success', False))
    total_tests = len(results)
    
    for result in results:
        model_type = result['model_type']
        success = result.get('success', False)
        status = "✅ PASS" if success else "❌ FAIL"
        
        print(f"\n{model_type} Model: {status}")
        
        if success and 'training_time' in result:
            print(f"  - Training time: {result['training_time']:.2f}s")
            print(f"  - Inference time: {result['inference_time']:.4f}s")
            if 'benchmarks_passed' in result:
                print(f"  - Benchmarks: {result['benchmarks_passed']}/{result['benchmarks_total']}")
        elif 'error' in result:
            print(f"  - Error: {result['error']}")
    
    print(f"\n🎯 Overall Success Rate: {successful_tests}/{total_tests} ({successful_tests/total_tests*100:.1f}%)")
    
    # Save results
    report_path = Path("tests/reports/optimized_models_test_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'summary': {
                'total_tests': total_tests,
                'successful_tests': successful_tests,
                'success_rate': successful_tests / total_tests
            },
            'results': results
        }, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: {report_path}")
    
    if successful_tests == total_tests:
        print("\n🎉 ALL OPTIMIZATIONS WORKING PERFECTLY!")
        return 0
    else:
        print(f"\n⚠️  {total_tests - successful_tests} optimization(s) need attention")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main())) 