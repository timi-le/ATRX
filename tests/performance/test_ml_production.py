#!/usr/bin/env python3
"""
Production-Ready ML Model Test Suite for FX AI-Quant Trading System.

This comprehensive test suite validates all ML models under production conditions,
including proper configuration, error handling, performance benchmarks, and
integration with the full system architecture.
"""

import os
# Disable oneDNN optimizations to avoid pooling issues
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import asyncio
import sys
import time
import json
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import structlog

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import all necessary modules
from models.predictor_interface import PredictionType, ModelType, ModelConfig, TrainingMetrics
from models import (
    get_available_models, create_model,
    LSTM_AVAILABLE, CNN_AVAILABLE, ENSEMBLE_AVAILABLE
)
from core.interfaces.ml_interfaces import Features, Prediction

# Configure comprehensive logging
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


class ProductionMLTestSuite:
    """Comprehensive production-ready ML test suite."""
    
    def __init__(self):
        self.test_results: Dict[str, Dict[str, Any]] = {}
        self.test_data_dir = Path("test_data")
        self.model_output_dir = Path("test_models_production")
        self.performance_benchmarks = {
            "training_time_max": 300.0,  # 5 minutes max
            "inference_time_max": 0.1,   # 100ms max
            "memory_usage_max": 1024.0,  # 1GB max
            "min_accuracy_threshold": 0.45,  # Minimum acceptable accuracy
            "min_sharpe_ratio": -2.0,    # Minimum Sharpe ratio
            "max_drawdown_threshold": -0.5  # Maximum acceptable drawdown
        }
        
        # Create directories
        self.test_data_dir.mkdir(exist_ok=True)
        self.model_output_dir.mkdir(exist_ok=True)
        
        logger.info("Production ML Test Suite initialized", 
                   benchmarks=self.performance_benchmarks)
    
    def generate_realistic_market_data(
        self, 
        n_samples: int = 5000, 
        sequence_length: int = 60, 
        n_features: int = 13
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Generate realistic market data with proper financial characteristics."""
        logger.info("Generating realistic market data", 
                   n_samples=n_samples, 
                   sequence_length=sequence_length, 
                   n_features=n_features)
        
        np.random.seed(42)  # Reproducible results
        
        # Feature names matching real FX data
        feature_names = [
            "price_return", "volume", "volatility", "rsi", "macd",
            "bollinger_upper", "bollinger_lower", "ema_short", "ema_long",
            "atr", "momentum", "stochastic", "williams_r"
        ][:n_features]
        
        # Generate features with realistic financial properties
        features = []
        returns = []
        volatility = []
        
        # Initial values
        price = 1.0
        vol = 0.01
        
        for i in range(n_samples + sequence_length):
            # Volatility clustering (GARCH-like)
            vol = 0.95 * vol + 0.05 * 0.01 + 0.1 * (returns[-1]**2 if returns else 0)
            vol = np.clip(vol, 0.005, 0.05)
            
            # Price return with fat tails
            if np.random.random() < 0.05:  # 5% chance of extreme event
                ret = np.random.normal(0, vol * 3)
            else:
                ret = np.random.normal(0, vol)
            
            returns.append(ret)
            volatility.append(vol)
            price *= (1 + ret)
            
            # Generate correlated features
            feature_vec = np.zeros(n_features)
            
            # Price return
            feature_vec[0] = ret
            
            # Volume (inverse correlation with returns)
            feature_vec[1] = np.random.lognormal(0, 0.5) * (1 + abs(ret) * 10)
            
            # Volatility
            if n_features > 2:
                feature_vec[2] = vol
            
            # Technical indicators (simplified)
            for j in range(3, n_features):
                # Add some autocorrelation and noise
                if i > 0:
                    feature_vec[j] = 0.7 * features[-1][j] + 0.3 * np.random.normal(0, 0.1)
                else:
                    feature_vec[j] = np.random.normal(0, 0.1)
            
            features.append(feature_vec)
        
        features = np.array(features)
        returns = np.array(returns)
        
        # Create sequences for training
        X_sequences = []
        y_targets = []
        
        for i in range(sequence_length, len(features)):
            X_sequences.append(features[i-sequence_length:i])
            y_targets.append(returns[i])
        
        X = np.array(X_sequences)
        y = np.array(y_targets)
        
        logger.info("Realistic market data generated", 
                   X_shape=X.shape, 
                   y_shape=y.shape,
                   y_mean=np.mean(y),
                   y_std=np.std(y),
                   feature_names=feature_names)
        
        return X, y, feature_names
    
    def create_production_configs(self) -> Dict[str, ModelConfig]:
        """Create production-ready model configurations."""
        configs = {}
        
        # LSTM Configuration
        if LSTM_AVAILABLE:
            from models.lstm_model import create_lstm_config
            configs["lstm"] = create_lstm_config(
                prediction_type=PredictionType.RETURN_REGRESSION,
                sequence_length=60,
                features_dim=13,
                lstm_units=64,
                num_layers=2,
                dropout_rate=0.2,
                recurrent_dropout=0.2,
                use_bidirectional=True,
                learning_rate=0.001,
                batch_size=64,
                epochs=20,  # Reduced for testing
                validation_split=0.2,
                early_stopping_patience=10
            )
        
        # CNN Configuration
        if CNN_AVAILABLE:
            from models.cnn_model import create_cnn_config
            configs["cnn"] = create_cnn_config(
                prediction_type=PredictionType.RETURN_REGRESSION,
                sequence_length=60,
                features_dim=13,
                cnn_filters=64,
                kernel_sizes=[3, 5, 7],
                num_conv_layers=3,
                dropout_rate=0.3,
                use_batch_norm=True,
                learning_rate=0.001,
                batch_size=64,
                epochs=20,  # Reduced for testing
                validation_split=0.2,
                early_stopping_patience=10
            )
        
        # XGBoost Configuration
        if ENSEMBLE_AVAILABLE:
            from models.ensemble_model import create_xgboost_config
            configs["xgboost"] = create_xgboost_config(
                prediction_type=PredictionType.RETURN_REGRESSION,
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
        
        # Ensemble Configuration
        if ENSEMBLE_AVAILABLE:
            from models.ensemble_model import create_ensemble_config
            configs["ensemble"] = create_ensemble_config(
                prediction_type=PredictionType.RETURN_REGRESSION,
                ensemble_method='weighted_average',
                meta_model_type='linear'
            )
        
        logger.info("Production configurations created", models=list(configs.keys()))
        return configs
    
    async def test_model_comprehensive(
        self, 
        model_type: str, 
        config: ModelConfig,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Comprehensive test of a single model."""
        logger.info(f"Starting comprehensive test for {model_type}")
        
        test_result = {
            "model_type": model_type,
            "config": config.__dict__,
            "training_metrics": {},
            "validation_metrics": {},
            "test_metrics": {},
            "performance_metrics": {},
            "production_tests": {},
            "errors": [],
            "warnings": [],
            "passed": False
        }
        
        try:
            # 1. Model Creation Test
            logger.info(f"Creating {model_type} model")
            start_time = time.time()
            model = create_model(model_type, config)
            model.feature_names = feature_names
            creation_time = time.time() - start_time
            
            test_result["performance_metrics"]["creation_time"] = creation_time
            logger.info(f"{model_type} model created", creation_time=creation_time)
            
            # 2. Training Test
            logger.info(f"Training {model_type} model")
            start_time = time.time()
            training_metrics = model.train(X_train, y_train, X_val, y_val)
            training_time = time.time() - start_time
            
            test_result["training_metrics"] = training_metrics.__dict__
            test_result["performance_metrics"]["training_time"] = training_time
            
            # Check training time benchmark
            if training_time > self.performance_benchmarks["training_time_max"]:
                test_result["warnings"].append(
                    f"Training time {training_time:.2f}s exceeds benchmark {self.performance_benchmarks['training_time_max']}s"
                )
            
            logger.info(f"{model_type} training completed", 
                       train_loss=training_metrics.train_loss,
                       val_loss=training_metrics.val_loss,
                       training_time=training_time)
            
            # 3. Inference Performance Test
            logger.info(f"Testing {model_type} inference performance")
            inference_times = []
            
            for i in range(100):  # 100 inference tests
                start_time = time.time()
                _ = model._predict_raw(X_test[i:i+1])
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
            
            avg_inference_time = np.mean(inference_times)
            max_inference_time = np.max(inference_times)
            
            test_result["performance_metrics"]["avg_inference_time"] = avg_inference_time
            test_result["performance_metrics"]["max_inference_time"] = max_inference_time
            
            # Check inference time benchmark
            if avg_inference_time > self.performance_benchmarks["inference_time_max"]:
                test_result["errors"].append(
                    f"Average inference time {avg_inference_time:.4f}s exceeds benchmark {self.performance_benchmarks['inference_time_max']}s"
                )
            
            logger.info(f"{model_type} inference performance", 
                       avg_time=avg_inference_time,
                       max_time=max_inference_time)
            
            # 4. Model Evaluation
            logger.info(f"Evaluating {model_type} model")
            eval_metrics = model.evaluate_model(X_test, y_test)
            test_result["test_metrics"] = eval_metrics
            
            # Check performance benchmarks
            if "accuracy" in eval_metrics:
                if eval_metrics["accuracy"] < self.performance_benchmarks["min_accuracy_threshold"]:
                    test_result["warnings"].append(
                        f"Accuracy {eval_metrics['accuracy']:.3f} below threshold {self.performance_benchmarks['min_accuracy_threshold']}"
                    )
            
            if "sharpe_ratio" in eval_metrics:
                if eval_metrics["sharpe_ratio"] < self.performance_benchmarks["min_sharpe_ratio"]:
                    test_result["warnings"].append(
                        f"Sharpe ratio {eval_metrics['sharpe_ratio']:.3f} below threshold {self.performance_benchmarks['min_sharpe_ratio']}"
                    )
            
            logger.info(f"{model_type} evaluation completed", **eval_metrics)
            
            # 5. Async Prediction Interface Test
            logger.info(f"Testing {model_type} async prediction interface")
            
            # Test single prediction
            test_features = Features(
                symbol="EURUSD",
                timestamp=datetime.now(),
                features={name: float(X_test[0, -1, i]) for i, name in enumerate(feature_names)},
                feature_names=feature_names
            )
            
            prediction = await model.predict(test_features)
            test_result["production_tests"]["single_prediction"] = {
                "prediction": prediction.prediction,
                "confidence": prediction.confidence,
                "symbol": prediction.symbol,
                "model_name": prediction.model_name
            }
            
            # Test batch prediction
            batch_features = [
                Features(
                    symbol="EURUSD",
                    timestamp=datetime.now(),
                    features={name: float(X_test[i, -1, j]) for j, name in enumerate(feature_names)},
                    feature_names=feature_names
                )
                for i in range(5)
            ]
            
            batch_predictions = await model.predict_batch(batch_features)
            test_result["production_tests"]["batch_prediction"] = {
                "num_predictions": len(batch_predictions),
                "predictions": [p.prediction for p in batch_predictions[:3]]
            }
            
            logger.info(f"{model_type} async prediction tests completed")
            
            # 6. Model Persistence Test
            logger.info(f"Testing {model_type} model persistence")
            
            model_path = self.model_output_dir / f"{model_type}_production_model"
            model.save_model(str(model_path))
            
            # Load model and test consistency
            new_model = create_model(model_type, config)
            new_model.load_model(str(model_path))
            
            # Test prediction consistency
            original_pred = model._predict_raw(X_test[:10])
            loaded_pred = new_model._predict_raw(X_test[:10])
            
            prediction_diff = np.abs(original_pred - loaded_pred)
            max_diff = np.max(prediction_diff)
            
            test_result["production_tests"]["persistence"] = {
                "max_prediction_diff": float(max_diff),
                "consistent": max_diff < 1e-5
            }
            
            if max_diff >= 1e-5:
                test_result["errors"].append(f"Model persistence test failed: max diff {max_diff}")
            
            logger.info(f"{model_type} persistence test completed", max_diff=max_diff)
            
            # 7. ONNX Export Test (if supported)
            try:
                onnx_path = self.model_output_dir / f"{model_type}_production_model.onnx"
                model.export_to_onnx(str(onnx_path))
                test_result["production_tests"]["onnx_export"] = True
                logger.info(f"{model_type} ONNX export successful")
            except Exception as e:
                test_result["production_tests"]["onnx_export"] = False
                test_result["warnings"].append(f"ONNX export failed: {str(e)}")
                logger.warning(f"{model_type} ONNX export failed", error=str(e))
            
            # 8. Feature Importance Test
            try:
                importance = await model.get_feature_importance()
                test_result["production_tests"]["feature_importance"] = {
                    "available": len(importance) > 0,
                    "num_features": len(importance),
                    "top_features": dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5])
                }
                logger.info(f"{model_type} feature importance test completed")
            except Exception as e:
                test_result["warnings"].append(f"Feature importance test failed: {str(e)}")
            
            # 9. Memory Usage Test
            try:
                import psutil
                import gc
                
                process = psutil.Process()
                memory_before = process.memory_info().rss / 1024 / 1024  # MB
                
                # Create multiple model instances to test memory usage
                temp_models = []
                for _ in range(5):
                    temp_model = create_model(model_type, config)
                    temp_models.append(temp_model)
                
                memory_after = process.memory_info().rss / 1024 / 1024  # MB
                memory_usage = memory_after - memory_before
                
                # Cleanup
                del temp_models
                gc.collect()
                
                test_result["performance_metrics"]["memory_usage_mb"] = memory_usage
                
                if memory_usage > self.performance_benchmarks["memory_usage_max"]:
                    test_result["warnings"].append(
                        f"Memory usage {memory_usage:.2f}MB exceeds benchmark {self.performance_benchmarks['memory_usage_max']}MB"
                    )
                
                logger.info(f"{model_type} memory usage test completed", memory_usage_mb=memory_usage)
                
            except ImportError:
                test_result["warnings"].append("psutil not available for memory testing")
            except Exception as e:
                test_result["warnings"].append(f"Memory usage test failed: {str(e)}")
            
            # 10. Edge Cases Test
            logger.info(f"Testing {model_type} edge cases")
            
            edge_cases = {
                "zero_input": np.zeros_like(X_test[:1]),
                "extreme_values": X_test[:1] * 100,
                "nan_handling": np.full_like(X_test[:1], np.nan),
                "inf_handling": np.full_like(X_test[:1], np.inf)
            }
            
            edge_case_results = {}
            for case_name, case_data in edge_cases.items():
                try:
                    if case_name in ["nan_handling", "inf_handling"]:
                        # These should raise errors or handle gracefully
                        try:
                            pred = model._predict_raw(case_data)
                            edge_case_results[case_name] = "handled_gracefully"
                        except Exception:
                            edge_case_results[case_name] = "error_raised"
                    else:
                        pred = model._predict_raw(case_data)
                        edge_case_results[case_name] = "success"
                except Exception as e:
                    edge_case_results[case_name] = f"error: {str(e)}"
            
            test_result["production_tests"]["edge_cases"] = edge_case_results
            logger.info(f"{model_type} edge cases test completed", results=edge_case_results)
            
            # Determine if test passed
            test_result["passed"] = len(test_result["errors"]) == 0
            
            logger.info(f"{model_type} comprehensive test completed", 
                       passed=test_result["passed"],
                       errors=len(test_result["errors"]),
                       warnings=len(test_result["warnings"]))
            
        except Exception as e:
            test_result["errors"].append(f"Critical error during testing: {str(e)}")
            test_result["passed"] = False
            logger.error(f"{model_type} test failed with critical error", error=str(e), exc_info=True)
        
        return test_result
    
    async def test_ensemble_integration(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Test ensemble model integration."""
        if not ENSEMBLE_AVAILABLE:
            return {"passed": False, "errors": ["Ensemble models not available"]}
        
        logger.info("Testing ensemble model integration")
        
        test_result = {
            "model_type": "ensemble_integration",
            "individual_models": {},
            "ensemble_performance": {},
            "errors": [],
            "warnings": [],
            "passed": False
        }
        
        try:
            from models.ensemble_model import EnsembleMLPredictor, create_ensemble_config
            
            # Create ensemble
            ensemble_config = create_ensemble_config(
                prediction_type=PredictionType.RETURN_REGRESSION,
                ensemble_method='weighted_average'
            )
            ensemble = EnsembleMLPredictor(ensemble_config, logger)
            
            # Train individual models and add to ensemble
            individual_models = []
            
            # Add LSTM if available
            if LSTM_AVAILABLE:
                from models.lstm_model import LSTMPredictor, create_lstm_config
                lstm_config = create_lstm_config(
                    prediction_type=PredictionType.RETURN_REGRESSION,
                    sequence_length=60, features_dim=13, 
                    lstm_units=32, epochs=10, batch_size=64
                )
                lstm_model = LSTMPredictor(lstm_config, logger)
                lstm_model.feature_names = feature_names
                lstm_metrics = lstm_model.train(X_train, y_train, X_val, y_val)
                await ensemble.add_model(lstm_model, weight=1.0)
                individual_models.append(("lstm", lstm_model, lstm_metrics))
                test_result["individual_models"]["lstm"] = lstm_metrics.__dict__
            
            # Add CNN if available
            if CNN_AVAILABLE:
                from models.cnn_model import CNNPredictor, create_cnn_config
                cnn_config = create_cnn_config(
                    prediction_type=PredictionType.RETURN_REGRESSION,
                    sequence_length=60, features_dim=13,
                    cnn_filters=32, epochs=10, batch_size=64
                )
                cnn_model = CNNPredictor(cnn_config, logger)
                cnn_model.feature_names = feature_names
                cnn_metrics = cnn_model.train(X_train, y_train, X_val, y_val)
                await ensemble.add_model(cnn_model, weight=1.0)
                individual_models.append(("cnn", cnn_model, cnn_metrics))
                test_result["individual_models"]["cnn"] = cnn_metrics.__dict__
            
            # Add XGBoost
            from models.ensemble_model import XGBoostPredictor, create_xgboost_config
            xgb_config = create_xgboost_config(
                prediction_type=PredictionType.RETURN_REGRESSION,
                n_estimators=50, max_depth=4
            )
            xgb_model = XGBoostPredictor(xgb_config, logger)
            xgb_model.feature_names = [f"feature_{i}" for i in range(60 * 13)]  # Flattened
            xgb_metrics = xgb_model.train(X_train, y_train, X_val, y_val)
            await ensemble.add_model(xgb_model, weight=1.0)
            individual_models.append(("xgboost", xgb_model, xgb_metrics))
            test_result["individual_models"]["xgboost"] = xgb_metrics.__dict__
            
            # Train ensemble
            logger.info("Training ensemble model")
            ensemble_metrics = ensemble.train(X_train, y_train, X_val, y_val)
            test_result["ensemble_performance"]["training"] = ensemble_metrics.__dict__
            
            # Evaluate ensemble
            ensemble_eval = ensemble.evaluate_model(X_test, y_test)
            test_result["ensemble_performance"]["evaluation"] = ensemble_eval
            
            # Compare ensemble vs individual models
            individual_predictions = {}
            for name, model, _ in individual_models:
                pred = model._predict_raw(X_test)
                individual_predictions[name] = pred
            
            ensemble_pred = ensemble._predict_raw(X_test)
            
            # Calculate improvement metrics
            ensemble_mse = np.mean((y_test - ensemble_pred) ** 2)
            individual_mses = {name: np.mean((y_test - pred) ** 2) 
                             for name, pred in individual_predictions.items()}
            
            best_individual_mse = min(individual_mses.values())
            improvement = (best_individual_mse - ensemble_mse) / best_individual_mse * 100
            
            test_result["ensemble_performance"]["improvement_vs_best"] = improvement
            test_result["ensemble_performance"]["ensemble_mse"] = ensemble_mse
            test_result["ensemble_performance"]["individual_mses"] = individual_mses
            
            # Test ensemble prediction interface
            test_features = Features(
                symbol="EURUSD",
                timestamp=datetime.now(),
                features={name: float(X_test[0, -1, i]) for i, name in enumerate(feature_names)},
                feature_names=feature_names
            )
            
            ensemble_prediction = await ensemble.predict(test_features)
            test_result["ensemble_performance"]["prediction_test"] = {
                "prediction": ensemble_prediction.prediction,
                "confidence": ensemble_prediction.confidence
            }
            
            test_result["passed"] = True
            logger.info("Ensemble integration test completed successfully", 
                       improvement=improvement,
                       ensemble_mse=ensemble_mse)
            
        except Exception as e:
            test_result["errors"].append(f"Ensemble integration test failed: {str(e)}")
            test_result["passed"] = False
            logger.error("Ensemble integration test failed", error=str(e), exc_info=True)
        
        return test_result
    
    async def run_production_test_suite(self) -> Dict[str, Any]:
        """Run the complete production test suite."""
        logger.info("Starting Production ML Test Suite")
        
        suite_results = {
            "start_time": datetime.now().isoformat(),
            "test_environment": {
                "available_models": get_available_models(),
                "lstm_available": LSTM_AVAILABLE,
                "cnn_available": CNN_AVAILABLE,
                "ensemble_available": ENSEMBLE_AVAILABLE
            },
            "data_generation": {},
            "model_tests": {},
            "ensemble_test": {},
            "summary": {},
            "passed": False
        }
        
        try:
            # 1. Generate realistic test data
            logger.info("Generating production test data")
            X, y, feature_names = self.generate_realistic_market_data(
                n_samples=5000, sequence_length=60, n_features=13
            )
            
            # Split data properly
            train_size = int(0.7 * len(X))
            val_size = int(0.15 * len(X))
            
            X_train = X[:train_size]
            y_train = y[:train_size]
            X_val = X[train_size:train_size + val_size]
            y_val = y[train_size:train_size + val_size]
            X_test = X[train_size + val_size:]
            y_test = y[train_size + val_size:]
            
            suite_results["data_generation"] = {
                "total_samples": len(X),
                "train_samples": len(X_train),
                "val_samples": len(X_val),
                "test_samples": len(X_test),
                "feature_names": feature_names,
                "data_characteristics": {
                    "y_mean": float(np.mean(y)),
                    "y_std": float(np.std(y)),
                    "y_min": float(np.min(y)),
                    "y_max": float(np.max(y))
                }
            }
            
            logger.info("Test data prepared", 
                       train_size=len(X_train),
                       val_size=len(X_val),
                       test_size=len(X_test))
            
            # 2. Create production configurations
            configs = self.create_production_configs()
            
            # 3. Test individual models
            for model_type, config in configs.items():
                if model_type == "ensemble":
                    continue  # Handle ensemble separately
                
                logger.info(f"Testing {model_type} model")
                test_result = await self.test_model_comprehensive(
                    model_type, config, X_train, y_train, X_val, y_val, X_test, y_test, feature_names
                )
                suite_results["model_tests"][model_type] = test_result
                self.test_results[model_type] = test_result
            
            # 4. Test ensemble integration
            logger.info("Testing ensemble integration")
            ensemble_result = await self.test_ensemble_integration(
                X_train, y_train, X_val, y_val, X_test, y_test, feature_names
            )
            suite_results["ensemble_test"] = ensemble_result
            self.test_results["ensemble"] = ensemble_result
            
            # 5. Generate summary
            total_tests = len(suite_results["model_tests"]) + 1  # +1 for ensemble
            passed_tests = sum(1 for result in suite_results["model_tests"].values() if result["passed"])
            if ensemble_result["passed"]:
                passed_tests += 1
            
            suite_results["summary"] = {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": total_tests - passed_tests,
                "success_rate": passed_tests / total_tests * 100,
                "performance_summary": self._generate_performance_summary(),
                "recommendations": self._generate_recommendations()
            }
            
            suite_results["passed"] = passed_tests == total_tests
            suite_results["end_time"] = datetime.now().isoformat()
            
            logger.info("Production test suite completed", 
                       passed_tests=passed_tests,
                       total_tests=total_tests,
                       success_rate=suite_results["summary"]["success_rate"])
            
        except Exception as e:
            suite_results["critical_error"] = str(e)
            suite_results["passed"] = False
            logger.error("Production test suite failed with critical error", error=str(e), exc_info=True)
        
        return suite_results
    
    def _generate_performance_summary(self) -> Dict[str, Any]:
        """Generate performance summary across all models."""
        summary = {
            "training_times": {},
            "inference_times": {},
            "memory_usage": {},
            "accuracy_metrics": {},
            "benchmark_compliance": {}
        }
        
        for model_type, result in self.test_results.items():
            if not result.get("passed", False):
                continue
            
            perf = result.get("performance_metrics", {})
            test_metrics = result.get("test_metrics", {})
            
            if "training_time" in perf:
                summary["training_times"][model_type] = perf["training_time"]
            
            if "avg_inference_time" in perf:
                summary["inference_times"][model_type] = perf["avg_inference_time"]
            
            if "memory_usage_mb" in perf:
                summary["memory_usage"][model_type] = perf["memory_usage_mb"]
            
            if "mse" in test_metrics:
                summary["accuracy_metrics"][model_type] = {
                    "mse": test_metrics["mse"],
                    "mae": test_metrics.get("mae", 0),
                    "sharpe_ratio": test_metrics.get("sharpe_ratio", 0)
                }
            
            # Check benchmark compliance
            compliance = {
                "training_time": perf.get("training_time", 0) <= self.performance_benchmarks["training_time_max"],
                "inference_time": perf.get("avg_inference_time", 0) <= self.performance_benchmarks["inference_time_max"],
                "memory_usage": perf.get("memory_usage_mb", 0) <= self.performance_benchmarks["memory_usage_max"]
            }
            summary["benchmark_compliance"][model_type] = compliance
        
        return summary
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        for model_type, result in self.test_results.items():
            if not result.get("passed", False):
                recommendations.append(f"❌ {model_type}: Failed critical tests - requires investigation")
                continue
            
            warnings = result.get("warnings", [])
            if warnings:
                recommendations.append(f"⚠️ {model_type}: {len(warnings)} warnings - consider optimization")
            
            perf = result.get("performance_metrics", {})
            
            # Performance recommendations
            if perf.get("training_time", 0) > self.performance_benchmarks["training_time_max"] * 0.8:
                recommendations.append(f"🐌 {model_type}: Training time approaching limit - consider reducing model complexity")
            
            if perf.get("avg_inference_time", 0) > self.performance_benchmarks["inference_time_max"] * 0.8:
                recommendations.append(f"🐌 {model_type}: Inference time approaching limit - optimize for production")
            
            if perf.get("memory_usage_mb", 0) > self.performance_benchmarks["memory_usage_max"] * 0.8:
                recommendations.append(f"💾 {model_type}: Memory usage high - consider model compression")
        
        # Overall recommendations
        passed_models = [k for k, v in self.test_results.items() if v.get("passed", False)]
        if len(passed_models) >= 2:
            recommendations.append("✅ Multiple models passed - ensemble approach recommended for production")
        
        if not recommendations:
            recommendations.append("🎉 All models performing excellently - ready for production deployment")
        
        return recommendations
    
    def save_test_report(self, results: Dict[str, Any], filepath: str = "ml_production_test_report.json"):
        """Save comprehensive test report."""
        report_path = Path(filepath)
        
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Test report saved to {report_path}")
        
        # Also save a human-readable summary
        summary_path = report_path.with_suffix('.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("FX AI-Quant Trading System - ML Production Test Report\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Test Date: {results.get('start_time', 'Unknown')}\n")
            f.write(f"Overall Result: {'PASSED' if results.get('passed', False) else 'FAILED'}\n\n")
            
            summary = results.get('summary', {})
            f.write(f"Tests Passed: {summary.get('passed_tests', 0)}/{summary.get('total_tests', 0)}\n")
            f.write(f"Success Rate: {summary.get('success_rate', 0):.1f}%\n\n")
            
            f.write("Model Test Results:\n")
            f.write("-" * 20 + "\n")
            for model_type, result in results.get('model_tests', {}).items():
                status = "PASSED" if result.get('passed', False) else "FAILED"
                f.write(f"{model_type.upper()}: {status}\n")
                
                errors = result.get('errors', [])
                warnings = result.get('warnings', [])
                if errors:
                    f.write(f"  Errors: {len(errors)}\n")
                if warnings:
                    f.write(f"  Warnings: {len(warnings)}\n")
            
            f.write("\nRecommendations:\n")
            f.write("-" * 15 + "\n")
            for rec in summary.get('recommendations', []):
                # Remove emoji characters for Windows compatibility
                clean_rec = rec.replace('❌', 'X').replace('⚠️', '!').replace('🐌', 'SLOW').replace('💾', 'MEM').replace('✅', 'OK').replace('🎉', 'GREAT')
                f.write(f"• {clean_rec}\n")
        
        logger.info(f"Human-readable summary saved to {summary_path}")
    
    def cleanup(self):
        """Clean up test artifacts."""
        if self.test_data_dir.exists():
            shutil.rmtree(self.test_data_dir)
        
        if self.model_output_dir.exists():
            shutil.rmtree(self.model_output_dir)
        
        logger.info("Test artifacts cleaned up")


async def main():
    """Main function to run the production test suite."""
    print("🚀 Starting FX AI-Quant Trading System - Production ML Test Suite")
    print("=" * 80)
    
    test_suite = ProductionMLTestSuite()
    
    try:
        # Run comprehensive test suite
        results = await test_suite.run_production_test_suite()
        
        # Save detailed report
        test_suite.save_test_report(results)
        
        # Print summary
        print("\n" + "=" * 80)
        print("PRODUCTION TEST SUITE RESULTS")
        print("=" * 80)
        
        summary = results.get('summary', {})
        passed_tests = summary.get('passed_tests', 0)
        total_tests = summary.get('total_tests', 0)
        success_rate = summary.get('success_rate', 0)
        
        print(f"Overall Result: {'🎉 PASSED' if results.get('passed', False) else '💥 FAILED'}")
        print(f"Tests Passed: {passed_tests}/{total_tests}")
        print(f"Success Rate: {success_rate:.1f}%")
        
        print("\nModel Test Results:")
        print("-" * 40)
        for model_type, result in results.get('model_tests', {}).items():
            status = "✅ PASSED" if result.get('passed', False) else "❌ FAILED"
            errors = len(result.get('errors', []))
            warnings = len(result.get('warnings', []))
            print(f"{model_type.upper():12} {status:10} (Errors: {errors}, Warnings: {warnings})")
        
        # Ensemble test
        ensemble_result = results.get('ensemble_test', {})
        if ensemble_result:
            status = "✅ PASSED" if ensemble_result.get('passed', False) else "❌ FAILED"
            errors = len(ensemble_result.get('errors', []))
            warnings = len(ensemble_result.get('warnings', []))
            print(f"{'ENSEMBLE':12} {status:10} (Errors: {errors}, Warnings: {warnings})")
        
        print("\nRecommendations:")
        print("-" * 20)
        for rec in summary.get('recommendations', []):
            print(f"• {rec}")
        
        print("\n" + "=" * 80)
        
        if results.get('passed', False):
            print("🎉 ALL TESTS PASSED - ML MODELS ARE PRODUCTION READY! 🎉")
        else:
            print("⚠️  SOME TESTS FAILED - REVIEW REQUIRED BEFORE PRODUCTION")
        
        print("=" * 80)
        
    except Exception as e:
        logger.error("Production test suite failed", error=str(e), exc_info=True)
        print(f"\n💥 CRITICAL ERROR: {str(e)}")
        print("Check logs for detailed error information.")
    
    finally:
        # Cleanup
        test_suite.cleanup()


if __name__ == "__main__":
    asyncio.run(main()) 