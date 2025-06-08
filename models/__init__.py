"""
ML Models Package for FX AI-Quant Trading System.

This package contains machine learning models for financial prediction including:
- LSTM models for sequential pattern recognition
- CNN models for localized pattern detection
- XGBoost models for tabular data
- Ensemble models combining multiple predictors
- ONNX export capabilities for production deployment
"""

from models.predictor_interface import (
    BasePredictorModel,
    ModelConfig,
    ModelType,
    PredictionType,
    TrainingMetrics,
)

try:
    from models.lstm_model import LSTMPredictor

    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False

try:
    from models.cnn_model import CNNPredictor

    CNN_AVAILABLE = True
except ImportError:
    CNN_AVAILABLE = False

try:
    from models.xgboost_model import XGBoostPredictor

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from models.ensemble_model import EnsembleMLPredictor

    ENSEMBLE_AVAILABLE = True
except ImportError:
    ENSEMBLE_AVAILABLE = False

__all__ = [
    # Base interfaces
    "BasePredictorModel",
    "ModelConfig",
    "TrainingMetrics",
    "PredictionType",
    "ModelType",
    # Model availability flags
    "LSTM_AVAILABLE",
    "CNN_AVAILABLE",
    "XGBOOST_AVAILABLE",
    "ENSEMBLE_AVAILABLE",
]

# Add available models to exports
if LSTM_AVAILABLE:
    __all__.extend(["LSTMPredictor", "create_lstm_config"])

if CNN_AVAILABLE:
    __all__.extend(["CNNPredictor", "create_cnn_config"])

if XGBOOST_AVAILABLE:
    __all__.extend(["XGBoostPredictor", "create_xgboost_config"])

if ENSEMBLE_AVAILABLE:
    __all__.extend(["EnsembleMLPredictor", "create_ensemble_config"])


def get_available_models():
    """Get list of available model types."""
    available = []

    if LSTM_AVAILABLE:
        available.append("lstm")
    if CNN_AVAILABLE:
        available.append("cnn")
    if XGBOOST_AVAILABLE:
        available.append("xgboost")
    if ENSEMBLE_AVAILABLE:
        available.append("ensemble")

    return available


def create_model(model_type: str, config: ModelConfig):
    """Factory function to create models by type."""
    model_type = model_type.lower()

    if model_type == "lstm" and LSTM_AVAILABLE:
        return LSTMPredictor(config)
    elif model_type == "cnn" and CNN_AVAILABLE:
        return CNNPredictor(config)
    elif model_type == "xgboost" and XGBOOST_AVAILABLE:
        return XGBoostPredictor(config)
    elif model_type == "ensemble" and ENSEMBLE_AVAILABLE:
        return EnsembleMLPredictor(config)
    else:
        raise ValueError(f"Model type '{model_type}' not available or not supported")


# Package metadata
__version__ = "1.0.0"
__author__ = "FX AI-Quant Trading System"
__description__ = "Machine learning models for financial prediction and trading"
