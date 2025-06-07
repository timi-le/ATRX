"""
Pydantic models for the Regime Detection API.

These models define the request and response schemas for the REST API endpoints.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator
from enum import Enum


class RegimeTypeAPI(str, Enum):
    """API representation of regime types."""
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
    CHOPPY = "choppy"


class RegimeFeaturesRequest(BaseModel):
    """Request model for regime features input."""
    
    # Volatility features (4)
    atr: float = Field(..., ge=0, le=1, description="Average True Range (normalized)")
    bb_width: float = Field(..., ge=0, le=1, description="Bollinger Band width (normalized)")
    realized_vol: float = Field(..., ge=0, le=1, description="Realized volatility (normalized)")
    vol_ratio: float = Field(..., ge=0, le=5, description="Current vol / historical vol")
    
    # Momentum features (5)
    macd_signal: float = Field(..., ge=-1, le=1, description="MACD signal strength")
    macd_histogram: float = Field(..., ge=-1, le=1, description="MACD histogram")
    adx: float = Field(..., ge=0, le=100, description="Average Directional Index")
    rsi: float = Field(..., ge=0, le=100, description="RSI")
    momentum: float = Field(..., ge=-1, le=1, description="Price momentum")
    
    # Macro features (2)
    macro_surprise: float = Field(..., ge=-1, le=1, description="Economic surprise index")
    macro_sentiment: float = Field(..., ge=-1, le=1, description="News sentiment")
    
    # Market structure features (2)
    trend_strength: float = Field(..., ge=0, le=1, description="Trend strength")
    mean_reversion: float = Field(..., ge=0, le=1, description="Mean reversion tendency")
    
    @validator('adx', 'rsi')
    def normalize_percentage_fields(cls, v):
        """Normalize ADX and RSI to 0-1 range for internal use."""
        return v / 100.0 if v > 1 else v
    
    class Config:
        schema_extra = {
            "example": {
                "atr": 0.5,
                "bb_width": 0.4,
                "realized_vol": 0.3,
                "vol_ratio": 1.1,
                "macd_signal": 0.2,
                "macd_histogram": 0.1,
                "adx": 60,
                "rsi": 65,
                "momentum": 0.3,
                "macro_surprise": 0.1,
                "macro_sentiment": 0.2,
                "trend_strength": 0.6,
                "mean_reversion": 0.3
            }
        }


class RegimeResponse(BaseModel):
    """Response model for regime detection."""
    
    regime: RegimeTypeAPI = Field(..., description="Detected market regime")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")
    probabilities: Dict[RegimeTypeAPI, float] = Field(..., description="Probability for each regime")
    transition_prob: Optional[float] = Field(None, ge=0, le=1, description="Probability of regime transition")
    features_used: Optional[List[str]] = Field(None, description="Features that contributed most")
    timestamp: datetime = Field(..., description="Timestamp of the prediction")
    
    class Config:
        schema_extra = {
            "example": {
                "regime": "trending",
                "confidence": 0.85,
                "probabilities": {
                    "trending": 0.85,
                    "mean_reverting": 0.10,
                    "choppy": 0.05
                },
                "transition_prob": 0.15,
                "features_used": ["adx", "momentum", "trend_strength"],
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


class RegimeHistoryItem(BaseModel):
    """Single item in regime history."""
    
    timestamp: datetime = Field(..., description="Timestamp of the regime detection")
    regime: RegimeTypeAPI = Field(..., description="Detected regime")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")
    probabilities: Dict[RegimeTypeAPI, float] = Field(..., description="Regime probabilities")


class RegimeHistoryResponse(BaseModel):
    """Response model for regime history."""
    
    history: List[RegimeHistoryItem] = Field(..., description="Historical regime data")
    window_size: int = Field(..., description="Number of items requested")
    total_items: int = Field(..., description="Total items returned")
    start_time: datetime = Field(..., description="Start time of the window")
    end_time: datetime = Field(..., description="End time of the window")
    
    class Config:
        schema_extra = {
            "example": {
                "history": [
                    {
                        "timestamp": "2024-01-15T10:30:00Z",
                        "regime": "trending",
                        "confidence": 0.85,
                        "probabilities": {
                            "trending": 0.85,
                            "mean_reverting": 0.10,
                            "choppy": 0.05
                        }
                    }
                ],
                "window_size": 100,
                "total_items": 1,
                "start_time": "2024-01-15T09:30:00Z",
                "end_time": "2024-01-15T10:30:00Z"
            }
        }


class TransitionMatrixResponse(BaseModel):
    """Response model for regime transition matrix."""
    
    transition_matrix: Optional[Dict[RegimeTypeAPI, Dict[RegimeTypeAPI, float]]] = Field(
        None, description="Transition probabilities between regimes"
    )
    model_type: str = Field(..., description="Type of model used (e.g., 'hmm', 'rule-based')")
    last_updated: datetime = Field(..., description="When the matrix was last computed")
    sample_size: Optional[int] = Field(None, description="Number of samples used to compute matrix")
    
    class Config:
        schema_extra = {
            "example": {
                "transition_matrix": {
                    "trending": {
                        "trending": 0.7,
                        "mean_reverting": 0.2,
                        "choppy": 0.1
                    },
                    "mean_reverting": {
                        "trending": 0.3,
                        "mean_reverting": 0.6,
                        "choppy": 0.1
                    },
                    "choppy": {
                        "trending": 0.4,
                        "mean_reverting": 0.3,
                        "choppy": 0.3
                    }
                },
                "model_type": "hmm",
                "last_updated": "2024-01-15T10:00:00Z",
                "sample_size": 1000
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Current timestamp")
    version: str = Field(..., description="API version")
    detector_ready: bool = Field(..., description="Whether the regime detector is ready")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")


class ErrorResponse(BaseModel):
    """Error response model."""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    timestamp: datetime = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")
    
    class Config:
        schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Invalid feature values provided",
                "timestamp": "2024-01-15T10:30:00Z",
                "request_id": "req_123456"
            }
        } 