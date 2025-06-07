"""
FastAPI application for the Regime Detection REST API.

This module provides the main FastAPI application with endpoints for:
- Current regime detection
- Historical regime data
- Transition matrix computation
- Health checks
"""

import time
import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any
import structlog
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from api.models import (
    RegimeFeaturesRequest,
    RegimeResponse,
    RegimeHistoryResponse,
    TransitionMatrixResponse,
    HealthResponse,
    ErrorResponse
)
from api.service import RegimeDetectionService, get_service, initialize_service


# Setup logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="ISO"),
        structlog.dev.ConsoleRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Regime Detection API")
    await initialize_service()
    logger.info("Service initialized with sample data")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Regime Detection API")


# Create FastAPI application
app = FastAPI(
    title="FX AI-Quant Regime Detection API",
    description="REST API for market regime detection and analysis",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


# Request ID middleware
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add request ID for tracking."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = str(process_time)
    
    return response


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error="HTTPException",
            message=exc.detail,
            timestamp=datetime.now(timezone.utc),
            request_id=getattr(request.state, 'request_id', None)
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="InternalServerError",
            message="An internal server error occurred",
            timestamp=datetime.now(timezone.utc),
            request_id=getattr(request.state, 'request_id', None)
        ).dict()
    )


# Dependency to get service
def get_regime_service() -> RegimeDetectionService:
    """Dependency to get the regime detection service."""
    return get_service()


# Health check endpoint
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check(service: RegimeDetectionService = Depends(get_regime_service)):
    """
    Health check endpoint.
    
    Returns the current status of the regime detection service.
    """
    health_data = service.get_health_status()
    return HealthResponse(**health_data)


# Current regime endpoint
@app.get("/regime/current", response_model=RegimeResponse, tags=["Regime Detection"])
async def get_current_regime(service: RegimeDetectionService = Depends(get_regime_service)):
    """
    Get the most recent regime detection result.
    
    Returns the latest regime classification with confidence scores.
    """
    current_regime = await service.get_current_regime()
    
    if current_regime is None:
        raise HTTPException(
            status_code=404,
            detail="No regime data available. Submit features first using POST /regime/predict"
        )
    
    return current_regime


# Regime prediction endpoint
@app.post("/regime/predict", response_model=RegimeResponse, tags=["Regime Detection"])
async def predict_regime(
    features: RegimeFeaturesRequest,
    service: RegimeDetectionService = Depends(get_regime_service)
):
    """
    Predict market regime based on provided features.
    
    Analyzes the provided market features and returns the detected regime
    along with confidence scores and contributing factors.
    """
    try:
        result = await service.predict_regime(features)
        return result
    except Exception as e:
        logger.error(f"Error in regime prediction: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to predict regime: {str(e)}"
        )


# Regime history endpoint
@app.get("/regime/history", response_model=RegimeHistoryResponse, tags=["Regime Detection"])
async def get_regime_history(
    window: int = Query(100, ge=1, le=1000, description="Number of recent regime states to return"),
    service: RegimeDetectionService = Depends(get_regime_service)
):
    """
    Get historical regime detection results.
    
    Returns the last N regime states with timestamps and confidence scores.
    """
    try:
        history = await service.get_regime_history(window_size=window)
        return history
    except Exception as e:
        logger.error(f"Error getting regime history: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve regime history: {str(e)}"
        )


# Transition matrix endpoint
@app.get("/regime/transitions", response_model=TransitionMatrixResponse, tags=["Regime Detection"])
async def get_transition_matrix(service: RegimeDetectionService = Depends(get_regime_service)):
    """
    Get regime transition matrix.
    
    Returns the transition probabilities between different market regimes
    based on historical data. Returns null if using non-HMM models or
    insufficient data is available.
    """
    try:
        transitions = await service.get_transition_matrix()
        return transitions
    except Exception as e:
        logger.error(f"Error getting transition matrix: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to compute transition matrix: {str(e)}"
        )


# Additional utility endpoints

@app.get("/regime/stats", tags=["Analytics"])
async def get_regime_statistics(service: RegimeDetectionService = Depends(get_regime_service)):
    """
    Get regime detection statistics.
    
    Returns summary statistics about regime detection performance and history.
    """
    try:
        # Get recent history for statistics
        history = await service.get_regime_history(window_size=1000)
        
        if not history.history:
            return {
                "total_predictions": 0,
                "regime_distribution": {},
                "average_confidence": 0.0,
                "last_updated": None
            }
        
        # Calculate statistics
        total_predictions = len(history.history)
        regime_counts = {}
        confidence_sum = 0.0
        
        for item in history.history:
            regime = item.regime.value
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
            confidence_sum += item.confidence
        
        # Calculate distribution percentages
        regime_distribution = {
            regime: count / total_predictions
            for regime, count in regime_counts.items()
        }
        
        average_confidence = confidence_sum / total_predictions if total_predictions > 0 else 0.0
        
        return {
            "total_predictions": total_predictions,
            "regime_distribution": regime_distribution,
            "average_confidence": round(average_confidence, 3),
            "last_updated": history.end_time,
            "window_analyzed": history.window_size
        }
        
    except Exception as e:
        logger.error(f"Error getting regime statistics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to compute statistics: {str(e)}"
        )


@app.post("/regime/simulate", tags=["Testing"])
async def simulate_regime_data(
    num_samples: int = Query(50, ge=1, le=500, description="Number of samples to simulate"),
    service: RegimeDetectionService = Depends(get_regime_service)
):
    """
    Simulate regime data for testing purposes.
    
    Generates synthetic regime detection data to populate the history
    for testing and demonstration purposes.
    """
    try:
        await service.simulate_regime_data(num_samples=num_samples)
        
        return {
            "message": f"Successfully simulated {num_samples} regime data points",
            "timestamp": datetime.now(timezone.utc),
            "history_size": service.history_store.size()
        }
        
    except Exception as e:
        logger.error(f"Error simulating regime data: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to simulate data: {str(e)}"
        )


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint with API information.
    """
    return {
        "name": "FX AI-Quant Regime Detection API",
        "version": "0.1.0",
        "description": "REST API for market regime detection and analysis",
        "docs_url": "/docs",
        "health_url": "/health",
        "endpoints": {
            "current_regime": "/regime/current",
            "predict_regime": "/regime/predict",
            "regime_history": "/regime/history",
            "transition_matrix": "/regime/transitions",
            "statistics": "/regime/stats",
            "simulate_data": "/regime/simulate"
        }
    }


# Development server
if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 