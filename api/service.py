"""
Service layer for the Regime Detection API.

This module provides the business logic for regime detection, including:
- Managing the regime detector instance
- Storing and retrieving regime history
- Computing transition matrices
- Caching for performance
"""

import asyncio
import time
from collections import deque
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Deque, Any
import structlog
import numpy as np
from dataclasses import asdict

from core.regime_detector import (
    RegimeFeatures,
    RegimeOutput,
    RuleBasedRegimeDetector,
    RegimeType
)
from api.models import (
    RegimeTypeAPI,
    RegimeResponse,
    RegimeHistoryItem,
    RegimeHistoryResponse,
    TransitionMatrixResponse,
    RegimeFeaturesRequest
)


class RegimeHistoryStore:
    """In-memory store for regime history with configurable retention."""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.history: Deque[RegimeOutput] = deque(maxlen=max_size)
        self._lock = asyncio.Lock()
    
    async def add_regime(self, regime_output: RegimeOutput) -> None:
        """Add a new regime detection result to history."""
        async with self._lock:
            self.history.append(regime_output)
    
    async def get_recent_history(self, window_size: int = 100) -> List[RegimeOutput]:
        """Get the most recent regime history."""
        async with self._lock:
            # Return the last window_size items
            return list(self.history)[-window_size:]
    
    async def get_history_by_time(
        self, 
        start_time: datetime, 
        end_time: datetime
    ) -> List[RegimeOutput]:
        """Get regime history within a time range."""
        async with self._lock:
            filtered_history = []
            for regime_output in self.history:
                # For now, we'll use current time as timestamp since RegimeOutput doesn't have it
                # In a real implementation, you'd add timestamp to RegimeOutput
                timestamp = datetime.now(timezone.utc)
                if start_time <= timestamp <= end_time:
                    filtered_history.append(regime_output)
            return filtered_history
    
    async def get_transition_counts(self) -> Dict[str, Dict[str, int]]:
        """Compute transition counts from history."""
        async with self._lock:
            if len(self.history) < 2:
                return {}
            
            # Initialize transition counts
            regimes = ["trending", "mean_reverting", "choppy"]
            transition_counts = {
                regime: {target: 0 for target in regimes}
                for regime in regimes
            }
            
            # Count transitions
            for i in range(len(self.history) - 1):
                current_regime = self.history[i].regime.value
                next_regime = self.history[i + 1].regime.value
                transition_counts[current_regime][next_regime] += 1
            
            return transition_counts
    
    def size(self) -> int:
        """Get current history size."""
        return len(self.history)


class RegimeDetectionService:
    """Main service for regime detection operations."""
    
    def __init__(
        self,
        detector: Optional[RuleBasedRegimeDetector] = None,
        history_store: Optional[RegimeHistoryStore] = None,
        logger: Optional[structlog.stdlib.BoundLogger] = None
    ):
        self.logger = logger or structlog.get_logger(__name__)
        self.detector = detector or RuleBasedRegimeDetector(logger=self.logger)
        self.history_store = history_store or RegimeHistoryStore()
        self.start_time = time.time()
        
        # Cache for transition matrix
        self._transition_matrix_cache: Optional[Dict[str, Any]] = None
        self._transition_matrix_last_update: Optional[datetime] = None
        self._cache_ttl_seconds = 300  # 5 minutes
        
        self.logger.info("RegimeDetectionService initialized")
    
    async def predict_regime(self, features_request: RegimeFeaturesRequest) -> RegimeResponse:
        """Predict regime for given features and store in history."""
        start_time = time.time()
        
        try:
            # Convert API request to internal features
            features = RegimeFeatures(
                atr=features_request.atr,
                bb_width=features_request.bb_width,
                realized_vol=features_request.realized_vol,
                vol_ratio=features_request.vol_ratio,
                macd_signal=features_request.macd_signal,
                macd_histogram=features_request.macd_histogram,
                adx=features_request.adx,
                rsi=features_request.rsi,
                momentum=features_request.momentum,
                macro_surprise=features_request.macro_surprise,
                macro_sentiment=features_request.macro_sentiment,
                trend_strength=features_request.trend_strength,
                mean_reversion=features_request.mean_reversion
            )
            
            # Get prediction from detector
            regime_output = self.detector.predict(features)
            
            # Store in history
            await self.history_store.add_regime(regime_output)
            
            # Convert to API response
            response = RegimeResponse(
                regime=RegimeTypeAPI(regime_output.regime.value),
                confidence=regime_output.confidence,
                probabilities={
                    RegimeTypeAPI(k.value): v 
                    for k, v in regime_output.probabilities.items()
                },
                transition_prob=regime_output.transition_prob,
                features_used=regime_output.features_used,
                timestamp=datetime.now(timezone.utc)
            )
            
            # Log performance
            elapsed_ms = (time.time() - start_time) * 1000
            self.logger.info(
                "Regime prediction completed",
                regime=regime_output.regime.value,
                confidence=regime_output.confidence,
                elapsed_ms=elapsed_ms
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error in regime prediction: {e}")
            raise
    
    async def get_current_regime(self) -> Optional[RegimeResponse]:
        """Get the most recent regime prediction."""
        try:
            recent_history = await self.history_store.get_recent_history(window_size=1)
            
            if not recent_history:
                return None
            
            latest = recent_history[0]
            return RegimeResponse(
                regime=RegimeTypeAPI(latest.regime.value),
                confidence=latest.confidence,
                probabilities={
                    RegimeTypeAPI(k.value): v 
                    for k, v in latest.probabilities.items()
                },
                transition_prob=latest.transition_prob,
                features_used=latest.features_used,
                timestamp=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            self.logger.error(f"Error getting current regime: {e}")
            raise
    
    async def get_regime_history(
        self, 
        window_size: int = 100,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> RegimeHistoryResponse:
        """Get regime history with optional time filtering."""
        try:
            if start_time and end_time:
                history = await self.history_store.get_history_by_time(start_time, end_time)
            else:
                history = await self.history_store.get_recent_history(window_size)
            
            # Convert to API format
            history_items = []
            for regime_output in history:
                item = RegimeHistoryItem(
                    timestamp=datetime.now(timezone.utc),  # Would use actual timestamp in real implementation
                    regime=RegimeTypeAPI(regime_output.regime.value),
                    confidence=regime_output.confidence,
                    probabilities={
                        RegimeTypeAPI(k.value): v 
                        for k, v in regime_output.probabilities.items()
                    }
                )
                history_items.append(item)
            
            # Determine time window
            if history_items:
                actual_start = min(item.timestamp for item in history_items)
                actual_end = max(item.timestamp for item in history_items)
            else:
                now = datetime.now(timezone.utc)
                actual_start = actual_end = now
            
            return RegimeHistoryResponse(
                history=history_items,
                window_size=window_size,
                total_items=len(history_items),
                start_time=actual_start,
                end_time=actual_end
            )
            
        except Exception as e:
            self.logger.error(f"Error getting regime history: {e}")
            raise
    
    async def get_transition_matrix(self) -> TransitionMatrixResponse:
        """Get regime transition matrix with caching."""
        try:
            # Check cache
            now = datetime.now(timezone.utc)
            if (self._transition_matrix_cache and 
                self._transition_matrix_last_update and
                (now - self._transition_matrix_last_update).total_seconds() < self._cache_ttl_seconds):
                
                return TransitionMatrixResponse(**self._transition_matrix_cache)
            
            # Compute transition matrix
            transition_counts = await self.history_store.get_transition_counts()
            
            if not transition_counts:
                # No history available
                return TransitionMatrixResponse(
                    transition_matrix=None,
                    model_type="rule-based",
                    last_updated=now,
                    sample_size=0
                )
            
            # Convert counts to probabilities
            transition_matrix = {}
            total_samples = 0
            
            for from_regime, to_counts in transition_counts.items():
                total_from = sum(to_counts.values())
                if total_from > 0:
                    transition_matrix[RegimeTypeAPI(from_regime)] = {
                        RegimeTypeAPI(to_regime): count / total_from
                        for to_regime, count in to_counts.items()
                    }
                    total_samples += total_from
            
            # Cache the result
            response_data = {
                "transition_matrix": transition_matrix if transition_matrix else None,
                "model_type": "rule-based",
                "last_updated": now,
                "sample_size": total_samples
            }
            
            self._transition_matrix_cache = response_data
            self._transition_matrix_last_update = now
            
            return TransitionMatrixResponse(**response_data)
            
        except Exception as e:
            self.logger.error(f"Error computing transition matrix: {e}")
            raise
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get service health status."""
        uptime = time.time() - self.start_time
        
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc),
            "version": "0.1.0",
            "detector_ready": True,
            "uptime_seconds": uptime,
            "history_size": self.history_store.size()
        }
    
    async def simulate_regime_data(self, num_samples: int = 50) -> None:
        """Simulate some regime data for testing purposes."""
        self.logger.info(f"Simulating {num_samples} regime data points")
        
        from core.regime_detector import create_sample_features
        
        # Generate sample features for different regimes
        for regime_type in RegimeType:
            samples = create_sample_features(
                n_samples=num_samples // 3,
                regime_type=regime_type
            )
            
            for sample in samples:
                regime_output = self.detector.predict(sample)
                await self.history_store.add_regime(regime_output)
                
                # Small delay to simulate real-time data
                await asyncio.sleep(0.001)
        
        self.logger.info(f"Simulation complete. History size: {self.history_store.size()}")


# Global service instance
_service_instance: Optional[RegimeDetectionService] = None


def get_service() -> RegimeDetectionService:
    """Get or create the global service instance."""
    global _service_instance
    
    if _service_instance is None:
        logger = structlog.get_logger(__name__)
        _service_instance = RegimeDetectionService(logger=logger)
    
    return _service_instance


async def initialize_service() -> RegimeDetectionService:
    """Initialize the service with some sample data."""
    service = get_service()
    
    # Simulate some historical data for demonstration
    await service.simulate_regime_data(num_samples=100)
    
    return service 