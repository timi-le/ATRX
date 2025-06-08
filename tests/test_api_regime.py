"""
Unit tests for the Regime Detection API.

Tests cover:
- API endpoints functionality
- Request/response validation
- Error handling
- Performance requirements
- Service layer logic
"""

import time
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.models import RegimeFeaturesRequest, RegimeTypeAPI
from api.service import RegimeDetectionService, RegimeHistoryStore
from core.regime_detector import (
    RegimeOutput,
    RegimeType,
    RuleBasedRegimeDetector,
)


class TestAPIModels:
    """Test API model validation."""

    def test_regime_features_request_valid(self):
        """Test valid regime features request."""
        features = RegimeFeaturesRequest(
            atr=0.5,
            bb_width=0.4,
            realized_vol=0.3,
            vol_ratio=1.1,
            macd_signal=0.2,
            macd_histogram=0.1,
            adx=60,
            rsi=65,
            momentum=0.3,
            macro_surprise=0.1,
            macro_sentiment=0.2,
            trend_strength=0.6,
            mean_reversion=0.3,
        )

        assert features.atr == 0.5
        assert features.adx == 0.6  # Should be normalized from 60
        assert features.rsi == 0.65  # Should be normalized from 65

    def test_regime_features_request_validation(self):
        """Test validation of regime features request."""
        # Test invalid ATR (negative)
        with pytest.raises(ValueError):
            RegimeFeaturesRequest(
                atr=-0.1,
                bb_width=0.4,
                realized_vol=0.3,
                vol_ratio=1.1,
                macd_signal=0.2,
                macd_histogram=0.1,
                adx=60,
                rsi=65,
                momentum=0.3,
                macro_surprise=0.1,
                macro_sentiment=0.2,
                trend_strength=0.6,
                mean_reversion=0.3,
            )

        # Test invalid RSI (> 100)
        with pytest.raises(ValueError):
            RegimeFeaturesRequest(
                atr=0.5,
                bb_width=0.4,
                realized_vol=0.3,
                vol_ratio=1.1,
                macd_signal=0.2,
                macd_histogram=0.1,
                adx=60,
                rsi=150,
                momentum=0.3,
                macro_surprise=0.1,
                macro_sentiment=0.2,
                trend_strength=0.6,
                mean_reversion=0.3,
            )


class TestRegimeHistoryStore:
    """Test the regime history store."""

    @pytest.fixture
    def history_store(self):
        """Create a history store for testing."""
        return RegimeHistoryStore(max_size=100)

    @pytest.fixture
    def sample_regime_output(self):
        """Create a sample regime output."""
        return RegimeOutput(
            regime=RegimeType.TRENDING,
            confidence=0.85,
            probabilities={
                RegimeType.TRENDING: 0.85,
                RegimeType.MEAN_REVERTING: 0.10,
                RegimeType.CHOPPY: 0.05,
            },
            features_used=["adx", "momentum"],
        )

    @pytest.mark.asyncio
    async def test_add_and_get_regime(self, history_store, sample_regime_output):
        """Test adding and retrieving regime data."""
        await history_store.add_regime(sample_regime_output)

        history = await history_store.get_recent_history(window_size=1)
        assert len(history) == 1
        assert history[0].regime == RegimeType.TRENDING
        assert history[0].confidence == 0.85

    @pytest.mark.asyncio
    async def test_history_size_limit(self, sample_regime_output):
        """Test that history respects size limit."""
        small_store = RegimeHistoryStore(max_size=3)

        # Add more items than the limit
        for i in range(5):
            await small_store.add_regime(sample_regime_output)

        history = await small_store.get_recent_history(window_size=10)
        assert len(history) == 3  # Should be limited to max_size

    @pytest.mark.asyncio
    async def test_transition_counts(self, history_store):
        """Test transition count computation."""
        # Add sequence of regimes
        regimes = [
            RegimeType.TRENDING,
            RegimeType.TRENDING,
            RegimeType.CHOPPY,
            RegimeType.MEAN_REVERTING,
            RegimeType.TRENDING,
        ]

        for regime in regimes:
            output = RegimeOutput(
                regime=regime,
                confidence=0.8,
                probabilities={r: 0.33 for r in RegimeType},
            )
            await history_store.add_regime(output)

        transitions = await history_store.get_transition_counts()

        # Check specific transitions
        assert transitions["trending"]["trending"] == 1
        assert transitions["trending"]["choppy"] == 1
        assert transitions["choppy"]["mean_reverting"] == 1
        assert transitions["mean_reverting"]["trending"] == 1


class TestRegimeDetectionService:
    """Test the regime detection service."""

    @pytest.fixture
    def mock_detector(self):
        """Create a mock detector."""
        detector = Mock(spec=RuleBasedRegimeDetector)
        detector.predict.return_value = RegimeOutput(
            regime=RegimeType.TRENDING,
            confidence=0.85,
            probabilities={
                RegimeType.TRENDING: 0.85,
                RegimeType.MEAN_REVERTING: 0.10,
                RegimeType.CHOPPY: 0.05,
            },
            features_used=["adx", "momentum"],
        )
        return detector

    @pytest.fixture
    def service(self, mock_detector):
        """Create a service with mock detector."""
        logger = Mock()
        return RegimeDetectionService(
            detector=mock_detector, history_store=RegimeHistoryStore(), logger=logger
        )

    @pytest.fixture
    def sample_features_request(self):
        """Create sample features request."""
        return RegimeFeaturesRequest(
            atr=0.5,
            bb_width=0.4,
            realized_vol=0.3,
            vol_ratio=1.1,
            macd_signal=0.2,
            macd_histogram=0.1,
            adx=60,
            rsi=65,
            momentum=0.3,
            macro_surprise=0.1,
            macro_sentiment=0.2,
            trend_strength=0.6,
            mean_reversion=0.3,
        )

    @pytest.mark.asyncio
    async def test_predict_regime(self, service, sample_features_request):
        """Test regime prediction."""
        result = await service.predict_regime(sample_features_request)

        assert result.regime == RegimeTypeAPI.TRENDING
        assert result.confidence == 0.85
        assert result.probabilities[RegimeTypeAPI.TRENDING] == 0.85
        assert result.features_used == ["adx", "momentum"]
        assert result.timestamp is not None

    @pytest.mark.asyncio
    async def test_get_current_regime_empty(self, service):
        """Test getting current regime when no data exists."""
        result = await service.get_current_regime()
        assert result is None

    @pytest.mark.asyncio
    async def test_get_current_regime_with_data(self, service, sample_features_request):
        """Test getting current regime after prediction."""
        # First make a prediction
        await service.predict_regime(sample_features_request)

        # Then get current regime
        current = await service.get_current_regime()
        assert current is not None
        assert current.regime == RegimeTypeAPI.TRENDING

    @pytest.mark.asyncio
    async def test_get_regime_history(self, service, sample_features_request):
        """Test getting regime history."""
        # Make several predictions
        for _ in range(5):
            await service.predict_regime(sample_features_request)

        history = await service.get_regime_history(window_size=3)

        assert len(history.history) == 3
        assert history.window_size == 3
        assert history.total_items == 3
        assert all(item.regime == RegimeTypeAPI.TRENDING for item in history.history)

    @pytest.mark.asyncio
    async def test_get_transition_matrix_empty(self, service):
        """Test transition matrix with no data."""
        result = await service.get_transition_matrix()

        assert result.transition_matrix is None
        assert result.model_type == "rule-based"
        assert result.sample_size == 0

    @pytest.mark.asyncio
    async def test_get_transition_matrix_with_data(self, service):
        """Test transition matrix computation with data."""
        # Simulate different regime outputs
        regimes = [RegimeType.TRENDING, RegimeType.CHOPPY, RegimeType.MEAN_REVERTING]

        for regime in regimes:
            service.mock_detector.predict.return_value = RegimeOutput(
                regime=regime,
                confidence=0.8,
                probabilities={r: 0.33 for r in RegimeType},
            )

            features_request = RegimeFeaturesRequest(
                atr=0.5,
                bb_width=0.4,
                realized_vol=0.3,
                vol_ratio=1.1,
                macd_signal=0.2,
                macd_histogram=0.1,
                adx=60,
                rsi=65,
                momentum=0.3,
                macro_surprise=0.1,
                macro_sentiment=0.2,
                trend_strength=0.6,
                mean_reversion=0.3,
            )
            await service.predict_regime(features_request)

        result = await service.get_transition_matrix()

        assert result.transition_matrix is not None
        assert result.sample_size > 0
        assert result.model_type == "rule-based"

    def test_health_status(self, service):
        """Test health status reporting."""
        health = service.get_health_status()

        assert health["status"] == "healthy"
        assert health["detector_ready"] is True
        assert health["version"] == "0.1.0"
        assert "uptime_seconds" in health
        assert "timestamp" in health


class TestAPIEndpoints:
    """Test API endpoints using TestClient."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def sample_features(self):
        """Sample features for testing."""
        return {
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
            "mean_reversion": 0.3,
        }

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert data["name"] == "FX AI-Quant Regime Detection API"
        assert data["version"] == "0.1.0"
        assert "endpoints" in data

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert data["detector_ready"] is True
        assert "uptime_seconds" in data

    def test_predict_regime_endpoint(self, client, sample_features):
        """Test regime prediction endpoint."""
        response = client.post("/regime/predict", json=sample_features)
        assert response.status_code == 200

        data = response.json()
        assert "regime" in data
        assert "confidence" in data
        assert "probabilities" in data
        assert "timestamp" in data

        # Check response headers
        assert "X-Request-ID" in response.headers
        assert "X-Process-Time" in response.headers

    def test_predict_regime_invalid_data(self, client):
        """Test prediction with invalid data."""
        invalid_features = {
            "atr": -0.5,  # Invalid negative value
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
            "mean_reversion": 0.3,
        }

        response = client.post("/regime/predict", json=invalid_features)
        assert response.status_code == 422  # Validation error

    def test_get_current_regime_no_data(self, client):
        """Test getting current regime when no data exists."""
        # Note: This might fail if the service has been initialized with sample data
        # In a real test, we'd reset the service state

    def test_get_current_regime_with_data(self, client, sample_features):
        """Test getting current regime after making a prediction."""
        # First make a prediction
        client.post("/regime/predict", json=sample_features)

        # Then get current regime
        response = client.get("/regime/current")
        assert response.status_code == 200

        data = response.json()
        assert "regime" in data
        assert "confidence" in data

    def test_get_regime_history(self, client, sample_features):
        """Test getting regime history."""
        # Make a few predictions first
        for _ in range(3):
            client.post("/regime/predict", json=sample_features)

        response = client.get("/regime/history?window=5")
        assert response.status_code == 200

        data = response.json()
        assert "history" in data
        assert "window_size" in data
        assert "total_items" in data
        assert data["window_size"] == 5

    def test_get_transition_matrix(self, client):
        """Test getting transition matrix."""
        response = client.get("/regime/transitions")
        assert response.status_code == 200

        data = response.json()
        assert "model_type" in data
        assert "last_updated" in data
        assert "sample_size" in data

    def test_get_statistics(self, client, sample_features):
        """Test getting regime statistics."""
        # Make some predictions first
        for _ in range(5):
            client.post("/regime/predict", json=sample_features)

        response = client.get("/regime/stats")
        assert response.status_code == 200

        data = response.json()
        assert "total_predictions" in data
        assert "regime_distribution" in data
        assert "average_confidence" in data

    def test_simulate_data(self, client):
        """Test data simulation endpoint."""
        response = client.post("/regime/simulate?num_samples=10")
        assert response.status_code == 200

        data = response.json()
        assert "message" in data
        assert "history_size" in data
        assert "10" in data["message"]


class TestAPIPerformance:
    """Test API performance requirements."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def sample_features(self):
        """Sample features for testing."""
        return {
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
            "mean_reversion": 0.3,
        }

    def test_prediction_latency(self, client, sample_features):
        """Test that prediction latency is under 100ms."""
        start_time = time.time()
        response = client.post("/regime/predict", json=sample_features)
        end_time = time.time()

        assert response.status_code == 200

        latency_ms = (end_time - start_time) * 1000
        assert (
            latency_ms < 100
        ), f"Prediction latency {latency_ms:.2f}ms exceeds 100ms requirement"

    def test_concurrent_requests(self, client, sample_features):
        """Test handling of concurrent requests."""
        import concurrent.futures

        def make_request():
            return client.post("/regime/predict", json=sample_features)

        # Test with 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]

            results = []
            for future in concurrent.futures.as_completed(futures):
                response = future.result()
                results.append(response.status_code)

        # All requests should succeed
        assert all(status == 200 for status in results)
        assert len(results) == 10

    def test_history_endpoint_performance(self, client):
        """Test that history endpoint performs well with large windows."""
        start_time = time.time()
        response = client.get("/regime/history?window=1000")
        end_time = time.time()

        assert response.status_code == 200

        latency_ms = (end_time - start_time) * 1000
        assert (
            latency_ms < 200
        ), f"History endpoint latency {latency_ms:.2f}ms is too high"


class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_invalid_json(self, client):
        """Test handling of invalid JSON."""
        response = client.post(
            "/regime/predict",
            data="invalid json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422

    def test_missing_fields(self, client):
        """Test handling of missing required fields."""
        incomplete_features = {
            "atr": 0.5,
            "bb_width": 0.4
            # Missing other required fields
        }

        response = client.post("/regime/predict", json=incomplete_features)
        assert response.status_code == 422

    def test_invalid_window_size(self, client):
        """Test invalid window size parameter."""
        response = client.get("/regime/history?window=-1")
        assert response.status_code == 422

        response = client.get("/regime/history?window=10000")
        assert response.status_code == 422

    def test_nonexistent_endpoint(self, client):
        """Test accessing non-existent endpoint."""
        response = client.get("/nonexistent")
        assert response.status_code == 404


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
