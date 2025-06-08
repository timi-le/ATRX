# FX AI-Quant Regime Detection API

A high-performance REST API for real-time market regime detection in FX trading systems.

## Overview

The Regime Detection API provides real-time classification of market conditions into three primary regimes:
- **Trending**: Strong directional movement with high momentum
- **Mean-Reverting**: Range-bound oscillation around a mean
- **Choppy**: High volatility with low directional bias

## Features

- ⚡ **Low Latency**: <100ms response time for regime predictions
- 🔄 **Async Support**: Handles concurrent requests efficiently
- 📊 **Historical Data**: Access to regime history and transition matrices
- 🛡️ **Robust Error Handling**: Comprehensive validation and error responses
- 📈 **Performance Monitoring**: Built-in metrics and health checks
- 🔧 **Easy Integration**: RESTful API with OpenAPI documentation

## Quick Start

### 1. Start the API Server

```bash
# Using uvicorn directly
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Or using the demo script
python test_api_demo.py
```

### 2. Check Health Status

```bash
curl http://localhost:8000/health
```

### 3. Make a Regime Prediction

```bash
curl -X POST "http://localhost:8000/regime/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

## API Endpoints

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check and service status |
| GET | `/regime/current` | Get the most recent regime detection |
| POST | `/regime/predict` | Predict regime for given features |
| GET | `/regime/history` | Get historical regime data |
| GET | `/regime/transitions` | Get regime transition matrix |

### Analytics Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/regime/stats` | Get regime detection statistics |
| POST | `/regime/simulate` | Simulate regime data for testing |

### Documentation

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API information and endpoint list |
| GET | `/docs` | Interactive OpenAPI documentation |
| GET | `/redoc` | Alternative API documentation |

## Request/Response Examples

### Regime Prediction Request

```json
{
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
```

### Regime Prediction Response

```json
{
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
```

### Historical Data Response

```json
{
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
```

## Feature Specifications

### Input Features (13 total)

#### Volatility Features (4)
- `atr`: Average True Range (0-1, normalized)
- `bb_width`: Bollinger Band width (0-1, normalized)
- `realized_vol`: Realized volatility (0-1, normalized)
- `vol_ratio`: Current vol / historical vol (0-5)

#### Momentum Features (5)
- `macd_signal`: MACD signal strength (-1 to 1)
- `macd_histogram`: MACD histogram (-1 to 1)
- `adx`: Average Directional Index (0-100)
- `rsi`: RSI (0-100)
- `momentum`: Price momentum (-1 to 1)

#### Macro Features (2)
- `macro_surprise`: Economic surprise index (-1 to 1)
- `macro_sentiment`: News sentiment (-1 to 1)

#### Market Structure Features (2)
- `trend_strength`: Trend strength (0 to 1)
- `mean_reversion`: Mean reversion tendency (0 to 1)

## Performance Requirements

- **Latency**: <100ms per prediction
- **Throughput**: >1000 requests/second
- **Concurrent Requests**: Supports multiple simultaneous connections
- **Memory Usage**: Efficient in-memory history storage
- **Error Rate**: <0.1% under normal conditions

## Error Handling

The API provides comprehensive error handling with structured error responses:

```json
{
  "error": "ValidationError",
  "message": "Invalid feature values provided",
  "timestamp": "2024-01-15T10:30:00Z",
  "request_id": "req_123456"
}
```

### Common Error Codes

- `400`: Bad Request - Invalid JSON or malformed request
- `422`: Validation Error - Invalid feature values or missing fields
- `404`: Not Found - Endpoint or resource not found
- `500`: Internal Server Error - Unexpected server error

## Testing

### Run Unit Tests

```bash
pytest tests/test_api_regime.py -v
```

### Run Comprehensive Demo

```bash
python test_api_demo.py
```

### Performance Testing

```bash
# Test latency
curl -w "@curl-format.txt" -X POST "http://localhost:8000/regime/predict" \
  -H "Content-Type: application/json" \
  -d @sample_features.json

# Load testing with Apache Bench
ab -n 1000 -c 10 -T application/json -p sample_features.json \
  http://localhost:8000/regime/predict
```

## Integration Examples

### Python Client

```python
import requests
import json

class RegimeDetectionClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()

    def predict_regime(self, features):
        response = self.session.post(
            f"{self.base_url}/regime/predict",
            json=features
        )
        return response.json()

    def get_current_regime(self):
        response = self.session.get(f"{self.base_url}/regime/current")
        return response.json()

# Usage
client = RegimeDetectionClient()
features = {
    "atr": 0.5, "bb_width": 0.4, "realized_vol": 0.3,
    # ... other features
}
result = client.predict_regime(features)
print(f"Detected regime: {result['regime']}")
```

### JavaScript/Node.js Client

```javascript
const axios = require('axios');

class RegimeDetectionClient {
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
        this.client = axios.create({
            baseURL: baseUrl,
            headers: { 'Content-Type': 'application/json' }
        });
    }

    async predictRegime(features) {
        const response = await this.client.post('/regime/predict', features);
        return response.data;
    }

    async getCurrentRegime() {
        const response = await this.client.get('/regime/current');
        return response.data;
    }
}

// Usage
const client = new RegimeDetectionClient();
const features = {
    atr: 0.5, bb_width: 0.4, realized_vol: 0.3,
    // ... other features
};

client.predictRegime(features)
    .then(result => console.log(`Detected regime: ${result.regime}`))
    .catch(error => console.error('Error:', error));
```

## Configuration

### Environment Variables

- `API_HOST`: Host to bind the server (default: 0.0.0.0)
- `API_PORT`: Port to bind the server (default: 8000)
- `LOG_LEVEL`: Logging level (default: info)
- `CORS_ORIGINS`: Allowed CORS origins (default: *)

### Production Deployment

```bash
# Using Gunicorn for production
gunicorn api.main:app -w 4 -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 --access-logfile - --error-logfile -

# Using Docker
docker build -t regime-detection-api .
docker run -p 8000:8000 regime-detection-api
```

## Monitoring

### Health Check

The `/health` endpoint provides comprehensive service status:

```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "0.1.0",
  "detector_ready": true,
  "uptime_seconds": 3600.5
}
```

### Metrics

- Request latency (via `X-Process-Time` header)
- Request tracking (via `X-Request-ID` header)
- Service uptime and status
- Regime detection statistics

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI App   │    │  Service Layer  │    │ Regime Detector │
│                 │    │                 │    │                 │
│ • Endpoints     │───▶│ • Business      │───▶│ • Rule-based    │
│ • Validation    │    │   Logic         │    │ • ML Models     │
│ • Error         │    │ • Caching       │    │ • Feature       │
│   Handling      │    │ • History       │    │   Processing    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Middleware    │    │ History Store   │    │ Configuration   │
│                 │    │                 │    │                 │
│ • CORS          │    │ • In-memory     │    │ • Thresholds    │
│ • Compression   │    │ • Thread-safe   │    │ • Model Params  │
│ • Request ID    │    │ • Size Limits   │    │ • Validation    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is part of the FX AI-Quant Trading System and is proprietary software.

## Support

For issues and questions:
- Check the `/docs` endpoint for interactive documentation
- Review the test files for usage examples
- Run the demo script for comprehensive testing
