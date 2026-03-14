# ATRX - Alpha Technology Risk Execution

A production AI-powered quantitative trading platform for FX markets combining regime detection, multi-model ML prediction, LLM-based decision validation, and automated trade execution with multi-layer risk management.

Live since November 2025. 600+ executed trades across XAUUSD, GBPUSD, and USDJPY.

---

## Architecture

```
Market Data Feeds (WebSocket / MT5 API)
         |
         v
+------------------+     +-------------------+     +------------------+
|  Data Ingestion  |---->|  Feature Engine    |---->| Regime Detector  |
|  (Live/Hist)     |     |  (JIT-optimized,  |     | (HMM, GMM,      |
+------------------+     |   70+ indicators)  |     |  KMeans, Rules)  |
                         +-------------------+     +------------------+
                                                          |
                              +---------------------------+
                              v
                    +-------------------+     +-------------------+
                    | ML Predictor      |<--->| Strategy Switcher |
                    | (XGBoost, LSTM,   |     | (Regime-adaptive) |
                    |  CNN, Ensemble)   |     +-------------------+
                    +-------------------+              |
                              |                        v
                    +-------------------+     +-------------------+
                    | LLM Validation    |     | Position Sizer    |
                    | (Gemini API,      |     | (Kelly Criterion, |
                    |  Scenario Gen)    |     |  ATR-based S/L)   |
                    +-------------------+     +-------------------+
                              |                        |
                              v                        v
                    +-------------------+     +-------------------+
                    | Risk Manager      |---->| Execution Engine  |
                    | (3-layer drawdown,|     | (MT5 API,         |
                    |  circuit breakers)|     |  order routing)   |
                    +-------------------+     +-------------------+
                                                       |
                                                       v
                                              +-------------------+
                                              | Monitoring & Logs |
                                              | (PostgreSQL,      |
                                              |  performance DB)  |
                                              +-------------------+
```

## Core Components

### Research Engine (`/core`, `/trainers`, `/backtester`)
25,000+ lines of Python implementing the full quantitative research pipeline:

- **Feature Engine** - 70+ technical indicators with Numba JIT optimization. Computes momentum, volatility, trend, Bollinger, Hurst exponent, Shannon entropy, kurtosis, spread, RSI, MACD, Stochastic, ADX, CCI, and more across configurable lookback windows.
- **Regime Detector** - Market regime classification (trending, mean-reverting, choppy) using Hidden Markov Models, Gaussian Mixture Models, KMeans clustering, DBSCAN, and rule-based fallback logic.
- **ML Predictor** - Multi-model prediction pipeline with XGBoost, LSTM, and CNN models. Includes cross-validation utilities, walk-forward validation, hyperparameter tuning, and ONNX export for optimized inference.
- **Macro Engine** - Economic surprise indices combined with news sentiment analysis to produce unified macro feature vectors for ML models and regime detection.
- **Backtester** - Full backtesting framework with historical market replay, execution simulation (slippage, spread, latency modeling), stress testing, and comprehensive performance metrics (Sharpe, Sortino, Calmar, max drawdown, win rate, profit factor).
- **Trainer Suite** - Dedicated training scripts for each model type with configurable training parameters, early stopping, model versioning, and automated retraining scheduling.

### Execution Layer (`/services`, `/strategies`)
- **Execution Engine** - TWAP and POV order slicing algorithms for large orders. Multi-broker routing with MT5 integration.
- **Risk Manager** - Three-layer drawdown control (position, daily, account). ATR-based dynamic stop-loss and take-profit. Circuit breakers for extreme market conditions. VaR calculations.
- **Position Sizer** - Kelly Criterion optimal sizing with risk-adjusted volatility targeting, maximum allocation caps, and correlation limits.
- **Strategy Library** - Breakout/trend following, time-based scalping, and grid/martingale strategies with regime-adaptive switching.

### LLM Decision Layer
- Multi-tier validation using Google Gemini API
- Structured prompt engineering for pre-session scenario generation
- Dual execution path generation based on macro-conditional analysis
- Structured output parsing with error handling and fallback logic

### API & Infrastructure (`/api`, `/config`)
- FastAPI REST API for system control, health monitoring, and data retrieval
- PostgreSQL-backed persistent state management
- WebSocket connections for live price feed ingestion
- YAML-based hierarchical configuration with environment overrides
- Encrypted secrets management with TLS support
- Audit logging for compliance and debugging

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| Core | Python 3.11+, NumPy, Pandas, SciPy |
| ML | Scikit-learn, XGBoost, TensorFlow/Keras, ONNX Runtime |
| Optimization | Numba JIT |
| API | FastAPI, Uvicorn, Pydantic |
| Database | PostgreSQL |
| Messaging | ZeroMQ |
| Broker | MetaTrader 5 API |
| AI | Google Gemini API |
| Infrastructure | Docker, Git, Linux |

## Performance

| Metric | Result |
|--------|--------|
| Live Trades | 600+ since Nov 2025 |
| Instruments | XAUUSD, GBPUSD, USDJPY |
| Regime Detection | HMM + GMM + KMeans ensemble |
| Feature Count | 70+ JIT-optimized indicators |
| Codebase | 25,000+ lines of Python |

## Project Structure

```
ATRX/
|-- api/                  # FastAPI REST endpoints
|-- backtester/           # Backtesting engine, market replay, stress testing
|-- config/               # YAML configs, secrets management, TLS
|-- core/                 # Feature engine, regime detector, risk manager,
|                         # execution engine, macro engine, ML predictor,
|                         # position sizer, order router, strategy switcher
|-- data/                 # Data connectors (MT5, OANDA, Dukascopy),
|                         # news sentiment, economic calendar
|-- docs/                 # System documentation, audit reports, devlog
|-- evaluation/           # Model evaluation and validation
|-- scripts/              # Training pipelines, data processing, utilities
|-- services/             # Execution service layer
|-- strategies/           # Trading strategy implementations
|-- tools/                # Parameter tuning framework
|-- trainers/             # ML model training (XGBoost, LSTM, CNN),
|                         # cross-validation, retraining scheduler
|-- test_models/          # Trained model artifacts
|-- test_reports/         # Backtest and integration test results
|-- outputs/              # Stress testing and backtest outputs
```

## Setup

```bash
git clone https://github.com/timi-le/ATRX.git
cd ATRX
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Configure environment variables in `config/.env` (see `config/env_template.txt`).

## Current Development

**v3.1** - Live production system with multi-factor alpha, HMM regime detection, LLM validation, and risk management.

**v4.0** (in progress) - Standardized model interface, per-symbol adaptive alpha engine, LangGraph agentic decision pipeline, macro intelligence integration with 109-path causal transmission matrix, Monte Carlo live calibration, overfitting monitor, and continuous model fine-tuning loop.

## Author

**Timilehin Olapade** - ML Engineer & Quantitative Developer
MScFE Candidate, WorldQuant University

## License

Proprietary. All rights reserved.
