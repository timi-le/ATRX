# atrx-public

Public reference for ATRX, a production AI trading system live since November 2025 with 600+ executed trades across XAUUSD, GBPUSD, and USDJPY.

_Status: reference · Last updated: 2026-05-13_

---

## Quick start

```bash
git clone https://github.com/timi-le/atrx-public.git
cd atrx-public
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py --help
```

This repository is reference code. Running it against live capital requires the trained model artifacts and broker credentials, which are not published.

## Why this exists

ATRX is a production multi-asset algorithmic trading system. It runs a regime-aware multi-model ML pipeline that ingests live market data, generates probabilistic signals, validates them through a three-tier LLM decision layer, sizes positions under Kelly with volatility targeting, and executes through MetaTrader 5 with three-layer drawdown control.

The live system has been in production since November 2025. This repository is the sanitized public reference. It documents the architecture, the component design, and the engineering approach for technical readers and prospective collaborators. Trained models, broker credentials, and operational tooling stay private.

The integration is the novelty. Regime detection, multi-model ML, and LLM-based decision validation are individually well-explored. Combining them inside a single live execution loop with proper risk discipline and walk-forward calibration is rare. ATRX does this end-to-end.

## How it works

```
              Market data (MT5, OANDA, Dukascopy, WebSocket)
                                 |
                                 v
                       +---------------------+
                       |  Feature engine     |
                       |  70+ JIT indicators |
                       +---------+-----------+
                                 |
              +------------------+------------------+
              v                                     v
      +---------------+                  +-------------------+
      | Regime        |                  | Macro engine      |
      | HMM, GMM,     |                  | Surprise indices, |
      | KMeans, rules |                  | news sentiment    |
      +-------+-------+                  +---------+---------+
              |                                    |
              +------------------+-----------------+
                                 |
                                 v
                       +--------------------+
                       | ML predictor       |
                       | XGBoost, LSTM, CNN |
                       | ONNX inference     |
                       +---------+----------+
                                 |
                                 v
                       +--------------------+
                       | LLM validation     |
                       | Gemini, three-tier |
                       +---------+----------+
                                 |
                                 v
                       +----------------------+
                       | Position sizer       |
                       | Kelly, ATR S/L, vol  |
                       | targeting            |
                       +----------+-----------+
                                  |
                                  v
                       +----------------------+
                       | Risk manager         |
                       | three-layer drawdown,|
                       | circuit breakers, VaR|
                       +----------+-----------+
                                  |
                                  v
                       +----------------------+
                       | Execution engine     |
                       | TWAP, POV, MT5       |
                       +----------+-----------+
                                  |
                                  v
                       +----------------------+
                       | Monitoring           |
                       | PostgreSQL, audit    |
                       +----------------------+
```

Each component is independently testable. The feature engine and regime detector run as JIT-compiled hot paths. The ML predictor and LLM validation are decoupled stages that can be swapped or disabled. Only the risk manager and execution engine touch broker state.

## Components

### Feature engine

70+ technical indicators with Numba JIT optimization. Momentum, volatility, trend, Bollinger bands, Hurst exponent, Shannon entropy, kurtosis, spread, RSI, MACD, Stochastic, ADX, CCI, and others across configurable lookback windows.

### Regime detector

Market regime classification across trending, mean-reverting, and choppy states. Ensemble of Hidden Markov Models, Gaussian Mixture Models, KMeans, DBSCAN, and rule-based fallback.

### ML predictor

XGBoost, LSTM, and CNN models trained per instrument and per regime. Walk-forward validation, hyperparameter tuning, and ONNX export for runtime inference.

### Macro engine

Economic surprise indices combined with news sentiment, fused into macro feature vectors that feed both the ML predictor and the regime detector.

### LLM decision validation

Three-tier validation pipeline against the Google Gemini API. Structured prompt engineering for pre-session scenario generation. Dual execution paths conditional on macro analysis. Structured-output parsing with deterministic fallback when the model returns malformed output.

### Position sizer

Kelly criterion sizing with volatility targeting, maximum allocation caps, and correlation limits. ATR-based dynamic stop-loss and take-profit.

### Risk manager

Three-layer drawdown control across position, daily, and account scopes. Circuit breakers for extreme conditions. Rolling-window VaR.

### Execution engine

TWAP and POV order-slicing for larger orders. Multi-broker routing with MT5 as primary. Latency-aware placement.

### Backtester

Full historical market replay with execution simulation (slippage, spread, latency). Stress testing and standard performance metrics (Sharpe, Sortino, Calmar, max drawdown, win rate, profit factor).

## Tech stack

Python 3.11+, NumPy, Pandas, SciPy, Scikit-learn, XGBoost, TensorFlow/Keras, ONNX Runtime, Numba JIT, FastAPI, Uvicorn, Pydantic, PostgreSQL, ZeroMQ, MetaTrader 5 API, Google Gemini API, Docker.

## Production behavior

| Metric | Value |
|---|---|
| Live since | November 2025 |
| Executed trades | 600+ |
| Instruments | XAUUSD, GBPUSD, USDJPY |
| Codebase | 25,000+ lines of Python |
| Regime detection | HMM, GMM, and KMeans ensemble |
| Feature count | 70+ JIT-optimized indicators |

Detailed P&L, Sharpe, and per-instrument performance figures are tracked privately and not published with this reference. The live system runs against private capital.

## What this is NOT

- Not a deployable trading bot. The runnable production system, trained model artifacts, and broker credentials are private.
- Not financial advice. The code is published for technical reference, not for investment use.
- Not a multi-broker abstraction. MT5 is the primary integration; other brokers are scaffolded but not production-tested in this repo.
- Not benchmarked against external trading systems. Internal benchmarks are tracked privately.

## Repository layout

```
api/            FastAPI REST endpoints
backtester/     Backtesting engine, market replay, stress testing
config/         YAML configs, secrets, TLS
core/           Feature engine, regime detector, risk manager,
                execution engine, macro engine, ML predictor,
                position sizer, order router, strategy switcher
data/           Data connectors (MT5, OANDA, Dukascopy),
                news sentiment, economic calendar
docs/           System documentation
examples/       Reference usage and demos
execution/      Execution and order routing
models/         Model loader and inference interfaces
monitoring/     Logging, metrics, health checks
notebooks/      Research notebooks
scripts/        Training pipelines, data utilities
services/       Long-running service layer
strategies/     Strategy implementations
tests/          Unit and integration tests
tools/          Parameter tuning framework
trainers/       Model training (XGBoost, LSTM, CNN), cross-validation
web_app/        Web interface (reference)
main.py         Entry point
```

## Development

```bash
pytest tests/
pre-commit run --all-files
```

## License

Proprietary. All rights reserved.
