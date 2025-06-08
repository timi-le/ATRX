# FX AI-Quant Trading System

A sophisticated algorithmic trading system for FX markets that combines regime detection, machine learning forecasting, and optimal position sizing using the Kelly criterion.

## 🏗️ System Architecture

The system is built with a modular, microservices-oriented architecture:

```
┌────────────┐     ┌──────────────┐     ┌────────────────┐
│ MarketData │──▶──│ Data Ingest  │──▶──│ Feature Engine │
│   Feeds    │     │  (Pub/Sub)   │     │ (Indicators,   │
└────────────┘     └──────────────┘     │  Surprises)    │
                                           └──────────────┘
                                                 │
                                                 ▼
                                          ┌──────────────┐
                                          │ Regime Detector│
                                          └──────────────┘
                                                 │
                                                 ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ ML Predictor │◀───▶│ Strategy      │◀───▶│ Position     │
│ (LSTM/CNN/   │     │  Switcher     │     │ Sizer (Kelly)│
│  Ensemble)   │     └──────────────┘     └──────────────┘
└──────────────┘                               │
        │                                       ▼
        ▼                                ┌──────────────┐
 ┌──────────────┐                         │ Execution     │
 │ Risk Manager │                         │ Engine &      │
 │ (Drawdown,   │                         │ Order Router  │
 │  Limits)     │                         └──────────────┘
 └──────────────┘                               │
        │                                       ▼
        └──────────────────────────────────▶─ P&L / Monitoring
```

## 📁 Project Structure

```
Fx_Quant_System/
├── core/                     # Core system interfaces and utilities
│   ├── interfaces/           # Abstract base classes and protocols
│   ├── messaging/            # ZeroMQ/Redis communication layer
│   └── config/               # Configuration management
├── data/                     # Data storage and processing
│   ├── raw/                  # Raw market data
│   ├── processed/            # Processed features
│   └── historical/           # Historical data cache
├── models/                   # ML models and regime detection
│   ├── regime/               # Regime detection models
│   ├── ml/                   # LSTM/CNN prediction models
│   └── ensemble/             # Ensemble model combinations
├── strategies/               # Trading strategies
│   ├── scalping/             # High-frequency scalping
│   ├── breakout/             # Momentum/breakout strategies
│   └── arbitrage/            # Statistical arbitrage
├── execution/                # Order execution and management
│   ├── order_management/     # Order lifecycle management
│   ├── brokers/              # Broker integrations (MT5, IBKR)
│   └── risk_management/      # Risk controls and limits
├── tests/                    # Test suites
│   ├── unit/                 # Unit tests
│   ├── integration/          # Integration tests
│   └── performance/          # Performance tests
├── monitoring/               # Monitoring and dashboards
│   ├── metrics/              # Prometheus metrics
│   └── dashboard/            # Grafana dashboards
├── scripts/                  # Utility scripts and PRD
└── tasks/                    # Task management (TaskMaster)
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- Git

### 1. Clone and Setup

```bash
git clone <repository-url>
cd Fx_Quant_System

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration

Create a `.env` file in the project root:

```bash
# Data Provider APIs
OANDA_API_KEY=your_oanda_api_key
OANDA_ACCOUNT_ID=your_account_id

# Database
POSTGRES_HOST=localhost
POSTGRES_DB=fx_trading
POSTGRES_USER=trading_user
POSTGRES_PASSWORD=trading_password

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# Environment
ENVIRONMENT=development
DEBUG=true
```

### 3. Start Infrastructure Services

```bash
# Start Redis, PostgreSQL, Prometheus, and Grafana
docker-compose up -d redis postgres prometheus grafana
```

### 4. Run the Trading System

```bash
# Start the main trading system
python main.py
```

## 📊 Monitoring and Dashboards

- **Grafana Dashboard**: http://localhost:3000 (admin/admin123)
- **Prometheus Metrics**: http://localhost:9090
- **System Health**: http://localhost:8000/health

## 🔧 Configuration

The system uses hierarchical configuration with environment variable overrides:

```python
from core.config import SystemConfig

config = SystemConfig(
    environment="development",
    data=DataConfig(
        fx_pairs=["EURUSD", "GBPUSD", "USDJPY"],
        oanda_enabled=True
    ),
    trading=TradingConfig(
        kelly_multiplier=0.25,
        max_position_size=0.1
    ),
    risk=RiskConfig(
        max_daily_drawdown=0.02,
        var_limit=0.05
    )
)
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/performance/

# Run with coverage
pytest --cov=. --cov-report=html
```

## 📈 Key Features

### 1. Regime Detection
- **Hidden Markov Models (HMM)** for regime classification
- **K-means clustering** on volatility/momentum features
- Three regimes: Trending, Mean-Reverting, Choppy

### 2. Machine Learning Prediction
- **LSTM** for sequential pattern learning
- **CNN** for local pattern detection
- **Ensemble methods** combining multiple models
- **ONNX Runtime** for optimized inference

### 3. Strategy Switching
- **Dynamic strategy allocation** based on regime
- **Configurable weights** for scalping/breakout/arbitrage
- **Signal aggregation** with confidence weighting

### 4. Position Sizing
- **Kelly Criterion** for optimal position sizing: `f* = (bp - q) / b`
- **Risk-adjusted sizing** with volatility targeting
- **Maximum allocation caps** and correlation limits

### 5. Risk Management
- **Real-time drawdown monitoring**
- **VaR calculations** with 95% confidence
- **Position limits** and emergency stops
- **Circuit breakers** for extreme market conditions

### 6. Execution Engine
- **TWAP/POV order slicing** for large orders
- **Multi-broker support** (MT5, Interactive Brokers)
- **Execution quality monitoring**
- **Latency optimization** (<100ms target)

## 🎯 Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| Sharpe Ratio | > 1.5 | 6-month live test |
| Regime Accuracy | > 80% | vs labeled data |
| ML Prediction | 20% MSE reduction | vs AR(1) baseline |
| Execution Slippage | < 5 bps | average per order |
| System Latency | < 100ms | tick→signal decision |
| Throughput | > 1000 ticks/sec | data processing |
| Uptime | > 99.5% | system availability |

## 🔒 Security

- **Encrypted API key storage**
- **TLS for all communications**
- **Access control and audit logging**
- **Container security with non-root users**

## 📝 Development Workflow

The project uses TaskMaster for task management:

```bash
# View current tasks
task-master list

# Get next task to work on
task-master next

# Update task status
task-master set-status --id=1 --status=in-progress

# Add new tasks
task-master add-task --prompt="Implement OANDA API integration"
```

## 🚧 Current Status

**Task 1: Project Setup & Architecture Design** - ✅ **COMPLETED**

- ✅ Monorepo structure created
- ✅ Core interfaces defined
- ✅ Communication protocols established
- ✅ Configuration management setup
- ✅ Docker infrastructure configured
- ✅ Main system orchestrator implemented

**Next Steps:**
- Task 2: Development Environment and Dependencies Setup
- Task 3: Data Ingest Module - Core Infrastructure

## 🤝 Contributing

1. Follow the task-driven development approach
2. Reference the PRD for requirements
3. Update tasks as implementation progresses
4. Ensure all tests pass before committing
5. Update documentation for new features

## 📄 License

[Specify your license here]

## 📞 Support

For questions and support, refer to:
- Project PRD: `scripts/PRD.txt`
- Task breakdown: `tasks/` directory
- System documentation: This README

---

*Built with precision for FX market trading excellence* 🎯
