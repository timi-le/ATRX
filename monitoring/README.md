# FX Quant Trading System - Monitoring Infrastructure

This directory contains the comprehensive Prometheus-based monitoring infrastructure for the FX Quant Trading System, providing real-time observability into trading performance, system health, and operational metrics.

## 📊 Overview

The monitoring system tracks:
- **Trading Performance**: PnL, equity curves, win rates, trade counts
- **Execution Quality**: Latency, slippage, fill rates, order execution metrics
- **Position Management**: Exposures, open positions, risk metrics
- **Strategy Analytics**: Signal generation, regime detection, strategy performance
- **System Health**: Resource usage, errors, market data lag
- **Risk Management**: Drawdowns, VaR, performance ratios

## 🚀 Quick Start

### 1. Start Metrics Server

```python
from monitoring.metrics_server import start_metrics_server

# Start the metrics server
metrics = start_metrics_server(port=9000)

# Server will run at http://localhost:9000/metrics
```

### 2. Update Metrics in Your Trading Code

```python
from monitoring.metrics_server import get_metrics, record_trade_execution

# Get metrics instance
metrics = get_metrics()

# Record a trade execution
record_trade_execution(
    symbol="EURUSD",
    side="BUY",
    status="filled",
    latency_seconds=0.025,
    slippage_pips=1.2,
    pnl=250.50
)

# Update session metrics
metrics.update_pnl(equity=102500.0, daily_pnl=2500.0)
metrics.update_regime("EURUSD", "trending", 0.65)
```

### 3. Configure Prometheus

Use the provided `prometheus.yml` configuration:

```bash
# Start Prometheus with our config
prometheus --config.file=monitoring/prometheus.yml
```

## 📋 Available Metrics

### Trading Performance Metrics

| Metric | Type | Description | Labels |
|--------|------|-------------|--------|
| `fxai_pnl_equity` | Gauge | Current PnL/equity in USD | - |
| `fxai_daily_pnl` | Gauge | Daily PnL in USD | - |
| `fxai_trades_total` | Counter | Total trades executed | symbol, side, status |
| `fxai_winning_trades_total` | Counter | Total winning trades | symbol |
| `fxai_win_rate_ratio` | Gauge | Win rate percentage | symbol |

### Execution Quality Metrics

| Metric | Type | Description | Labels |
|--------|------|-------------|--------|
| `fxai_execution_latency_seconds` | Histogram | Order execution latency | - |
| `fxai_fill_latency_seconds` | Histogram | Time from order to fill | - |
| `fxai_order_slippage_pips` | Histogram | Order slippage in pips | - |
| `fxai_fill_rate_ratio` | Gauge | Percentage of filled orders | symbol |

### Position & Exposure Metrics

| Metric | Type | Description | Labels |
|--------|------|-------------|--------|
| `fxai_position_exposure_usd` | Gauge | Position exposure in USD | symbol, side |
| `fxai_total_exposure_usd` | Gauge | Total exposure across symbols | - |
| `fxai_open_positions` | Gauge | Number of open positions | symbol |

### Strategy & Regime Metrics

| Metric | Type | Description | Labels |
|--------|------|-------------|--------|
| `fxai_regime_ratio` | Gauge | Regime classification proportion | regime, symbol |
| `fxai_strategy_signals_total` | Counter | Strategy signals generated | strategy, signal_type, symbol |
| `fxai_strategy_pnl` | Gauge | PnL attributed to strategy | strategy |

### System Health Metrics

| Metric | Type | Description | Labels |
|--------|------|-------------|--------|
| `fxai_errors_total` | Counter | Total system errors | error_type, component |
| `fxai_memory_usage_mb` | Gauge | Memory usage in MB | - |
| `fxai_cpu_usage_percent` | Gauge | CPU usage percentage | - |
| `fxai_market_data_lag_seconds` | Gauge | Market data lag | symbol |

### Risk Management Metrics

| Metric | Type | Description | Labels |
|--------|------|-------------|--------|
| `fxai_current_drawdown_ratio` | Gauge | Current drawdown ratio | - |
| `fxai_max_drawdown_ratio` | Gauge | Maximum drawdown experienced | - |
| `fxai_var_estimate` | Gauge | Value at Risk estimate | confidence_level |
| `fxai_sharpe_ratio` | Gauge | Current Sharpe ratio | - |

## 🔧 Usage Examples

### Basic Integration

```python
from monitoring.metrics_server import FXTradingMetrics, start_metrics_server

# Start monitoring
metrics = start_metrics_server(port=9000)

# In your trading loop
for trade_result in trading_engine.execute_trades():
    # Record trade execution
    metrics.record_trade(
        symbol=trade_result.symbol,
        side=trade_result.side,
        status=trade_result.status,
        pnl=trade_result.pnl
    )

    # Record execution metrics
    metrics.record_execution_latency(trade_result.latency)
    metrics.record_slippage(trade_result.slippage_pips)

    # Update equity
    current_equity = portfolio.get_equity()
    metrics.update_pnl(current_equity)
```

### Advanced Usage

```python
# Update regime analysis
regime_detector = RegimeDetector()
for symbol in ['EURUSD', 'GBPUSD']:
    regime_probs = regime_detector.analyze(symbol)
    metrics.update_regime(symbol, 'trending', regime_probs['trending'])
    metrics.update_regime(symbol, 'ranging', regime_probs['ranging'])

# Record strategy signals
strategy_engine = StrategyEngine()
for signal in strategy_engine.generate_signals():
    metrics.record_strategy_signal(
        strategy=signal.strategy_name,
        signal_type=signal.type,  # 'buy', 'sell', 'hold'
        symbol=signal.symbol
    )

# Update risk metrics
risk_manager = RiskManager()
risk_metrics = risk_manager.calculate_metrics()
metrics.update_risk_metrics(
    current_drawdown=risk_metrics.current_dd,
    max_drawdown=risk_metrics.max_dd,
    var_95=risk_metrics.var_95,
    var_99=risk_metrics.var_99
)
```

## 🔍 Testing & Validation

Run the comprehensive test suite:

```bash
python test_metrics_server.py
```

Run the interactive demonstration:

```bash
python demo_metrics_monitoring.py
```

## 🎯 Prometheus Configuration

The provided `prometheus.yml` configures Prometheus to:
- Scrape metrics every 5 seconds from `localhost:9000`
- Apply appropriate labels and relabeling rules
- Include self-monitoring for Prometheus

### Running Prometheus

1. **Download Prometheus** from https://prometheus.io/download/
2. **Start with our config**:
   ```bash
   prometheus --config.file=monitoring/prometheus.yml
   ```
3. **Access Web UI**: http://localhost:9090

### Example Queries

```promql
# Current equity
fxai_pnl_equity

# Trade rate (trades per minute)
rate(fxai_trades_total[1m]) * 60

# Average execution latency
rate(fxai_execution_latency_seconds_sum[5m]) / rate(fxai_execution_latency_seconds_count[5m])

# Win rate by symbol
fxai_win_rate_ratio

# Current drawdown
fxai_current_drawdown_ratio

# Error rate by component
rate(fxai_errors_total[5m])
```

## 📈 Grafana Integration

Create dashboards with panels for:

### Trading Performance Dashboard
- Equity curve over time
- Daily PnL chart
- Trade count and win rate gauges
- Position exposure breakdown

### Execution Quality Dashboard
- Execution latency histogram
- Slippage distribution
- Fill rate by symbol
- Trade volume metrics

### System Health Dashboard
- Memory and CPU usage
- Error rates by component
- Market data lag metrics
- Alert status overview

### Risk Management Dashboard
- Current vs maximum drawdown
- VaR/CVaR estimates
- Sharpe ratio evolution
- Position concentration metrics

## 🚨 Alerting

Example alerting rules for `alert_rules.yml`:

```yaml
groups:
- name: fx_trading_alerts
  rules:
  - alert: HighDrawdown
    expr: fxai_current_drawdown_ratio > 0.05
    for: 1m
    labels:
      severity: warning
    annotations:
      summary: "High drawdown detected"

  - alert: LowFillRate
    expr: fxai_fill_rate_ratio < 0.8
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "Fill rate below 80%"

  - alert: HighLatency
    expr: rate(fxai_execution_latency_seconds_sum[5m]) / rate(fxai_execution_latency_seconds_count[5m]) > 0.1
    for: 1m
    labels:
      severity: warning
    annotations:
      summary: "High execution latency detected"
```

## 🛠 Architecture

```
Trading System  →  Metrics Server  →  Prometheus  →  Grafana
     ↓                   ↓               ↓           ↓
  Trade Events      HTTP /metrics    Time Series   Dashboards
  System Health      Port 9000        Database     & Alerts
  Performance                         Port 9090    Port 3000
```

## 📦 Files

- `metrics_server.py` - Core metrics collection and HTTP server
- `prometheus.yml` - Prometheus configuration
- `test_metrics_server.py` - Comprehensive test suite
- `demo_metrics_monitoring.py` - Interactive demonstration
- `README.md` - This documentation

## 🔗 Next Steps

1. **Install Prometheus** and configure with `prometheus.yml`
2. **Set up Grafana** with custom dashboards
3. **Configure Alertmanager** for critical alerts
4. **Integrate with live trading system** using the provided APIs
5. **Set up remote storage** for long-term metrics retention

## 📞 Support

For questions or issues with the monitoring infrastructure:
- Review test cases in `test_metrics_server.py`
- Run the demo with `demo_metrics_monitoring.py`
- Check Prometheus metrics at `http://localhost:9000/metrics`
