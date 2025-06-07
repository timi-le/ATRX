# Risk Manager - Core Risk Controls

## Overview

The Risk Manager is a comprehensive risk management module for the FX AI-Quant Trading System that provides real-time monitoring, risk limit enforcement, and emergency controls to protect trading capital.

## Features

### 🛡️ Core Risk Controls
- **Real-time P&L Monitoring**: Track realized and unrealized profits/losses
- **Drawdown Protection**: Monitor daily and total drawdown with configurable limits
- **Position Limits**: Enforce exposure limits per symbol, strategy, and total portfolio
- **Leverage Controls**: Monitor and limit portfolio leverage
- **VaR Calculations**: Value at Risk monitoring with multiple calculation methods

### ⚡ Emergency Controls
- **Kill Switch**: Automatic trading halt when limits are breached
- **Manual Override**: Temporary bypass of risk controls with time limits
- **Emergency Stop**: Immediate halt of all trading activities
- **Recovery Conditions**: Automatic reset when conditions normalize

### 📊 Real-time Monitoring
- **Risk Metrics Dashboard**: Current exposure, drawdown, and risk status
- **Alert System**: Multi-channel alerts (log, console, pubsub)
- **Performance Tracking**: Risk-adjusted performance metrics
- **Historical Analysis**: Risk metrics history and trends

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Strategy       │    │  Position       │    │  Risk Manager   │
│  Switcher       │───▶│  Sizer          │───▶│  (Core)         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                                               ┌───────▼───────┐
                                               │  Risk Alerts  │
                                               │  (PubSub)     │
                                               └───────────────┘
```

## Configuration

### Risk Limits Configuration (`config/risk_limits.yaml`)

```yaml
# Drawdown Limits
drawdown_limits:
  max_daily_drawdown: 0.05          # 5% daily drawdown limit
  max_total_drawdown: 0.15          # 15% total drawdown limit
  warning_daily_drawdown: 0.03      # 3% daily warning
  warning_total_drawdown: 0.10      # 10% total warning

# Position Limits
position_limits:
  max_position_per_symbol: 0.10     # 10% per symbol
  max_total_exposure: 0.50          # 50% total exposure
  max_concurrent_positions: 8       # Maximum 8 positions
  max_leverage: 10.0                # Maximum 10:1 leverage

# VaR Parameters
var_parameters:
  confidence_level: 0.95            # 95% confidence level
  horizon_days: 1                   # 1-day VaR
  max_portfolio_var: 0.03           # 3% portfolio VaR limit
  calculation_method: "historical"   # VaR calculation method

# Emergency Controls
emergency_controls:
  kill_switch_triggers:
    daily_loss_threshold: 0.05      # 5% daily loss triggers kill switch
    total_loss_threshold: 0.15      # 15% total loss triggers kill switch
  
  manual_override:
    enabled: true                   # Allow manual override
    override_duration_hours: 24     # Override duration
```

## Usage

### Basic Usage

```python
from core.risk_manager import create_risk_manager
from core.interfaces.trading_interfaces import Order, Position, OrderSide, OrderType

# Create risk manager
risk_manager = create_risk_manager(
    config_path="config/risk_limits.yaml",
    initial_capital=100000.0
)

# Check pre-trade risk
order = Order(
    order_id="order_1",
    symbol="EURUSD",
    side=OrderSide.BUY,
    order_type=OrderType.MARKET,
    quantity=10000,
    price=1.1000
)

current_positions = {}  # Your current positions
account_balance = 100000.0

# Pre-trade risk check
risk_approved = await risk_manager.check_pre_trade_risk(
    order, current_positions, account_balance
)

if risk_approved:
    # Execute trade
    print("Trade approved by risk manager")
else:
    print("Trade rejected by risk manager")
```

### Real-time Monitoring

```python
# Update positions
await risk_manager.update_positions(current_positions)

# Update P&L
total_pnl = 1500.0
realized_pnl = 500.0
unrealized_pnl = 1000.0
await risk_manager.update_pnl(total_pnl, realized_pnl, unrealized_pnl)

# Calculate VaR
portfolio_var = await risk_manager.calculate_var(current_positions)
print(f"Portfolio VaR: {portfolio_var:.2%}")

# Get risk metrics
metrics = await risk_manager.get_risk_metrics()
print(f"Risk Status: {metrics.risk_status.value}")
print(f"Current Drawdown: {metrics.current_drawdown:.2%}")
print(f"Total Exposure: {metrics.total_exposure:.2%}")
```

### Emergency Controls

```python
# Trigger emergency stop
await risk_manager.emergency_stop("Market volatility spike")

# Set manual override
await risk_manager.set_manual_override(
    duration_hours=12, 
    reason="News event trading"
)

# Clear manual override
await risk_manager.clear_manual_override()

# Reset emergency mode (if conditions allow)
await risk_manager.reset_emergency_mode()
```

### Integration with Trading System

```python
from examples.risk_manager_integration import TradingSystemWithRiskManager

# Create integrated trading system
trading_system = TradingSystemWithRiskManager()

# Start the system
await trading_system.start()

# Process trading signals with risk management
signal = Signal(
    symbol="EURUSD",
    side=OrderSide.BUY,
    strength=0.8,
    confidence=0.75,
    strategy_name="breakout_trend",
    timestamp=datetime.now()
)

success = await trading_system.process_signal(signal)
```

## Risk Metrics

### Core Metrics

| Metric | Description | Threshold |
|--------|-------------|-----------|
| **Daily Drawdown** | Loss from daily start | 5% (limit), 3% (warning) |
| **Total Drawdown** | Loss from peak equity | 15% (limit), 10% (warning) |
| **Position Exposure** | Capital at risk per symbol | 10% per symbol |
| **Total Exposure** | Total capital at risk | 50% of capital |
| **Leverage** | Portfolio leverage ratio | 10:1 (limit), 7:1 (warning) |
| **Portfolio VaR** | Value at Risk (95% confidence) | 3% (limit), 2% (warning) |

### Risk Status Levels

- **🟢 HEALTHY**: All metrics within normal ranges
- **🟡 WARNING**: One or more metrics approaching limits
- **🟠 CRITICAL**: One or more metrics near breach
- **🔴 HALTED**: Trading halted due to risk controls
- **🚨 EMERGENCY**: Emergency stop triggered

## Alert System

### Alert Levels

1. **INFO**: Informational messages (overrides, resets)
2. **WARNING**: Risk metrics approaching limits
3. **CRITICAL**: Risk metrics near breach
4. **EMERGENCY**: Emergency stop triggered

### Alert Channels

- **Log**: Structured logging with configurable levels
- **Console**: Real-time console output
- **PubSub**: Message bus integration for system-wide alerts
- **Slack**: Optional Slack notifications (webhook required)
- **Email**: Optional email alerts (SMTP required)

### Alert Frequency Limiting

- Duplicate alert cooldown: 5 minutes
- Emergency alert cooldown: 1 minute
- Maximum alerts per minute: 10

## Testing

### Running Tests

```bash
# Run all risk manager tests
pytest tests/test_risk_manager.py -v

# Run specific test categories
pytest tests/test_risk_manager.py::TestPreTradeRiskChecks -v
pytest tests/test_risk_manager.py::TestDrawdownMonitoring -v
pytest tests/test_risk_manager.py::TestVaRCalculation -v
pytest tests/test_risk_manager.py::TestEmergencyControls -v
```

### Test Coverage

The test suite covers:
- ✅ Risk limit enforcement
- ✅ Drawdown calculations
- ✅ VaR computations
- ✅ Emergency stop mechanisms
- ✅ Manual override functionality
- ✅ Alert generation and frequency limiting
- ✅ Position tracking and exposure calculations
- ✅ P&L tracking and rapid loss detection
- ✅ Edge cases and error handling

## Performance Considerations

### Latency Requirements

- **Pre-trade risk checks**: < 1ms typical
- **P&L updates**: < 0.5ms typical
- **Risk metrics calculation**: < 10ms typical
- **VaR calculation**: < 100ms typical

### Memory Usage

- **P&L history**: 1000 records (configurable)
- **VaR history**: 100 records (configurable)
- **Drawdown history**: 1000 records (configurable)
- **Alert history**: Unlimited (consider cleanup)

### Scalability

- Supports up to 100 concurrent positions
- Handles 1000+ risk checks per second
- Configurable update frequencies for optimization

## Best Practices

### Configuration

1. **Start Conservative**: Begin with lower limits and gradually increase
2. **Regular Review**: Review and adjust limits based on strategy performance
3. **Backtesting**: Test risk parameters on historical data
4. **Documentation**: Document all configuration changes

### Monitoring

1. **Real-time Dashboards**: Monitor risk metrics continuously
2. **Alert Response**: Have procedures for different alert levels
3. **Regular Audits**: Review risk manager logs and performance
4. **Stress Testing**: Test emergency procedures regularly

### Integration

1. **Pre-trade Checks**: Always check risk before executing trades
2. **Position Updates**: Update positions immediately after execution
3. **P&L Tracking**: Update P&L in real-time with market data
4. **Error Handling**: Implement robust error handling and fallbacks

## Troubleshooting

### Common Issues

#### Risk Manager Rejecting Valid Trades

**Symptoms**: All trades rejected despite normal market conditions

**Solutions**:
1. Check if emergency mode is active: `risk_manager.get_status()`
2. Verify position limits aren't exceeded
3. Check if manual override is needed
4. Review recent alerts for trigger causes

#### High VaR Calculations

**Symptoms**: VaR consistently above limits

**Solutions**:
1. Review position concentration
2. Check correlation between positions
3. Consider reducing position sizes
4. Verify VaR calculation parameters

#### Emergency Mode Not Resetting

**Symptoms**: Emergency mode persists after conditions improve

**Solutions**:
1. Check recovery conditions in configuration
2. Verify stability period requirements
3. Manually reset if conditions are met
4. Review alert history for ongoing issues

### Debug Mode

Enable debug logging for detailed troubleshooting:

```python
import logging
logging.getLogger('core.risk_manager').setLevel(logging.DEBUG)
```

## API Reference

### Core Classes

#### `CoreRiskManager`

Main risk manager implementation.

**Methods**:
- `check_pre_trade_risk(order, positions, balance)`: Pre-trade risk check
- `monitor_drawdown(current_pnl, peak_pnl)`: Monitor drawdown levels
- `calculate_var(positions, confidence_level, horizon_days)`: Calculate VaR
- `update_positions(positions)`: Update position tracking
- `update_pnl(total_pnl, realized_pnl, unrealized_pnl)`: Update P&L
- `get_risk_metrics()`: Get current risk metrics
- `emergency_stop(reason)`: Trigger emergency stop
- `set_manual_override(duration_hours, reason)`: Set manual override
- `clear_manual_override()`: Clear manual override
- `reset_emergency_mode()`: Reset emergency mode

#### `RiskManagerConfig`

Configuration management for risk parameters.

#### `RiskMetrics`

Risk metrics data structure containing current risk state.

#### `RiskAlert`

Alert data structure for risk notifications.

### Factory Functions

#### `create_risk_manager(config_path, initial_capital, publisher, logger)`

Factory function to create a configured risk manager instance.

## Contributing

### Development Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Run tests: `pytest tests/test_risk_manager.py`
3. Check code style: `flake8 core/risk_manager.py`
4. Type checking: `mypy core/risk_manager.py`

### Adding New Risk Controls

1. Add configuration parameters to `risk_limits.yaml`
2. Implement risk check logic in `CoreRiskManager`
3. Add comprehensive tests
4. Update documentation
5. Consider performance impact

### Extending Alert System

1. Add new alert channels in configuration
2. Implement channel-specific logic in `_send_alert`
3. Add tests for new functionality
4. Update documentation

## License

This module is part of the FX AI-Quant Trading System and follows the project's licensing terms. 