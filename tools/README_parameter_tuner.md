# FX AI-Quant Parameter Tuning Interface

## Overview

The Parameter Tuning Interface provides a robust, CLI-based system for adjusting and optimizing trading strategy parameters with comprehensive validation, history tracking, and optimization suggestions.

## Features

### 🔧 Core Functionality
- **Safe Parameter Management**: Set parameters with automatic validation and type checking
- **Configuration Versioning**: Automatic backup and version history with timestamp tracking
- **Change Tracking**: Complete audit trail of parameter modifications
- **Optimization Suggestions**: AI-powered recommendations for parameter improvements
- **Validation System**: Comprehensive parameter validation with range and type checks

### 🎯 Supported Parameters

#### Strategy Parameters
- `grid_step_factor` (0.1 - 2.0): Grid step multiplier for grid-based strategies
- `martingale_multiplier` (1.0 - 5.0): Position size multiplier for martingale strategies
- `max_levels` (1 - 10): Maximum number of grid levels
- `breakout_zone_buffer` (0.0001 - 0.01): Buffer zone for breakout detection
- `risk_per_trade` (0.001 - 0.1): Risk percentage per individual trade
- `max_drawdown` (0.01 - 0.5): Maximum allowed portfolio drawdown
- `max_positions` (1 - 50): Maximum concurrent positions
- `volatility_filter_threshold` (0.0 - 1.0): Volatility filter threshold
- `session_filter_enabled` (true/false): Enable session-based filtering
- `tp_type` (fixed/dynamic/trailing): Take profit strategy type
- `sl_type` (fixed/dynamic/trailing): Stop loss strategy type
- `tp_ratio` (0.5 - 10.0): Take profit ratio
- `sl_ratio` (0.1 - 5.0): Stop loss ratio

#### Strategy Weights
- `momentum_weight` (0.0 - 1.0): Weight for momentum strategy component
- `mean_reversion_weight` (0.0 - 1.0): Weight for mean reversion strategy
- `breakout_weight` (0.0 - 1.0): Weight for breakout strategy component
- `regime_adaptive_weight` (0.0 - 1.0): Weight for regime-adaptive strategies

#### Technical Indicators
- `rsi_period` (5 - 50): RSI calculation period
- `ma_fast_period` (5 - 50): Fast moving average period
- `ma_slow_period` (10 - 200): Slow moving average period
- `bollinger_period` (10 - 50): Bollinger bands calculation period
- `bollinger_std` (1.0 - 3.0): Bollinger bands standard deviation multiplier

## Usage

### CLI Interface

#### Basic Parameter Operations

```bash
# Set parameters with validation
python tools/parameter_tuner.py --set "grid_step_factor=0.6" --set "max_levels=7"

# Get parameter values
python tools/parameter_tuner.py --get "strategy_parameters.grid_step_factor"

# Get parameter information and validation rules
python tools/parameter_tuner.py --info grid_step_factor

# Get optimization suggestions
python tools/parameter_tuner.py --suggest grid_step_factor
```

#### Configuration Management

```bash
# List all configuration versions
python tools/parameter_tuner.py --list-versions

# Restore from a specific version
python tools/parameter_tuner.py --restore 20240601_143000

# Validate all parameters
python tools/parameter_tuner.py --validate-all

# Save configuration manually
python tools/parameter_tuner.py --save
```

#### Advanced Options

```bash
# Set parameters without validation (dangerous!)
python tools/parameter_tuner.py --set "grid_step_factor=0.6" --no-validate

# Save without creating version backup
python tools/parameter_tuner.py --set "max_levels=5" --no-version

# Use custom configuration file
python tools/parameter_tuner.py --config "custom_config.yaml" --get "grid_step_factor"
```

### Programmatic Interface

```python
from tools.parameter_tuner import ParameterTuner, ParameterValidationError

# Initialize tuner
tuner = ParameterTuner()

# Set parameters with validation
try:
    tuner.set_parameter("strategy_parameters.grid_step_factor", 0.6)
    tuner.set_parameter("strategy_parameters.max_levels", 7)
except ParameterValidationError as e:
    print(f"Validation error: {e}")

# Get parameter values
value = tuner.get_parameter("strategy_parameters.grid_step_factor")

# Get parameter information
info = tuner.get_parameter_info("grid_step_factor")

# Get optimization suggestions
suggestion = tuner.suggest_optimization("grid_step_factor")

# Save configuration with versioning
saved_path = tuner.save_config(versioned=True)

# Validate all parameters
results = tuner.validate_all_parameters()
```

## Configuration Structure

```yaml
strategy_parameters:
  grid_step_factor: 0.5
  martingale_multiplier: 1.5
  max_levels: 5
  risk_per_trade: 0.02
  # ... more parameters

strategy_weights:
  momentum_weight: 0.3
  mean_reversion_weight: 0.3
  breakout_weight: 0.2
  regime_adaptive_weight: 0.2

technical_indicators:
  rsi_period: 14
  ma_fast_period: 12
  ma_slow_period: 26
  # ... more indicators

metadata:
  created: "2024-06-04T21:00:00"
  version: "1.0.0"
  last_modified: "2024-06-04T21:17:49"
  changes_count: 4
```

## File Structure

```
├── config/
│   ├── live_config.yaml          # Main configuration file
│   └── versions/                 # Configuration version history
│       ├── config_20240604_210000.yaml
│       └── config_20240604_211749.yaml
├── tools/
│   ├── parameter_tuner.py        # Main tuning interface
│   ├── parameter_tuner_demo.py   # Demonstration script
│   └── README_parameter_tuner.md # This documentation
├── tests/
│   └── test_parameter_tuner.py   # Comprehensive test suite
└── logs/
    └── param_edit.log            # Parameter change audit log
```

## Validation System

### Type Validation
- Automatic type conversion (string to int/float/bool)
- Strict type checking with helpful error messages

### Range Validation
- Minimum and maximum value enforcement
- Boundary value testing and validation

### Allowed Values
- Enumeration validation for categorical parameters
- Clear error messages for invalid choices

### Example Validation Errors
```bash
❌ Validation error for grid_step_factor: Parameter 'grid_step_factor' value 10.0 is above maximum 2.0
❌ Validation error for max_levels: Parameter 'max_levels' value 0 is below minimum 1
❌ Validation error for tp_type: Parameter 'tp_type' value 'invalid_type' not in allowed values: ['fixed', 'dynamic', 'trailing']
```

## Version Management

### Automatic Versioning
- Every save operation creates a timestamped backup
- Configurable versioning behavior (can be disabled)
- Version metadata includes size and modification time

### Version Restoration
- Restore any previous configuration version
- Automatic backup before restoration
- Complete configuration reload after restoration

### Version Listing
```bash
📅 Configuration Versions:
  20250604_210134 - 710 bytes - 2025-06-04T21:01:33.977944
  20250604_210141 - 815 bytes - 2025-06-04T21:01:34.018946
  20250604_211749 - 815 bytes - 2025-06-04T21:17:49.077462
```

## Optimization Suggestions

### Intelligence Features
- Parameter range analysis based on validation rules
- Extreme value detection (parameters at min/max boundaries)
- Confidence scoring for suggestions
- Performance-based recommendations (when historical data available)

### Example Suggestions
```
💡 Optimization Suggestion: grid_step_factor
Current Value: 0.1
Suggested Range: [0.1, 2.0]
Suggested Value: 1.05
Reasoning: Valid range based on system constraints: 0.1 to 2.0. Current value (0.1) is near minimum, consider increasing.
Confidence: medium
```

## Testing

Run the comprehensive test suite:
```bash
python tests/test_parameter_tuner.py
```

Test coverage includes:
- Parameter validation and setting
- Configuration management and versioning
- CLI interface functionality
- Optimization suggestions
- Error handling and edge cases

## Demo

Run the interactive demonstration:
```bash
python tools/parameter_tuner_demo.py
```

The demo showcases:
- Parameter information and validation
- Valid and invalid parameter changes
- Optimization suggestions
- Change tracking and version management
- Complete configuration validation

## Security and Safety

### Input Validation
- All parameters are validated before application
- Type checking and range enforcement
- Protection against configuration injection

### Change Tracking
- Complete audit trail of all parameter modifications
- Timestamp tracking for all changes
- Validation status recording

### Rollback Capability
- Version-based rollback to any previous configuration
- Automatic backup before risky operations
- Safe parameter experimentation

## Performance

### Efficiency Features
- Fast YAML-based configuration loading
- Incremental change tracking
- Efficient validation rule processing
- Minimal memory footprint

### Scalability
- Support for complex nested configurations
- Extensible validation rule system
- Configurable logging and output verbosity

## Future Enhancements

### Planned Features
- Web-based UI with parameter sliders
- Integration with optimization libraries (Optuna, etc.)
- Performance-based parameter recommendations
- A/B testing framework for parameter variants
- Real-time parameter updates for live trading

### Integration Points
- Strategy performance monitoring integration
- Backtesting result analysis
- Risk management system integration
- Portfolio optimization feedback loops

---

**Created**: June 4, 2025
**Version**: 1.0.0
**Authors**: FX AI-Quant Development Team
**Status**: Production Ready ✅
