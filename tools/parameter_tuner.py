#!/usr/bin/env python3
"""
Parameter Tuning Interface for FX AI-Quant Trading System

This module provides a comprehensive interface for adjusting strategy parameters
with validation, history tracking, and optimization suggestions.

Features:
- CLI and programmatic interface for parameter adjustment
- YAML configuration management with validation
- Version history tracking with timestamps
- Parameter validation and range checking
- Optimization suggestions based on performance data
- Safe parameter updates with rollback capability
"""

import argparse
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))


class ParameterValidationError(Exception):
    """Parameter validation error."""



class ParameterTuner:
    """
    Comprehensive parameter tuning system for strategy optimization.

    Provides safe parameter adjustment with validation, history tracking,
    and optimization suggestions for trading strategy parameters.
    """

    # Parameter validation rules
    VALIDATION_RULES = {
        # Grid/Martingale Strategy Parameters
        "grid_step_factor": {
            "type": float,
            "min": 0.1,
            "max": 2.0,
            "description": "Grid step multiplier",
        },
        "martingale_multiplier": {
            "type": float,
            "min": 1.0,
            "max": 5.0,
            "description": "Martingale position multiplier",
        },
        "max_levels": {
            "type": int,
            "min": 1,
            "max": 10,
            "description": "Maximum grid levels",
        },
        "breakout_zone_buffer": {
            "type": float,
            "min": 0.0001,
            "max": 0.01,
            "description": "Breakout zone buffer in price units",
        },
        # Risk Management Parameters
        "risk_per_trade": {
            "type": float,
            "min": 0.001,
            "max": 0.1,
            "description": "Risk percentage per trade",
        },
        "max_drawdown": {
            "type": float,
            "min": 0.01,
            "max": 0.5,
            "description": "Maximum allowed drawdown",
        },
        "max_positions": {
            "type": int,
            "min": 1,
            "max": 50,
            "description": "Maximum concurrent positions",
        },
        "position_size_limit": {
            "type": float,
            "min": 0.01,
            "max": 10.0,
            "description": "Maximum position size",
        },
        # Volatility and Market Filters
        "volatility_filter_threshold": {
            "type": float,
            "min": 0.0,
            "max": 1.0,
            "description": "Volatility filter threshold",
        },
        "trend_filter_strength": {
            "type": float,
            "min": 0.0,
            "max": 1.0,
            "description": "Trend filter strength",
        },
        "session_filter_enabled": {
            "type": bool,
            "description": "Enable session-based filtering",
        },
        "spread_filter_max": {
            "type": float,
            "min": 0.0,
            "max": 0.01,
            "description": "Maximum allowed spread",
        },
        # Take Profit / Stop Loss
        "tp_type": {
            "type": str,
            "allowed_values": ["fixed", "dynamic", "trailing"],
            "description": "Take profit type",
        },
        "sl_type": {
            "type": str,
            "allowed_values": ["fixed", "dynamic", "trailing"],
            "description": "Stop loss type",
        },
        "tp_ratio": {
            "type": float,
            "min": 0.5,
            "max": 10.0,
            "description": "Take profit ratio",
        },
        "sl_ratio": {
            "type": float,
            "min": 0.1,
            "max": 5.0,
            "description": "Stop loss ratio",
        },
        # Strategy Weights
        "momentum_weight": {
            "type": float,
            "min": 0.0,
            "max": 1.0,
            "description": "Momentum strategy weight",
        },
        "mean_reversion_weight": {
            "type": float,
            "min": 0.0,
            "max": 1.0,
            "description": "Mean reversion strategy weight",
        },
        "breakout_weight": {
            "type": float,
            "min": 0.0,
            "max": 1.0,
            "description": "Breakout strategy weight",
        },
        "regime_adaptive_weight": {
            "type": float,
            "min": 0.0,
            "max": 1.0,
            "description": "Regime adaptive weight",
        },
        # Technical Indicator Parameters
        "rsi_period": {
            "type": int,
            "min": 5,
            "max": 50,
            "description": "RSI calculation period",
        },
        "ma_fast_period": {
            "type": int,
            "min": 5,
            "max": 50,
            "description": "Fast moving average period",
        },
        "ma_slow_period": {
            "type": int,
            "min": 10,
            "max": 200,
            "description": "Slow moving average period",
        },
        "bollinger_period": {
            "type": int,
            "min": 10,
            "max": 50,
            "description": "Bollinger bands period",
        },
        "bollinger_std": {
            "type": float,
            "min": 1.0,
            "max": 3.0,
            "description": "Bollinger bands standard deviation",
        },
    }

    def __init__(
        self,
        config_path: str = "config/live_config.yaml",
        versions_dir: str = "config/versions",
        log_file: str = "logs/param_edit.log",
    ):
        """
        Initialize parameter tuner.

        Args:
            config_path: Path to the main configuration file
            versions_dir: Directory for storing configuration versions
            log_file: Path to parameter edit log file
        """
        self.config_path = Path(config_path)
        self.versions_dir = Path(versions_dir)
        self.log_file = Path(log_file)

        # Create directories if they don't exist
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self._setup_logging()

        # Load current configuration
        self.config = self._load_config()

        # Track changes for this session
        self.changes_made = []

        self.logger.info("Parameter tuner initialized")

    def _setup_logging(self):
        """Setup logging for parameter changes."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(self.log_file), logging.StreamHandler()],
        )
        self.logger = logging.getLogger(__name__)

    def _load_config(self) -> dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            self.logger.warning(
                f"Config file {self.config_path} not found. Creating default config."
            )
            self._create_default_config()

        try:
            with open(self.config_path) as f:
                config = yaml.safe_load(f)
            self.logger.info(f"Loaded configuration from {self.config_path}")
            return config or {}
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            return {}

    def _create_default_config(self):
        """Create a default configuration file."""
        default_config = {
            "strategy_parameters": {
                "grid_step_factor": 0.5,
                "martingale_multiplier": 1.5,
                "max_levels": 5,
                "breakout_zone_buffer": 0.001,
                "risk_per_trade": 0.02,
                "max_drawdown": 0.2,
                "max_positions": 10,
                "volatility_filter_threshold": 0.7,
                "session_filter_enabled": True,
                "tp_type": "dynamic",
                "sl_type": "dynamic",
                "tp_ratio": 2.0,
                "sl_ratio": 1.0,
            },
            "strategy_weights": {
                "momentum_weight": 0.3,
                "mean_reversion_weight": 0.3,
                "breakout_weight": 0.2,
                "regime_adaptive_weight": 0.2,
            },
            "technical_indicators": {
                "rsi_period": 14,
                "ma_fast_period": 12,
                "ma_slow_period": 26,
                "bollinger_period": 20,
                "bollinger_std": 2.0,
            },
            "metadata": {
                "created": datetime.now().isoformat(),
                "version": "1.0.0",
                "description": "Default parameter configuration",
            },
        }

        self._save_config(default_config)
        self.logger.info("Created default configuration file")

    def get_parameter(self, key: str) -> Any:
        """
        Get a parameter value from the configuration.

        Args:
            key: Parameter key (supports dot notation like 'strategy_parameters.grid_step_factor')

        Returns:
            Parameter value
        """
        keys = key.split(".")
        value = self.config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            raise ParameterValidationError(
                f"Parameter '{key}' not found in configuration"
            )

    def set_parameter(self, key: str, value: Any, validate: bool = True) -> bool:
        """
        Set a parameter value with validation.

        Args:
            key: Parameter key (supports dot notation)
            value: New parameter value
            validate: Whether to validate the parameter

        Returns:
            True if parameter was set successfully
        """
        # Extract the base parameter name for validation
        base_key = key.split(".")[-1]

        # Validate parameter if requested
        if validate:
            self._validate_parameter(base_key, value)

        # Store old value for rollback
        try:
            old_value = self.get_parameter(key)
        except ParameterValidationError:
            old_value = None

        # Set the parameter
        keys = key.split(".")
        target = self.config

        # Navigate to the parent dictionary
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]

        # Set the value
        target[keys[-1]] = value

        # Log the change
        change_record = {
            "timestamp": datetime.now().isoformat(),
            "parameter": key,
            "old_value": old_value,
            "new_value": value,
            "validated": validate,
        }

        self.changes_made.append(change_record)
        self.logger.info(f"Set parameter '{key}': {old_value} -> {value}")

        return True

    def _validate_parameter(self, key: str, value: Any):
        """
        Validate a parameter value against defined rules.

        Args:
            key: Parameter key
            value: Value to validate
        """
        if key not in self.VALIDATION_RULES:
            self.logger.warning(f"No validation rule found for parameter '{key}'")
            return

        rule = self.VALIDATION_RULES[key]

        # Type validation
        expected_type = rule["type"]
        if not isinstance(value, expected_type):
            try:
                # Try to convert the value
                if expected_type == int:
                    value = int(value)
                elif expected_type == float:
                    value = float(value)
                elif expected_type == bool:
                    if isinstance(value, str):
                        value = value.lower() in ["true", "1", "yes", "on"]
                    else:
                        value = bool(value)
                elif expected_type == str:
                    value = str(value)
            except (ValueError, TypeError):
                raise ParameterValidationError(
                    f"Parameter '{key}' must be of type {expected_type.__name__}, got {type(value).__name__}"
                )

        # Range validation for numeric types
        if expected_type in [int, float]:
            if "min" in rule and value < rule["min"]:
                raise ParameterValidationError(
                    f"Parameter '{key}' value {value} is below minimum {rule['min']}"
                )
            if "max" in rule and value > rule["max"]:
                raise ParameterValidationError(
                    f"Parameter '{key}' value {value} is above maximum {rule['max']}"
                )

        # Allowed values validation
        if "allowed_values" in rule and value not in rule["allowed_values"]:
            raise ParameterValidationError(
                f"Parameter '{key}' value '{value}' not in allowed values: {rule['allowed_values']}"
            )

        self.logger.debug(f"Parameter '{key}' validated successfully: {value}")

    def save_config(self, versioned: bool = True) -> str:
        """
        Save the current configuration.

        Args:
            versioned: Whether to create a versioned backup

        Returns:
            Path to the saved configuration file
        """
        # Update metadata
        if "metadata" not in self.config:
            self.config["metadata"] = {}

        self.config["metadata"]["last_modified"] = datetime.now().isoformat()
        self.config["metadata"]["changes_count"] = len(self.changes_made)

        # Create versioned backup if requested
        if versioned and self.config_path.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.versions_dir / f"config_{timestamp}.yaml"
            shutil.copy2(self.config_path, backup_path)
            self.logger.info(f"Created versioned backup: {backup_path}")

        # Save current configuration
        saved_path = self._save_config(self.config)

        # Log the save operation
        self.logger.info(f"Configuration saved to {saved_path}")
        if self.changes_made:
            self.logger.info(f"Changes made in this session: {len(self.changes_made)}")
            for change in self.changes_made:
                self.logger.info(
                    f"  {change['parameter']}: {change['old_value']} -> {change['new_value']}"
                )

        # Clear changes after save
        self.changes_made.clear()

        return str(saved_path)

    def _save_config(self, config: dict[str, Any]) -> Path:
        """Save configuration to YAML file."""
        with open(self.config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, indent=2, sort_keys=False)
        return self.config_path

    def list_versions(self) -> list[dict[str, Any]]:
        """
        List all configuration versions.

        Returns:
            List of version information dictionaries
        """
        versions = []

        for version_file in sorted(self.versions_dir.glob("config_*.yaml")):
            try:
                stat = version_file.stat()
                timestamp_str = version_file.stem.replace("config_", "")

                version_info = {
                    "file": str(version_file),
                    "timestamp": timestamp_str,
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                }
                versions.append(version_info)
            except Exception as e:
                self.logger.warning(f"Error reading version file {version_file}: {e}")

        return versions

    def restore_version(self, timestamp: str) -> bool:
        """
        Restore a configuration from a specific version.

        Args:
            timestamp: Timestamp string from version filename

        Returns:
            True if restore was successful
        """
        version_file = self.versions_dir / f"config_{timestamp}.yaml"

        if not version_file.exists():
            raise FileNotFoundError(f"Version file not found: {version_file}")

        # Create backup of current config before restore
        self.save_config(versioned=True)

        # Restore the version
        shutil.copy2(version_file, self.config_path)

        # Reload configuration
        self.config = self._load_config()

        self.logger.info(f"Restored configuration from version: {timestamp}")
        return True

    def suggest_optimization(self, param_name: str) -> dict[str, Any]:
        """
        Suggest optimization for a parameter based on validation rules and performance data.

        Args:
            param_name: Parameter name to suggest optimization for

        Returns:
            Dictionary with optimization suggestions
        """
        suggestion = {
            "parameter": param_name,
            "current_value": None,
            "suggested_range": None,
            "reasoning": "",
            "confidence": "low",
        }

        # Get current value
        try:
            # Try different possible locations for the parameter
            possible_paths = [
                f"strategy_parameters.{param_name}",
                f"strategy_weights.{param_name}",
                f"technical_indicators.{param_name}",
                param_name,
            ]

            current_value = None
            for path in possible_paths:
                try:
                    current_value = self.get_parameter(path)
                    break
                except ParameterValidationError:
                    continue

            suggestion["current_value"] = current_value
        except Exception:
            pass

        # Get validation rule
        if param_name in self.VALIDATION_RULES:
            rule = self.VALIDATION_RULES[param_name]

            if rule["type"] in [int, float]:
                min_val = rule.get("min", 0)
                max_val = rule.get("max", 1)

                suggestion["suggested_range"] = [min_val, max_val]
                suggestion[
                    "reasoning"
                ] = f"Valid range based on system constraints: {min_val} to {max_val}"
                suggestion["confidence"] = "medium"

                # If current value is at extremes, suggest moving toward center
                if current_value is not None:
                    mid_point = (min_val + max_val) / 2
                    if current_value <= min_val * 1.1:
                        suggestion["suggested_value"] = mid_point
                        suggestion[
                            "reasoning"
                        ] += f". Current value ({current_value}) is near minimum, consider increasing."
                    elif current_value >= max_val * 0.9:
                        suggestion["suggested_value"] = mid_point
                        suggestion[
                            "reasoning"
                        ] += f". Current value ({current_value}) is near maximum, consider decreasing."

            elif "allowed_values" in rule:
                suggestion["suggested_range"] = rule["allowed_values"]
                suggestion["reasoning"] = f"Allowed values: {rule['allowed_values']}"
                suggestion["confidence"] = "high"

        else:
            suggestion[
                "reasoning"
            ] = "No optimization rules available for this parameter"

        return suggestion

    def get_parameter_info(self, param_name: str) -> dict[str, Any]:
        """
        Get comprehensive information about a parameter.

        Args:
            param_name: Parameter name

        Returns:
            Dictionary with parameter information
        """
        info = {
            "name": param_name,
            "description": "No description available",
            "type": "unknown",
            "current_value": None,
            "validation_rule": None,
            "last_modified": None,
        }

        # Get validation rule
        if param_name in self.VALIDATION_RULES:
            rule = self.VALIDATION_RULES[param_name]
            info["validation_rule"] = rule
            info["description"] = rule.get("description", info["description"])
            info["type"] = rule["type"].__name__

        # Get current value
        try:
            possible_paths = [
                f"strategy_parameters.{param_name}",
                f"strategy_weights.{param_name}",
                f"technical_indicators.{param_name}",
                param_name,
            ]

            for path in possible_paths:
                try:
                    info["current_value"] = self.get_parameter(path)
                    break
                except ParameterValidationError:
                    continue
        except Exception:
            pass

        # Check for recent changes
        for change in self.changes_made:
            if param_name in change["parameter"]:
                info["last_modified"] = change["timestamp"]
                break

        return info

    def validate_all_parameters(self) -> dict[str, Any]:
        """
        Validate all parameters in the current configuration.

        Returns:
            Dictionary with validation results
        """
        results = {"valid": [], "invalid": [], "warnings": [], "total_checked": 0}

        def check_params(params: dict[str, Any], prefix: str = ""):
            for key, value in params.items():
                if isinstance(value, dict):
                    check_params(value, f"{prefix}{key}.")
                else:
                    full_key = f"{prefix}{key}"
                    base_key = key
                    results["total_checked"] += 1

                    try:
                        self._validate_parameter(base_key, value)
                        results["valid"].append(
                            {"parameter": full_key, "value": value, "status": "valid"}
                        )
                    except ParameterValidationError as e:
                        results["invalid"].append(
                            {
                                "parameter": full_key,
                                "value": value,
                                "error": str(e),
                                "status": "invalid",
                            }
                        )
                    except Exception as e:
                        results["warnings"].append(
                            {
                                "parameter": full_key,
                                "value": value,
                                "warning": str(e),
                                "status": "warning",
                            }
                        )

        check_params(self.config)

        return results


def main():
    """CLI interface for parameter tuning."""
    parser = argparse.ArgumentParser(
        description="FX AI-Quant Parameter Tuning Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --set grid_step_factor=0.4 --set max_levels=5
  %(prog)s --get strategy_parameters.risk_per_trade
  %(prog)s --list-versions
  %(prog)s --restore 20240601_1430
  %(prog)s --suggest grid_step_factor
  %(prog)s --validate-all
        """,
    )

    parser.add_argument(
        "--config", default="config/live_config.yaml", help="Path to configuration file"
    )
    parser.add_argument(
        "--set",
        action="append",
        dest="set_params",
        help="Set parameter (key=value format)",
    )
    parser.add_argument("--get", dest="get_param", help="Get parameter value")
    parser.add_argument("--info", dest="param_info", help="Get parameter information")
    parser.add_argument(
        "--suggest",
        dest="suggest_param",
        help="Get optimization suggestion for parameter",
    )
    parser.add_argument("--save", action="store_true", help="Save configuration")
    parser.add_argument(
        "--no-version",
        action="store_true",
        help="Don't create versioned backup when saving",
    )
    parser.add_argument(
        "--list-versions", action="store_true", help="List all configuration versions"
    )
    parser.add_argument(
        "--restore",
        dest="restore_version",
        help="Restore configuration from version (timestamp)",
    )
    parser.add_argument(
        "--validate-all", action="store_true", help="Validate all parameters"
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip parameter validation when setting",
    )

    args = parser.parse_args()

    try:
        # Initialize parameter tuner
        tuner = ParameterTuner(config_path=args.config)

        # Handle parameter setting
        if args.set_params:
            for param_setting in args.set_params:
                if "=" not in param_setting:
                    print(f"Error: Invalid parameter setting format: {param_setting}")
                    print("Use format: key=value")
                    continue

                key, value_str = param_setting.split("=", 1)

                # Try to parse value as appropriate type
                try:
                    # Try int first
                    if value_str.isdigit() or (
                        value_str.startswith("-") and value_str[1:].isdigit()
                    ):
                        value = int(value_str)
                    # Try float
                    elif "." in value_str:
                        value = float(value_str)
                    # Try boolean
                    elif value_str.lower() in ["true", "false"]:
                        value = value_str.lower() == "true"
                    # Keep as string
                    else:
                        value = value_str
                except ValueError:
                    value = value_str

                try:
                    tuner.set_parameter(key, value, validate=not args.no_validate)
                    print(f"✅ Set {key} = {value}")
                except ParameterValidationError as e:
                    print(f"❌ Validation error for {key}: {e}")
                except Exception as e:
                    print(f"❌ Error setting {key}: {e}")

        # Handle parameter getting
        if args.get_param:
            try:
                value = tuner.get_parameter(args.get_param)
                print(f"{args.get_param} = {value}")
            except Exception as e:
                print(f"❌ Error getting parameter: {e}")

        # Handle parameter info
        if args.param_info:
            info = tuner.get_parameter_info(args.param_info)
            print(f"\n📊 Parameter Information: {args.param_info}")
            print(f"Description: {info['description']}")
            print(f"Type: {info['type']}")
            print(f"Current Value: {info['current_value']}")
            if info["validation_rule"]:
                rule = info["validation_rule"]
                print(f"Validation Rule:")
                if "min" in rule:
                    print(f"  Min: {rule['min']}")
                if "max" in rule:
                    print(f"  Max: {rule['max']}")
                if "allowed_values" in rule:
                    print(f"  Allowed: {rule['allowed_values']}")

        # Handle optimization suggestion
        if args.suggest_param:
            suggestion = tuner.suggest_optimization(args.suggest_param)
            print(f"\n💡 Optimization Suggestion: {args.suggest_param}")
            print(f"Current Value: {suggestion['current_value']}")
            print(f"Suggested Range: {suggestion['suggested_range']}")
            if "suggested_value" in suggestion:
                print(f"Suggested Value: {suggestion['suggested_value']}")
            print(f"Reasoning: {suggestion['reasoning']}")
            print(f"Confidence: {suggestion['confidence']}")

        # Handle validation
        if args.validate_all:
            results = tuner.validate_all_parameters()
            print(f"\n🔍 Parameter Validation Results")
            print(f"Total Parameters Checked: {results['total_checked']}")
            print(f"✅ Valid: {len(results['valid'])}")
            print(f"❌ Invalid: {len(results['invalid'])}")
            print(f"⚠️  Warnings: {len(results['warnings'])}")

            if results["invalid"]:
                print("\n❌ Invalid Parameters:")
                for item in results["invalid"]:
                    print(f"  {item['parameter']}: {item['error']}")

            if results["warnings"]:
                print("\n⚠️  Warnings:")
                for item in results["warnings"]:
                    print(f"  {item['parameter']}: {item['warning']}")

        # Handle version listing
        if args.list_versions:
            versions = tuner.list_versions()
            if versions:
                print("\n📅 Configuration Versions:")
                for version in versions:
                    timestamp = version["timestamp"]
                    size = version["size"]
                    modified = version["modified"]
                    print(f"  {timestamp} - {size} bytes - {modified}")
            else:
                print("No configuration versions found.")

        # Handle version restore
        if args.restore_version:
            try:
                tuner.restore_version(args.restore_version)
                print(f"✅ Restored configuration from version: {args.restore_version}")
            except Exception as e:
                print(f"❌ Error restoring version: {e}")

        # Handle saving
        if args.save or args.set_params:
            if tuner.changes_made or args.save:
                saved_path = tuner.save_config(versioned=not args.no_version)
                print(f"✅ Configuration saved to: {saved_path}")
            else:
                print("ℹ️  No changes to save.")

    except Exception as e:
        print(f"❌ Fatal error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
