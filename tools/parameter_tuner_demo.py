#!/usr/bin/env python3
"""
Parameter Tuner Demonstration Script

This script demonstrates the comprehensive parameter tuning capabilities
of the FX AI-Quant Trading System parameter management interface.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from tools.parameter_tuner import ParameterTuner, ParameterValidationError


def demo_parameter_tuning():
    """Demonstrate parameter tuning capabilities."""
    
    print("🎯 FX AI-Quant Parameter Tuning System Demo")
    print("=" * 60)
    
    # Initialize parameter tuner
    print("\n📦 1. Initializing Parameter Tuner...")
    tuner = ParameterTuner()
    print(f"✅ Loaded configuration with {len(tuner.config)} main sections")
    
    # Show current configuration overview
    print("\n📊 2. Current Configuration Overview:")
    for section, params in tuner.config.items():
        if isinstance(params, dict) and section != "metadata":
            print(f"  📁 {section}: {len(params)} parameters")
    
    # Get parameter information
    print("\n💡 3. Parameter Information Example:")
    info = tuner.get_parameter_info("grid_step_factor")
    print(f"  📋 Parameter: {info['name']}")
    print(f"  📝 Description: {info['description']}")
    print(f"  🔢 Type: {info['type']}")
    print(f"  📈 Current Value: {info['current_value']}")
    if info['validation_rule']:
        rule = info['validation_rule']
        if 'min' in rule: print(f"  📉 Min: {rule['min']}")
        if 'max' in rule: print(f"  📊 Max: {rule['max']}")
    
    # Demonstrate parameter setting with validation
    print("\n🔧 4. Setting Parameters with Validation:")
    
    # Valid parameter changes
    valid_changes = [
        ("strategy_parameters.grid_step_factor", 0.8),
        ("strategy_parameters.max_levels", 6),
        ("strategy_weights.momentum_weight", 0.4),
        ("strategy_parameters.session_filter_enabled", False)
    ]
    
    for param, value in valid_changes:
        try:
            old_value = tuner.get_parameter(param)
            tuner.set_parameter(param, value)
            print(f"  ✅ {param}: {old_value} → {value}")
        except ParameterValidationError as e:
            print(f"  ❌ {param}: {e}")
        except Exception as e:
            print(f"  ⚠️  {param}: {e}")
    
    # Invalid parameter changes (to show validation)
    print("\n🚫 5. Testing Validation (Invalid Values):")
    invalid_changes = [
        ("strategy_parameters.grid_step_factor", 10.0),  # Above max
        ("strategy_parameters.max_levels", 0),  # Below min
        ("strategy_parameters.tp_type", "invalid_type")  # Not allowed
    ]
    
    for param, value in invalid_changes:
        try:
            tuner.set_parameter(param, value)
            print(f"  ⚠️  {param}: {value} (should have failed!)")
        except ParameterValidationError as e:
            print(f"  ✅ {param}: Correctly rejected - {e}")
        except Exception as e:
            print(f"  ❓ {param}: Unexpected error - {e}")
    
    # Show optimization suggestions
    print("\n💡 6. Optimization Suggestions:")
    suggestion_params = ["grid_step_factor", "max_levels", "momentum_weight"]
    
    for param in suggestion_params:
        suggestion = tuner.suggest_optimization(param)
        print(f"  📈 {param}:")
        print(f"    Current: {suggestion['current_value']}")
        print(f"    Range: {suggestion['suggested_range']}")
        if 'suggested_value' in suggestion:
            print(f"    Suggested: {suggestion['suggested_value']}")
        print(f"    Reason: {suggestion['reasoning']}")
        print(f"    Confidence: {suggestion['confidence']}")
        print()
    
    # Show change tracking
    print("\n📝 7. Change Tracking:")
    print(f"  📊 Changes made this session: {len(tuner.changes_made)}")
    for i, change in enumerate(tuner.changes_made[-3:], 1):  # Show last 3 changes
        print(f"  {i}. {change['parameter']}: {change['old_value']} → {change['new_value']}")
        print(f"     ⏰ {change['timestamp']}")
    
    # Save configuration and show version info
    print("\n💾 8. Configuration Management:")
    saved_path = tuner.save_config(versioned=True)
    print(f"  ✅ Configuration saved to: {saved_path}")
    
    versions = tuner.list_versions()
    print(f"  📅 Available versions: {len(versions)}")
    if versions:
        latest = versions[-1]
        print(f"  📊 Latest version: {latest['timestamp']} ({latest['size']} bytes)")
    
    # Validate all parameters
    print("\n🔍 9. Full Configuration Validation:")
    results = tuner.validate_all_parameters()
    print(f"  📊 Total parameters checked: {results['total_checked']}")
    print(f"  ✅ Valid: {len(results['valid'])}")
    print(f"  ❌ Invalid: {len(results['invalid'])}")
    print(f"  ⚠️  Warnings: {len(results['warnings'])}")
    
    if results['invalid']:
        print("\n  ❌ Invalid parameters found:")
        for item in results['invalid'][:3]:  # Show first 3
            print(f"    - {item['parameter']}: {item['error']}")
    
    # Summary
    print("\n🎉 Demo Complete!")
    print("=" * 60)
    print("🔧 Key Features Demonstrated:")
    print("  ✅ Parameter validation with type checking and range limits")
    print("  ✅ Configuration management with versioning")
    print("  ✅ Change tracking and audit trail")
    print("  ✅ Optimization suggestions")
    print("  ✅ Comprehensive error handling")
    print("  ✅ CLI interface for easy parameter adjustments")
    
    print("\n🚀 Next Steps:")
    print("  1. Use CLI: python tools/parameter_tuner.py --help")
    print("  2. Set parameters: --set grid_step_factor=0.6")
    print("  3. Get parameter info: --info grid_step_factor")
    print("  4. List versions: --list-versions")
    print("  5. Validate config: --validate-all")


def main():
    """Run the parameter tuning demonstration."""
    try:
        demo_parameter_tuning()
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main()) 