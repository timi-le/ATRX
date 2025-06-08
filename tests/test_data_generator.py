#!/usr/bin/env python3
"""
Test Data Generator for Integration Testing Demonstration

This module generates sample test results to demonstrate the
integration testing framework and report generation capabilities.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def generate_sample_integration_results() -> dict[str, Any]:
    """Generate sample integration test results"""

    scenarios = {}

    # Normal trading scenario
    scenarios["normal_trading"] = {
        "duration_seconds": 300.5,
        "success": True,
        "throughput_tps": 1150.2,
        "avg_latency_ms": 42.3,
        "max_latency_ms": 89.7,
        "error_count": 0,
        "ticks_processed": 345600,
        "trades_executed": 28,
        "pnl": 150.75,
        "sharpe_ratio": 1.85,
    }

    # High volatility scenario
    scenarios["high_volatility"] = {
        "duration_seconds": 305.8,
        "success": True,
        "throughput_tps": 980.5,
        "avg_latency_ms": 78.1,
        "max_latency_ms": 145.2,
        "error_count": 2,
        "ticks_processed": 299700,
        "trades_executed": 45,
        "pnl": -23.40,
        "sharpe_ratio": 0.92,
    }

    # Low liquidity scenario
    scenarios["low_liquidity"] = {
        "duration_seconds": 298.2,
        "success": False,  # Failed due to high latency
        "throughput_tps": 750.3,
        "avg_latency_ms": 125.8,
        "max_latency_ms": 234.5,
        "error_count": 5,
        "ticks_processed": 223800,
        "trades_executed": 12,
        "pnl": -45.20,
        "sharpe_ratio": -0.15,
    }

    # News events scenario
    scenarios["news_events"] = {
        "duration_seconds": 302.1,
        "success": True,
        "throughput_tps": 1200.8,
        "avg_latency_ms": 55.4,
        "max_latency_ms": 98.3,
        "error_count": 1,
        "ticks_processed": 362640,
        "trades_executed": 67,
        "pnl": 234.60,
        "sharpe_ratio": 2.34,
    }

    # Session transitions scenario
    scenarios["session_transitions"] = {
        "duration_seconds": 295.7,
        "success": True,
        "throughput_tps": 890.2,
        "avg_latency_ms": 38.9,
        "max_latency_ms": 72.1,
        "error_count": 0,
        "ticks_processed": 263280,
        "trades_executed": 19,
        "pnl": 89.30,
        "sharpe_ratio": 1.67,
    }

    return {
        "test_type": "integration_testing",
        "timestamp": datetime.now().isoformat(),
        "total_duration": sum(s["duration_seconds"] for s in scenarios.values()),
        "scenarios": scenarios,
        "overall_success": 0.8,  # 4 out of 5 scenarios passed = 80% success rate
    }


def generate_sample_stress_results() -> dict[str, Any]:
    """Generate sample stress test results"""

    individual_results = {}

    # NFP stress test
    individual_results["nfp_stress"] = {
        "scenario": "NFP_stress_test",
        "duration_seconds": 600.5,
        "ticks_processed": 6000,
        "trades_executed": 89,
        "avg_latency_ms": 67.8,
        "max_latency_ms": 198.5,
        "throughput_tps": 9.98,
        "success": True,
    }

    # Flash crash test
    individual_results["flash_crash"] = {
        "scenario": "flash_crash_test",
        "duration_seconds": 510.2,
        "ticks_processed": 5100,
        "risk_breaches": 8,
        "emergency_stops": 1,
        "success": True,
    }

    # Component failure test
    individual_results["component_failure"] = {
        "scenario": "component_failure_test",
        "duration_seconds": 300.8,
        "operations_attempted": 600,
        "operations_successful": 540,
        "success_rate": 0.9,
        "success": True,
    }

    return {
        "total_stress_tests": 3,
        "passed_tests": 3,
        "success_rate": 1.0,
        "individual_results": individual_results,
        "overall_success": True,
    }


def save_sample_results():
    """Save sample results to logs directory"""

    # Create logs directory
    logs_dir = Path("../logs")
    logs_dir.mkdir(exist_ok=True)

    # Generate and save integration results
    integration_results = generate_sample_integration_results()
    with open(logs_dir / "integration_test_results.json", "w") as f:
        json.dump(integration_results, f, indent=2)

    # Generate and save stress test results
    stress_results = generate_sample_stress_results()
    with open(logs_dir / "stress_test_results.json", "w") as f:
        json.dump(stress_results, f, indent=2)

    print("Sample test results generated:")
    print(f"  Integration: {logs_dir}/integration_test_results.json")
    print(f"  Stress: {logs_dir}/stress_test_results.json")

    return integration_results, stress_results


if __name__ == "__main__":
    integration_results, stress_results = save_sample_results()

    print("\n" + "=" * 60)
    print("SAMPLE TEST RESULTS GENERATED")
    print("=" * 60)
    print(f"Integration Test Scenarios: {len(integration_results['scenarios'])}")
    print(f"Stress Test Scenarios: {stress_results['total_stress_tests']}")
    print(f"Overall Integration Success: {integration_results['overall_success']}")
    print(f"Overall Stress Success: {stress_results['overall_success']}")
    print("=" * 60)
