#!/usr/bin/env python3
"""
Integration Test Runner for FX AI-Quant Trading System

This module orchestrates multiple integration test scenarios and provides comprehensive reporting

This module orchestrates multiple integration test scenarios:
- Normal trading conditions
- High volatility scenarios
- Low liquidity conditions
- News event simulations
- System stress testing
- Latency degradation testing

Provides comprehensive reporting and analysis across all scenarios.
"""

import os
import sys
import asyncio
import time
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Fix import path
try:
    from test_integration_pipeline import (
        SystemIntegrationTester, 
        IntegrationTestConfig, 
        PerformanceMetrics
    )
except ImportError:
    # Alternative import method
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "test_integration_pipeline", 
        Path(__file__).parent / "test_integration_pipeline.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    SystemIntegrationTester = module.SystemIntegrationTester
    IntegrationTestConfig = module.IntegrationTestConfig
    PerformanceMetrics = module.PerformanceMetrics


@dataclass
class TestScenario:
    """Configuration for a specific test scenario."""
    name: str
    description: str
    config: IntegrationTestConfig
    expected_latency_ms: float = 100.0
    expected_throughput_tps: float = 1000.0
    expected_error_rate: float = 0.001
    market_conditions: str = "normal"
    stress_level: str = "low"


class IntegrationTestRunner:
    """Orchestrates multiple integration test scenarios."""
    
    def __init__(self, output_dir: str = "test_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.logger = self._setup_logging()
        self.test_results = {}
        self.summary_metrics = {}
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for test runner."""
        logger = logging.getLogger("integration_test_runner")
        logger.setLevel(logging.INFO)
        
        # File handler
        log_file = self.output_dir / f"test_runner_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def create_test_scenarios(self) -> List[TestScenario]:
        """Create comprehensive test scenarios."""
        scenarios = []
        
        # 1. Normal Trading Conditions
        scenarios.append(TestScenario(
            name="normal_trading",
            description="Standard trading conditions with typical market behavior",
            config=IntegrationTestConfig(
                test_duration_minutes=10,
                tick_rate_per_second=500,
                instruments=["EURUSD", "GBPUSD"],
                latency_target_ms=100,
                throughput_target_tps=500,
                enable_stress_testing=False,
                log_level="INFO"
            ),
            expected_latency_ms=80.0,
            expected_throughput_tps=450.0,
            expected_error_rate=0.0001,
            market_conditions="normal",
            stress_level="low"
        ))
        
        # 2. High Frequency Trading
        scenarios.append(TestScenario(
            name="high_frequency",
            description="High frequency trading with 1000+ TPS",
            config=IntegrationTestConfig(
                test_duration_minutes=5,
                tick_rate_per_second=1200,
                instruments=["EURUSD", "GBPUSD", "USDJPY"],
                latency_target_ms=50,
                throughput_target_tps=1200,
                enable_stress_testing=False,
                log_level="WARNING"
            ),
            expected_latency_ms=45.0,
            expected_throughput_tps=1100.0,
            expected_error_rate=0.0005,
            market_conditions="normal",
            stress_level="medium"
        ))
        
        # 3. High Volatility Scenario
        scenarios.append(TestScenario(
            name="high_volatility",
            description="High volatility market conditions (NFP, FOMC, Brexit-like events)",
            config=IntegrationTestConfig(
                test_duration_minutes=8,
                tick_rate_per_second=800,
                instruments=["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"],
                latency_target_ms=150,
                throughput_target_tps=800,
                enable_stress_testing=True,
                log_level="INFO"
            ),
            expected_latency_ms=120.0,
            expected_throughput_tps=700.0,
            expected_error_rate=0.002,
            market_conditions="volatile",
            stress_level="high"
        ))
        
        # 4. Low Liquidity Conditions
        scenarios.append(TestScenario(
            name="low_liquidity",
            description="Low liquidity market conditions with wide spreads",
            config=IntegrationTestConfig(
                test_duration_minutes=6,
                tick_rate_per_second=300,
                instruments=["EURUSD", "GBPUSD"],
                latency_target_ms=200,
                throughput_target_tps=300,
                enable_stress_testing=True,
                log_level="INFO"
            ),
            expected_latency_ms=150.0,
            expected_throughput_tps=250.0,
            expected_error_rate=0.003,
            market_conditions="illiquid",
            stress_level="medium"
        ))
        
        # 5. News Event Simulation
        scenarios.append(TestScenario(
            name="news_event",
            description="Major news event with price spikes and increased volatility",
            config=IntegrationTestConfig(
                test_duration_minutes=4,
                tick_rate_per_second=1500,
                instruments=["EURUSD", "GBPUSD", "USDJPY"],
                latency_target_ms=100,
                throughput_target_tps=1500,
                enable_stress_testing=True,
                log_level="DEBUG"
            ),
            expected_latency_ms=90.0,
            expected_throughput_tps=1300.0,
            expected_error_rate=0.005,
            market_conditions="news_spike",
            stress_level="very_high"
        ))
        
        # 6. System Stress Test
        scenarios.append(TestScenario(
            name="system_stress",
            description="Maximum system stress with all components under load",
            config=IntegrationTestConfig(
                test_duration_minutes=3,
                tick_rate_per_second=2000,
                instruments=["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"],
                latency_target_ms=200,
                throughput_target_tps=2000,
                enable_stress_testing=True,
                enable_metrics_collection=True,
                log_level="ERROR"  # Reduce log noise during stress
            ),
            expected_latency_ms=180.0,
            expected_throughput_tps=1600.0,
            expected_error_rate=0.01,
            market_conditions="extreme",
            stress_level="maximum"
        ))
        
        # 7. Latency Degradation Test
        scenarios.append(TestScenario(
            name="latency_degradation",
            description="Test system behavior under artificial latency constraints",
            config=IntegrationTestConfig(
                test_duration_minutes=5,
                tick_rate_per_second=600,
                instruments=["EURUSD", "GBPUSD"],
                latency_target_ms=300,  # Relaxed target
                throughput_target_tps=600,
                enable_stress_testing=True,
                log_level="INFO"
            ),
            expected_latency_ms=250.0,
            expected_throughput_tps=500.0,
            expected_error_rate=0.002,
            market_conditions="normal",
            stress_level="latency_constrained"
        ))
        
        return scenarios
    
    async def run_scenario(self, scenario: TestScenario) -> Dict[str, Any]:
        """Run a single test scenario and return results."""
        self.logger.info(f"🚀 Starting scenario: {scenario.name}")
        self.logger.info(f"📝 Description: {scenario.description}")
        
        start_time = time.time()
        
        try:
            # Create tester instance
            tester = SystemIntegrationTester(scenario.config)
            
            # Add scenario-specific market conditions
            if hasattr(tester.market_simulator, 'current_scenario'):
                tester.market_simulator.current_scenario = scenario.market_conditions
            
            # Run the integration test
            metrics = await tester.run_integration_test()
            
            # Calculate scenario results
            duration = time.time() - start_time
            
            result = {
                'scenario': asdict(scenario),
                'metrics': asdict(metrics),
                'duration_seconds': duration,
                'success': self._evaluate_scenario_success(scenario, metrics),
                'performance_score': self._calculate_performance_score(scenario, metrics),
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"✅ Scenario {scenario.name} completed in {duration:.1f}s")
            self.logger.info(f"📊 Performance score: {result['performance_score']:.2f}/100")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Scenario {scenario.name} failed: {e}")
            
            return {
                'scenario': asdict(scenario),
                'metrics': asdict(PerformanceMetrics()),  # Empty metrics
                'duration_seconds': time.time() - start_time,
                'success': False,
                'performance_score': 0.0,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _evaluate_scenario_success(self, scenario: TestScenario, metrics: PerformanceMetrics) -> bool:
        """Evaluate if a scenario passed its success criteria."""
        criteria = [
            metrics.avg_latency_ms <= scenario.expected_latency_ms * 1.5,  # 50% tolerance
            metrics.throughput_tps >= scenario.expected_throughput_tps * 0.7,  # 30% tolerance
            (metrics.total_errors / max(metrics.total_ticks_processed, 1)) <= scenario.expected_error_rate * 2,
            metrics.total_ticks_processed > 0,
            metrics.max_latency_ms < 1000,  # Absolute maximum
        ]
        
        return all(criteria)
    
    def _calculate_performance_score(self, scenario: TestScenario, metrics: PerformanceMetrics) -> float:
        """Calculate a performance score (0-100) for the scenario."""
        scores = []
        
        # Latency score (0-25 points)
        latency_ratio = min(metrics.avg_latency_ms / scenario.expected_latency_ms, 2.0)
        latency_score = max(0, 25 * (2.0 - latency_ratio))
        scores.append(latency_score)
        
        # Throughput score (0-25 points)
        throughput_ratio = metrics.throughput_tps / scenario.expected_throughput_tps
        throughput_score = min(25, 25 * throughput_ratio)
        scores.append(throughput_score)
        
        # Error rate score (0-25 points)
        if metrics.total_ticks_processed > 0:
            error_rate = metrics.total_errors / metrics.total_ticks_processed
            error_ratio = min(error_rate / scenario.expected_error_rate, 2.0)
            error_score = max(0, 25 * (2.0 - error_ratio))
        else:
            error_score = 0
        scores.append(error_score)
        
        # Stability score (0-25 points)
        if metrics.max_latency_ms > 0:
            stability_ratio = metrics.avg_latency_ms / metrics.max_latency_ms
            stability_score = 25 * stability_ratio
        else:
            stability_score = 25
        scores.append(min(25, stability_score))
        
        return sum(scores)
    
    async def run_all_scenarios(self) -> Dict[str, Any]:
        """Run all test scenarios and generate comprehensive report."""
        self.logger.info("🎯 Starting FX AI-Quant Integration Test Suite")
        
        scenarios = self.create_test_scenarios()
        self.logger.info(f"📋 Created {len(scenarios)} test scenarios")
        
        suite_start_time = time.time()
        results = {}
        
        # Run each scenario
        for i, scenario in enumerate(scenarios, 1):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"SCENARIO {i}/{len(scenarios)}: {scenario.name.upper()}")
            self.logger.info(f"{'='*60}")
            
            result = await self.run_scenario(scenario)
            results[scenario.name] = result
            
            # Brief pause between scenarios
            await asyncio.sleep(2)
        
        # Calculate suite summary
        suite_duration = time.time() - suite_start_time
        suite_summary = self._calculate_suite_summary(results, suite_duration)
        
        # Generate comprehensive report
        await self._generate_comprehensive_report(results, suite_summary)
        
        self.logger.info(f"\n🎉 Integration test suite completed in {suite_duration:.1f}s")
        self.logger.info(f"📊 Overall score: {suite_summary['overall_score']:.1f}/100")
        
        return {
            'scenarios': results,
            'summary': suite_summary,
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_suite_summary(self, results: Dict[str, Any], duration: float) -> Dict[str, Any]:
        """Calculate summary metrics for the entire test suite."""
        total_scenarios = len(results)
        passed_scenarios = sum(1 for r in results.values() if r['success'])
        
        # Aggregate metrics
        total_ticks = sum(r['metrics']['total_ticks_processed'] for r in results.values())
        total_trades = sum(r['metrics']['total_trades_executed'] for r in results.values())
        total_errors = sum(r['metrics']['total_errors'] for r in results.values())
        
        # Calculate averages
        performance_scores = [r['performance_score'] for r in results.values()]
        latencies = [r['metrics']['avg_latency_ms'] for r in results.values() if r['metrics']['avg_latency_ms'] > 0]
        throughputs = [r['metrics']['throughput_tps'] for r in results.values() if r['metrics']['throughput_tps'] > 0]
        
        summary = {
            'total_scenarios': total_scenarios,
            'passed_scenarios': passed_scenarios,
            'success_rate': passed_scenarios / total_scenarios if total_scenarios > 0 else 0,
            'overall_score': np.mean(performance_scores) if performance_scores else 0,
            'total_duration_seconds': duration,
            'aggregate_metrics': {
                'total_ticks_processed': total_ticks,
                'total_trades_executed': total_trades,
                'total_errors': total_errors,
                'overall_error_rate': total_errors / total_ticks if total_ticks > 0 else 0,
                'avg_latency_ms': np.mean(latencies) if latencies else 0,
                'max_latency_ms': max(r['metrics']['max_latency_ms'] for r in results.values()) if results else 0,
                'avg_throughput_tps': np.mean(throughputs) if throughputs else 0,
                'max_throughput_tps': max(throughputs) if throughputs else 0
            },
            'scenario_breakdown': {
                name: {
                    'success': result['success'],
                    'score': result['performance_score'],
                    'latency': result['metrics']['avg_latency_ms'],
                    'throughput': result['metrics']['throughput_tps'],
                    'error_rate': result['metrics']['total_errors'] / max(result['metrics']['total_ticks_processed'], 1)
                }
                for name, result in results.items()
            }
        }
        
        return summary
    
    async def _generate_comprehensive_report(self, results: Dict[str, Any], summary: Dict[str, Any]):
        """Generate comprehensive test report with visualizations and analysis."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 1. JSON Summary Report
        summary_file = self.output_dir / f"integration_test_suite_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump({
                'summary': summary,
                'detailed_results': results
            }, f, indent=2)
        
        # 2. CSV Performance Matrix
        performance_data = []
        for name, result in results.items():
            performance_data.append({
                'scenario': name,
                'success': result['success'],
                'performance_score': result['performance_score'],
                'duration_seconds': result['duration_seconds'],
                'ticks_processed': result['metrics']['total_ticks_processed'],
                'trades_executed': result['metrics']['total_trades_executed'],
                'avg_latency_ms': result['metrics']['avg_latency_ms'],
                'max_latency_ms': result['metrics']['max_latency_ms'],
                'throughput_tps': result['metrics']['throughput_tps'],
                'error_rate': result['metrics']['total_errors'] / max(result['metrics']['total_ticks_processed'], 1),
                'market_conditions': result['scenario']['market_conditions'],
                'stress_level': result['scenario']['stress_level']
            })
        
        performance_df = pd.DataFrame(performance_data)
        performance_df.to_csv(self.output_dir / f"performance_matrix_{timestamp}.csv", index=False)
        
        # 3. Latency Analysis Report
        latency_data = []
        for name, result in results.items():
            if result['metrics']['avg_latency_ms'] > 0:
                latency_data.append({
                    'scenario': name,
                    'avg_latency_ms': result['metrics']['avg_latency_ms'],
                    'max_latency_ms': result['metrics']['max_latency_ms'],
                    'expected_latency_ms': result['scenario']['expected_latency_ms'],
                    'latency_target_ms': result['scenario']['config']['latency_target_ms'],
                    'meets_target': result['metrics']['avg_latency_ms'] <= result['scenario']['config']['latency_target_ms']
                })
        
        if latency_data:
            latency_df = pd.DataFrame(latency_data)
            latency_df.to_csv(self.output_dir / f"latency_analysis_{timestamp}.csv", index=False)
        
        # 4. Throughput Analysis Report
        throughput_data = []
        for name, result in results.items():
            if result['metrics']['throughput_tps'] > 0:
                throughput_data.append({
                    'scenario': name,
                    'actual_throughput_tps': result['metrics']['throughput_tps'],
                    'expected_throughput_tps': result['scenario']['expected_throughput_tps'],
                    'target_throughput_tps': result['scenario']['config']['throughput_target_tps'],
                    'throughput_efficiency': result['metrics']['throughput_tps'] / result['scenario']['expected_throughput_tps'],
                    'meets_target': result['metrics']['throughput_tps'] >= result['scenario']['config']['throughput_target_tps'] * 0.8
                })
        
        if throughput_data:
            throughput_df = pd.DataFrame(throughput_data)
            throughput_df.to_csv(self.output_dir / f"throughput_analysis_{timestamp}.csv", index=False)
        
        # 5. Executive Summary Report
        exec_summary = self._generate_executive_summary(summary, results)
        with open(self.output_dir / f"executive_summary_{timestamp}.md", 'w') as f:
            f.write(exec_summary)
        
        self.logger.info(f"📄 Comprehensive test report generated in {self.output_dir}")
    
    def _generate_executive_summary(self, summary: Dict[str, Any], results: Dict[str, Any]) -> str:
        """Generate executive summary in Markdown format."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        report = f"""# FX AI-Quant Integration Test Suite - Executive Summary

**Generated:** {timestamp}  
**Duration:** {summary['total_duration_seconds']:.1f} seconds  
**Overall Score:** {summary['overall_score']:.1f}/100  

## 🎯 Test Results Overview

- **Total Scenarios:** {summary['total_scenarios']}
- **Passed Scenarios:** {summary['passed_scenarios']}
- **Success Rate:** {summary['success_rate']:.1%}
- **Overall Performance Score:** {summary['overall_score']:.1f}/100

## 📊 Aggregate Performance Metrics

- **Total Ticks Processed:** {summary['aggregate_metrics']['total_ticks_processed']:,}
- **Total Trades Executed:** {summary['aggregate_metrics']['total_trades_executed']:,}
- **Overall Error Rate:** {summary['aggregate_metrics']['overall_error_rate']:.4%}
- **Average Latency:** {summary['aggregate_metrics']['avg_latency_ms']:.1f}ms
- **Maximum Latency:** {summary['aggregate_metrics']['max_latency_ms']:.1f}ms
- **Average Throughput:** {summary['aggregate_metrics']['avg_throughput_tps']:.0f} TPS
- **Peak Throughput:** {summary['aggregate_metrics']['max_throughput_tps']:.0f} TPS

## 🏆 Scenario Performance Breakdown

| Scenario | Success | Score | Latency (ms) | Throughput (TPS) | Error Rate | Market Conditions |
|----------|---------|-------|--------------|------------------|------------|-------------------|
"""
        
        for name, breakdown in summary['scenario_breakdown'].items():
            success_icon = "✅" if breakdown['success'] else "❌"
            report += f"| {name} | {success_icon} | {breakdown['score']:.1f} | {breakdown['latency']:.1f} | {breakdown['throughput']:.0f} | {breakdown['error_rate']:.4%} | {results[name]['scenario']['market_conditions']} |\n"
        
        report += f"""
## 🎯 Key Findings

### ✅ System Strengths
"""
        
        # Identify strengths
        high_performing_scenarios = [name for name, breakdown in summary['scenario_breakdown'].items() if breakdown['score'] >= 80]
        if high_performing_scenarios:
            report += f"- **High Performance Scenarios:** {', '.join(high_performing_scenarios)}\n"
        
        if summary['aggregate_metrics']['avg_latency_ms'] < 100:
            report += f"- **Low Latency Achievement:** Average latency of {summary['aggregate_metrics']['avg_latency_ms']:.1f}ms meets performance targets\n"
        
        if summary['aggregate_metrics']['overall_error_rate'] < 0.001:
            report += f"- **High Reliability:** Overall error rate of {summary['aggregate_metrics']['overall_error_rate']:.4%} demonstrates system stability\n"
        
        report += f"""
### ⚠️ Areas for Improvement
"""
        
        # Identify areas for improvement
        failed_scenarios = [name for name, breakdown in summary['scenario_breakdown'].items() if not breakdown['success']]
        if failed_scenarios:
            report += f"- **Failed Scenarios:** {', '.join(failed_scenarios)} require investigation\n"
        
        low_performing_scenarios = [name for name, breakdown in summary['scenario_breakdown'].items() if breakdown['score'] < 50]
        if low_performing_scenarios:
            report += f"- **Low Performance Scenarios:** {', '.join(low_performing_scenarios)} need optimization\n"
        
        if summary['aggregate_metrics']['max_latency_ms'] > 500:
            report += f"- **Latency Spikes:** Maximum latency of {summary['aggregate_metrics']['max_latency_ms']:.1f}ms indicates potential bottlenecks\n"
        
        report += f"""
## 📈 Recommendations

### Immediate Actions
1. **Investigate Failed Scenarios:** Review logs and error patterns for failed test scenarios
2. **Optimize High-Latency Components:** Focus on components contributing to latency spikes
3. **Enhance Error Handling:** Improve robustness in high-stress scenarios

### Performance Optimization
1. **Throughput Enhancement:** Target scenarios with throughput below expectations
2. **Latency Reduction:** Optimize critical path components for sub-50ms latency
3. **Resource Scaling:** Consider horizontal scaling for high-frequency scenarios

### System Monitoring
1. **Real-time Monitoring:** Implement continuous monitoring of key performance metrics
2. **Alerting System:** Set up alerts for latency and error rate thresholds
3. **Regular Testing:** Schedule periodic integration tests to catch regressions

## 🔗 Supporting Documentation

- **Detailed Results:** `integration_test_suite_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json`
- **Performance Matrix:** `performance_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv`
- **Latency Analysis:** `latency_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv`
- **Throughput Analysis:** `throughput_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv`

---
*Report generated by FX AI-Quant Integration Test Suite*
"""
        
        return report


def main():
    """Main function to run integration test suite."""
    import argparse
    
    parser = argparse.ArgumentParser(description="FX AI-Quant Integration Test Suite Runner")
    parser.add_argument("--output", default="test_reports", help="Output directory for test reports")
    parser.add_argument("--scenarios", nargs="+", help="Specific scenarios to run (default: all)")
    parser.add_argument("--quick", action="store_true", help="Run quick version with reduced durations")
    
    args = parser.parse_args()
    
    async def run_tests():
        runner = IntegrationTestRunner(args.output)
        
        if args.quick:
            # Reduce test durations for quick testing
            runner.logger.info("🚀 Running quick integration test suite")
            scenarios = runner.create_test_scenarios()
            for scenario in scenarios:
                scenario.config.test_duration_minutes = max(1, scenario.config.test_duration_minutes // 3)
        
        results = await runner.run_all_scenarios()
        
        # Print summary
        print("\n" + "="*80)
        print("FX AI-QUANT INTEGRATION TEST SUITE RESULTS")
        print("="*80)
        print(f"Overall Score: {results['summary']['overall_score']:.1f}/100")
        print(f"Success Rate: {results['summary']['success_rate']:.1%}")
        print(f"Total Duration: {results['summary']['total_duration_seconds']:.1f}s")
        print(f"Scenarios Passed: {results['summary']['passed_scenarios']}/{results['summary']['total_scenarios']}")
        print("="*80)
        
        return 0 if results['summary']['success_rate'] >= 0.8 else 1
    
    # Run the async main function
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(run_tests())
        return result
    finally:
        loop.close()


if __name__ == "__main__":
    import sys
    sys.exit(main()) 