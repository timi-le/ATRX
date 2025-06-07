#!/usr/bin/env python3
"""
Test Report Generator for FX AI-Quant Trading System

This module generates comprehensive test reports by consolidating results
from all integration tests, stress tests, and performance benchmarks.

Features:
- HTML and PDF report generation
- Performance metrics visualization
- Test coverage analysis
- Trend analysis across test runs
- Executive summary generation
- Detailed technical appendices
"""

import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, asdict
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

@dataclass
class TestMetrics:
    """Container for test metrics"""
    test_name: str
    duration_seconds: float
    success: bool
    throughput_tps: float = 0.0
    avg_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    error_count: int = 0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0

@dataclass
class SystemPerformance:
    """Container for system performance metrics"""
    total_tests: int
    passed_tests: int
    failed_tests: int
    success_rate: float
    total_duration: float
    avg_throughput: float
    avg_latency: float
    max_latency: float
    total_errors: int

class TestReportGenerator:
    """Generates comprehensive test reports"""
    
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Set up matplotlib for report generation
        try:
            plt.style.use('seaborn')
        except OSError:
            # Fallback to default style if seaborn is not available
            plt.style.use('default')
        sns.set_palette("husl")
    
    def load_test_results(self) -> Dict[str, Any]:
        """Load all test results from various test files"""
        results = {}
        
        # Load integration test results
        integration_file = Path('logs/integration_test_results.json')
        if integration_file.exists():
            with open(integration_file, 'r') as f:
                results['integration'] = json.load(f)
        
        # Load stress test results
        stress_file = Path('logs/stress_test_results.json')
        if stress_file.exists():
            with open(stress_file, 'r') as f:
                results['stress'] = json.load(f)
        
        # Load performance test results
        performance_file = Path('logs/performance_test_results.json')
        if performance_file.exists():
            with open(performance_file, 'r') as f:
                results['performance'] = json.load(f)
        
        return results
    
    def extract_metrics(self, test_results: Dict[str, Any]) -> List[TestMetrics]:
        """Extract standardized metrics from test results"""
        metrics = []
        
        for test_type, results in test_results.items():
            if test_type == 'integration':
                # Extract integration test metrics
                for scenario_name, scenario_data in results.get('scenarios', {}).items():
                    metric = TestMetrics(
                        test_name=f"integration_{scenario_name}",
                        duration_seconds=scenario_data.get('duration_seconds', 0),
                        success=scenario_data.get('success', False),
                        throughput_tps=scenario_data.get('throughput_tps', 0),
                        avg_latency_ms=scenario_data.get('avg_latency_ms', 0),
                        max_latency_ms=scenario_data.get('max_latency_ms', 0),
                        error_count=scenario_data.get('error_count', 0)
                    )
                    metrics.append(metric)
            
            elif test_type == 'stress':
                # Extract stress test metrics
                for test_name, test_data in results.get('individual_results', {}).items():
                    metric = TestMetrics(
                        test_name=f"stress_{test_name}",
                        duration_seconds=test_data.get('duration_seconds', 0),
                        success=test_data.get('success', False),
                        throughput_tps=test_data.get('throughput_tps', 0),
                        avg_latency_ms=test_data.get('avg_latency_ms', 0),
                        max_latency_ms=test_data.get('max_latency_ms', 0)
                    )
                    metrics.append(metric)
        
        return metrics
    
    def calculate_system_performance(self, metrics: List[TestMetrics]) -> SystemPerformance:
        """Calculate overall system performance metrics"""
        if not metrics:
            return SystemPerformance(0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
        
        total_tests = len(metrics)
        passed_tests = sum(1 for m in metrics if m.success)
        failed_tests = total_tests - passed_tests
        success_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        total_duration = sum(m.duration_seconds for m in metrics)
        avg_throughput = np.mean([m.throughput_tps for m in metrics if m.throughput_tps > 0])
        avg_latency = np.mean([m.avg_latency_ms for m in metrics if m.avg_latency_ms > 0])
        max_latency = max([m.max_latency_ms for m in metrics], default=0)
        total_errors = sum(m.error_count for m in metrics)
        
        return SystemPerformance(
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            success_rate=success_rate,
            total_duration=total_duration,
            avg_throughput=avg_throughput if not np.isnan(avg_throughput) else 0,
            avg_latency=avg_latency if not np.isnan(avg_latency) else 0,
            max_latency=max_latency,
            total_errors=total_errors
        )
    
    def generate_performance_charts(self, metrics: List[TestMetrics]) -> List[str]:
        """Generate performance visualization charts"""
        chart_files = []
        
        if not metrics:
            return chart_files
        
        # Create DataFrame for easier plotting
        df = pd.DataFrame([asdict(m) for m in metrics])
        
        # 1. Test Success Rate Chart
        fig, ax = plt.subplots(figsize=(10, 6))
        success_counts = df['success'].value_counts()
        
        # Handle case where all tests passed or all failed
        if len(success_counts) == 1:
            if success_counts.index[0]:  # All tests passed
                labels = ['Passed']
                colors = ['#51cf66']
            else:  # All tests failed
                labels = ['Failed']
                colors = ['#ff6b6b']
            values = [100]  # 100% of whichever category
        else:
            # Both passed and failed tests exist
            # Reorder to have Failed first, then Passed for consistent coloring
            if True in success_counts.index and False in success_counts.index:
                labels = ['Failed', 'Passed']
                values = [success_counts[False], success_counts[True]]
                colors = ['#ff6b6b', '#51cf66']
            else:
                labels = ['Passed' if success_counts.index[0] else 'Failed']
                values = success_counts.values
                colors = ['#51cf66' if success_counts.index[0] else '#ff6b6b']
        
        ax.pie(values, labels=labels, autopct='%1.1f%%', 
               colors=colors, startangle=90)
        ax.set_title('Test Success Rate Distribution', fontsize=14, fontweight='bold')
        
        chart_file = self.output_dir / 'test_success_rate.png'
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        chart_files.append(str(chart_file))
        
        # 2. Latency Distribution Chart
        if df['avg_latency_ms'].sum() > 0:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Filter out zero latencies for better visualization
            latency_data = df[df['avg_latency_ms'] > 0]
            
            if not latency_data.empty:
                ax.bar(range(len(latency_data)), latency_data['avg_latency_ms'], 
                       color='skyblue', alpha=0.7)
                ax.set_xlabel('Test Index')
                ax.set_ylabel('Average Latency (ms)')
                ax.set_title('Average Latency by Test', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                
                # Add horizontal line for target latency (100ms)
                ax.axhline(y=100, color='red', linestyle='--', alpha=0.7, 
                          label='Target (100ms)')
                ax.legend()
            
            chart_file = self.output_dir / 'latency_distribution.png'
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            plt.close()
            chart_files.append(str(chart_file))
        
        # 3. Throughput Performance Chart
        if df['throughput_tps'].sum() > 0:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            throughput_data = df[df['throughput_tps'] > 0]
            
            if not throughput_data.empty:
                ax.bar(range(len(throughput_data)), throughput_data['throughput_tps'], 
                       color='lightgreen', alpha=0.7)
                ax.set_xlabel('Test Index')
                ax.set_ylabel('Throughput (TPS)')
                ax.set_title('Throughput Performance by Test', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                
                # Add horizontal line for target throughput (1000 TPS)
                ax.axhline(y=1000, color='red', linestyle='--', alpha=0.7, 
                          label='Target (1000 TPS)')
                ax.legend()
            
            chart_file = self.output_dir / 'throughput_performance.png'
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            plt.close()
            chart_files.append(str(chart_file))
        
        # 4. Test Duration Comparison
        fig, ax = plt.subplots(figsize=(12, 8))
        
        test_names = [m.test_name.replace('_', ' ').title() for m in metrics]
        durations = [m.duration_seconds for m in metrics]
        colors = ['green' if m.success else 'red' for m in metrics]
        
        bars = ax.barh(range(len(test_names)), durations, color=colors, alpha=0.7)
        ax.set_yticks(range(len(test_names)))
        ax.set_yticklabels(test_names)
        ax.set_xlabel('Duration (seconds)')
        ax.set_title('Test Duration by Test Type', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='green', alpha=0.7, label='Passed'),
                          Patch(facecolor='red', alpha=0.7, label='Failed')]
        ax.legend(handles=legend_elements)
        
        chart_file = self.output_dir / 'test_duration_comparison.png'
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        chart_files.append(str(chart_file))
        
        return chart_files
    
    def generate_html_report(self, test_results: Dict[str, Any], 
                           metrics: List[TestMetrics], 
                           performance: SystemPerformance,
                           chart_files: List[str]) -> str:
        """Generate comprehensive HTML report"""
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FX AI-Quant Trading System - Integration Test Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            border-bottom: 3px solid #007acc;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            color: #007acc;
            margin: 0;
            font-size: 2.5em;
        }}
        .header p {{
            color: #666;
            margin: 10px 0 0 0;
            font-size: 1.1em;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .metric-card h3 {{
            margin: 0 0 10px 0;
            font-size: 1.1em;
            opacity: 0.9;
        }}
        .metric-card .value {{
            font-size: 2em;
            font-weight: bold;
            margin: 0;
        }}
        .section {{
            margin-bottom: 40px;
        }}
        .section h2 {{
            color: #333;
            border-left: 4px solid #007acc;
            padding-left: 15px;
            margin-bottom: 20px;
        }}
        .test-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }}
        .test-card {{
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 20px;
            background-color: #fafafa;
        }}
        .test-card.passed {{
            border-left: 4px solid #28a745;
        }}
        .test-card.failed {{
            border-left: 4px solid #dc3545;
        }}
        .test-card h4 {{
            margin: 0 0 15px 0;
            color: #333;
        }}
        .test-detail {{
            display: flex;
            justify-content: space-between;
            margin: 5px 0;
            font-size: 0.9em;
        }}
        .test-detail .label {{
            font-weight: bold;
            color: #666;
        }}
        .status-badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            font-weight: bold;
            text-transform: uppercase;
        }}
        .status-badge.passed {{
            background-color: #d4edda;
            color: #155724;
        }}
        .status-badge.failed {{
            background-color: #f8d7da;
            color: #721c24;
        }}
        .chart-container {{
            text-align: center;
            margin: 20px 0;
        }}
        .chart-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .recommendations {{
            background-color: #e7f3ff;
            border: 1px solid #b3d9ff;
            border-radius: 8px;
            padding: 20px;
            margin-top: 30px;
        }}
        .recommendations h3 {{
            color: #0066cc;
            margin-top: 0;
        }}
        .recommendations ul {{
            margin: 0;
            padding-left: 20px;
        }}
        .recommendations li {{
            margin: 8px 0;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>FX AI-Quant Trading System</h1>
            <p>End-to-End Integration Test Report</p>
            <p>Generated on {timestamp}</p>
        </div>
        
        <div class="section">
            <h2>Executive Summary</h2>
            <div class="summary">
                <div class="metric-card">
                    <h3>Total Tests</h3>
                    <p class="value">{performance.total_tests}</p>
                </div>
                <div class="metric-card">
                    <h3>Success Rate</h3>
                    <p class="value">{performance.success_rate:.1%}</p>
                </div>
                <div class="metric-card">
                    <h3>Avg Latency</h3>
                    <p class="value">{performance.avg_latency:.1f}ms</p>
                </div>
                <div class="metric-card">
                    <h3>Avg Throughput</h3>
                    <p class="value">{performance.avg_throughput:.0f} TPS</p>
                </div>
                <div class="metric-card">
                    <h3>Total Duration</h3>
                    <p class="value">{performance.total_duration:.1f}s</p>
                </div>
                <div class="metric-card">
                    <h3>Total Errors</h3>
                    <p class="value">{performance.total_errors}</p>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>Performance Visualizations</h2>
        """
        
        # Add charts
        for chart_file in chart_files:
            chart_name = Path(chart_file).stem.replace('_', ' ').title()
            html_content += f"""
            <div class="chart-container">
                <h3>{chart_name}</h3>
                <img src="{Path(chart_file).name}" alt="{chart_name}">
            </div>
            """
        
        html_content += """
        </div>
        
        <div class="section">
            <h2>Detailed Test Results</h2>
            <div class="test-grid">
        """
        
        # Add individual test results
        for metric in metrics:
            status_class = "passed" if metric.success else "failed"
            status_text = "PASSED" if metric.success else "FAILED"
            
            html_content += f"""
                <div class="test-card {status_class}">
                    <h4>{metric.test_name.replace('_', ' ').title()}</h4>
                    <div class="test-detail">
                        <span class="label">Status:</span>
                        <span class="status-badge {status_class}">{status_text}</span>
                    </div>
                    <div class="test-detail">
                        <span class="label">Duration:</span>
                        <span>{metric.duration_seconds:.2f}s</span>
                    </div>
                    <div class="test-detail">
                        <span class="label">Throughput:</span>
                        <span>{metric.throughput_tps:.0f} TPS</span>
                    </div>
                    <div class="test-detail">
                        <span class="label">Avg Latency:</span>
                        <span>{metric.avg_latency_ms:.1f}ms</span>
                    </div>
                    <div class="test-detail">
                        <span class="label">Max Latency:</span>
                        <span>{metric.max_latency_ms:.1f}ms</span>
                    </div>
                    <div class="test-detail">
                        <span class="label">Errors:</span>
                        <span>{metric.error_count}</span>
                    </div>
                </div>
            """
        
        # Generate recommendations
        recommendations = self.generate_recommendations(performance, metrics)
        
        html_content += f"""
            </div>
        </div>
        
        <div class="recommendations">
            <h3>Recommendations & Next Steps</h3>
            <ul>
        """
        
        for rec in recommendations:
            html_content += f"<li>{rec}</li>"
        
        html_content += """
            </ul>
        </div>
        
        <div class="footer">
            <p>This report was automatically generated by the FX AI-Quant Trading System Integration Test Suite.</p>
            <p>For technical details and raw data, please refer to the individual test log files.</p>
        </div>
    </div>
</body>
</html>
        """
        
        # Save HTML report
        report_file = self.output_dir / f'integration_test_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(report_file)
    
    def generate_recommendations(self, performance: SystemPerformance, 
                               metrics: List[TestMetrics]) -> List[str]:
        """Generate actionable recommendations based on test results"""
        recommendations = []
        
        # Success rate recommendations
        if performance.success_rate < 0.8:
            recommendations.append(
                f"Success rate is {performance.success_rate:.1%}, which is below the 80% target. "
                "Review failed tests and address underlying issues."
            )
        elif performance.success_rate < 0.95:
            recommendations.append(
                f"Success rate is {performance.success_rate:.1%}. Consider investigating "
                "intermittent failures to achieve >95% reliability."
            )
        else:
            recommendations.append(
                f"Excellent success rate of {performance.success_rate:.1%}. "
                "System demonstrates high reliability."
            )
        
        # Latency recommendations
        if performance.avg_latency > 100:
            recommendations.append(
                f"Average latency of {performance.avg_latency:.1f}ms exceeds the 100ms target. "
                "Consider optimizing data processing pipelines and reducing I/O operations."
            )
        elif performance.avg_latency > 50:
            recommendations.append(
                f"Average latency of {performance.avg_latency:.1f}ms is acceptable but could be improved. "
                "Monitor for performance degradation under higher loads."
            )
        
        # Throughput recommendations
        if performance.avg_throughput < 1000:
            recommendations.append(
                f"Average throughput of {performance.avg_throughput:.0f} TPS is below the 1000 TPS target. "
                "Consider implementing parallel processing and optimizing bottlenecks."
            )
        
        # Error recommendations
        if performance.total_errors > 0:
            recommendations.append(
                f"System generated {performance.total_errors} errors during testing. "
                "Review error logs and implement additional error handling."
            )
        
        # Stress test specific recommendations
        stress_tests = [m for m in metrics if m.test_name.startswith('stress_')]
        if stress_tests:
            failed_stress = [m for m in stress_tests if not m.success]
            if failed_stress:
                recommendations.append(
                    f"{len(failed_stress)} stress tests failed. "
                    "System may not handle extreme market conditions well. "
                    "Implement additional circuit breakers and failsafe mechanisms."
                )
        
        # General recommendations
        recommendations.extend([
            "Implement continuous monitoring of key performance metrics in production.",
            "Set up automated alerts for latency spikes and throughput degradation.",
            "Consider implementing adaptive load balancing for high-frequency scenarios.",
            "Regular performance regression testing should be conducted before releases."
        ])
        
        return recommendations
    
    def generate_summary_json(self, test_results: Dict[str, Any], 
                            performance: SystemPerformance) -> str:
        """Generate machine-readable summary in JSON format"""
        summary = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'system': 'FX AI-Quant Trading System',
                'test_type': 'End-to-End Integration Testing'
            },
            'performance_summary': asdict(performance),
            'test_results': test_results,
            'compliance': {
                'latency_target_100ms': performance.avg_latency <= 100,
                'throughput_target_1000tps': performance.avg_throughput >= 1000,
                'success_rate_target_95pct': performance.success_rate >= 0.95,
                'zero_errors': performance.total_errors == 0
            }
        }
        
        summary_file = self.output_dir / f'test_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        return str(summary_file)
    
    def generate_complete_report(self) -> Dict[str, str]:
        """Generate complete test report with all components"""
        self.logger.info("Generating comprehensive test report...")
        
        # Load all test results
        test_results = self.load_test_results()
        
        if not test_results:
            self.logger.warning("No test results found. Run integration tests first.")
            return {}
        
        # Extract metrics
        metrics = self.extract_metrics(test_results)
        
        # Calculate performance
        performance = self.calculate_system_performance(metrics)
        
        # Generate charts
        chart_files = self.generate_performance_charts(metrics)
        
        # Generate reports
        html_report = self.generate_html_report(test_results, metrics, performance, chart_files)
        json_summary = self.generate_summary_json(test_results, performance)
        
        report_files = {
            'html_report': html_report,
            'json_summary': json_summary,
            'charts': chart_files
        }
        
        self.logger.info(f"Test report generated successfully:")
        self.logger.info(f"  HTML Report: {html_report}")
        self.logger.info(f"  JSON Summary: {json_summary}")
        self.logger.info(f"  Charts: {len(chart_files)} files")
        
        return report_files

def main():
    """Main entry point for report generation"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    generator = TestReportGenerator()
    report_files = generator.generate_complete_report()
    
    if report_files:
        print("\n" + "="*60)
        print("FX AI-QUANT TRADING SYSTEM - TEST REPORT GENERATED")
        print("="*60)
        print(f"HTML Report: {report_files.get('html_report', 'N/A')}")
        print(f"JSON Summary: {report_files.get('json_summary', 'N/A')}")
        print(f"Charts Generated: {len(report_files.get('charts', []))}")
        print("="*60)
    else:
        print("No test results found. Please run integration tests first.")

if __name__ == '__main__':
    main() 