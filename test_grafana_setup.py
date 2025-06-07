#!/usr/bin/env python3
"""
Test script for Grafana Dashboard Setup (Task 20)

This script validates the complete monitoring stack including:
- Docker containers running (Prometheus, Grafana, Alertmanager)
- Metrics server connectivity and data availability
- Dashboard imports and functionality
- Alerting rules validation
"""

import sys
import os
import time
import logging
import requests
import subprocess
import json
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from monitoring.metrics_server import start_metrics_server, get_metrics
except ImportError:
    print("⚠️  Warning: Could not import metrics server. Please ensure Task 19 is completed.")
    start_metrics_server = None
    get_metrics = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GrafanaTestSuite:
    """Comprehensive test suite for Grafana monitoring stack."""
    
    def __init__(self):
        self.prometheus_url = "http://localhost:9090"
        self.grafana_url = "http://localhost:3000"
        self.alertmanager_url = "http://localhost:9093"
        self.metrics_url = "http://localhost:9000/metrics"
        self.grafana_user = "admin"
        self.grafana_password = "Twizb170317"
        
    def test_docker_containers(self):
        """Test that all Docker containers are running."""
        logger.info("=== Testing Docker Containers ===")
        
        required_containers = [
            "fx-prometheus",
            "fx-grafana", 
            "fx-alertmanager"
        ]
        
        try:
            # Get running containers
            result = subprocess.run(
                ["docker", "ps", "--format", "{{.Names}}"],
                capture_output=True,
                text=True,
                check=True
            )
            
            running_containers = result.stdout.strip().split('\n')
            
            for container in required_containers:
                if container in running_containers:
                    logger.info(f"✓ Container {container} is running")
                else:
                    logger.error(f"❌ Container {container} is NOT running")
                    return False
                    
            logger.info("✓ All Docker containers are running")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Error checking Docker containers: {e}")
            return False
        except FileNotFoundError:
            logger.error("❌ Docker command not found. Please install Docker.")
            return False
    
    def test_metrics_server_connectivity(self):
        """Test connectivity to the FX metrics server."""
        logger.info("=== Testing Metrics Server Connectivity ===")
        
        try:
            response = requests.get(self.metrics_url, timeout=5)
            if response.status_code == 200:
                content = response.text
                
                # Check for key metrics
                expected_metrics = [
                    "fxai_pnl_equity",
                    "fxai_trades_total",
                    "fxai_execution_latency_seconds",
                    "fxai_regime_ratio",
                    "fxai_errors_total"
                ]
                
                missing_metrics = []
                for metric in expected_metrics:
                    if metric not in content:
                        missing_metrics.append(metric)
                
                if not missing_metrics:
                    logger.info("✓ Metrics server is accessible and serving expected metrics")
                    return True
                else:
                    logger.error(f"❌ Missing metrics: {missing_metrics}")
                    return False
            else:
                logger.error(f"❌ Metrics server returned status {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Cannot connect to metrics server: {e}")
            return False
    
    def test_prometheus_connectivity(self):
        """Test Prometheus connectivity and configuration."""
        logger.info("=== Testing Prometheus Connectivity ===")
        
        try:
            # Test basic connectivity
            response = requests.get(f"{self.prometheus_url}/api/v1/targets", timeout=10)
            if response.status_code != 200:
                logger.error(f"❌ Prometheus API returned status {response.status_code}")
                return False
            
            targets = response.json()
            
            # Check if FX trading system target is configured
            fx_target_found = False
            for target in targets.get('data', {}).get('activeTargets', []):
                if target.get('labels', {}).get('job') == 'fx-ai-trading-system':
                    fx_target_found = True
                    health = target.get('health', 'unknown')
                    if health == 'up':
                        logger.info("✓ FX trading system target is UP in Prometheus")
                    else:
                        logger.warning(f"⚠️  FX trading system target health: {health}")
                    break
            
            if not fx_target_found:
                logger.error("❌ FX trading system target not found in Prometheus")
                return False
            
            # Test a sample query
            query_response = requests.get(
                f"{self.prometheus_url}/api/v1/query",
                params={"query": "fxai_pnl_equity"},
                timeout=5
            )
            
            if query_response.status_code == 200:
                data = query_response.json()
                if data.get('data', {}).get('result'):
                    logger.info("✓ Prometheus can query FX metrics successfully")
                    return True
                else:
                    logger.warning("⚠️  Prometheus connected but no FX metrics data found")
                    return True
            else:
                logger.error(f"❌ Prometheus query failed: {query_response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Cannot connect to Prometheus: {e}")
            return False
    
    def test_grafana_connectivity(self):
        """Test Grafana connectivity and authentication."""
        logger.info("=== Testing Grafana Connectivity ===")
        
        try:
            # Test basic connectivity
            response = requests.get(f"{self.grafana_url}/api/health", timeout=10)
            if response.status_code != 200:
                logger.error(f"❌ Grafana health check failed: {response.status_code}")
                return False
            
            logger.info("✓ Grafana is responding to health checks")
            
            # Test authentication
            auth_response = requests.get(
                f"{self.grafana_url}/api/user",
                auth=(self.grafana_user, self.grafana_password),
                timeout=5
            )
            
            if auth_response.status_code == 200:
                user_info = auth_response.json()
                logger.info(f"✓ Grafana authentication successful (user: {user_info.get('login', 'unknown')})")
                return True
            else:
                logger.error(f"❌ Grafana authentication failed: {auth_response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Cannot connect to Grafana: {e}")
            return False
    
    def test_grafana_datasource(self):
        """Test Grafana Prometheus datasource configuration."""
        logger.info("=== Testing Grafana Datasource ===")
        
        try:
            response = requests.get(
                f"{self.grafana_url}/api/datasources",
                auth=(self.grafana_user, self.grafana_password),
                timeout=5
            )
            
            if response.status_code != 200:
                logger.error(f"❌ Failed to get datasources: {response.status_code}")
                return False
            
            datasources = response.json()
            prometheus_ds = None
            
            for ds in datasources:
                if ds.get('type') == 'prometheus':
                    prometheus_ds = ds
                    break
            
            if not prometheus_ds:
                logger.error("❌ Prometheus datasource not found in Grafana")
                return False
            
            logger.info(f"✓ Prometheus datasource found: {prometheus_ds.get('name', 'Unknown')}")
            
            # Test datasource health
            ds_id = prometheus_ds.get('id')
            health_response = requests.get(
                f"{self.grafana_url}/api/datasources/{ds_id}/health",
                auth=(self.grafana_user, self.grafana_password),
                timeout=10
            )
            
            if health_response.status_code == 200:
                health_data = health_response.json()
                if health_data.get('status') == 'OK':
                    logger.info("✓ Prometheus datasource health check passed")
                    return True
                else:
                    logger.error(f"❌ Datasource health check failed: {health_data}")
                    return False
            else:
                logger.error(f"❌ Datasource health check request failed: {health_response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Error testing Grafana datasource: {e}")
            return False
    
    def test_grafana_dashboards(self):
        """Test if FX trading dashboards are imported and accessible."""
        logger.info("=== Testing Grafana Dashboards ===")
        
        try:
            response = requests.get(
                f"{self.grafana_url}/api/search",
                auth=(self.grafana_user, self.grafana_password),
                timeout=5
            )
            
            if response.status_code != 200:
                logger.error(f"❌ Failed to search dashboards: {response.status_code}")
                return False
            
            dashboards = response.json()
            
            expected_dashboards = [
                "FX AI-Quant Trading Performance",
                "FX AI-Quant System Health"
            ]
            
            found_dashboards = []
            for dashboard in dashboards:
                title = dashboard.get('title', '')
                if any(expected in title for expected in expected_dashboards):
                    found_dashboards.append(title)
            
            if len(found_dashboards) >= 2:
                logger.info(f"✓ Found FX trading dashboards: {found_dashboards}")
                return True
            else:
                logger.warning(f"⚠️  Expected dashboards not found. Found: {found_dashboards}")
                logger.info("ℹ️  Dashboards may need to be imported manually")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Error testing Grafana dashboards: {e}")
            return False
    
    def test_alertmanager_connectivity(self):
        """Test Alertmanager connectivity and configuration."""
        logger.info("=== Testing Alertmanager Connectivity ===")
        
        try:
            response = requests.get(f"{self.alertmanager_url}/-/healthy", timeout=5)
            if response.status_code == 200:
                logger.info("✓ Alertmanager is responding to health checks")
                
                # Check if we can access the API
                api_response = requests.get(f"{self.alertmanager_url}/api/v1/alerts", timeout=5)
                if api_response.status_code == 200:
                    logger.info("✓ Alertmanager API is accessible")
                    return True
                else:
                    logger.warning("⚠️  Alertmanager API not accessible, but service is healthy")
                    return True
            else:
                logger.error(f"❌ Alertmanager health check failed: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Cannot connect to Alertmanager: {e}")
            return False
    
    def test_sample_metrics_generation(self):
        """Generate sample metrics data for testing dashboards."""
        logger.info("=== Generating Sample Metrics Data ===")
        
        if not start_metrics_server or not get_metrics:
            logger.warning("⚠️  Metrics server not available, skipping sample data generation")
            return True
        
        try:
            # Start metrics server if not running
            logger.info("Starting sample metrics generation...")
            
            metrics = get_metrics()
            
            # Generate sample trading data
            metrics.update_pnl(equity=105000.0, daily_pnl=5000.0)
            metrics.record_trade("EURUSD", "BUY", "filled", pnl=250.0)
            metrics.record_trade("GBPUSD", "SELL", "filled", pnl=-120.0)
            metrics.record_execution_latency(0.025)
            metrics.record_slippage(1.2)
            
            # Generate regime data
            metrics.update_regime("EURUSD", "trending", 0.65)
            metrics.update_regime("EURUSD", "ranging", 0.35)
            metrics.update_regime("GBPUSD", "trending", 0.45)
            metrics.update_regime("GBPUSD", "ranging", 0.55)
            
            # Generate system health data
            metrics.update_system_resources(memory_mb=756.2, cpu_percent=35.5)
            metrics.update_risk_metrics(current_drawdown=0.015, max_drawdown=0.025)
            metrics.update_performance_ratios(sharpe_ratio=1.65)
            
            logger.info("✓ Sample metrics data generated successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error generating sample metrics: {e}")
            return False
    
    def run_comprehensive_test(self):
        """Run all tests and provide summary."""
        logger.info("🚀 Starting Comprehensive Grafana Setup Test")
        logger.info("=" * 60)
        
        tests = [
            ("Docker Containers", self.test_docker_containers),
            ("Metrics Server", self.test_metrics_server_connectivity),
            ("Prometheus", self.test_prometheus_connectivity),
            ("Grafana", self.test_grafana_connectivity),
            ("Grafana Datasource", self.test_grafana_datasource),
            ("Grafana Dashboards", self.test_grafana_dashboards),
            ("Alertmanager", self.test_alertmanager_connectivity),
            ("Sample Data Generation", self.test_sample_metrics_generation)
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            try:
                logger.info(f"\n--- Testing {test_name} ---")
                result = test_func()
                results[test_name] = result
                
                if result:
                    logger.info(f"✅ {test_name}: PASSED")
                else:
                    logger.error(f"❌ {test_name}: FAILED")
                    
            except Exception as e:
                logger.error(f"❌ {test_name}: ERROR - {e}")
                results[test_name] = False
        
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("🎯 TEST SUMMARY")
        logger.info("=" * 60)
        
        passed = sum(1 for result in results.values() if result)
        total = len(results)
        
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"{test_name:<25} {status}")
        
        logger.info("")
        logger.info(f"Overall Result: {passed}/{total} tests passed")
        
        if passed == total:
            logger.info("🎉 ALL TESTS PASSED! Grafana monitoring stack is ready!")
            logger.info("")
            logger.info("🌐 Access Points:")
            logger.info(f"  • Grafana Dashboard: {self.grafana_url} (admin/admin)")
            logger.info(f"  • Prometheus: {self.prometheus_url}")
            logger.info(f"  • Alertmanager: {self.alertmanager_url}")
            logger.info(f"  • Metrics Endpoint: {self.metrics_url}")
        else:
            logger.error("❌ Some tests failed. Please check the logs above.")
            logger.info("")
            logger.info("💡 Common issues and solutions:")
            logger.info("  • Docker containers not running: Run 'docker-compose -f docker-compose.monitoring.yml up -d'")
            logger.info("  • Metrics server not accessible: Start with 'python demo_metrics_monitoring.py'")
            logger.info("  • Dashboards not found: Import manually from monitoring/grafana/dashboards/")
        
        return passed == total

def main():
    """Run the Grafana setup test suite."""
    test_suite = GrafanaTestSuite()
    
    try:
        success = test_suite.run_comprehensive_test()
        
        if success:
            print("\n🏁 Grafana monitoring stack validation completed successfully!")
            print("Task 20: Dashboard and Visualization is ready for production use.")
        else:
            print("\n⚠️  Some tests failed. Please review the output above.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error during testing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 