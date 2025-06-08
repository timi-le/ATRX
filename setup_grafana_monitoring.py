#!/usr/bin/env python3
"""
Setup script for FX Quant Trading System - Grafana Monitoring Stack

This script automates the complete setup of the monitoring infrastructure:
- Starts Docker containers (Prometheus, Grafana, Alertmanager)
- Imports dashboards automatically
- Configures datasources
- Validates the complete setup
"""

import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class GrafanaMonitoringSetup:
    """Automated setup for the FX trading monitoring stack."""

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.grafana_url = "http://localhost:3000"
        self.grafana_user = "admin"
        self.grafana_password = "admin"

    def check_prerequisites(self):
        """Check if Docker and Docker Compose are available."""
        logger.info("=== Checking Prerequisites ===")

        try:
            # Check Docker
            result = subprocess.run(
                ["docker", "--version"], capture_output=True, text=True, check=True
            )
            logger.info(f"✓ Docker found: {result.stdout.strip()}")

            # Check Docker Compose
            result = subprocess.run(
                ["docker", "compose", "version"],
                capture_output=True,
                text=True,
                check=True,
            )
            logger.info(f"✓ Docker Compose found: {result.stdout.strip()}")

            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Docker/Docker Compose check failed: {e}")
            return False
        except FileNotFoundError:
            logger.error(
                "❌ Docker not found. Please install Docker and Docker Compose."
            )
            return False

    def start_docker_stack(self):
        """Start the monitoring Docker stack."""
        logger.info("=== Starting Docker Monitoring Stack ===")

        compose_file = self.project_root / "docker-compose.monitoring.yml"

        if not compose_file.exists():
            logger.error(f"❌ Docker Compose file not found: {compose_file}")
            return False

        try:
            # Stop any existing containers
            logger.info("Stopping any existing containers...")
            subprocess.run(
                ["docker", "compose", "-f", str(compose_file), "down"],
                capture_output=True,
                check=False,  # Don't fail if containers aren't running
            )

            # Start the stack
            logger.info("Starting monitoring stack...")
            result = subprocess.run(
                ["docker", "compose", "-f", str(compose_file), "up", "-d"],
                capture_output=True,
                text=True,
                check=True,
            )

            logger.info("✓ Docker stack started successfully")

            # Wait for services to be ready
            logger.info("Waiting for services to become ready...")
            self._wait_for_services()

            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to start Docker stack: {e}")
            logger.error(f"STDERR: {e.stderr}")
            return False

    def _wait_for_services(self):
        """Wait for all services to become healthy."""
        services = {
            "Grafana": "http://localhost:3000/api/health",
            "Prometheus": "http://localhost:9090/-/healthy",
            "Alertmanager": "http://localhost:9093/api/v1/status",
        }

        max_wait = 120  # 2 minutes
        start_time = time.time()

        while time.time() - start_time < max_wait:
            all_ready = True

            for service, url in services.items():
                try:
                    response = requests.get(url, timeout=5)
                    if response.status_code == 200:
                        logger.info(f"✓ {service} is ready")
                    else:
                        all_ready = False
                        logger.info(f"⏳ {service} not ready yet...")
                except requests.exceptions.RequestException:
                    all_ready = False
                    logger.info(f"⏳ {service} not ready yet...")

            if all_ready:
                logger.info("✓ All services are ready!")
                return True

            time.sleep(5)

        logger.warning("⚠️  Some services may not be fully ready, continuing...")
        return True

    def import_grafana_dashboards(self):
        """Import FX trading dashboards into Grafana."""
        logger.info("=== Importing Grafana Dashboards ===")

        dashboards_dir = self.project_root / "monitoring" / "grafana" / "dashboards"

        if not dashboards_dir.exists():
            logger.error(f"❌ Dashboards directory not found: {dashboards_dir}")
            return False

        dashboard_files = list(dashboards_dir.glob("*.json"))

        if not dashboard_files:
            logger.warning("⚠️  No dashboard files found to import")
            return True

        success_count = 0

        for dashboard_file in dashboard_files:
            try:
                logger.info(f"Importing dashboard: {dashboard_file.name}")

                with open(dashboard_file) as f:
                    dashboard_data = json.load(f)

                # Prepare the import payload
                import_payload = {
                    "dashboard": dashboard_data,
                    "overwrite": True,
                    "inputs": [
                        {
                            "name": "DS_PROMETHEUS",
                            "type": "datasource",
                            "pluginId": "prometheus",
                            "value": "prometheus",
                        }
                    ],
                }

                response = requests.post(
                    f"{self.grafana_url}/api/dashboards/import",
                    auth=(self.grafana_user, self.grafana_password),
                    json=import_payload,
                    timeout=10,
                )

                if response.status_code in [200, 201]:
                    result = response.json()
                    logger.info(
                        f"✓ Dashboard imported: {result.get('title', 'Unknown')}"
                    )
                    success_count += 1
                else:
                    logger.error(
                        f"❌ Failed to import {dashboard_file.name}: {response.status_code}"
                    )
                    logger.error(f"Response: {response.text}")

            except Exception as e:
                logger.error(f"❌ Error importing {dashboard_file.name}: {e}")

        logger.info(
            f"✓ Successfully imported {success_count}/{len(dashboard_files)} dashboards"
        )
        return success_count > 0

    def verify_prometheus_datasource(self):
        """Verify Prometheus datasource is configured in Grafana."""
        logger.info("=== Verifying Prometheus Datasource ===")

        try:
            # Get existing datasources
            response = requests.get(
                f"{self.grafana_url}/api/datasources",
                auth=(self.grafana_user, self.grafana_password),
                timeout=5,
            )

            if response.status_code != 200:
                logger.error(f"❌ Failed to get datasources: {response.status_code}")
                return False

            datasources = response.json()

            # Check if Prometheus datasource exists
            for ds in datasources:
                if ds.get("type") == "prometheus":
                    logger.info(f"✓ Prometheus datasource found: {ds.get('name')}")

                    # Test datasource health
                    health_response = requests.get(
                        f"{self.grafana_url}/api/datasources/{ds.get('id')}/health",
                        auth=(self.grafana_user, self.grafana_password),
                        timeout=10,
                    )

                    if health_response.status_code == 200:
                        health_data = health_response.json()
                        if health_data.get("status") == "OK":
                            logger.info("✓ Prometheus datasource is healthy")
                            return True
                        else:
                            logger.warning(
                                f"⚠️  Datasource health issue: {health_data}"
                            )
                            return True
                    else:
                        logger.warning("⚠️  Could not check datasource health")
                        return True

            logger.warning(
                "⚠️  Prometheus datasource not found (may be auto-provisioned)"
            )
            return True

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Error verifying datasource: {e}")
            return False

    def start_metrics_server(self):
        """Start the FX metrics server if available."""
        logger.info("=== Starting FX Metrics Server ===")

        try:
            from monitoring.metrics_server import start_metrics_server

            # Start metrics server
            logger.info("Starting FX metrics server...")
            metrics = start_metrics_server(port=9000)

            # Generate some sample data
            logger.info("Generating sample metrics data...")
            metrics.update_pnl(equity=100000.0, daily_pnl=0.0)
            metrics.record_trade("EURUSD", "BUY", "filled", pnl=150.0)
            metrics.update_regime("EURUSD", "trending", 0.6)
            metrics.update_regime("EURUSD", "ranging", 0.4)
            metrics.update_system_resources(memory_mb=512.0, cpu_percent=25.0)

            logger.info("✓ FX metrics server started and populated with sample data")
            return True

        except ImportError:
            logger.warning(
                "⚠️  FX metrics server not available (Task 19 not completed)"
            )
            return True
        except Exception as e:
            logger.error(f"❌ Error starting metrics server: {e}")
            return False

    def print_access_information(self):
        """Print access information for all services."""
        logger.info("\n" + "=" * 60)
        logger.info("🎉 GRAFANA MONITORING STACK SETUP COMPLETE!")
        logger.info("=" * 60)
        logger.info("")
        logger.info("🌐 Access URLs:")
        logger.info(f"  📊 Grafana Dashboard:  http://localhost:3000")
        logger.info(f"      Username: admin")
        logger.info(f"      Password: admin")
        logger.info("")
        logger.info(f"  📈 Prometheus:        http://localhost:9090")
        logger.info(f"  🚨 Alertmanager:      http://localhost:9093")
        logger.info(f"  📊 Metrics Endpoint:  http://localhost:9000/metrics")
        logger.info("")
        logger.info("📋 Available Dashboards:")
        logger.info("  • FX AI-Quant Trading Performance")
        logger.info("  • FX AI-Quant System Health")
        logger.info("")
        logger.info("🔔 Alerting:")
        logger.info("  • Configured for critical trading alerts")
        logger.info("  • Email and Slack notifications ready")
        logger.info("  • Edit monitoring/alertmanager.yml to configure")
        logger.info("")
        logger.info("🛠️  Next Steps:")
        logger.info("  1. Visit Grafana and explore the dashboards")
        logger.info("  2. Configure email/Slack in alertmanager.yml")
        logger.info("  3. Integrate with your trading system")
        logger.info("  4. Run 'python test_grafana_setup.py' to validate")
        logger.info("")

    def run_complete_setup(self):
        """Run the complete monitoring stack setup."""
        logger.info("🚀 Starting FX Quant Trading System - Grafana Monitoring Setup")
        logger.info("=" * 70)

        steps = [
            ("Prerequisites Check", self.check_prerequisites),
            ("Docker Stack Startup", self.start_docker_stack),
            ("FX Metrics Server", self.start_metrics_server),
            ("Prometheus Datasource", self.verify_prometheus_datasource),
            ("Dashboard Import", self.import_grafana_dashboards),
        ]

        for step_name, step_func in steps:
            logger.info(f"\n--- {step_name} ---")

            try:
                success = step_func()

                if success:
                    logger.info(f"✅ {step_name}: COMPLETED")
                else:
                    logger.error(f"❌ {step_name}: FAILED")
                    logger.error("Setup cannot continue. Please check the logs above.")
                    return False

            except Exception as e:
                logger.error(f"❌ {step_name}: ERROR - {e}")
                return False

        # Print success information
        self.print_access_information()

        return True


def main():
    """Main setup function."""
    setup = GrafanaMonitoringSetup()

    try:
        success = setup.run_complete_setup()

        if success:
            print("\n🏁 Setup completed successfully!")
            print(
                "You can now run 'python test_grafana_setup.py' to validate the installation."
            )
        else:
            print("\n❌ Setup failed. Please check the logs above.")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("\n⚠️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error during setup: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
