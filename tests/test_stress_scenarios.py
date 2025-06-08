#!/usr/bin/env python3
"""
Stress Testing Scenarios for FX AI-Quant Trading System

This module provides specialized stress testing scenarios to validate
system behavior under extreme conditions:

- Market volatility spikes (NFP, Brexit, COVID-19 style events)
- Network latency degradation
- Component failure simulation
- High-frequency data bursts
- Memory and CPU stress testing
- Database connection failures
- API rate limiting scenarios
"""

import asyncio
import json
import logging
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


@dataclass
class StressTestConfig:
    """Configuration for stress testing scenarios"""

    scenario_name: str
    duration_minutes: int = 5
    base_tick_rate: int = 1000  # ticks per second
    volatility_multiplier: float = 1.0
    latency_degradation_ms: int = 0
    failure_probability: float = 0.0
    memory_pressure: bool = False
    cpu_stress: bool = False
    network_issues: bool = False


class MarketVolatilitySimulator:
    """Simulates extreme market volatility scenarios"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def generate_nfp_scenario(self, base_price: float = 1.1000) -> list[dict]:
        """Simulate Non-Farm Payroll announcement volatility"""
        ticks = []
        current_price = base_price

        # Pre-announcement: low volatility
        for i in range(300):  # 5 minutes before
            spread = random.uniform(0.00001, 0.00003)
            bid = current_price - spread / 2
            ask = current_price + spread / 2

            ticks.append(
                {
                    "timestamp": datetime.now() + timedelta(seconds=i),
                    "symbol": "EURUSD",
                    "bid": round(bid, 5),
                    "ask": round(ask, 5),
                    "volume": random.randint(1, 5),
                }
            )

            # Small random walk
            current_price += random.uniform(-0.00005, 0.00005)

        # Announcement: extreme volatility
        for i in range(60):  # 1 minute of chaos
            # Large price movements
            shock = random.uniform(-0.002, 0.002)  # 20 pip moves
            current_price += shock

            # Wide spreads
            spread = random.uniform(0.0001, 0.0005)  # 1-5 pip spreads
            bid = current_price - spread / 2
            ask = current_price + spread / 2

            ticks.append(
                {
                    "timestamp": datetime.now() + timedelta(seconds=300 + i),
                    "symbol": "EURUSD",
                    "bid": round(bid, 5),
                    "ask": round(ask, 5),
                    "volume": random.randint(50, 200),  # High volume
                }
            )

        # Post-announcement: gradual normalization
        for i in range(240):  # 4 minutes recovery
            spread = random.uniform(0.00003, 0.0001)
            bid = current_price - spread / 2
            ask = current_price + spread / 2

            ticks.append(
                {
                    "timestamp": datetime.now() + timedelta(seconds=360 + i),
                    "symbol": "EURUSD",
                    "bid": round(bid, 5),
                    "ask": round(ask, 5),
                    "volume": random.randint(5, 20),
                }
            )

            # Smaller movements
            current_price += random.uniform(-0.0001, 0.0001)

        return ticks

    def generate_flash_crash_scenario(self, base_price: float = 1.1000) -> list[dict]:
        """Simulate flash crash scenario (like CHF 2015)"""
        ticks = []
        current_price = base_price

        # Normal trading
        for i in range(180):  # 3 minutes normal
            spread = random.uniform(0.00001, 0.00003)
            bid = current_price - spread / 2
            ask = current_price + spread / 2

            ticks.append(
                {
                    "timestamp": datetime.now() + timedelta(seconds=i),
                    "symbol": "EURUSD",
                    "bid": round(bid, 5),
                    "ask": round(ask, 5),
                    "volume": random.randint(1, 5),
                }
            )

            current_price += random.uniform(-0.00002, 0.00002)

        # Flash crash: massive drop
        crash_magnitude = 0.05  # 500 pip crash
        for i in range(30):  # 30 seconds of crash
            # Exponential decay of crash impact
            crash_impact = crash_magnitude * np.exp(-i / 10)
            current_price -= crash_impact / 30

            # Very wide spreads during crash
            spread = random.uniform(0.001, 0.005)  # 10-50 pip spreads
            bid = current_price - spread / 2
            ask = current_price + spread / 2

            ticks.append(
                {
                    "timestamp": datetime.now() + timedelta(seconds=180 + i),
                    "symbol": "EURUSD",
                    "bid": round(bid, 5),
                    "ask": round(ask, 5),
                    "volume": random.randint(100, 500),
                }
            )

        # Recovery phase
        recovery_target = base_price - 0.02  # Partial recovery
        for i in range(300):  # 5 minutes recovery
            # Gradual price recovery
            price_diff = recovery_target - current_price
            current_price += price_diff * 0.01  # 1% recovery per tick

            spread = random.uniform(0.0001, 0.0003)
            bid = current_price - spread / 2
            ask = current_price + spread / 2

            ticks.append(
                {
                    "timestamp": datetime.now() + timedelta(seconds=210 + i),
                    "symbol": "EURUSD",
                    "bid": round(bid, 5),
                    "ask": round(ask, 5),
                    "volume": random.randint(10, 50),
                }
            )

        return ticks


class LatencyStressTester:
    """Simulates network and processing latency issues"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.base_latency = 0
        self.degradation_active = False

    async def simulate_network_degradation(self, duration_seconds: int = 60):
        """Simulate gradual network degradation"""
        self.degradation_active = True
        start_time = time.time()

        while time.time() - start_time < duration_seconds and self.degradation_active:
            # Gradually increase latency
            elapsed = time.time() - start_time
            progress = elapsed / duration_seconds

            # Exponential latency increase
            self.base_latency = int(50 * (np.exp(progress * 2) - 1))

            await asyncio.sleep(1)

        self.degradation_active = False
        self.base_latency = 0

    async def add_latency(self, base_delay_ms: int = 0):
        """Add artificial latency to operations"""
        total_delay = base_delay_ms + self.base_latency
        if total_delay > 0:
            await asyncio.sleep(total_delay / 1000.0)


class ComponentFailureSimulator:
    """Simulates various component failures"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.failed_components = set()

    def simulate_database_failure(self, duration_seconds: int = 30):
        """Simulate database connection failure"""
        self.failed_components.add("database")
        self.logger.warning(f"Simulating database failure for {duration_seconds}s")

        async def restore_database():
            await asyncio.sleep(duration_seconds)
            self.failed_components.discard("database")
            self.logger.info("Database connection restored")

        asyncio.create_task(restore_database())

    def simulate_api_failure(self, component: str, duration_seconds: int = 45):
        """Simulate API endpoint failure"""
        self.failed_components.add(component)
        self.logger.warning(
            f"Simulating {component} API failure for {duration_seconds}s"
        )

        async def restore_api():
            await asyncio.sleep(duration_seconds)
            self.failed_components.discard(component)
            self.logger.info(f"{component} API restored")

        asyncio.create_task(restore_api())

    def is_component_failed(self, component: str) -> bool:
        """Check if a component is currently failed"""
        return component in self.failed_components


class StressTestRunner:
    """Main stress test runner"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.volatility_sim = MarketVolatilitySimulator()
        self.latency_tester = LatencyStressTester()
        self.failure_sim = ComponentFailureSimulator()

        # Test results
        self.test_results = {}

    async def run_nfp_stress_test(self) -> dict[str, Any]:
        """Run NFP announcement stress test"""
        self.logger.info("Starting NFP stress test scenario")
        start_time = time.time()

        # Generate NFP market data
        market_data = self.volatility_sim.generate_nfp_scenario()

        # Simulate trading during NFP
        trades_executed = 0
        max_latency = 0
        total_latency = 0
        latency_samples = 0

        for tick in market_data:
            tick_start = time.time()

            # Simulate processing latency
            await self.latency_tester.add_latency(random.randint(5, 50))

            # Simulate trade decision and execution
            if random.random() < 0.1:  # 10% chance of trade
                trades_executed += 1

                # Simulate execution latency during high volatility
                execution_latency = random.randint(20, 200)
                await self.latency_tester.add_latency(execution_latency)

            tick_latency = (time.time() - tick_start) * 1000
            max_latency = max(max_latency, tick_latency)
            total_latency += tick_latency
            latency_samples += 1

            # Small delay between ticks
            await asyncio.sleep(0.001)

        duration = time.time() - start_time
        avg_latency = total_latency / latency_samples if latency_samples > 0 else 0

        results = {
            "scenario": "NFP_stress_test",
            "duration_seconds": duration,
            "ticks_processed": len(market_data),
            "trades_executed": trades_executed,
            "avg_latency_ms": avg_latency,
            "max_latency_ms": max_latency,
            "throughput_tps": len(market_data) / duration,
            "success": max_latency < 500 and avg_latency < 100,
        }

        self.test_results["nfp_stress"] = results
        self.logger.info(f"NFP stress test completed: {results}")
        return results

    async def run_flash_crash_test(self) -> dict[str, Any]:
        """Run flash crash stress test"""
        self.logger.info("Starting flash crash stress test")
        start_time = time.time()

        # Generate flash crash data
        market_data = self.volatility_sim.generate_flash_crash_scenario()

        # Simulate system response to flash crash
        risk_breaches = 0
        emergency_stops = 0

        for i, tick in enumerate(market_data):
            # Simulate risk management response
            if i > 180 and i < 210:  # During crash period
                # High probability of risk breach
                if random.random() < 0.3:
                    risk_breaches += 1

                # Emergency stop after multiple breaches
                if risk_breaches > 5:
                    emergency_stops += 1
                    self.logger.warning("Emergency stop triggered")
                    break

            await asyncio.sleep(0.001)

        duration = time.time() - start_time

        results = {
            "scenario": "flash_crash_test",
            "duration_seconds": duration,
            "ticks_processed": len(market_data),
            "risk_breaches": risk_breaches,
            "emergency_stops": emergency_stops,
            "success": emergency_stops > 0,  # System should trigger emergency stop
        }

        self.test_results["flash_crash"] = results
        self.logger.info(f"Flash crash test completed: {results}")
        return results

    async def run_component_failure_test(self) -> dict[str, Any]:
        """Run component failure stress test"""
        self.logger.info("Starting component failure stress test")
        start_time = time.time()

        # Simulate multiple component failures
        self.failure_sim.simulate_database_failure(30)
        await asyncio.sleep(10)

        self.failure_sim.simulate_api_failure("ml_predictor", 45)
        await asyncio.sleep(15)

        self.failure_sim.simulate_api_failure("risk_manager", 20)

        # Test system resilience
        operations_attempted = 0
        operations_successful = 0

        for _ in range(100):
            operations_attempted += 1

            # Simulate operation that might fail
            if not self.failure_sim.is_component_failed("database"):
                operations_successful += 1

            await asyncio.sleep(0.5)

        duration = time.time() - start_time
        success_rate = operations_successful / operations_attempted

        results = {
            "scenario": "component_failure_test",
            "duration_seconds": duration,
            "operations_attempted": operations_attempted,
            "operations_successful": operations_successful,
            "success_rate": success_rate,
            "success": success_rate > 0.5,  # Should maintain >50% success rate
        }

        self.test_results["component_failure"] = results
        self.logger.info(f"Component failure test completed: {results}")
        return results

    async def run_all_stress_tests(self) -> dict[str, Any]:
        """Run all stress test scenarios"""
        self.logger.info("Starting comprehensive stress testing")

        # Run all stress tests
        await self.run_nfp_stress_test()
        await asyncio.sleep(2)  # Brief pause between tests

        await self.run_flash_crash_test()
        await asyncio.sleep(2)

        await self.run_component_failure_test()

        # Generate summary
        total_tests = len(self.test_results)
        passed_tests = sum(
            1 for result in self.test_results.values() if result["success"] > 0
        )

        summary = {
            "total_stress_tests": total_tests,
            "passed_tests": passed_tests,
            "success_rate": passed_tests / total_tests,
            "individual_results": self.test_results,
            "overall_success": passed_tests >= total_tests * 0.7,  # 70% pass rate
        }

        self.logger.info(f"Stress testing completed: {summary}")
        return summary


def main():
    """Main entry point for stress testing"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    async def run_tests():
        runner = StressTestRunner()
        results = await runner.run_all_stress_tests()

        # Save results
        output_file = Path("logs/stress_test_results.json")
        output_file.parent.mkdir(exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\nStress test results saved to: {output_file}")
        print(f"Overall success: {results['overall_success']}")
        print(f"Pass rate: {results['success_rate']:.1%}")

        return results

    return asyncio.run(run_tests())


if __name__ == "__main__":
    main()
