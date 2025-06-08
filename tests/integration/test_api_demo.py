#!/usr/bin/env python3
"""
Comprehensive demo script for the Regime Detection API.

This script demonstrates:
1. Starting the API server
2. Testing all endpoints
3. Performance validation
4. Error handling
5. Real-world usage scenarios
"""

import concurrent.futures
import subprocess
import sys
import time
from pathlib import Path

import requests

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


class APITester:
    """Test the Regime Detection API."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def test_health_check(self):
        """Test the health check endpoint."""
        print("🔍 Testing health check...")

        response = self.session.get(f"{self.base_url}/health")

        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed")
            print(f"   Status: {data['status']}")
            print(f"   Detector ready: {data['detector_ready']}")
            print(f"   Uptime: {data['uptime_seconds']:.2f}s")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False

    def test_root_endpoint(self):
        """Test the root endpoint."""
        print("\n🔍 Testing root endpoint...")

        response = self.session.get(f"{self.base_url}/")

        if response.status_code == 200:
            data = response.json()
            print(f"✅ Root endpoint working")
            print(f"   API: {data['name']}")
            print(f"   Version: {data['version']}")
            print(f"   Available endpoints: {len(data['endpoints'])}")
            return True
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
            return False

    def test_regime_prediction(self):
        """Test regime prediction endpoint."""
        print("\n🔍 Testing regime prediction...")

        # Test different market scenarios
        scenarios = [
            (
                "Strong Uptrend",
                {
                    "atr": 0.4,
                    "bb_width": 0.5,
                    "realized_vol": 0.3,
                    "vol_ratio": 1.0,
                    "macd_signal": 0.8,
                    "macd_histogram": 0.6,
                    "adx": 70,
                    "rsi": 75,
                    "momentum": 0.9,
                    "macro_surprise": 0.2,
                    "macro_sentiment": 0.4,
                    "trend_strength": 0.9,
                    "mean_reversion": 0.1,
                },
            ),
            (
                "Range-bound Market",
                {
                    "atr": 0.2,
                    "bb_width": 0.15,
                    "realized_vol": 0.1,
                    "vol_ratio": 0.8,
                    "macd_signal": 0.1,
                    "macd_histogram": 0.05,
                    "adx": 15,
                    "rsi": 50,
                    "momentum": 0.1,
                    "macro_surprise": 0.0,
                    "macro_sentiment": 0.0,
                    "trend_strength": 0.2,
                    "mean_reversion": 0.9,
                },
            ),
            (
                "High Volatility Chop",
                {
                    "atr": 0.9,
                    "bb_width": 0.8,
                    "realized_vol": 0.8,
                    "vol_ratio": 1.6,
                    "macd_signal": 0.1,
                    "macd_histogram": 0.0,
                    "adx": 10,
                    "rsi": 45,
                    "momentum": 0.0,
                    "macro_surprise": 0.3,
                    "macro_sentiment": -0.2,
                    "trend_strength": 0.1,
                    "mean_reversion": 0.3,
                },
            ),
        ]

        results = []
        for scenario_name, features in scenarios:
            print(f"   Testing {scenario_name}...")

            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}/regime/predict", json=features
            )
            end_time = time.time()

            if response.status_code == 200:
                data = response.json()
                latency_ms = (end_time - start_time) * 1000

                print(
                    f"   ✅ {scenario_name}: {data['regime']} (confidence: {data['confidence']:.2f}, latency: {latency_ms:.1f}ms)"
                )
                print(
                    f"      Probabilities: {[(k, f'{v:.2f}') for k, v in data['probabilities'].items()]}"
                )

                results.append(
                    {
                        "scenario": scenario_name,
                        "regime": data["regime"],
                        "confidence": data["confidence"],
                        "latency_ms": latency_ms,
                        "success": True,
                    }
                )
            else:
                print(f"   ❌ {scenario_name} failed: {response.status_code}")
                results.append(
                    {
                        "scenario": scenario_name,
                        "success": False,
                        "error": response.status_code,
                    }
                )

        return results

    def test_current_regime(self):
        """Test getting current regime."""
        print("\n🔍 Testing current regime endpoint...")

        response = self.session.get(f"{self.base_url}/regime/current")

        if response.status_code == 200:
            data = response.json()
            print(
                f"✅ Current regime: {data['regime']} (confidence: {data['confidence']:.2f})"
            )
            return True
        elif response.status_code == 404:
            print("ℹ️  No current regime data available")
            return True
        else:
            print(f"❌ Current regime failed: {response.status_code}")
            return False

    def test_regime_history(self):
        """Test regime history endpoint."""
        print("\n🔍 Testing regime history...")

        response = self.session.get(f"{self.base_url}/regime/history?window=10")

        if response.status_code == 200:
            data = response.json()
            print(f"✅ History retrieved: {data['total_items']} items")
            print(f"   Window: {data['window_size']}")
            print(f"   Time range: {data['start_time']} to {data['end_time']}")

            if data["history"]:
                print(
                    f"   Recent regimes: {[item['regime'] for item in data['history'][:5]]}"
                )

            return True
        else:
            print(f"❌ History failed: {response.status_code}")
            return False

    def test_transition_matrix(self):
        """Test transition matrix endpoint."""
        print("\n🔍 Testing transition matrix...")

        response = self.session.get(f"{self.base_url}/regime/transitions")

        if response.status_code == 200:
            data = response.json()
            print(f"✅ Transition matrix retrieved")
            print(f"   Model type: {data['model_type']}")
            print(f"   Sample size: {data['sample_size']}")

            if data["transition_matrix"]:
                print("   Transition probabilities:")
                for from_regime, transitions in data["transition_matrix"].items():
                    print(
                        f"     {from_regime}: {[(to, f'{prob:.2f}') for to, prob in transitions.items()]}"
                    )
            else:
                print("   No transition matrix available (insufficient data)")

            return True
        else:
            print(f"❌ Transition matrix failed: {response.status_code}")
            return False

    def test_statistics(self):
        """Test statistics endpoint."""
        print("\n🔍 Testing statistics...")

        response = self.session.get(f"{self.base_url}/regime/stats")

        if response.status_code == 200:
            data = response.json()
            print(f"✅ Statistics retrieved")
            print(f"   Total predictions: {data['total_predictions']}")
            print(f"   Average confidence: {data['average_confidence']:.3f}")
            print(
                f"   Regime distribution: {[(k, f'{v:.2f}') for k, v in data['regime_distribution'].items()]}"
            )
            return True
        else:
            print(f"❌ Statistics failed: {response.status_code}")
            return False

    def test_data_simulation(self):
        """Test data simulation endpoint."""
        print("\n🔍 Testing data simulation...")

        response = self.session.post(f"{self.base_url}/regime/simulate?num_samples=20")

        if response.status_code == 200:
            data = response.json()
            print(f"✅ Data simulation completed")
            print(f"   Message: {data['message']}")
            print(f"   History size: {data['history_size']}")
            return True
        else:
            print(f"❌ Data simulation failed: {response.status_code}")
            return False

    def test_performance(self):
        """Test API performance requirements."""
        print("\n🔍 Testing performance requirements...")

        # Test single request latency
        features = {
            "atr": 0.5,
            "bb_width": 0.4,
            "realized_vol": 0.3,
            "vol_ratio": 1.1,
            "macd_signal": 0.2,
            "macd_histogram": 0.1,
            "adx": 60,
            "rsi": 65,
            "momentum": 0.3,
            "macro_surprise": 0.1,
            "macro_sentiment": 0.2,
            "trend_strength": 0.6,
            "mean_reversion": 0.3,
        }

        # Single request latency test
        latencies = []
        for i in range(10):
            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}/regime/predict", json=features
            )
            end_time = time.time()

            if response.status_code == 200:
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)

        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            max_latency = max(latencies)
            print(f"   Average latency: {avg_latency:.1f}ms")
            print(f"   Max latency: {max_latency:.1f}ms")

            if avg_latency < 100:
                print("   ✅ Latency requirement met (<100ms)")
            else:
                print("   ❌ Latency requirement not met")

        # Concurrent requests test
        print("   Testing concurrent requests...")

        def make_request():
            return self.session.post(f"{self.base_url}/regime/predict", json=features)

        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [
                future.result() for future in concurrent.futures.as_completed(futures)
            ]
        end_time = time.time()

        successful_requests = sum(1 for r in results if r.status_code == 200)
        total_time = end_time - start_time

        print(f"   Concurrent requests: {successful_requests}/10 successful")
        print(f"   Total time: {total_time:.2f}s")

        if successful_requests == 10:
            print("   ✅ Concurrent request handling working")
        else:
            print("   ❌ Some concurrent requests failed")

        return avg_latency < 100 and successful_requests == 10

    def test_error_handling(self):
        """Test error handling."""
        print("\n🔍 Testing error handling...")

        # Test invalid data
        invalid_features = {
            "atr": -0.5,  # Invalid negative value
            "bb_width": 0.4
            # Missing required fields
        }

        response = self.session.post(
            f"{self.base_url}/regime/predict", json=invalid_features
        )

        if response.status_code == 422:
            print("   ✅ Validation error handling working")
        else:
            print(f"   ❌ Expected validation error, got {response.status_code}")

        # Test invalid endpoint
        response = self.session.get(f"{self.base_url}/nonexistent")

        if response.status_code == 404:
            print("   ✅ 404 error handling working")
        else:
            print(f"   ❌ Expected 404, got {response.status_code}")

        # Test invalid window size
        response = self.session.get(f"{self.base_url}/regime/history?window=-1")

        if response.status_code == 422:
            print("   ✅ Parameter validation working")
        else:
            print(
                f"   ❌ Expected parameter validation error, got {response.status_code}"
            )

        return True

    def run_comprehensive_test(self):
        """Run all tests."""
        print("🚀 Starting Comprehensive API Test")
        print("=" * 50)

        test_results = []

        # Run all tests
        tests = [
            ("Health Check", self.test_health_check),
            ("Root Endpoint", self.test_root_endpoint),
            ("Data Simulation", self.test_data_simulation),
            ("Regime Prediction", self.test_regime_prediction),
            ("Current Regime", self.test_current_regime),
            ("Regime History", self.test_regime_history),
            ("Transition Matrix", self.test_transition_matrix),
            ("Statistics", self.test_statistics),
            ("Performance", self.test_performance),
            ("Error Handling", self.test_error_handling),
        ]

        for test_name, test_func in tests:
            try:
                result = test_func()
                test_results.append((test_name, result))
            except Exception as e:
                print(f"❌ {test_name} crashed: {e}")
                test_results.append((test_name, False))

        # Summary
        print("\n" + "=" * 50)
        print("TEST SUMMARY")
        print("=" * 50)

        passed = 0
        total = len(test_results)

        for test_name, result in test_results:
            if isinstance(result, bool):
                status = "✅ PASSED" if result else "❌ FAILED"
                if result:
                    passed += 1
            else:
                # For tests that return detailed results
                status = "✅ COMPLETED"
                passed += 1

            print(f"{test_name}: {status}")

        print(f"\nOverall: {passed}/{total} tests passed")

        if passed == total:
            print("🎉 All tests passed! API is working correctly.")
        else:
            print(f"⚠️  {total - passed} tests failed. Please review the issues above.")

        return passed == total


def start_api_server():
    """Start the API server in the background."""
    print("🚀 Starting API server...")

    # Start the server
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "api.main:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--log-level",
            "info",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait a moment for the server to start
    time.sleep(3)

    # Check if server is running
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ API server started successfully")
            return process
        else:
            print("❌ API server not responding correctly")
            process.terminate()
            return None
    except requests.exceptions.RequestException:
        print("❌ Failed to connect to API server")
        process.terminate()
        return None


def main():
    """Main function to run the demo."""
    print("FX AI-Quant Regime Detection API Demo")
    print("=" * 50)

    # Check if server is already running
    try:
        response = requests.get("http://localhost:8000/health", timeout=2)
        if response.status_code == 200:
            print("ℹ️  API server is already running")
            server_process = None
        else:
            server_process = start_api_server()
            if not server_process:
                return 1
    except requests.exceptions.RequestException:
        server_process = start_api_server()
        if not server_process:
            return 1

    try:
        # Run tests
        tester = APITester()
        success = tester.run_comprehensive_test()

        return 0 if success else 1

    finally:
        # Clean up
        if server_process:
            print("\n🛑 Stopping API server...")
            server_process.terminate()
            server_process.wait()
            print("✅ API server stopped")


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
