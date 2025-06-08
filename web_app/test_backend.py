#!/usr/bin/env python3
"""
Test script for the Parameter Tuner Web Application Backend

This script demonstrates the backend API functionality and can be used
to verify that the API is working correctly before connecting the frontend.
"""


import requests

API_BASE_URL = "http://localhost:8000"


def test_health_check():
    """Test the health check endpoint."""
    print("🔍 Testing health check...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data['message']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to backend. Is the server running?")
        return False


def test_get_configuration():
    """Test getting the current configuration."""
    print("\n📋 Testing configuration retrieval...")
    try:
        response = requests.get(f"{API_BASE_URL}/config")
        if response.status_code == 200:
            data = response.json()
            print("✅ Configuration loaded successfully")
            print(f"   Parameters in config: {len(data.get('config', {}))}")
            return data
        else:
            print(f"❌ Failed to get configuration: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error getting configuration: {e}")
        return None


def test_get_parameters():
    """Test getting all parameters with metadata."""
    print("\n🎛️ Testing parameter metadata retrieval...")
    try:
        response = requests.get(f"{API_BASE_URL}/config/parameters")
        if response.status_code == 200:
            data = response.json()
            if data["success"]:
                params = data["data"]
                print(f"✅ Parameters loaded successfully: {len(params)} parameters")

                # Show some example parameters
                print("   Example parameters:")
                for i, (name, info) in enumerate(list(params.items())[:3]):
                    print(
                        f"   - {name}: {info.get('current_value')} ({info.get('type')})"
                    )

                return params
            else:
                print(f"❌ Failed to get parameters: {data['message']}")
                return None
        else:
            print(f"❌ Failed to get parameters: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error getting parameters: {e}")
        return None


def test_update_parameter():
    """Test updating a single parameter."""
    print("\n🔧 Testing parameter update...")
    try:
        # Test updating grid_step_factor
        update_data = {"key": "grid_step_factor", "value": 0.75, "validate": True}

        response = requests.post(f"{API_BASE_URL}/config/parameter", json=update_data)
        if response.status_code == 200:
            data = response.json()
            if data["success"]:
                print(f"✅ Parameter updated successfully: {data['message']}")
                return True
            else:
                print(f"❌ Parameter update failed: {data['message']}")
                return False
        else:
            print(f"❌ Parameter update failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error updating parameter: {e}")
        return False


def test_bulk_update():
    """Test bulk parameter update."""
    print("\n📦 Testing bulk parameter update...")
    try:
        # Test updating multiple parameters
        update_data = {
            "parameters": {
                "risk_per_trade": 0.025,
                "max_levels": 6,
                "session_filter_enabled": True,
            },
            "validate": True,
        }

        response = requests.post(
            f"{API_BASE_URL}/config/parameters/bulk", json=update_data
        )
        if response.status_code == 200:
            data = response.json()
            if data["success"]:
                summary = data["data"]["summary"]
                print(
                    f"✅ Bulk update completed: {summary['successful']}/{summary['total_updates']} successful"
                )
                return True
            else:
                print(f"❌ Bulk update failed: {data['message']}")
                return False
        else:
            print(f"❌ Bulk update failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error in bulk update: {e}")
        return False


def test_save_configuration():
    """Test saving the configuration."""
    print("\n💾 Testing configuration save...")
    try:
        response = requests.post(f"{API_BASE_URL}/config/save?versioned=true")
        if response.status_code == 200:
            data = response.json()
            if data["success"]:
                print(f"✅ Configuration saved successfully: {data['message']}")
                return True
            else:
                print(f"❌ Configuration save failed: {data['message']}")
                return False
        else:
            print(f"❌ Configuration save failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error saving configuration: {e}")
        return False


def test_get_versions():
    """Test getting configuration versions."""
    print("\n📚 Testing version history retrieval...")
    try:
        response = requests.get(f"{API_BASE_URL}/config/versions")
        if response.status_code == 200:
            data = response.json()
            if data["success"]:
                versions = data["data"]["versions"]
                print(f"✅ Version history loaded: {len(versions)} versions available")
                if versions:
                    latest = versions[0]
                    print(f"   Latest version: {latest.get('timestamp', 'Unknown')}")
                return versions
            else:
                print(f"❌ Failed to get versions: {data['message']}")
                return None
        else:
            print(f"❌ Failed to get versions: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error getting versions: {e}")
        return None


def test_validation():
    """Test configuration validation."""
    print("\n✅ Testing configuration validation...")
    try:
        response = requests.get(f"{API_BASE_URL}/config/validate")
        if response.status_code == 200:
            data = response.json()
            if data["success"]:
                validation_data = data["data"]
                if validation_data.get("valid", False):
                    print("✅ Configuration is valid")
                else:
                    print("⚠️ Configuration has validation issues")
                    if "errors" in validation_data:
                        for error in validation_data["errors"]:
                            print(f"   - {error}")
                return validation_data
            else:
                print(f"❌ Validation failed: {data['message']}")
                return None
        else:
            print(f"❌ Validation failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error in validation: {e}")
        return None


def run_complete_test():
    """Run all tests in sequence."""
    print("🚀 Starting Backend API Test Suite")
    print("=" * 50)

    tests_passed = 0
    total_tests = 7

    # Run all tests
    if test_health_check():
        tests_passed += 1

    if test_get_configuration():
        tests_passed += 1

    if test_get_parameters():
        tests_passed += 1

    if test_update_parameter():
        tests_passed += 1

    if test_bulk_update():
        tests_passed += 1

    if test_save_configuration():
        tests_passed += 1

    if test_validation():
        tests_passed += 1

    print("\n" + "=" * 50)
    print(f"🎯 Test Results: {tests_passed}/{total_tests} tests passed")

    if tests_passed == total_tests:
        print("🎉 All tests passed! The backend is working correctly.")
        print("✨ You can now start the frontend and begin parameter tuning.")
    else:
        print("⚠️ Some tests failed. Please check the backend configuration.")

    return tests_passed == total_tests


if __name__ == "__main__":
    print("FX AI-Quant Parameter Tuner - Backend Test")
    print("=========================================")
    print()
    print("This script tests the backend API functionality.")
    print("Make sure the backend server is running on http://localhost:8000")
    print()

    input("Press Enter to start the tests...")
    print()

    success = run_complete_test()

    if success:
        print("\n🚀 Ready to launch the web application!")
        print("Next steps:")
        print("1. Keep the backend running")
        print("2. Open a new terminal")
        print("3. Navigate to: cd web_app/frontend")
        print("4. Run: npm install && npm start")
        print("5. Open: http://localhost:3000")
    else:
        print("\n🔧 Backend needs attention before proceeding.")
