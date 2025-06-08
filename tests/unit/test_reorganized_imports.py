#!/usr/bin/env python3
"""
Test script to verify all imports work correctly after architecture reorganization.
"""

import sys
import traceback


def test_import(module_name, description):
    """Test importing a module and return success status."""
    try:
        exec(f"import {module_name}")
        print(f"✅ {description}: SUCCESS")
        return True
    except Exception as e:
        print(f"❌ {description}: FAILED - {e}")
        traceback.print_exc()
        return False


def test_specific_import(import_statement, description):
    """Test a specific import statement."""
    try:
        exec(import_statement)
        print(f"✅ {description}: SUCCESS")
        return True
    except Exception as e:
        print(f"❌ {description}: FAILED - {e}")
        traceback.print_exc()
        return False


def main():
    """Run all import tests."""
    print("🔧 Testing Architecture Reorganization - Import Verification")
    print("=" * 60)

    success_count = 0
    total_tests = 0

    # Test basic model imports
    tests = [
        ("models", "Models package"),
        ("models.predictor_interface", "Predictor interface"),
        ("models.lstm_model", "LSTM model"),
        ("models.cnn_model", "CNN model"),
        ("models.xgboost_model", "XGBoost model (new separate file)"),
        ("models.ensemble_model", "Ensemble model (updated)"),
    ]

    for module, desc in tests:
        total_tests += 1
        if test_import(module, desc):
            success_count += 1

    # Test specific class imports
    specific_imports = [
        (
            "from models import BasePredictorModel, ModelConfig, TrainingMetrics",
            "Base interfaces",
        ),
        ("from models import LSTMPredictor, create_lstm_config", "LSTM predictor"),
        ("from models import CNNPredictor, create_cnn_config", "CNN predictor"),
        (
            "from models import XGBoostPredictor, create_xgboost_config",
            "XGBoost predictor",
        ),
        (
            "from models import EnsembleMLPredictor, create_ensemble_config",
            "Ensemble predictor",
        ),
        ("from models import get_available_models, create_model", "Factory functions"),
    ]

    for import_stmt, desc in specific_imports:
        total_tests += 1
        if test_specific_import(import_stmt, desc):
            success_count += 1

    # Test model availability flags
    availability_tests = [
        (
            "from models import LSTM_AVAILABLE, CNN_AVAILABLE, XGBOOST_AVAILABLE, ENSEMBLE_AVAILABLE",
            "Availability flags",
        ),
    ]

    for import_stmt, desc in availability_tests:
        total_tests += 1
        if test_specific_import(import_stmt, desc):
            success_count += 1

    # Test model creation
    print("\n🏗️ Testing Model Creation")
    print("-" * 30)

    try:
        from models import (
            create_cnn_config,
            create_ensemble_config,
            create_lstm_config,
            create_model,
            create_xgboost_config,
        )

        # Test config creation
        lstm_config = create_lstm_config()
        cnn_config = create_cnn_config()
        xgboost_config = create_xgboost_config()
        ensemble_config = create_ensemble_config()

        print("✅ Config creation: SUCCESS")
        success_count += 1
        total_tests += 1

        # Test model factory
        available_models = ["lstm", "cnn", "xgboost", "ensemble"]
        for model_type in available_models:
            try:
                if model_type == "lstm":
                    create_model(model_type, lstm_config)
                elif model_type == "cnn":
                    create_model(model_type, cnn_config)
                elif model_type == "xgboost":
                    create_model(model_type, xgboost_config)
                elif model_type == "ensemble":
                    create_model(model_type, ensemble_config)

                print(f"✅ {model_type.upper()} model creation: SUCCESS")
                success_count += 1
            except Exception as e:
                print(f"❌ {model_type.upper()} model creation: FAILED - {e}")
            total_tests += 1

    except Exception as e:
        print(f"❌ Model creation test setup: FAILED - {e}")
        total_tests += 1

    # Test other core imports
    print("\n🔧 Testing Other Core Imports")
    print("-" * 30)

    other_tests = [
        ("core", "Core package"),
        ("data", "Data package"),
        ("api", "API package"),
    ]

    for module, desc in other_tests:
        total_tests += 1
        if test_import(module, desc):
            success_count += 1

    # Summary
    print("\n" + "=" * 60)
    print(f"📊 IMPORT TEST SUMMARY")
    print(f"✅ Successful: {success_count}/{total_tests}")
    print(f"❌ Failed: {total_tests - success_count}/{total_tests}")
    print(f"📈 Success Rate: {(success_count/total_tests)*100:.1f}%")

    if success_count == total_tests:
        print("\n🎉 ALL IMPORTS WORKING! Architecture reorganization successful!")
        return True
    else:
        print(
            f"\n⚠️  {total_tests - success_count} import(s) failed. Please check the errors above."
        )
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
