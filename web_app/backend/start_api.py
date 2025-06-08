#!/usr/bin/env python3
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    print("Starting Parameter Tuner API...")
    print(f"Python path: {sys.path}")
    print(f"Current working directory: {os.getcwd()}")

    # Test imports
    print("Testing imports...")
    import uvicorn

    print("✓ uvicorn imported")

    from api_main import app

    print("✓ app imported")

    # Check if parameter tuner can be imported
    try:
        from tools.parameter_tuner import ParameterTuner

        print("✓ ParameterTuner imported")

        # Test creating parameter tuner
        tuner = ParameterTuner()
        print("✓ ParameterTuner created successfully")
    except Exception as e:
        print(f"⚠️ ParameterTuner issue: {e}")

    print("\nStarting server on http://127.0.0.1:8080")
    uvicorn.run(app, host="127.0.0.1", port=8080, log_level="debug")

except Exception as e:
    print(f"❌ Error starting API: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
