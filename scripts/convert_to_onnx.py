#!/usr/bin/env python3
"""
ONNX Conversion Script for FX AI-Quant Trading System.

This script converts trained ML models to ONNX format for fast,
framework-agnostic inference in production environments.
"""

import argparse
import sys
from pathlib import Path

import structlog

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.config.settings import SystemConfig
from models.cnn_model import CNNPredictor, create_cnn_config
from models.ensemble_model import XGBoostPredictor, create_xgboost_config
from models.lstm_model import LSTMPredictor, create_lstm_config

try:
    import onnxruntime as ort

    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_AVAILABLE = False

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


class ONNXConverter:
    """ONNX model converter and validator."""

    def __init__(self, config: SystemConfig):
        self.config = config
        self.logger = logger.bind(component="ONNXConverter")

        # Model and ONNX directories
        self.model_dir = Path(config.ml.model_path)
        self.onnx_dir = self.model_dir / "onnx"
        self.onnx_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(
            "ONNX Converter initialized",
            model_dir=str(self.model_dir),
            onnx_dir=str(self.onnx_dir),
            onnxruntime_available=ONNXRUNTIME_AVAILABLE,
        )

    def load_model(self, model_path: str, model_type: str) -> object | None:
        """Load a trained model."""
        try:
            model_path = Path(model_path)

            if model_type == "lstm":
                # Create dummy config for loading
                config = create_lstm_config()
                model = LSTMPredictor(config, self.logger)
                model.load_model(str(model_path))

            elif model_type == "cnn":
                config = create_cnn_config()
                model = CNNPredictor(config, self.logger)
                model.load_model(str(model_path))

            elif model_type == "xgboost":
                config = create_xgboost_config()
                model = XGBoostPredictor(config, self.logger)
                model.load_model(str(model_path))

            else:
                self.logger.error(f"Unsupported model type: {model_type}")
                return None

            self.logger.info(f"Loaded {model_type} model from {model_path}")
            return model

        except Exception as e:
            self.logger.error(f"Failed to load {model_type} model: {e}", exc_info=True)
            return None

    def convert_model_to_onnx(
        self, model, model_name: str, onnx_filename: str | None = None
    ) -> str | None:
        """Convert a model to ONNX format."""
        try:
            if onnx_filename is None:
                onnx_filename = f"{model_name}.onnx"

            onnx_path = self.onnx_dir / onnx_filename

            # Export model to ONNX
            model.export_to_onnx(str(onnx_path))

            self.logger.info(
                f"Successfully converted {model_name} to ONNX", onnx_path=str(onnx_path)
            )

            return str(onnx_path)

        except Exception as e:
            self.logger.error(
                f"Failed to convert {model_name} to ONNX: {e}", exc_info=True
            )
            return None

    def validate_onnx_model(
        self, onnx_path: str, original_model, test_input_shape: tuple = (1, 60, 13)
    ) -> bool:
        """Validate ONNX model by comparing outputs with original model."""
        if not ONNXRUNTIME_AVAILABLE:
            self.logger.warning("ONNX Runtime not available, skipping validation")
            return True

        try:
            # Load ONNX model
            ort_session = ort.InferenceSession(onnx_path)

            # Generate test input
            import numpy as np

            test_input = np.random.randn(*test_input_shape).astype(np.float32)

            # Get prediction from original model
            original_pred = original_model._predict_raw(test_input)

            # Get prediction from ONNX model
            ort_inputs = {ort_session.get_inputs()[0].name: test_input}
            onnx_pred = ort_session.run(None, ort_inputs)[0]

            # Compare predictions
            diff = np.abs(original_pred - onnx_pred.flatten())
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)

            # Tolerance for floating point differences
            tolerance = 1e-5

            if max_diff < tolerance:
                self.logger.info(
                    "ONNX model validation passed",
                    max_diff=float(max_diff),
                    mean_diff=float(mean_diff),
                    tolerance=tolerance,
                )
                return True
            else:
                self.logger.warning(
                    "ONNX model validation failed",
                    max_diff=float(max_diff),
                    mean_diff=float(mean_diff),
                    tolerance=tolerance,
                )
                return False

        except Exception as e:
            self.logger.error(f"ONNX validation failed: {e}", exc_info=True)
            return False

    def benchmark_onnx_model(
        self,
        onnx_path: str,
        input_shape: tuple = (1, 60, 13),
        num_iterations: int = 1000,
    ) -> dict[str, float]:
        """Benchmark ONNX model inference performance."""
        if not ONNXRUNTIME_AVAILABLE:
            self.logger.warning("ONNX Runtime not available, skipping benchmark")
            return {}

        try:
            import time

            import numpy as np

            # Load ONNX model
            ort_session = ort.InferenceSession(onnx_path)

            # Generate test input
            test_input = np.random.randn(*input_shape).astype(np.float32)
            ort_inputs = {ort_session.get_inputs()[0].name: test_input}

            # Warmup
            for _ in range(10):
                ort_session.run(None, ort_inputs)

            # Benchmark
            start_time = time.time()
            for _ in range(num_iterations):
                ort_session.run(None, ort_inputs)
            end_time = time.time()

            total_time = end_time - start_time
            avg_time_ms = (total_time / num_iterations) * 1000
            throughput = num_iterations / total_time

            metrics = {
                "total_time_seconds": total_time,
                "average_inference_time_ms": avg_time_ms,
                "throughput_inferences_per_second": throughput,
            }

            self.logger.info(
                "ONNX model benchmark completed",
                **metrics,
                num_iterations=num_iterations,
            )

            return metrics

        except Exception as e:
            self.logger.error(f"ONNX benchmark failed: {e}", exc_info=True)
            return {}

    def convert_all_models(
        self,
        model_types: list[str] | None = None,
        validate: bool = True,
        benchmark: bool = True,
    ) -> dict[str, dict[str, any]]:
        """Convert all available models to ONNX format."""

        if model_types is None:
            model_types = ["lstm", "cnn", "xgboost"]

        results = {}

        for model_type in model_types:
            self.logger.info(f"Converting {model_type} model to ONNX")

            # Find model file
            model_pattern = f"{model_type}_model"
            model_files = list(self.model_dir.glob(f"{model_pattern}*"))

            if not model_files:
                self.logger.warning(f"No {model_type} model found")
                continue

            # Use the first matching file (without extension)
            model_path = str(model_files[0]).split(".")[0]

            # Load model
            model = self.load_model(model_path, model_type)
            if model is None:
                continue

            # Convert to ONNX
            onnx_path = self.convert_model_to_onnx(model, model_type)
            if onnx_path is None:
                continue

            result = {"onnx_path": onnx_path, "conversion_success": True}

            # Validate ONNX model
            if validate:
                validation_success = self.validate_onnx_model(onnx_path, model)
                result["validation_success"] = validation_success

            # Benchmark ONNX model
            if benchmark:
                benchmark_metrics = self.benchmark_onnx_model(onnx_path)
                result["benchmark_metrics"] = benchmark_metrics

            results[model_type] = result

        return results

    def create_onnx_inference_example(self, onnx_path: str, output_file: str) -> None:
        """Create an example script for ONNX inference."""

        example_code = f'''#!/usr/bin/env python3
"""
ONNX Inference Example for FX AI-Quant Trading System.

This script demonstrates how to use the ONNX model for fast inference.
"""

import numpy as np
import onnxruntime as ort
from typing import List, Tuple

class ONNXPredictor:
    """ONNX-based predictor for fast inference."""

    def __init__(self, onnx_path: str):
        self.session = ort.InferenceSession(onnx_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Get input shape info
        input_shape = self.session.get_inputs()[0].shape
        print(f"Model input shape: {{input_shape}}")
        print(f"Model input name: {{self.input_name}}")
        print(f"Model output name: {{self.output_name}}")

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Make prediction using ONNX model."""
        # Ensure correct input shape
        if len(features.shape) == 2:
            features = features.reshape(1, features.shape[0], features.shape[1])

        # Run inference
        ort_inputs = {{self.input_name: features.astype(np.float32)}}
        prediction = self.session.run([self.output_name], ort_inputs)[0]

        return prediction.flatten()

    def predict_batch(self, features_batch: np.ndarray) -> np.ndarray:
        """Make batch predictions."""
        ort_inputs = {{self.input_name: features_batch.astype(np.float32)}}
        predictions = self.session.run([self.output_name], ort_inputs)[0]

        return predictions.flatten()


def main():
    """Example usage."""
    # Load ONNX model
    predictor = ONNXPredictor("{onnx_path}")

    # Generate sample input (sequence_length=60, features=13)
    sample_input = np.random.randn(60, 13)

    # Make prediction
    prediction = predictor.predict(sample_input)
    print(f"Prediction: {{prediction[0]:.6f}}")

    # Batch prediction example
    batch_input = np.random.randn(5, 60, 13)  # 5 samples
    batch_predictions = predictor.predict_batch(batch_input)
    print(f"Batch predictions: {{batch_predictions}}")


if __name__ == "__main__":
    main()
'''

        with open(output_file, "w") as f:
            f.write(example_code)

        self.logger.info(f"Created ONNX inference example: {output_file}")


def main():
    """Main conversion function."""
    parser = argparse.ArgumentParser(description="Convert ML models to ONNX format")

    parser.add_argument(
        "--model-types",
        nargs="+",
        choices=["lstm", "cnn", "xgboost", "all"],
        default=["all"],
        help="Types of models to convert",
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        default=True,
        help="Validate ONNX models against original models",
    )

    parser.add_argument(
        "--benchmark",
        action="store_true",
        default=True,
        help="Benchmark ONNX model performance",
    )

    parser.add_argument(
        "--create-example",
        action="store_true",
        help="Create ONNX inference example script",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for ONNX models (default: models/onnx/)",
    )

    args = parser.parse_args()

    # Initialize configuration
    config = SystemConfig()

    # Override output directory if specified
    if args.output_dir:
        config.ml.model_path = Path(args.output_dir).parent

    # Initialize converter
    converter = ONNXConverter(config)

    # Determine model types to convert
    if "all" in args.model_types:
        model_types = ["lstm", "cnn", "xgboost"]
    else:
        model_types = args.model_types

    # Convert models
    logger.info("Starting ONNX conversion process")
    results = converter.convert_all_models(
        model_types=model_types, validate=args.validate, benchmark=args.benchmark
    )

    # Print summary
    logger.info("ONNX Conversion Summary:")
    for model_type, result in results.items():
        status = "✅" if result.get("conversion_success", False) else "❌"
        validation = "✅" if result.get("validation_success", True) else "❌"

        logger.info(
            f"{model_type.upper()}: Conversion {status}, Validation {validation}"
        )

        if "benchmark_metrics" in result:
            metrics = result["benchmark_metrics"]
            avg_time = metrics.get("average_inference_time_ms", 0)
            throughput = metrics.get("throughput_inferences_per_second", 0)
            logger.info(
                f"  Performance: {avg_time:.2f}ms avg, {throughput:.0f} inferences/sec"
            )

    # Create inference example if requested
    if args.create_example and results:
        # Use the first successfully converted model for the example
        for model_type, result in results.items():
            if result.get("conversion_success", False):
                onnx_path = result["onnx_path"]
                example_file = converter.onnx_dir / "inference_example.py"
                converter.create_onnx_inference_example(onnx_path, str(example_file))
                break

    logger.info("ONNX conversion process completed!")


if __name__ == "__main__":
    main()
