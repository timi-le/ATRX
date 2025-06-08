#!/usr/bin/env python3
"""
Comprehensive test suite for Parameter Tuning Interface

Tests all aspects of the parameter tuning system including:
- Parameter validation and setting
- Configuration management and versioning
- CLI interface functionality
- Optimization suggestions
- Error handling and edge cases
"""

import shutil
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import yaml

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from tools.parameter_tuner import ParameterTuner, ParameterValidationError


class TestParameterTuner(unittest.TestCase):
    """Test suite for ParameterTuner class."""

    def setUp(self):
        """Set up test environment."""
        # Create temporary directory for testing
        self.test_dir = tempfile.mkdtemp()
        self.config_path = Path(self.test_dir) / "test_config.yaml"
        self.versions_dir = Path(self.test_dir) / "versions"
        self.log_file = Path(self.test_dir) / "test.log"

        # Create test configuration
        self.test_config = {
            "strategy_parameters": {
                "grid_step_factor": 0.5,
                "max_levels": 5,
                "risk_per_trade": 0.02,
                "volatility_filter_threshold": 0.7,
                "session_filter_enabled": True,
                "tp_type": "dynamic",
            },
            "strategy_weights": {"momentum_weight": 0.3, "mean_reversion_weight": 0.3},
            "metadata": {"created": datetime.now().isoformat(), "version": "1.0.0"},
        }

        # Save test configuration
        with open(self.config_path, "w") as f:
            yaml.dump(self.test_config, f)

        # Initialize parameter tuner
        self.tuner = ParameterTuner(
            config_path=str(self.config_path),
            versions_dir=str(self.versions_dir),
            log_file=str(self.log_file),
        )

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)

    def test_initialization(self):
        """Test parameter tuner initialization."""
        self.assertTrue(self.config_path.exists())
        self.assertTrue(self.versions_dir.exists())
        self.assertTrue(self.log_file.parent.exists())
        self.assertIsInstance(self.tuner.config, dict)
        self.assertEqual(len(self.tuner.changes_made), 0)

    def test_get_parameter(self):
        """Test parameter retrieval."""
        # Test simple parameter
        value = self.tuner.get_parameter("strategy_parameters.grid_step_factor")
        self.assertEqual(value, 0.5)

        # Test nested parameter
        value = self.tuner.get_parameter("metadata.version")
        self.assertEqual(value, "1.0.0")

        # Test non-existent parameter
        with self.assertRaises(ParameterValidationError):
            self.tuner.get_parameter("non_existent.parameter")

    def test_set_parameter_valid(self):
        """Test setting valid parameters."""
        # Test float parameter
        result = self.tuner.set_parameter("strategy_parameters.grid_step_factor", 0.8)
        self.assertTrue(result)
        self.assertEqual(
            self.tuner.get_parameter("strategy_parameters.grid_step_factor"), 0.8
        )

        # Test integer parameter
        result = self.tuner.set_parameter("strategy_parameters.max_levels", 7)
        self.assertTrue(result)
        self.assertEqual(self.tuner.get_parameter("strategy_parameters.max_levels"), 7)

        # Test boolean parameter
        result = self.tuner.set_parameter(
            "strategy_parameters.session_filter_enabled", False
        )
        self.assertTrue(result)
        self.assertEqual(
            self.tuner.get_parameter("strategy_parameters.session_filter_enabled"),
            False,
        )

        # Test string parameter
        result = self.tuner.set_parameter("strategy_parameters.tp_type", "fixed")
        self.assertTrue(result)
        self.assertEqual(
            self.tuner.get_parameter("strategy_parameters.tp_type"), "fixed"
        )

        # Check changes were tracked
        self.assertGreater(len(self.tuner.changes_made), 0)

    def test_set_parameter_invalid(self):
        """Test setting invalid parameters."""
        # Test out of range value
        with self.assertRaises(ParameterValidationError):
            self.tuner.set_parameter(
                "strategy_parameters.grid_step_factor", 5.0
            )  # Max is 2.0

        with self.assertRaises(ParameterValidationError):
            self.tuner.set_parameter("strategy_parameters.max_levels", 0)  # Min is 1

        # Test invalid type
        with self.assertRaises(ParameterValidationError):
            self.tuner.set_parameter("strategy_parameters.max_levels", "invalid")

        # Test invalid allowed value
        with self.assertRaises(ParameterValidationError):
            self.tuner.set_parameter("strategy_parameters.tp_type", "invalid_type")

    def test_parameter_validation(self):
        """Test parameter validation rules."""
        # Test numeric range validation
        self.tuner._validate_parameter("grid_step_factor", 1.5)  # Valid

        with self.assertRaises(ParameterValidationError):
            self.tuner._validate_parameter("grid_step_factor", 0.05)  # Below min

        with self.assertRaises(ParameterValidationError):
            self.tuner._validate_parameter("grid_step_factor", 3.0)  # Above max

        # Test allowed values validation
        self.tuner._validate_parameter("tp_type", "dynamic")  # Valid

        with self.assertRaises(ParameterValidationError):
            self.tuner._validate_parameter("tp_type", "invalid")  # Not allowed

        # Test type conversion
        self.tuner._validate_parameter("max_levels", "5")  # String to int
        self.tuner._validate_parameter("grid_step_factor", "1.5")  # String to float
        self.tuner._validate_parameter(
            "session_filter_enabled", "true"
        )  # String to bool

    def test_config_save_and_load(self):
        """Test configuration saving and loading."""
        # Make changes
        self.tuner.set_parameter("strategy_parameters.grid_step_factor", 0.8)
        self.tuner.set_parameter("strategy_parameters.max_levels", 7)

        # Save configuration
        saved_path = self.tuner.save_config(versioned=True)
        self.assertEqual(saved_path, str(self.config_path))

        # Check versioned backup was created
        version_files = list(self.versions_dir.glob("config_*.yaml"))
        self.assertGreater(len(version_files), 0)

        # Load configuration in new tuner instance
        new_tuner = ParameterTuner(
            config_path=str(self.config_path),
            versions_dir=str(self.versions_dir),
            log_file=str(self.log_file),
        )

        # Verify changes were persisted
        self.assertEqual(
            new_tuner.get_parameter("strategy_parameters.grid_step_factor"), 0.8
        )
        self.assertEqual(new_tuner.get_parameter("strategy_parameters.max_levels"), 7)

    def test_version_management(self):
        """Test configuration version management."""
        # Create initial version
        self.tuner.save_config(versioned=True)

        # Make changes and save again
        self.tuner.set_parameter("strategy_parameters.grid_step_factor", 0.9)
        self.tuner.save_config(versioned=True)

        # List versions
        versions = self.tuner.list_versions()
        self.assertGreaterEqual(len(versions), 1)

        for version in versions:
            self.assertIn("timestamp", version)
            self.assertIn("size", version)
            self.assertIn("file", version)

        # Test restore functionality
        if versions:
            # Change parameter
            self.tuner.set_parameter("strategy_parameters.grid_step_factor", 1.5)
            self.assertEqual(
                self.tuner.get_parameter("strategy_parameters.grid_step_factor"), 1.5
            )

            # Restore previous version
            timestamp = versions[0]["timestamp"]
            self.tuner.restore_version(timestamp)

            # Verify restoration
            restored_value = self.tuner.get_parameter(
                "strategy_parameters.grid_step_factor"
            )
            self.assertNotEqual(restored_value, 1.5)  # Should be restored value

    def test_optimization_suggestions(self):
        """Test optimization suggestion functionality."""
        # Test suggestion for existing parameter
        suggestion = self.tuner.suggest_optimization("grid_step_factor")

        self.assertEqual(suggestion["parameter"], "grid_step_factor")
        self.assertIsNotNone(suggestion["current_value"])
        self.assertIsNotNone(suggestion["suggested_range"])
        self.assertIn(suggestion["confidence"], ["low", "medium", "high"])

        # Test suggestion for parameter at minimum
        self.tuner.set_parameter("strategy_parameters.grid_step_factor", 0.1)
        suggestion = self.tuner.suggest_optimization("grid_step_factor")
        self.assertIn("suggested_value", suggestion)
        self.assertIn("near minimum", suggestion["reasoning"])

        # Test suggestion for parameter at maximum
        self.tuner.set_parameter("strategy_parameters.grid_step_factor", 2.0)
        suggestion = self.tuner.suggest_optimization("grid_step_factor")
        self.assertIn("suggested_value", suggestion)
        self.assertIn("near maximum", suggestion["reasoning"])

    def test_parameter_info(self):
        """Test parameter information retrieval."""
        info = self.tuner.get_parameter_info("grid_step_factor")

        self.assertEqual(info["name"], "grid_step_factor")
        self.assertIn("description", info)
        self.assertIn("type", info)
        self.assertIsNotNone(info["current_value"])
        self.assertIsNotNone(info["validation_rule"])

        # Test parameter that doesn't exist in validation rules
        info = self.tuner.get_parameter_info("unknown_parameter")
        self.assertEqual(info["type"], "unknown")
        self.assertIsNone(info["validation_rule"])

    def test_validate_all_parameters(self):
        """Test validation of all parameters in configuration."""
        results = self.tuner.validate_all_parameters()

        self.assertIn("valid", results)
        self.assertIn("invalid", results)
        self.assertIn("warnings", results)
        self.assertIn("total_checked", results)

        self.assertGreater(results["total_checked"], 0)
        self.assertIsInstance(results["valid"], list)
        self.assertIsInstance(results["invalid"], list)
        self.assertIsInstance(results["warnings"], list)

        # Add invalid parameter and test
        self.tuner.set_parameter(
            "strategy_parameters.grid_step_factor", 10.0, validate=False
        )
        results = self.tuner.validate_all_parameters()

        self.assertGreater(len(results["invalid"]), 0)

        # Check invalid parameter details
        invalid_param = results["invalid"][0]
        self.assertIn("parameter", invalid_param)
        self.assertIn("value", invalid_param)
        self.assertIn("error", invalid_param)

    def test_create_default_config(self):
        """Test default configuration creation."""
        # Create tuner with non-existent config path
        empty_config_path = Path(self.test_dir) / "empty_config.yaml"

        tuner = ParameterTuner(
            config_path=str(empty_config_path),
            versions_dir=str(self.versions_dir),
            log_file=str(self.log_file),
        )

        # Check that default config was created
        self.assertTrue(empty_config_path.exists())

        # Verify default config contains expected sections
        self.assertIn("strategy_parameters", tuner.config)
        self.assertIn("strategy_weights", tuner.config)
        self.assertIn("technical_indicators", tuner.config)
        self.assertIn("metadata", tuner.config)

    def test_change_tracking(self):
        """Test change tracking functionality."""
        initial_changes = len(self.tuner.changes_made)

        # Make several changes
        self.tuner.set_parameter("strategy_parameters.grid_step_factor", 0.8)
        self.tuner.set_parameter("strategy_parameters.max_levels", 7)
        self.tuner.set_parameter("strategy_weights.momentum_weight", 0.4)

        # Check changes were tracked
        self.assertEqual(len(self.tuner.changes_made), initial_changes + 3)

        # Check change details
        for change in self.tuner.changes_made:
            self.assertIn("timestamp", change)
            self.assertIn("parameter", change)
            self.assertIn("old_value", change)
            self.assertIn("new_value", change)
            self.assertIn("validated", change)

        # Save and check changes are cleared
        self.tuner.save_config()
        self.assertEqual(len(self.tuner.changes_made), 0)

    def test_dot_notation_parameters(self):
        """Test dot notation parameter access."""
        # Test setting new nested parameter
        self.tuner.set_parameter("new_section.new_param", 42)
        self.assertEqual(self.tuner.get_parameter("new_section.new_param"), 42)

        # Test deeper nesting
        self.tuner.set_parameter("deep.nested.parameter", "value")
        self.assertEqual(self.tuner.get_parameter("deep.nested.parameter"), "value")

        # Verify structure was created correctly
        self.assertIn("new_section", self.tuner.config)
        self.assertIn("new_param", self.tuner.config["new_section"])
        self.assertIn("deep", self.tuner.config)
        self.assertIn("nested", self.tuner.config["deep"])
        self.assertIn("parameter", self.tuner.config["deep"]["nested"])


class TestParameterTunerCLI(unittest.TestCase):
    """Test suite for CLI interface."""

    def setUp(self):
        """Set up test environment for CLI testing."""
        self.test_dir = tempfile.mkdtemp()
        self.config_path = Path(self.test_dir) / "cli_test_config.yaml"

        # Create test configuration
        test_config = {
            "strategy_parameters": {"grid_step_factor": 0.5, "max_levels": 5}
        }

        with open(self.config_path, "w") as f:
            yaml.dump(test_config, f)

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)

    @patch("sys.argv")
    def test_cli_set_parameter(self, mock_argv):
        """Test CLI parameter setting."""
        mock_argv.__getitem__.side_effect = [
            "parameter_tuner.py",
            "--config",
            str(self.config_path),
            "--set",
            "strategy_parameters.grid_step_factor=0.8",
            "--set",
            "strategy_parameters.max_levels=7",
        ]

        # Import main function
        from tools.parameter_tuner import main

        # Capture output
        with patch("builtins.print") as mock_print:
            result = main()

        self.assertEqual(result, 0)

        # Verify parameters were set
        tuner = ParameterTuner(config_path=str(self.config_path))
        self.assertEqual(
            tuner.get_parameter("strategy_parameters.grid_step_factor"), 0.8
        )
        self.assertEqual(tuner.get_parameter("strategy_parameters.max_levels"), 7)

    @patch("sys.argv")
    def test_cli_get_parameter(self, mock_argv):
        """Test CLI parameter getting."""
        mock_argv.__getitem__.side_effect = [
            "parameter_tuner.py",
            "--config",
            str(self.config_path),
            "--get",
            "strategy_parameters.grid_step_factor",
        ]

        from tools.parameter_tuner import main

        with patch("builtins.print") as mock_print:
            result = main()

        self.assertEqual(result, 0)
        # Check that print was called with parameter value
        mock_print.assert_called()

    @patch("sys.argv")
    def test_cli_validation_error(self, mock_argv):
        """Test CLI validation error handling."""
        mock_argv.__getitem__.side_effect = [
            "parameter_tuner.py",
            "--config",
            str(self.config_path),
            "--set",
            "strategy_parameters.grid_step_factor=10.0",  # Invalid value
        ]

        from tools.parameter_tuner import main

        with patch("builtins.print") as mock_print:
            result = main()

        self.assertEqual(result, 0)  # Should not exit with error
        # Should print validation error
        mock_print.assert_called()


class TestParameterValidation(unittest.TestCase):
    """Test parameter validation edge cases."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.tuner = ParameterTuner(
            config_path=str(Path(self.test_dir) / "validation_test.yaml")
        )

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)

    def test_type_conversion(self):
        """Test automatic type conversion during validation."""
        # String to int
        self.tuner._validate_parameter("max_levels", "5")

        # String to float
        self.tuner._validate_parameter("grid_step_factor", "1.5")

        # String to bool
        self.tuner._validate_parameter("session_filter_enabled", "true")
        self.tuner._validate_parameter("session_filter_enabled", "false")
        self.tuner._validate_parameter("session_filter_enabled", "1")
        self.tuner._validate_parameter("session_filter_enabled", "0")

        # Invalid conversions should raise errors
        with self.assertRaises(ParameterValidationError):
            self.tuner._validate_parameter("max_levels", "not_a_number")

        with self.assertRaises(ParameterValidationError):
            self.tuner._validate_parameter("grid_step_factor", "not_a_float")

    def test_boundary_values(self):
        """Test boundary value validation."""
        # Test exact min/max values
        self.tuner._validate_parameter("grid_step_factor", 0.1)  # Min value
        self.tuner._validate_parameter("grid_step_factor", 2.0)  # Max value

        # Test just outside boundaries
        with self.assertRaises(ParameterValidationError):
            self.tuner._validate_parameter("grid_step_factor", 0.09)  # Below min

        with self.assertRaises(ParameterValidationError):
            self.tuner._validate_parameter("grid_step_factor", 2.1)  # Above max

    def test_unknown_parameters(self):
        """Test handling of unknown parameters."""
        # Should not raise error, but log warning
        try:
            self.tuner._validate_parameter("unknown_param", "value")
        except Exception as e:
            self.fail(f"Unknown parameter validation should not raise error: {e}")


if __name__ == "__main__":
    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test cases
    test_suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestParameterTuner))
    test_suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestParameterTunerCLI))
    test_suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestParameterValidation))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Parameter Tuner Test Results:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(
        f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%"
    )
    print(f"{'='*60}")

    # Exit with error code if tests failed
    sys.exit(len(result.failures) + len(result.errors))
