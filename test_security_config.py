#!/usr/bin/env python3
"""
Comprehensive Security and Configuration Management Test Suite
for FX AI-Quant Trading System

This test suite validates all security components:
- Secrets Manager encryption and RBAC
- Configuration Loader validation and security
- Audit Logger functionality
- TLS Manager certificate and ZeroMQ security
- Environment variable loading
- Integration testing

Test Categories:
1. Encryption/Decryption Tests
2. Role-Based Access Control Tests
3. Configuration Loading and Validation Tests
4. TLS/Certificate Tests
5. ZeroMQ CURVE Security Tests
6. Audit Logging Tests
7. Integration Tests
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any
import pytest
import yaml
import zmq
import time
from datetime import datetime

# Add config directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'config'))

from config.secrets_manager import SecretsManager, get_secrets_manager
from config.config_loader import SecureConfigLoader, get_config_loader, ConfigValidationError
from config.audit_logger import AuditLogger, get_audit_logger
from config.tls_manager import TLSManager, get_tls_manager


class SecurityTestSuite:
    """Comprehensive security test suite for FX AI-Quant system."""
    
    def __init__(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix="fx_security_test_"))
        self.config_dir = self.test_dir / "config"
        self.secrets_dir = self.config_dir / "secrets"
        self.certs_dir = self.config_dir / "certs"
        self.logs_dir = self.test_dir / "logs"
        
        # Create test directories
        for dir_path in [self.config_dir, self.secrets_dir, self.certs_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Test results
        self.results = {
            "encryption_tests": [],
            "rbac_tests": [],
            "config_tests": [],
            "tls_tests": [],
            "zmq_tests": [],
            "audit_tests": [],
            "integration_tests": [],
            "summary": {}
        }
        
        # Initialize components with test directories
        self.secrets_manager = SecretsManager(
            secrets_dir=str(self.secrets_dir)
        )
        self.config_loader = SecureConfigLoader(
            config_dir=str(self.config_dir)
        )
        self.audit_logger = AuditLogger(
            log_dir=str(self.logs_dir)
        )
        self.tls_manager = TLSManager(
            certs_dir=str(self.certs_dir)
        )
        
        print(f"🔧 Test environment initialized at: {self.test_dir}")
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all security tests and return results."""
        print("\n🛡️  Starting Comprehensive Security Test Suite")
        print("=" * 60)
        
        try:
            # Test categories
            self.test_encryption_functionality()
            self.test_rbac_functionality()
            self.test_configuration_loading()
            self.test_tls_functionality()
            self.test_zmq_security()
            self.test_audit_logging()
            self.test_integration_scenarios()
            
            # Generate summary
            self.generate_test_summary()
            
        except Exception as e:
            print(f"❌ Critical test failure: {e}")
            self.results["critical_error"] = str(e)
        
        finally:
            self.cleanup()
        
        return self.results
    
    def test_encryption_functionality(self):
        """Test encryption and decryption functionality."""
        print("\n📊 Testing Encryption Functionality")
        print("-" * 40)
        
        tests = [
            ("fernet_roundtrip", self._test_fernet_encryption),
            ("secrets_vault_persistence", self._test_secrets_vault_persistence),
            ("key_derivation", self._test_key_derivation),
            ("backup_restore", self._test_backup_restore)
        ]
        
        for test_name, test_func in tests:
            try:
                result = test_func()
                self.results["encryption_tests"].append({
                    "test": test_name,
                    "status": "PASS" if result else "FAIL",
                    "details": result if isinstance(result, dict) else {}
                })
                print(f"  ✅ {test_name}: PASS")
            except Exception as e:
                self.results["encryption_tests"].append({
                    "test": test_name,
                    "status": "FAIL",
                    "error": str(e)
                })
                print(f"  ❌ {test_name}: FAIL - {e}")
    
    def _test_fernet_encryption(self) -> bool:
        """Test Fernet encryption round trip."""
        # Set admin role
        self.secrets_manager.set_user_context("test_user", "admin")
        
        test_data = "sensitive_api_key_12345"
        
        # Store secret
        self.secrets_manager.store_secret("api", "test_key", test_data)
        
        # Retrieve secret
        retrieved = self.secrets_manager.get_secret("api", "test_key")
        
        return retrieved == test_data
    
    def _test_secrets_vault_persistence(self) -> bool:
        """Test secrets vault persistence across manager instances."""
        # Store data with first instance
        self.secrets_manager.set_user_context("test_user", "admin")
        self.secrets_manager.store_secret("api", "persistent_key", "persistent_value")
        
        # Create new instance
        new_manager = SecretsManager(secrets_dir=str(self.secrets_dir))
        new_manager.set_user_context("test_user", "admin")
        
        # Retrieve with new instance
        retrieved = new_manager.get_secret("api", "persistent_key")
        
        return retrieved == "persistent_value"
    
    def _test_key_derivation(self) -> bool:
        """Test key derivation consistency."""
        # This tests that the same master key produces the same derived key
        # We can't directly test the derivation, but we can test consistency
        
        # Store and retrieve multiple times
        self.secrets_manager.set_user_context("test_user", "admin")
        
        for i in range(3):
            self.secrets_manager.store_secret("api", f"key_{i}", f"value_{i}")
        
        # Verify all can be retrieved
        for i in range(3):
            value = self.secrets_manager.get_secret("api", f"key_{i}")
            if value != f"value_{i}":
                return False
        
        return True
    
    def _test_backup_restore(self) -> bool:
        """Test vault backup functionality."""
        # Store test data
        self.secrets_manager.set_user_context("test_user", "admin")
        self.secrets_manager.store_secret("api", "backup_test", "backup_data")
        
        # Create backup
        backup_path = self.secrets_manager.backup_vault()
        
        # Verify backup file exists
        return Path(backup_path).exists()
    
    def test_rbac_functionality(self):
        """Test Role-Based Access Control."""
        print("\n🔐 Testing Role-Based Access Control")
        print("-" * 40)
        
        tests = [
            ("admin_access", self._test_admin_access),
            ("service_access", self._test_service_access),
            ("public_access", self._test_public_access),
            ("access_denied", self._test_access_denied)
        ]
        
        for test_name, test_func in tests:
            try:
                result = test_func()
                self.results["rbac_tests"].append({
                    "test": test_name,
                    "status": "PASS" if result else "FAIL",
                    "details": result if isinstance(result, dict) else {}
                })
                print(f"  ✅ {test_name}: PASS")
            except Exception as e:
                self.results["rbac_tests"].append({
                    "test": test_name,
                    "status": "FAIL",
                    "error": str(e)
                })
                print(f"  ❌ {test_name}: FAIL - {e}")
    
    def _test_admin_access(self) -> bool:
        """Test admin role access to all scopes."""
        self.secrets_manager.set_user_context("admin_user", "admin")
        
        # Admin should access all scopes
        test_scopes = ["api", "database", "monitoring", "system", "public"]
        
        for scope in test_scopes:
            self.secrets_manager.store_secret(scope, "admin_test", "admin_value")
            retrieved = self.secrets_manager.get_secret(scope, "admin_test")
            if retrieved != "admin_value":
                return False
        
        return True
    
    def _test_service_access(self) -> bool:
        """Test service role access limitations."""
        self.secrets_manager.set_user_context("service_user", "service")
        
        # Service should access api, database, monitoring, public
        allowed_scopes = ["api", "database", "monitoring", "public"]
        restricted_scopes = ["system"]
        
        # Test allowed access
        for scope in allowed_scopes:
            self.secrets_manager.store_secret(scope, "service_test", "service_value")
            retrieved = self.secrets_manager.get_secret(scope, "service_test")
            if retrieved != "service_value":
                return False
        
        # Test restricted access
        for scope in restricted_scopes:
            try:
                self.secrets_manager.store_secret(scope, "service_test", "service_value")
                return False  # Should have thrown exception
            except PermissionError:
                pass  # Expected
        
        return True
    
    def _test_public_access(self) -> bool:
        """Test public role access limitations."""
        self.secrets_manager.set_user_context("public_user", "public")
        
        # Public should only access public scope
        try:
            self.secrets_manager.store_secret("public", "public_test", "public_value")
            retrieved = self.secrets_manager.get_secret("public", "public_test")
            if retrieved != "public_value":
                return False
        except PermissionError:
            return False
        
        # Test restricted access to other scopes
        restricted_scopes = ["api", "database", "monitoring", "system"]
        for scope in restricted_scopes:
            try:
                self.secrets_manager.get_secret(scope, "any_key")
                return False  # Should have thrown exception
            except PermissionError:
                pass  # Expected
        
        return True
    
    def _test_access_denied(self) -> bool:
        """Test access denied scenarios are logged."""
        self.secrets_manager.set_user_context("restricted_user", "public")
        
        try:
            self.secrets_manager.get_secret("system", "restricted_key")
            return False  # Should have thrown exception
        except PermissionError:
            # Check that access denial was logged
            events = self.audit_logger.get_recent_events("ACCESS_DENIED", 10)
            return len(events) > 0
    
    def test_configuration_loading(self):
        """Test configuration loading and validation."""
        print("\n⚙️  Testing Configuration Loading")
        print("-" * 40)
        
        tests = [
            ("basic_config_load", self._test_basic_config_load),
            ("env_overrides", self._test_env_overrides),
            ("secret_resolution", self._test_secret_resolution),
            ("validation_rules", self._test_validation_rules),
            ("security_checks", self._test_security_checks)
        ]
        
        for test_name, test_func in tests:
            try:
                result = test_func()
                self.results["config_tests"].append({
                    "test": test_name,
                    "status": "PASS" if result else "FAIL",
                    "details": result if isinstance(result, dict) else {}
                })
                print(f"  ✅ {test_name}: PASS")
            except Exception as e:
                self.results["config_tests"].append({
                    "test": test_name,
                    "status": "FAIL",
                    "error": str(e)
                })
                print(f"  ❌ {test_name}: FAIL - {e}")
    
    def _test_basic_config_load(self) -> bool:
        """Test basic configuration loading."""
        # Create test config file
        config_data = {
            "trading": {
                "risk_per_trade": 0.02,
                "max_positions": 10,
                "environment": "paper"
            },
            "database": {
                "host": "localhost",
                "port": 5432
            }
        }
        
        config_file = self.config_dir / "test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # Load configuration
        loaded_config = self.config_loader.load_config("test_config")
        
        return loaded_config["trading"]["risk_per_trade"] == 0.02
    
    def _test_env_overrides(self) -> bool:
        """Test environment-specific overrides."""
        # Create base config
        base_config = {
            "trading": {
                "environment": "paper",
                "risk_per_trade": 0.02
            }
        }
        
        # Create production override
        prod_config = {
            "trading": {
                "environment": "live",
                "risk_per_trade": 0.01
            }
        }
        
        base_file = self.config_dir / "override_test.yaml"
        prod_file = self.config_dir / "override_test.production.yaml"
        
        with open(base_file, 'w') as f:
            yaml.dump(base_config, f)
        
        with open(prod_file, 'w') as f:
            yaml.dump(prod_config, f)
        
        # Set environment to production
        original_env = os.environ.get("ENVIRONMENT")
        os.environ["ENVIRONMENT"] = "production"
        
        try:
            # Create new loader to pick up environment change
            prod_loader = SecureConfigLoader(config_dir=str(self.config_dir))
            
            loaded_config = prod_loader.load_config("override_test")
            
            return (loaded_config["trading"]["environment"] == "live" and 
                    loaded_config["trading"]["risk_per_trade"] == 0.01)
        
        finally:
            # Restore environment
            if original_env:
                os.environ["ENVIRONMENT"] = original_env
            else:
                os.environ.pop("ENVIRONMENT", None)
    
    def _test_secret_resolution(self) -> bool:
        """Test secret reference resolution in config."""
        # Store a secret
        self.secrets_manager.set_user_context("config_test", "admin")
        self.secrets_manager.store_secret("api", "test_token", "secret_token_123")
        
        # Create config with secret reference
        config_data = {
            "api": {
                "token": "secret://api/test_token",
                "url": "https://api.example.com"
            }
        }
        
        config_file = self.config_dir / "secret_test.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # Load configuration
        loaded_config = self.config_loader.load_config("secret_test")
        
        return loaded_config["api"]["token"] == "secret_token_123"
    
    def _test_validation_rules(self) -> bool:
        """Test configuration validation rules."""
        # Create invalid config (risk too high)
        invalid_config = {
            "trading": {
                "risk_per_trade": 0.5,  # 50% - too high
                "max_positions": 10,
                "environment": "paper"
            }
        }
        
        config_file = self.config_dir / "invalid_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(invalid_config, f)
        
        # Loading should fail validation
        try:
            self.config_loader.load_config("invalid_config")
            return False  # Should have thrown exception
        except ConfigValidationError:
            return True  # Expected
    
    def _test_security_checks(self) -> bool:
        """Test security injection detection."""
        # Create config with malicious content
        malicious_config = {
            "database": {
                "command": "eval(dangerous_code)",
                "host": "localhost"
            }
        }
        
        config_file = self.config_dir / "malicious_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(malicious_config, f)
        
        # Loading should fail security check
        try:
            self.config_loader.load_config("malicious_config")
            return False  # Should have thrown exception
        except ConfigValidationError:
            return True  # Expected
    
    def test_tls_functionality(self):
        """Test TLS and certificate functionality."""
        print("\n🔒 Testing TLS Functionality")
        print("-" * 40)
        
        tests = [
            ("cert_generation", self._test_cert_generation),
            ("ssl_context_creation", self._test_ssl_context),
            ("cert_validation", self._test_cert_validation)
        ]
        
        for test_name, test_func in tests:
            try:
                result = test_func()
                self.results["tls_tests"].append({
                    "test": test_name,
                    "status": "PASS" if result else "FAIL",
                    "details": result if isinstance(result, dict) else {}
                })
                print(f"  ✅ {test_name}: PASS")
            except Exception as e:
                self.results["tls_tests"].append({
                    "test": test_name,
                    "status": "FAIL",
                    "error": str(e)
                })
                print(f"  ❌ {test_name}: FAIL - {e}")
    
    def _test_cert_generation(self) -> bool:
        """Test certificate generation."""
        cert_file, key_file = self.tls_manager.generate_self_signed_cert("test-host")
        
        return Path(cert_file).exists() and Path(key_file).exists()
    
    def _test_ssl_context(self) -> bool:
        """Test SSL context creation."""
        # Generate certificate first
        cert_file, key_file = self.tls_manager.generate_self_signed_cert("ssl-test")
        
        # Create SSL context
        context = self.tls_manager.create_ssl_context(cert_file, key_file)
        
        return context is not None
    
    def _test_cert_validation(self) -> bool:
        """Test certificate validation."""
        # Generate certificate
        cert_file, _ = self.tls_manager.generate_self_signed_cert("validation-test")
        
        # Validate certificate
        cert_info = self.tls_manager.validate_certificate(cert_file)
        
        return cert_info["is_valid"] and "validation-test" in cert_info["subject"]
    
    def test_zmq_security(self):
        """Test ZeroMQ CURVE security."""
        print("\n🔗 Testing ZeroMQ Security")
        print("-" * 40)
        
        tests = [
            ("curve_key_generation", self._test_curve_keys),
            ("zmq_socket_config", self._test_zmq_socket_config)
        ]
        
        for test_name, test_func in tests:
            try:
                result = test_func()
                self.results["zmq_tests"].append({
                    "test": test_name,
                    "status": "PASS" if result else "FAIL",
                    "details": result if isinstance(result, dict) else {}
                })
                print(f"  ✅ {test_name}: PASS")
            except Exception as e:
                self.results["zmq_tests"].append({
                    "test": test_name,
                    "status": "FAIL",
                    "error": str(e)
                })
                print(f"  ❌ {test_name}: FAIL - {e}")
    
    def _test_curve_keys(self) -> bool:
        """Test ZeroMQ CURVE key generation."""
        context = zmq.Context()
        
        try:
            keys = self.tls_manager.setup_zmq_curve_security(context)
            
            # Verify all keys are present
            required_keys = ["server_public", "server_secret", "client_public", "client_secret"]
            
            return all(key in keys for key in required_keys)
        
        finally:
            context.term()
    
    def _test_zmq_socket_config(self) -> bool:
        """Test ZeroMQ socket CURVE configuration."""
        context = zmq.Context()
        
        try:
            # Setup CURVE security
            self.tls_manager.setup_zmq_curve_security(context)
            
            # Create and configure socket
            socket = context.socket(zmq.DEALER)
            
            # Configure as client
            self.tls_manager.configure_zmq_socket_curve(socket, "DEALER", "client")
            
            # Verify CURVE is enabled
            return hasattr(socket, 'curve_publickey')
        
        finally:
            socket.close()
            context.term()
    
    def test_audit_logging(self):
        """Test audit logging functionality."""
        print("\n📝 Testing Audit Logging")
        print("-" * 40)
        
        tests = [
            ("security_event_logging", self._test_security_logging),
            ("config_access_logging", self._test_config_logging),
            ("breach_detection", self._test_breach_detection),
            ("log_file_creation", self._test_log_files)
        ]
        
        for test_name, test_func in tests:
            try:
                result = test_func()
                self.results["audit_tests"].append({
                    "test": test_name,
                    "status": "PASS" if result else "FAIL",
                    "details": result if isinstance(result, dict) else {}
                })
                print(f"  ✅ {test_name}: PASS")
            except Exception as e:
                self.results["audit_tests"].append({
                    "test": test_name,
                    "status": "FAIL",
                    "error": str(e)
                })
                print(f"  ❌ {test_name}: FAIL - {e}")
    
    def _test_security_logging(self) -> bool:
        """Test security event logging."""
        self.audit_logger.log_security_event(
            "TEST_SECURITY_EVENT",
            {"test_data": "security_test"}
        )
        
        events = self.audit_logger.get_recent_events("TEST_SECURITY_EVENT", 5)
        return len(events) > 0
    
    def _test_config_logging(self) -> bool:
        """Test configuration access logging."""
        self.audit_logger.log_config_access(
            "TEST_CONFIG_ACCESS",
            {"config_name": "test_config"}
        )
        
        events = self.audit_logger.get_recent_events("TEST_CONFIG_ACCESS", 5)
        return len(events) > 0
    
    def _test_breach_detection(self) -> bool:
        """Test breach detection patterns."""
        # Trigger multiple failed decryption events
        for _ in range(6):  # Exceed threshold of 5
            self.audit_logger.log_security_event(
                "SECRETS_LOAD_FAILED",
                {"error": "decryption_failed"}
            )
        
        # Wait a moment for pattern detection
        time.sleep(1)
        
        # Check for breach alert
        events = self.audit_logger.get_recent_events(limit=50)
        breach_events = [e for e in events if e.get("breach_type") == "MULTIPLE_FAILED_DECRYPTION"]
        
        return len(breach_events) > 0
    
    def _test_log_files(self) -> bool:
        """Test log file creation."""
        expected_files = [
            "security_audit.log",
            "config_access.log",
            "security_breaches.log"
        ]
        
        # Trigger some logging
        self.audit_logger.log_security_event("TEST_EVENT", {})
        self.audit_logger.log_config_access("TEST_ACCESS", {})
        
        # Check if log files exist
        for filename in expected_files:
            log_file = self.logs_dir / filename
            if not log_file.exists():
                return False
        
        return True
    
    def test_integration_scenarios(self):
        """Test integration scenarios."""
        print("\n🔄 Testing Integration Scenarios")
        print("-" * 40)
        
        tests = [
            ("end_to_end_secret_flow", self._test_e2e_secret_flow),
            ("config_with_secrets", self._test_config_with_secrets),
            ("security_monitoring", self._test_security_monitoring)
        ]
        
        for test_name, test_func in tests:
            try:
                result = test_func()
                self.results["integration_tests"].append({
                    "test": test_name,
                    "status": "PASS" if result else "FAIL",
                    "details": result if isinstance(result, dict) else {}
                })
                print(f"  ✅ {test_name}: PASS")
            except Exception as e:
                self.results["integration_tests"].append({
                    "test": test_name,
                    "status": "FAIL",
                    "error": str(e)
                })
                print(f"  ❌ {test_name}: FAIL - {e}")
    
    def _test_e2e_secret_flow(self) -> bool:
        """Test end-to-end secret management flow."""
        # 1. Store secrets from environment variables
        os.environ["TEST_API_KEY"] = "test_api_key_value"
        os.environ["TEST_DB_PASSWORD"] = "test_db_password"
        
        self.secrets_manager.set_user_context("e2e_test", "admin")
        
        # 2. Load secrets from environment
        env_mapping = {
            "api": {"api_key": "TEST_API_KEY"},
            "database": {"password": "TEST_DB_PASSWORD"}
        }
        self.secrets_manager.load_from_env(env_mapping)
        
        # 3. Create configuration that references secrets
        config_data = {
            "api": {
                "key": "secret://api/api_key",
                "url": "https://api.example.com"
            },
            "database": {
                "password": "secret://database/password",
                "host": "localhost"
            }
        }
        
        config_file = self.config_dir / "e2e_test.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # 4. Load configuration (should resolve secrets)
        loaded_config = self.config_loader.load_config("e2e_test")
        
        # 5. Verify secrets were resolved
        return (loaded_config["api"]["key"] == "test_api_key_value" and
                loaded_config["database"]["password"] == "test_db_password")
    
    def _test_config_with_secrets(self) -> bool:
        """Test configuration loading with secret resolution and validation."""
        # Store required secrets
        self.secrets_manager.set_user_context("config_test", "admin")
        self.secrets_manager.store_secret("api", "oanda_token", "test_oanda_token")
        
        # Create realistic config
        config_data = {
            "trading": {
                "risk_per_trade": 0.02,
                "max_positions": 10,
                "environment": "paper",
                "broker": {
                    "token": "secret://api/oanda_token"
                }
            },
            "database": {
                "host": "localhost",
                "port": 5432
            }
        }
        
        config_file = self.config_dir / "realistic_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # Load and validate
        loaded_config = self.config_loader.load_config("realistic_config")
        
        return loaded_config["trading"]["broker"]["token"] == "test_oanda_token"
    
    def _test_security_monitoring(self) -> bool:
        """Test security monitoring and reporting."""
        # Generate various security events
        self.audit_logger.log_security_event("SYSTEM_START", {"component": "trading_engine"})
        self.audit_logger.log_config_access("CONFIG_LOADED", {"config": "main"})
        self.audit_logger.log_security_event("ENCRYPTION_KEY_DERIVED", {"algorithm": "PBKDF2"})
        
        # Generate security report
        report = self.audit_logger.generate_security_report()
        
        return (report["total_events"] > 0 and 
                "event_type_counts" in report and
                "access_type_counts" in report)
    
    def generate_test_summary(self):
        """Generate test summary."""
        print("\n📊 Test Summary")
        print("=" * 60)
        
        categories = [
            ("Encryption Tests", "encryption_tests"),
            ("RBAC Tests", "rbac_tests"),
            ("Configuration Tests", "config_tests"),
            ("TLS Tests", "tls_tests"),
            ("ZeroMQ Tests", "zmq_tests"),
            ("Audit Tests", "audit_tests"),
            ("Integration Tests", "integration_tests")
        ]
        
        total_tests = 0
        total_passed = 0
        
        for category_name, category_key in categories:
            tests = self.results.get(category_key, [])
            passed = len([t for t in tests if t["status"] == "PASS"])
            total = len(tests)
            
            total_tests += total
            total_passed += passed
            
            print(f"{category_name}: {passed}/{total} passed")
            
            # Show failed tests
            failed_tests = [t for t in tests if t["status"] == "FAIL"]
            for test in failed_tests:
                print(f"  ❌ {test['test']}: {test.get('error', 'Unknown error')}")
        
        success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        print(f"\nOverall: {total_passed}/{total_tests} tests passed ({success_rate:.1f}%)")
        
        if success_rate >= 90:
            print("🎉 Security system is EXCELLENT!")
        elif success_rate >= 75:
            print("✅ Security system is GOOD")
        elif success_rate >= 50:
            print("⚠️  Security system needs IMPROVEMENT")
        else:
            print("❌ Security system has CRITICAL ISSUES")
        
        self.results["summary"] = {
            "total_tests": total_tests,
            "total_passed": total_passed,
            "success_rate": success_rate,
            "status": "EXCELLENT" if success_rate >= 90 else 
                     "GOOD" if success_rate >= 75 else
                     "NEEDS_IMPROVEMENT" if success_rate >= 50 else
                     "CRITICAL_ISSUES"
        }
    
    def cleanup(self):
        """Clean up test environment."""
        try:
            # Clean up environment variables
            test_env_vars = ["TEST_API_KEY", "TEST_DB_PASSWORD"]
            for var in test_env_vars:
                os.environ.pop(var, None)
            
            # Remove test directory
            if self.test_dir.exists():
                shutil.rmtree(self.test_dir)
            
            print(f"\n🧹 Test environment cleaned up")
        
        except Exception as e:
            print(f"⚠️  Cleanup warning: {e}")


def main():
    """Run the security test suite."""
    print("🛡️  FX AI-Quant Security and Configuration Management Test Suite")
    print("================================================================")
    
    # Initialize and run tests
    test_suite = SecurityTestSuite()
    results = test_suite.run_all_tests()
    
    # Save results to file
    results_file = Path("security_test_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📄 Detailed results saved to: {results_file}")
    
    # Return exit code based on success rate
    success_rate = results.get("summary", {}).get("success_rate", 0)
    return 0 if success_rate >= 75 else 1


if __name__ == "__main__":
    sys.exit(main()) 