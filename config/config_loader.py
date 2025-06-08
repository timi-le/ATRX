"""
Secure Configuration Loader for FX AI-Quant Trading System

This module provides secure configuration management with environment-based overrides,
validation, and integration with the encrypted secrets manager.

Features:
- Environment-aware configuration loading (dev, staging, prod)
- Configuration validation and type checking
- Integration with secrets manager for sensitive data
- Configuration injection protection
- Change detection and monitoring
- Configuration caching with invalidation
"""

import copy
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from .audit_logger import get_audit_logger
from .secrets_manager import get_secrets_manager


@dataclass
class ConfigValidationRule:
    """Configuration validation rule definition."""

    field_path: str
    field_type: type
    required: bool = True
    min_value: int | float | None = None
    max_value: int | float | None = None
    allowed_values: list[Any] | None = None
    pattern: str | None = None
    description: str = ""


class ConfigValidationError(Exception):
    """Configuration validation error."""



class SecureConfigLoader:
    """
    Secure configuration loader with validation and secrets integration.

    Provides environment-aware configuration loading with comprehensive
    validation, security checks, and integration with the secrets manager.
    """

    def __init__(self, config_dir: str = "config", cache_ttl: int = 300):
        self.config_dir = Path(config_dir)
        self.cache_ttl = cache_ttl  # Cache time-to-live in seconds
        self.secrets_manager = get_secrets_manager()
        self.audit_logger = get_audit_logger()

        # Configuration cache
        self._config_cache = {}
        self._cache_timestamps = {}
        self._config_hashes = {}

        # Environment detection
        self.environment = self._detect_environment()

        # Validation rules
        self.validation_rules = self._load_validation_rules()

        self.audit_logger.log_security_event(
            "CONFIG_LOADER_INITIALIZED",
            {
                "config_dir": str(self.config_dir),
                "environment": self.environment,
                "cache_ttl": cache_ttl,
            },
        )

    def _detect_environment(self) -> str:
        """Detect the current environment."""
        # Check environment variable first
        env = os.getenv("ENVIRONMENT", os.getenv("ENV", "development")).lower()

        # Validate environment
        valid_environments = [
            "development",
            "dev",
            "staging",
            "stage",
            "production",
            "prod",
        ]
        if env not in valid_environments:
            env = "development"

        # Normalize environment names
        if env in ["dev", "development"]:
            env = "development"
        elif env in ["stage", "staging"]:
            env = "staging"
        elif env in ["prod", "production"]:
            env = "production"

        return env

    def _load_validation_rules(self) -> list[ConfigValidationRule]:
        """Load configuration validation rules."""
        rules = [
            # Trading configuration
            ConfigValidationRule(
                "trading.risk_per_trade",
                float,
                required=True,
                min_value=0.001,
                max_value=0.1,
                description="Risk percentage per trade (0.1% to 10%)",
            ),
            ConfigValidationRule(
                "trading.max_positions",
                int,
                required=True,
                min_value=1,
                max_value=50,
                description="Maximum number of open positions",
            ),
            ConfigValidationRule(
                "trading.environment",
                str,
                required=True,
                allowed_values=["paper", "live"],
                description="Trading environment mode",
            ),
            # Database configuration
            ConfigValidationRule(
                "database.host", str, required=True, description="Database host address"
            ),
            ConfigValidationRule(
                "database.port",
                int,
                required=True,
                min_value=1,
                max_value=65535,
                description="Database port number",
            ),
            # API configuration
            ConfigValidationRule(
                "api.rate_limit",
                int,
                required=False,
                min_value=1,
                max_value=10000,
                description="API rate limit per minute",
            ),
            # Monitoring configuration
            ConfigValidationRule(
                "monitoring.prometheus.enabled",
                bool,
                required=False,
                description="Enable Prometheus monitoring",
            ),
            ConfigValidationRule(
                "monitoring.prometheus.port",
                int,
                required=False,
                min_value=1024,
                max_value=65535,
                description="Prometheus metrics port",
            ),
            # Security configuration
            ConfigValidationRule(
                "security.tls.enabled",
                bool,
                required=False,
                description="Enable TLS encryption",
            ),
        ]

        return rules

    def load_config(
        self, config_name: str = "config", reload: bool = False
    ) -> dict[str, Any]:
        """
        Load configuration with environment overrides and caching.

        Args:
            config_name: Name of the configuration file (without extension)
            reload: Force reload from disk, bypassing cache

        Returns:
            Loaded and validated configuration dictionary
        """
        cache_key = f"{config_name}_{self.environment}"

        # Check cache first
        if not reload and self._is_cache_valid(cache_key):
            self.audit_logger.log_config_access(
                "CONFIG_CACHE_HIT",
                {"config_name": config_name, "environment": self.environment},
            )
            return copy.deepcopy(self._config_cache[cache_key])

        try:
            # Load base configuration
            base_config = self._load_config_file(f"{config_name}.yaml")

            # Apply environment-specific overrides
            env_config = self._load_environment_overrides(config_name)
            config = self._merge_configs(base_config, env_config)

            # Resolve secrets
            config = self._resolve_secrets(config)

            # Validate configuration
            self._validate_config(config)

            # Check for malicious content
            self._security_check(config)

            # Cache the configuration
            self._cache_config(cache_key, config)

            self.audit_logger.log_config_access(
                "CONFIG_LOADED",
                {
                    "config_name": config_name,
                    "environment": self.environment,
                    "has_secrets": self._has_secrets(config),
                    "config_size": len(str(config)),
                },
            )

            return copy.deepcopy(config)

        except Exception as e:
            self.audit_logger.log_config_access(
                "CONFIG_LOAD_FAILED",
                {
                    "config_name": config_name,
                    "environment": self.environment,
                    "error": str(e),
                },
            )
            raise

    def _load_config_file(self, filename: str) -> dict[str, Any]:
        """Load a configuration file."""
        config_path = self.config_dir / filename

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path, encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}

            # Calculate file hash for change detection
            with open(config_path, "rb") as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()

            self._config_hashes[str(config_path)] = file_hash

            return config

        except yaml.YAMLError as e:
            raise ConfigValidationError(f"Invalid YAML in {filename}: {e}")
        except Exception as e:
            raise ConfigValidationError(f"Error loading {filename}: {e}")

    def _load_environment_overrides(self, config_name: str) -> dict[str, Any]:
        """Load environment-specific configuration overrides."""
        env_filename = f"{config_name}.{self.environment}.yaml"
        env_path = self.config_dir / env_filename

        if env_path.exists():
            return self._load_config_file(env_filename)

        return {}

    def _merge_configs(
        self, base: dict[str, Any], override: dict[str, Any]
    ) -> dict[str, Any]:
        """Deep merge configuration dictionaries."""
        result = copy.deepcopy(base)

        def deep_merge(target: dict[str, Any], source: dict[str, Any]):
            for key, value in source.items():
                if (
                    key in target
                    and isinstance(target[key], dict)
                    and isinstance(value, dict)
                ):
                    deep_merge(target[key], value)
                else:
                    target[key] = value

        deep_merge(result, override)
        return result

    def _resolve_secrets(self, config: dict[str, Any]) -> dict[str, Any]:
        """Resolve secret references in configuration."""

        def resolve_recursive(obj):
            if isinstance(obj, dict):
                return {key: resolve_recursive(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [resolve_recursive(item) for item in obj]
            elif isinstance(obj, str) and obj.startswith("secret://"):
                # Parse secret reference: secret://scope/key
                try:
                    secret_ref = obj[9:]  # Remove "secret://" prefix
                    if "/" in secret_ref:
                        scope, key = secret_ref.split("/", 1)
                        return self.secrets_manager.get_secret(scope, key)
                    else:
                        # Default to 'api' scope for backward compatibility
                        return self.secrets_manager.get_secret("api", secret_ref)
                except Exception as e:
                    self.audit_logger.log_config_access(
                        "SECRET_RESOLUTION_FAILED", {"secret_ref": obj, "error": str(e)}
                    )
                    raise ConfigValidationError(f"Failed to resolve secret {obj}: {e}")
            else:
                return obj

        return resolve_recursive(config)

    def _validate_config(self, config: dict[str, Any]) -> None:
        """Validate configuration against rules."""
        errors = []

        for rule in self.validation_rules:
            try:
                value = self._get_nested_value(config, rule.field_path)

                # Check if required field is missing
                if rule.required and value is None:
                    errors.append(f"Required field '{rule.field_path}' is missing")
                    continue

                # Skip validation if field is optional and missing
                if not rule.required and value is None:
                    continue

                # Type validation
                if not isinstance(value, rule.field_type):
                    errors.append(
                        f"Field '{rule.field_path}' should be {rule.field_type.__name__}, "
                        f"got {type(value).__name__}"
                    )
                    continue

                # Range validation for numbers
                if (
                    rule.min_value is not None
                    and hasattr(value, "__lt__")
                    and value < rule.min_value
                ):
                    errors.append(
                        f"Field '{rule.field_path}' value {value} is below minimum {rule.min_value}"
                    )

                if (
                    rule.max_value is not None
                    and hasattr(value, "__gt__")
                    and value > rule.max_value
                ):
                    errors.append(
                        f"Field '{rule.field_path}' value {value} is above maximum {rule.max_value}"
                    )

                # Allowed values validation
                if rule.allowed_values is not None and value not in rule.allowed_values:
                    errors.append(
                        f"Field '{rule.field_path}' value '{value}' is not in allowed values: "
                        f"{rule.allowed_values}"
                    )

                # Pattern validation for strings
                if rule.pattern is not None and isinstance(value, str):
                    import re

                    if not re.match(rule.pattern, value):
                        errors.append(
                            f"Field '{rule.field_path}' does not match pattern {rule.pattern}"
                        )

            except Exception as e:
                errors.append(f"Validation error for '{rule.field_path}': {e}")

        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(
                f"  - {error}" for error in errors
            )
            raise ConfigValidationError(error_msg)

    def _get_nested_value(self, data: dict[str, Any], path: str) -> Any:
        """Get value from nested dictionary using dot notation."""
        keys = path.split(".")
        current = data

        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None

        return current

    def _security_check(self, config: dict[str, Any]) -> None:
        """Perform security checks on configuration."""
        config_str = json.dumps(config, default=str).lower()

        # Check for potential injection patterns
        dangerous_patterns = [
            "eval(",
            "exec(",
            "__import__",
            "os.system",
            "subprocess",
            "<script",
            "javascript:",
            "${",  # Template injection
            "#{",  # Expression injection
        ]

        for pattern in dangerous_patterns:
            if pattern in config_str:
                self.audit_logger.log_breach_attempt(
                    "CONFIG_INJECTION_DETECTED",
                    {
                        "pattern": pattern,
                        "config_snippet": config_str[
                            max(0, config_str.find(pattern) - 20) : config_str.find(
                                pattern
                            )
                            + 20
                        ],
                    },
                    "CRITICAL",
                )
                raise ConfigValidationError(
                    f"Potentially malicious pattern detected in configuration: {pattern}"
                )

    def _has_secrets(self, config: dict[str, Any]) -> bool:
        """Check if configuration contains secret references."""
        config_str = json.dumps(config, default=str)
        return "secret://" in config_str

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached configuration is still valid."""
        if cache_key not in self._config_cache:
            return False

        # Check TTL
        cache_time = self._cache_timestamps.get(cache_key, 0)
        if datetime.now().timestamp() - cache_time > self.cache_ttl:
            return False

        # Check for file changes
        for file_path, stored_hash in self._config_hashes.items():
            if Path(file_path).exists():
                with open(file_path, "rb") as f:
                    current_hash = hashlib.sha256(f.read()).hexdigest()
                if current_hash != stored_hash:
                    return False

        return True

    def _cache_config(self, cache_key: str, config: dict[str, Any]) -> None:
        """Cache configuration data."""
        self._config_cache[cache_key] = copy.deepcopy(config)
        self._cache_timestamps[cache_key] = datetime.now().timestamp()

    def invalidate_cache(self, config_name: str | None = None) -> None:
        """
        Invalidate configuration cache.

        Args:
            config_name: Specific config to invalidate, or None for all
        """
        if config_name:
            keys_to_remove = [
                key
                for key in self._config_cache.keys()
                if key.startswith(f"{config_name}_")
            ]
        else:
            keys_to_remove = list(self._config_cache.keys())

        for key in keys_to_remove:
            self._config_cache.pop(key, None)
            self._cache_timestamps.pop(key, None)

        self.audit_logger.log_config_access(
            "CONFIG_CACHE_INVALIDATED",
            {"config_name": config_name, "invalidated_keys": keys_to_remove},
        )

    def get_config_info(self, config_name: str = "config") -> dict[str, Any]:
        """Get information about a configuration."""
        cache_key = f"{config_name}_{self.environment}"

        return {
            "config_name": config_name,
            "environment": self.environment,
            "cached": cache_key in self._config_cache,
            "cache_timestamp": self._cache_timestamps.get(cache_key),
            "cache_ttl": self.cache_ttl,
            "base_file": str(self.config_dir / f"{config_name}.yaml"),
            "env_file": str(self.config_dir / f"{config_name}.{self.environment}.yaml"),
            "validation_rules_count": len(self.validation_rules),
        }

    def get_environment_info(self) -> dict[str, Any]:
        """Get information about the current environment."""
        return {
            "environment": self.environment,
            "environment_var": os.getenv("ENVIRONMENT", "not_set"),
            "env_var": os.getenv("ENV", "not_set"),
            "config_dir": str(self.config_dir),
            "cache_ttl": self.cache_ttl,
        }

    def reload_validation_rules(self) -> None:
        """Reload validation rules (useful for dynamic rule updates)."""
        self.validation_rules = self._load_validation_rules()

        self.audit_logger.log_config_access(
            "VALIDATION_RULES_RELOADED", {"rules_count": len(self.validation_rules)}
        )


# Global instance
_config_loader = None


def get_config_loader() -> SecureConfigLoader:
    """Get the global configuration loader instance."""
    global _config_loader
    if _config_loader is None:
        _config_loader = SecureConfigLoader()
    return _config_loader
