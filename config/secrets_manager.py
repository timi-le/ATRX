"""
Secure Secrets Manager for FX AI-Quant Trading System

This module provides encrypted storage and retrieval of sensitive configuration
data with role-based access control and comprehensive audit logging.

Features:
- Fernet (AES 128) encryption for sensitive data
- Role-based access control (RBAC)
- Environment variable integration
- Secure key derivation
- Comprehensive audit logging
- Backup and recovery capabilities
"""

import base64
import json
import os
import secrets
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from dotenv import load_dotenv

from .audit_logger import get_audit_logger


class SecretsManager:
    """
    Secure secrets manager with encryption, RBAC, and audit logging.

    Provides encrypted storage for sensitive configuration data with
    role-based access control and comprehensive security logging.
    """

    # Define access scopes and their allowed roles
    SCOPE_PERMISSIONS = {
        "api": ["admin", "service", "trading"],
        "database": ["admin", "service"],
        "monitoring": ["admin", "service", "monitoring"],
        "system": ["admin"],
        "public": ["admin", "service", "trading", "monitoring", "public"],
    }

    def __init__(
        self,
        secrets_dir: str = "config/secrets",
        master_key_env: str = "MASTER_ENCRYPTION_KEY",
    ):
        self.secrets_dir = Path(secrets_dir)
        self.master_key_env = master_key_env
        self.audit_logger = get_audit_logger()

        # Create secrets directory
        self.secrets_dir.mkdir(parents=True, exist_ok=True)

        # Load environment variables
        load_dotenv()

        # Initialize encryption
        self._fernet = self._initialize_encryption()

        # Current user context (can be set by authentication system)
        self.current_user = None
        self.current_role = "public"  # Default role

        # Load existing secrets vault
        self._secrets_vault = self._load_secrets_vault()

        self.audit_logger.log_security_event(
            "SECRETS_MANAGER_INITIALIZED",
            {
                "secrets_dir": str(self.secrets_dir),
                "vault_exists": self._vault_file_exists(),
                "role": self.current_role,
            },
        )

    def _initialize_encryption(self) -> Fernet:
        """Initialize encryption with master key."""
        try:
            # Try to get master key from environment
            master_key = os.getenv(self.master_key_env)

            if not master_key:
                # Generate a new master key if none exists
                master_key = self._generate_master_key()
                self.audit_logger.log_security_event(
                    "MASTER_KEY_GENERATED", {"env_var": self.master_key_env}
                )
            else:
                self.audit_logger.log_security_event(
                    "MASTER_KEY_LOADED", {"env_var": self.master_key_env}
                )

            # Derive encryption key using PBKDF2
            encryption_key = self._derive_encryption_key(master_key)

            return Fernet(encryption_key)

        except Exception as e:
            self.audit_logger.log_security_event(
                "ENCRYPTION_INIT_FAILED", {"error": str(e)}
            )
            raise

    def _generate_master_key(self) -> str:
        """Generate a new master key."""
        # Generate 32 random bytes and encode as base64
        key_bytes = secrets.token_bytes(32)
        master_key = base64.urlsafe_b64encode(key_bytes).decode()

        # Save to .env file for persistence
        env_file = Path(".env")

        if env_file.exists():
            # Read existing .env content
            with open(env_file) as f:
                content = f.read()
        else:
            content = ""

        # Add or update master key
        lines = content.split("\n")
        key_line = f"{self.master_key_env}={master_key}"

        # Check if key already exists
        updated = False
        for i, line in enumerate(lines):
            if line.startswith(f"{self.master_key_env}="):
                lines[i] = key_line
                updated = True
                break

        if not updated:
            lines.append(key_line)

        # Write back to .env
        with open(env_file, "w") as f:
            f.write("\n".join(lines))

        return master_key

    def _derive_encryption_key(self, master_key: str) -> bytes:
        """Derive encryption key from master key using PBKDF2."""
        try:
            # Use a fixed salt for consistency (in production, use random salt per secret)
            salt = b"fx_quant_salt_2024"

            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,  # Industry standard
            )

            key = base64.urlsafe_b64encode(kdf.derive(master_key.encode()))

            self.audit_logger.log_security_event(
                "ENCRYPTION_KEY_DERIVED",
                {"iterations": 100000, "algorithm": "PBKDF2-SHA256"},
            )

            return key

        except Exception as e:
            self.audit_logger.log_security_event(
                "KEY_DERIVATION_FAILED", {"error": str(e)}
            )
            raise

    def _load_secrets_vault(self) -> dict[str, Any]:
        """Load the encrypted secrets vault."""
        vault_file = self.secrets_dir / "vault.enc"

        if not vault_file.exists():
            self.audit_logger.log_security_event(
                "NEW_SECRETS_VAULT_CREATED", {"vault_file": str(vault_file)}
            )
            return {}

        try:
            with open(vault_file, "rb") as f:
                encrypted_data = f.read()

            # Decrypt the vault
            decrypted_data = self._fernet.decrypt(encrypted_data)
            vault = json.loads(decrypted_data.decode())

            self.audit_logger.log_security_event(
                "SECRETS_VAULT_LOADED",
                {
                    "vault_file": str(vault_file),
                    "num_secrets": len(vault),
                    "scopes": list(vault.keys()),
                },
            )

            return vault

        except Exception as e:
            self.audit_logger.log_security_event(
                "SECRETS_LOAD_FAILED", {"vault_file": str(vault_file), "error": str(e)}
            )
            # Return empty vault if decryption fails
            return {}

    def _save_secrets_vault(self) -> None:
        """Save the encrypted secrets vault."""
        vault_file = self.secrets_dir / "vault.enc"

        try:
            # Serialize and encrypt
            vault_json = json.dumps(self._secrets_vault, indent=2)
            encrypted_data = self._fernet.encrypt(vault_json.encode())

            # Write to file
            with open(vault_file, "wb") as f:
                f.write(encrypted_data)

            self.audit_logger.log_security_event(
                "SECRETS_VAULT_SAVED",
                {
                    "vault_file": str(vault_file),
                    "num_secrets": len(self._secrets_vault),
                    "scopes": list(self._secrets_vault.keys()),
                },
            )

        except Exception as e:
            self.audit_logger.log_security_event(
                "SECRETS_SAVE_FAILED", {"vault_file": str(vault_file), "error": str(e)}
            )
            raise

    def _vault_file_exists(self) -> bool:
        """Check if vault file exists."""
        return (self.secrets_dir / "vault.enc").exists()

    def set_user_context(self, user: str, role: str) -> None:
        """
        Set the current user context for access control.

        Args:
            user: Username or identifier
            role: User role (admin, service, trading, monitoring, public)
        """
        self.current_user = user
        self.current_role = role

        self.audit_logger.log_security_event(
            "USER_CONTEXT_SET", {"user": user, "role": role}
        )

    def _check_access(self, scope: str) -> bool:
        """
        Check if current user/role has access to a scope.

        Args:
            scope: The scope to check access for

        Returns:
            True if access is allowed, False otherwise
        """
        allowed_roles = self.SCOPE_PERMISSIONS.get(scope, [])
        has_access = self.current_role in allowed_roles

        if not has_access:
            self.audit_logger.log_config_access(
                "ACCESS_DENIED",
                {
                    "scope": scope,
                    "user": self.current_user,
                    "role": self.current_role,
                    "allowed_roles": allowed_roles,
                },
            )

        return has_access

    def store_secret(
        self, scope: str, key: str, value: str | dict[str, Any]
    ) -> None:
        """
        Store a secret in the encrypted vault.

        Args:
            scope: The scope/category of the secret
            key: The secret key
            value: The secret value
        """
        # Check access
        if not self._check_access(scope):
            raise PermissionError(
                f"Access denied to scope '{scope}' for role '{self.current_role}'"
            )

        # Initialize scope if it doesn't exist
        if scope not in self._secrets_vault:
            self._secrets_vault[scope] = {}

        # Store the secret
        self._secrets_vault[scope][key] = value

        # Save vault
        self._save_secrets_vault()

        self.audit_logger.log_config_access(
            "SECRET_STORED",
            {
                "scope": scope,
                "key": key,
                "user": self.current_user,
                "role": self.current_role,
                "value_type": type(value).__name__,
            },
        )

    def get_secret(self, scope: str, key: str, default: Any = None) -> Any:
        """
        Retrieve a secret from the encrypted vault.

        Args:
            scope: The scope/category of the secret
            key: The secret key
            default: Default value if secret not found

        Returns:
            The secret value or default
        """
        # Check access
        if not self._check_access(scope):
            self.audit_logger.log_config_access(
                "SECRET_ACCESS_FAILED",
                {
                    "scope": scope,
                    "key": key,
                    "user": self.current_user,
                    "role": self.current_role,
                    "reason": "access_denied",
                },
            )
            raise PermissionError(
                f"Access denied to scope '{scope}' for role '{self.current_role}'"
            )

        # Get the secret
        value = self._secrets_vault.get(scope, {}).get(key, default)

        self.audit_logger.log_config_access(
            "SECRET_ACCESSED",
            {
                "scope": scope,
                "key": key,
                "user": self.current_user,
                "role": self.current_role,
                "found": value is not default,
            },
        )

        return value

    def delete_secret(self, scope: str, key: str) -> bool:
        """
        Delete a secret from the vault.

        Args:
            scope: The scope/category of the secret
            key: The secret key

        Returns:
            True if deleted, False if not found
        """
        # Check access
        if not self._check_access(scope):
            raise PermissionError(
                f"Access denied to scope '{scope}' for role '{self.current_role}'"
            )

        if scope in self._secrets_vault and key in self._secrets_vault[scope]:
            del self._secrets_vault[scope][key]

            # Remove empty scope
            if not self._secrets_vault[scope]:
                del self._secrets_vault[scope]

            # Save vault
            self._save_secrets_vault()

            self.audit_logger.log_config_access(
                "SECRET_DELETED",
                {
                    "scope": scope,
                    "key": key,
                    "user": self.current_user,
                    "role": self.current_role,
                },
            )

            return True

        return False

    def list_secrets(self, scope: str | None = None) -> dict[str, list[str]]:
        """
        List available secrets (keys only, not values).

        Args:
            scope: Optional scope to filter by

        Returns:
            Dictionary of scope -> list of keys
        """
        result = {}

        scopes_to_check = [scope] if scope else self._secrets_vault.keys()

        for scope_name in scopes_to_check:
            if self._check_access(scope_name):
                result[scope_name] = list(
                    self._secrets_vault.get(scope_name, {}).keys()
                )

        self.audit_logger.log_config_access(
            "SECRETS_LISTED",
            {
                "requested_scope": scope,
                "returned_scopes": list(result.keys()),
                "user": self.current_user,
                "role": self.current_role,
            },
        )

        return result

    def load_from_env(self, mapping: dict[str, dict[str, str]]) -> None:
        """
        Load secrets from environment variables.

        Args:
            mapping: Dict of scope -> {key: env_var_name}
        """
        loaded_secrets = {}

        for scope, env_mapping in mapping.items():
            if not self._check_access(scope):
                continue

            scope_secrets = {}
            for key, env_var in env_mapping.items():
                value = os.getenv(env_var)
                if value:
                    scope_secrets[key] = value

            if scope_secrets:
                if scope not in self._secrets_vault:
                    self._secrets_vault[scope] = {}
                self._secrets_vault[scope].update(scope_secrets)
                loaded_secrets[scope] = list(scope_secrets.keys())

        if loaded_secrets:
            self._save_secrets_vault()

        self.audit_logger.log_config_access(
            "SECRETS_LOADED_FROM_ENV",
            {
                "loaded_secrets": loaded_secrets,
                "user": self.current_user,
                "role": self.current_role,
            },
        )

    def backup_vault(self, backup_path: str | None = None) -> str:
        """
        Create a backup of the encrypted vault.

        Args:
            backup_path: Optional custom backup path

        Returns:
            Path to the backup file
        """
        if backup_path is None:
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = str(self.secrets_dir / f"vault_backup_{timestamp}.enc")

        vault_file = self.secrets_dir / "vault.enc"

        if vault_file.exists():
            import shutil

            shutil.copy2(vault_file, backup_path)

            self.audit_logger.log_security_event(
                "VAULT_BACKUP_CREATED",
                {
                    "backup_path": backup_path,
                    "original_path": str(vault_file),
                    "user": self.current_user,
                    "role": self.current_role,
                },
            )

        return backup_path

    def get_health_status(self) -> dict[str, Any]:
        """Get health status of the secrets manager."""
        vault_file = self.secrets_dir / "vault.enc"

        status = {
            "vault_exists": vault_file.exists(),
            "vault_readable": False,
            "encryption_working": False,
            "scopes_count": 0,
            "secrets_count": 0,
        }

        try:
            # Test encryption/decryption
            test_data = b"health_check_test"
            encrypted = self._fernet.encrypt(test_data)
            decrypted = self._fernet.decrypt(encrypted)
            status["encryption_working"] = decrypted == test_data

            # Check vault readability
            if vault_file.exists():
                status["vault_readable"] = True
                status["scopes_count"] = len(self._secrets_vault)
                status["secrets_count"] = sum(
                    len(scope_secrets) for scope_secrets in self._secrets_vault.values()
                )

        except Exception as e:
            self.audit_logger.log_security_event(
                "HEALTH_CHECK_FAILED", {"error": str(e)}
            )

        return status


# Global instance
_secrets_manager = None


def get_secrets_manager() -> SecretsManager:
    """Get the global secrets manager instance."""
    global _secrets_manager
    if _secrets_manager is None:
        _secrets_manager = SecretsManager()
    return _secrets_manager
