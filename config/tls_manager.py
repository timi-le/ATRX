"""
TLS and Secure Communication Manager for FX AI-Quant Trading System

This module provides TLS/SSL configuration and secure communication setup
for various components including ZeroMQ CURVE security, Redis TLS, and HTTP/API TLS.

Features:
- ZeroMQ CURVE key pair generation and management
- TLS certificate management
- Secure Redis connections
- HTTPS/API server TLS configuration
- Certificate validation and rotation
"""

import hashlib
import socket
import ssl
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import zmq
import zmq.auth
from zmq.auth.thread import ThreadAuthenticator

from .audit_logger import get_audit_logger
from .secrets_manager import get_secrets_manager


class TLSManager:
    """
    Comprehensive TLS and secure communication manager.

    Handles certificate generation, ZeroMQ CURVE security, Redis TLS,
    and API server TLS configuration with comprehensive security logging.
    """

    def __init__(self, certs_dir: str = "config/certs"):
        self.certs_dir = Path(certs_dir)
        self.secrets_manager = get_secrets_manager()
        self.audit_logger = get_audit_logger()

        # Create certificates directory
        self.certs_dir.mkdir(parents=True, exist_ok=True)

        # ZeroMQ authenticator
        self._zmq_authenticator = None

        self.audit_logger.log_security_event(
            "TLS_MANAGER_INITIALIZED", {"certs_dir": str(self.certs_dir)}
        )

    def setup_zmq_curve_security(self, context: zmq.Context) -> dict[str, bytes]:
        """
        Set up ZeroMQ CURVE security with key pair generation.

        Args:
            context: ZeroMQ context

        Returns:
            Dictionary containing public and secret keys
        """
        try:
            # Generate or load server key pair
            server_keys = self._get_or_generate_zmq_keys("server")

            # Generate or load client key pair
            client_keys = self._get_or_generate_zmq_keys("client")

            # Start authenticator
            self._zmq_authenticator = ThreadAuthenticator(context)
            self._zmq_authenticator.start()

            # Configure CURVE authentication
            self._zmq_authenticator.configure_curve(
                domain="*", location=str(self.certs_dir)
            )

            # Allow client public key
            client_public_dir = self.certs_dir / "client_public_keys"
            client_public_dir.mkdir(exist_ok=True)

            # Save client public key for server authentication
            client_public_file = client_public_dir / "client.key"
            with open(client_public_file, "wb") as f:
                f.write(client_keys["public"])

            self.audit_logger.log_security_event(
                "ZMQ_CURVE_SECURITY_CONFIGURED",
                {
                    "server_public_key_hash": hashlib.sha256(
                        server_keys["public"]
                    ).hexdigest()[:16],
                    "client_public_key_hash": hashlib.sha256(
                        client_keys["public"]
                    ).hexdigest()[:16],
                    "authenticator_started": True,
                },
            )

            # Store keys in secrets manager
            self._store_zmq_keys("server", server_keys)
            self._store_zmq_keys("client", client_keys)

            return {
                "server_public": server_keys["public"],
                "server_secret": server_keys["secret"],
                "client_public": client_keys["public"],
                "client_secret": client_keys["secret"],
            }

        except Exception as e:
            self.audit_logger.log_security_event(
                "ZMQ_CURVE_SETUP_FAILED", {"error": str(e)}
            )
            raise

    def _get_or_generate_zmq_keys(self, key_type: str) -> dict[str, bytes]:
        """Generate or load ZeroMQ CURVE key pair."""
        try:
            # Try to load existing keys from secrets
            public_key = self.secrets_manager.get_secret(
                "system", f"zmq_{key_type}_public"
            )
            secret_key = self.secrets_manager.get_secret(
                "system", f"zmq_{key_type}_secret"
            )

            if public_key and secret_key:
                # Convert from base64 if needed
                if isinstance(public_key, str):
                    import base64

                    public_key = base64.b64decode(public_key.encode())
                    secret_key = base64.b64decode(secret_key.encode())

                self.audit_logger.log_security_event(
                    "ZMQ_KEYS_LOADED",
                    {
                        "key_type": key_type,
                        "public_key_hash": hashlib.sha256(public_key).hexdigest()[:16],
                    },
                )

                return {"public": public_key, "secret": secret_key}

        except Exception as e:
            self.audit_logger.log_security_event(
                "ZMQ_KEYS_LOAD_FAILED", {"key_type": key_type, "error": str(e)}
            )

        # Generate new key pair
        public_key, secret_key = zmq.curve_keypair()

        self.audit_logger.log_security_event(
            "ZMQ_KEYS_GENERATED",
            {
                "key_type": key_type,
                "public_key_hash": hashlib.sha256(public_key).hexdigest()[:16],
            },
        )

        return {"public": public_key, "secret": secret_key}

    def _store_zmq_keys(self, key_type: str, keys: dict[str, bytes]) -> None:
        """Store ZeroMQ keys in secrets manager."""
        import base64

        # Set admin role for storing system secrets
        original_role = self.secrets_manager.current_role
        self.secrets_manager.set_user_context("tls_manager", "admin")

        try:
            # Store as base64 encoded strings
            self.secrets_manager.store_secret(
                "system",
                f"zmq_{key_type}_public",
                base64.b64encode(keys["public"]).decode(),
            )
            self.secrets_manager.store_secret(
                "system",
                f"zmq_{key_type}_secret",
                base64.b64encode(keys["secret"]).decode(),
            )
        finally:
            # Restore original role
            self.secrets_manager.current_role = original_role

    def configure_zmq_socket_curve(
        self, socket: zmq.Socket, socket_type: str, role: str = "client"
    ) -> None:
        """
        Configure a ZeroMQ socket with CURVE security.

        Args:
            socket: ZeroMQ socket to configure
            socket_type: Type of socket (DEALER, ROUTER, PUB, SUB, etc.)
            role: Role - "server" or "client"
        """
        try:
            if role == "server":
                # Server configuration
                server_secret = self.secrets_manager.get_secret(
                    "system", "zmq_server_secret"
                )
                if isinstance(server_secret, str):
                    import base64

                    server_secret = base64.b64decode(server_secret.encode())

                socket.curve_secretkey = server_secret
                socket.curve_server = True

                self.audit_logger.log_security_event(
                    "ZMQ_SOCKET_CONFIGURED_AS_SERVER",
                    {"socket_type": socket_type, "curve_enabled": True},
                )

            else:  # client
                # Client configuration
                client_public = self.secrets_manager.get_secret(
                    "system", "zmq_client_public"
                )
                client_secret = self.secrets_manager.get_secret(
                    "system", "zmq_client_secret"
                )
                server_public = self.secrets_manager.get_secret(
                    "system", "zmq_server_public"
                )

                if isinstance(client_public, str):
                    import base64

                    client_public = base64.b64decode(client_public.encode())
                    client_secret = base64.b64decode(client_secret.encode())
                    server_public = base64.b64decode(server_public.encode())

                socket.curve_publickey = client_public
                socket.curve_secretkey = client_secret
                socket.curve_serverkey = server_public

                self.audit_logger.log_security_event(
                    "ZMQ_SOCKET_CONFIGURED_AS_CLIENT",
                    {"socket_type": socket_type, "curve_enabled": True},
                )

        except Exception as e:
            self.audit_logger.log_security_event(
                "ZMQ_SOCKET_CONFIGURATION_FAILED",
                {"socket_type": socket_type, "role": role, "error": str(e)},
            )
            raise

    def generate_self_signed_cert(
        self, hostname: str = "localhost", days: int = 365
    ) -> tuple[str, str]:
        """
        Generate a self-signed certificate for development/testing.

        Args:
            hostname: Hostname for the certificate
            days: Certificate validity in days

        Returns:
            Tuple of (cert_path, key_path)
        """
        cert_file = self.certs_dir / f"{hostname}.crt"
        key_file = self.certs_dir / f"{hostname}.key"

        try:
            # Generate private key and certificate using OpenSSL
            cmd = [
                "openssl",
                "req",
                "-x509",
                "-newkey",
                "rsa:4096",
                "-keyout",
                str(key_file),
                "-out",
                str(cert_file),
                "-days",
                str(days),
                "-nodes",  # No passphrase
                "-subj",
                f"/C=US/ST=State/L=City/O=FX-Quant/OU=Trading/CN={hostname}",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            self.audit_logger.log_security_event(
                "SELF_SIGNED_CERT_GENERATED",
                {
                    "hostname": hostname,
                    "days": days,
                    "cert_file": str(cert_file),
                    "key_file": str(key_file),
                },
            )

            # Store certificate paths in secrets
            original_role = self.secrets_manager.current_role
            self.secrets_manager.set_user_context("tls_manager", "admin")

            try:
                self.secrets_manager.store_secret(
                    "system", "tls_cert_path", str(cert_file)
                )
                self.secrets_manager.store_secret(
                    "system", "tls_key_path", str(key_file)
                )
            finally:
                self.secrets_manager.current_role = original_role

            return str(cert_file), str(key_file)

        except subprocess.CalledProcessError as e:
            self.audit_logger.log_security_event(
                "CERT_GENERATION_FAILED", {"hostname": hostname, "error": e.stderr}
            )
            raise
        except FileNotFoundError:
            # OpenSSL not available, use Python cryptography library
            return self._generate_cert_with_cryptography(hostname, days)

    def _generate_cert_with_cryptography(
        self, hostname: str, days: int
    ) -> tuple[str, str]:
        """Generate certificate using Python cryptography library."""
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID

        # Generate private key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=4096,
        )

        # Create certificate
        subject = issuer = x509.Name(
            [
                x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
                x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "State"),
                x509.NameAttribute(NameOID.LOCALITY_NAME, "City"),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, "FX-Quant"),
                x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, "Trading"),
                x509.NameAttribute(NameOID.COMMON_NAME, hostname),
            ]
        )

        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(private_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.utcnow())
            .not_valid_after(datetime.utcnow() + timedelta(days=days))
            .add_extension(
                x509.SubjectAlternativeName(
                    [
                        x509.DNSName(hostname),
                        x509.DNSName("localhost"),
                        x509.IPAddress(socket.inet_aton("127.0.0.1")),
                    ]
                ),
                critical=False,
            )
            .sign(private_key, hashes.SHA256())
        )

        # Write certificate and key files
        cert_file = self.certs_dir / f"{hostname}.crt"
        key_file = self.certs_dir / f"{hostname}.key"

        with open(cert_file, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))

        with open(key_file, "wb") as f:
            f.write(
                private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption(),
                )
            )

        self.audit_logger.log_security_event(
            "CERT_GENERATED_WITH_CRYPTOGRAPHY",
            {
                "hostname": hostname,
                "days": days,
                "cert_file": str(cert_file),
                "key_file": str(key_file),
            },
        )

        return str(cert_file), str(key_file)

    def create_ssl_context(
        self, cert_file: str | None = None, key_file: str | None = None
    ) -> ssl.SSLContext:
        """
        Create an SSL context for secure connections.

        Args:
            cert_file: Path to certificate file
            key_file: Path to private key file

        Returns:
            Configured SSL context
        """
        try:
            # Create SSL context
            context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)

            # Configure security settings
            context.check_hostname = False  # For self-signed certificates
            context.verify_mode = ssl.CERT_NONE  # For development

            # Use strong ciphers
            context.set_ciphers(
                "ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS"
            )

            # Set minimum TLS version
            context.minimum_version = ssl.TLSVersion.TLSv1_2

            if cert_file and key_file:
                context.load_cert_chain(cert_file, key_file)

            self.audit_logger.log_security_event(
                "SSL_CONTEXT_CREATED",
                {
                    "cert_file": cert_file,
                    "key_file": key_file,
                    "min_tls_version": "1.2",
                },
            )

            return context

        except Exception as e:
            self.audit_logger.log_security_event(
                "SSL_CONTEXT_CREATION_FAILED",
                {"cert_file": cert_file, "key_file": key_file, "error": str(e)},
            )
            raise

    def get_redis_tls_config(self) -> dict[str, Any]:
        """Get Redis TLS configuration."""
        try:
            # Get certificate paths
            cert_file = self.secrets_manager.get_secret("system", "tls_cert_path")
            key_file = self.secrets_manager.get_secret("system", "tls_key_path")

            if not cert_file or not key_file:
                # Generate certificates if they don't exist
                cert_file, key_file = self.generate_self_signed_cert("redis-server")

            ssl_config = {
                "ssl_certfile": cert_file,
                "ssl_keyfile": key_file,
                "ssl_check_hostname": False,
                "ssl_cert_reqs": ssl.CERT_NONE,
                "ssl_ca_certs": None,
            }

            self.audit_logger.log_security_event(
                "REDIS_TLS_CONFIG_GENERATED",
                {"cert_file": cert_file, "ssl_enabled": True},
            )

            return ssl_config

        except Exception as e:
            self.audit_logger.log_security_event(
                "REDIS_TLS_CONFIG_FAILED", {"error": str(e)}
            )
            raise

    def validate_certificate(self, cert_path: str) -> dict[str, Any]:
        """
        Validate a certificate and return its information.

        Args:
            cert_path: Path to the certificate file

        Returns:
            Certificate information dictionary
        """
        try:
            from cryptography import x509

            with open(cert_path, "rb") as f:
                cert_data = f.read()

            cert = x509.load_pem_x509_certificate(cert_data)

            # Extract certificate information
            info = {
                "subject": cert.subject.rfc4514_string(),
                "issuer": cert.issuer.rfc4514_string(),
                "serial_number": str(cert.serial_number),
                "not_valid_before": cert.not_valid_before.isoformat(),
                "not_valid_after": cert.not_valid_after.isoformat(),
                "is_valid": datetime.utcnow() < cert.not_valid_after,
                "signature_algorithm": cert.signature_algorithm_oid._name,
                "public_key_size": cert.public_key().key_size
                if hasattr(cert.public_key(), "key_size")
                else None,
            }

            # Check for SAN extension
            try:
                san_ext = cert.extensions.get_extension_for_oid(
                    x509.oid.ExtensionOID.SUBJECT_ALTERNATIVE_NAME
                )
                info["subject_alt_names"] = [name.value for name in san_ext.value]
            except x509.ExtensionNotFound:
                info["subject_alt_names"] = []

            self.audit_logger.log_security_event(
                "CERTIFICATE_VALIDATED",
                {
                    "cert_path": cert_path,
                    "is_valid": info["is_valid"],
                    "expires": info["not_valid_after"],
                },
            )

            return info

        except Exception as e:
            self.audit_logger.log_security_event(
                "CERTIFICATE_VALIDATION_FAILED",
                {"cert_path": cert_path, "error": str(e)},
            )
            raise

    def cleanup(self) -> None:
        """Clean up resources."""
        if self._zmq_authenticator:
            self._zmq_authenticator.stop()
            self._zmq_authenticator = None

            self.audit_logger.log_security_event("ZMQ_AUTHENTICATOR_STOPPED", {})


# Global instance
_tls_manager = None


def get_tls_manager() -> TLSManager:
    """Get the global TLS manager instance."""
    global _tls_manager
    if _tls_manager is None:
        _tls_manager = TLSManager()
    return _tls_manager
