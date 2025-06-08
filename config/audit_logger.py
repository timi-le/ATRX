"""
Security Audit Logger for FX AI-Quant Trading System

This module provides comprehensive audit logging for security events,
configuration access, and potential security breaches.

Features:
- Structured logging with timestamps
- Security event categorization
- Breach detection patterns
- Log file rotation and integrity
- Real-time alerting capabilities
"""

import hashlib
import json
import logging
import os
import threading
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import structlog


class AuditLogger:
    """
    Comprehensive audit logging system for security events and configuration access.

    Provides structured logging with security event categorization, breach detection,
    and real-time monitoring capabilities.
    """

    def __init__(self, log_dir: str = "logs", max_file_size: int = 10 * 1024 * 1024):
        self.log_dir = Path(log_dir)
        self.max_file_size = max_file_size
        self.lock = threading.Lock()

        # Create logs directory
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize structured logger
        self._setup_structured_logging()

        # Security event patterns for breach detection
        self.breach_patterns = {
            "MULTIPLE_FAILED_DECRYPTION": {
                "threshold": 5,
                "window": 300,
            },  # 5 failures in 5 minutes
            "RAPID_SECRET_ACCESS": {
                "threshold": 50,
                "window": 60,
            },  # 50 accesses in 1 minute
            "UNAUTHORIZED_CONFIG_CHANGE": {
                "threshold": 1,
                "window": 1,
            },  # Immediate alert
        }

        # Recent events for pattern matching
        self.recent_events = []

        # Alert callbacks
        self.alert_callbacks = []

    def _setup_structured_logging(self):
        """Setup structured logging with multiple handlers."""

        # Configure structlog
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="ISO"),
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

        # Security audit logger
        self.security_logger = self._create_file_logger(
            "security_audit.log", "security_audit"
        )

        # Config access logger
        self.config_logger = self._create_file_logger(
            "config_access.log", "config_access"
        )

        # Breach detection logger
        self.breach_logger = self._create_file_logger(
            "security_breaches.log", "security_breaches"
        )

    def _create_file_logger(self, filename: str, logger_name: str) -> logging.Logger:
        """Create a rotating file logger."""
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)

        # Avoid duplicate handlers
        if logger.handlers:
            return logger

        # Rotating file handler
        handler = RotatingFileHandler(
            self.log_dir / filename,
            maxBytes=self.max_file_size,
            backupCount=10,
            encoding="utf-8",
        )

        # JSON formatter for structured logs
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)

        logger.addHandler(handler)
        return logger

    def log_security_event(self, event_type: str, details: dict[str, Any]) -> None:
        """
        Log a security event with structured data.

        Args:
            event_type: Type of security event (e.g., "ENCRYPTION_KEY_DERIVED")
            details: Additional event details
        """
        with self.lock:
            event = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": event_type,
                "details": details,
                "session_id": self._get_session_id(),
                "process_id": os.getpid(),
                "thread_id": threading.get_ident(),
            }

            # Add integrity hash
            event["integrity_hash"] = self._calculate_event_hash(event)

            # Log the event
            self.security_logger.info(json.dumps(event))

            # Store for breach detection
            self.recent_events.append(event)
            self._cleanup_old_events()

            # Check for breach patterns
            self._check_breach_patterns(event)

    def log_config_access(self, access_type: str, details: dict[str, Any]) -> None:
        """
        Log configuration access events.

        Args:
            access_type: Type of access (e.g., "SECRET_ACCESSED")
            details: Access details
        """
        with self.lock:
            event = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "access_type": access_type,
                "details": details,
                "session_id": self._get_session_id(),
                "process_id": os.getpid(),
                "thread_id": threading.get_ident(),
                "caller_info": self._get_caller_info(),
            }

            # Add integrity hash
            event["integrity_hash"] = self._calculate_event_hash(event)

            # Log the event
            self.config_logger.info(json.dumps(event))

            # Store for breach detection
            self.recent_events.append(event)
            self._cleanup_old_events()

            # Check for suspicious patterns
            self._check_config_access_patterns(event)

    def log_breach_attempt(
        self, breach_type: str, details: dict[str, Any], severity: str = "HIGH"
    ) -> None:
        """
        Log a security breach attempt.

        Args:
            breach_type: Type of breach detected
            details: Breach details
            severity: Severity level (LOW, MEDIUM, HIGH, CRITICAL)
        """
        with self.lock:
            breach_event = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "breach_type": breach_type,
                "severity": severity,
                "details": details,
                "session_id": self._get_session_id(),
                "process_id": os.getpid(),
                "thread_id": threading.get_ident(),
                "system_info": self._get_system_info(),
            }

            # Add integrity hash
            breach_event["integrity_hash"] = self._calculate_event_hash(breach_event)

            # Log the breach
            self.breach_logger.error(json.dumps(breach_event))

            # Trigger alerts
            self._trigger_alerts(breach_event)

    def _get_session_id(self) -> str:
        """Generate a unique session identifier."""
        if not hasattr(self, "_session_id"):
            import uuid

            self._session_id = str(uuid.uuid4())[:8]
        return self._session_id

    def _get_caller_info(self) -> dict[str, str]:
        """Get information about the calling code."""
        import inspect

        frame = inspect.currentframe()
        try:
            # Go up the stack to find the actual caller
            for _ in range(3):  # Skip this method and the logging method
                frame = frame.f_back
                if frame is None:
                    break

            if frame:
                return {
                    "file": frame.f_code.co_filename,
                    "function": frame.f_code.co_name,
                    "line": frame.f_lineno,
                }
        finally:
            del frame

        return {"file": "unknown", "function": "unknown", "line": 0}

    def _get_system_info(self) -> dict[str, Any]:
        """Get system information for breach context."""
        import platform

        return {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "hostname": platform.node(),
            "user": os.getenv("USERNAME", "unknown"),
            "pwd": os.getcwd(),
        }

    def _calculate_event_hash(self, event: dict[str, Any]) -> str:
        """Calculate integrity hash for an event."""
        # Create a copy without the hash field
        event_copy = {k: v for k, v in event.items() if k != "integrity_hash"}
        event_str = json.dumps(event_copy, sort_keys=True)
        return hashlib.sha256(event_str.encode()).hexdigest()[:16]

    def _cleanup_old_events(self) -> None:
        """Clean up old events to prevent memory bloat."""
        current_time = datetime.now(timezone.utc)
        max_age = 3600  # Keep events for 1 hour

        self.recent_events = [
            event
            for event in self.recent_events
            if (
                current_time
                - datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00"))
            ).total_seconds()
            < max_age
        ]

    def _check_breach_patterns(self, event: dict[str, Any]) -> None:
        """Check for security breach patterns."""
        current_time = datetime.now(timezone.utc)

        for pattern_name, pattern_config in self.breach_patterns.items():
            # Count matching events in the time window
            matching_events = 0
            window_start = current_time.timestamp() - pattern_config["window"]

            for recent_event in self.recent_events:
                event_time = datetime.fromisoformat(
                    recent_event["timestamp"].replace("Z", "+00:00")
                ).timestamp()
                if event_time >= window_start:
                    if self._event_matches_pattern(recent_event, pattern_name):
                        matching_events += 1

            # Check if threshold exceeded
            if matching_events >= pattern_config["threshold"]:
                self.log_breach_attempt(
                    pattern_name,
                    {
                        "matching_events": matching_events,
                        "threshold": pattern_config["threshold"],
                        "window_seconds": pattern_config["window"],
                        "latest_event": event,
                    },
                    "HIGH",
                )

    def _event_matches_pattern(self, event: dict[str, Any], pattern_name: str) -> bool:
        """Check if an event matches a specific breach pattern."""
        if pattern_name == "MULTIPLE_FAILED_DECRYPTION":
            return event.get("event_type") in [
                "SECRETS_LOAD_FAILED",
                "SECRET_ACCESS_FAILED",
            ]
        elif pattern_name == "RAPID_SECRET_ACCESS":
            return event.get("access_type") == "SECRET_ACCESSED"
        elif pattern_name == "UNAUTHORIZED_CONFIG_CHANGE":
            return event.get("access_type") in ["SECRET_STORED", "SECRET_DELETED"]

        return False

    def _check_config_access_patterns(self, event: dict[str, Any]) -> None:
        """Check for suspicious configuration access patterns."""
        # Example: Check for access to sensitive scopes
        if event.get("details", {}).get("scope") in ["api", "database"]:
            caller_info = event.get("caller_info", {})
            if "test" in caller_info.get("file", "").lower():
                # Potential test accessing production secrets
                self.log_breach_attempt(
                    "TEST_ACCESSING_PRODUCTION_SECRETS",
                    {"caller": caller_info, "scope": event["details"]["scope"]},
                    "MEDIUM",
                )

    def _trigger_alerts(self, breach_event: dict[str, Any]) -> None:
        """Trigger real-time alerts for security breaches."""
        for callback in self.alert_callbacks:
            try:
                callback(breach_event)
            except Exception as e:
                # Log alert failure but don't raise
                self.security_logger.error(f"Alert callback failed: {e}")

    def add_alert_callback(self, callback) -> None:
        """Add a callback function for breach alerts."""
        self.alert_callbacks.append(callback)

    def get_recent_events(
        self, event_type: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        """
        Get recent events for monitoring.

        Args:
            event_type: Filter by event type (optional)
            limit: Maximum number of events to return

        Returns:
            List of recent events
        """
        with self.lock:
            events = self.recent_events[-limit:]

            if event_type:
                events = [
                    event
                    for event in events
                    if event.get("event_type") == event_type
                    or event.get("access_type") == event_type
                ]

            return events

    def generate_security_report(self) -> dict[str, Any]:
        """Generate a comprehensive security report."""
        current_time = datetime.now(timezone.utc)

        # Count events by type in the last 24 hours
        day_ago = current_time.timestamp() - 86400
        recent_day_events = [
            event
            for event in self.recent_events
            if datetime.fromisoformat(
                event["timestamp"].replace("Z", "+00:00")
            ).timestamp()
            >= day_ago
        ]

        event_counts = {}
        access_counts = {}

        for event in recent_day_events:
            event_type = event.get("event_type")
            access_type = event.get("access_type")

            if event_type:
                event_counts[event_type] = event_counts.get(event_type, 0) + 1
            if access_type:
                access_counts[access_type] = access_counts.get(access_type, 0) + 1

        return {
            "report_timestamp": current_time.isoformat(),
            "period": "24_hours",
            "total_events": len(recent_day_events),
            "event_type_counts": event_counts,
            "access_type_counts": access_counts,
            "log_files": {
                "security_audit": str(self.log_dir / "security_audit.log"),
                "config_access": str(self.log_dir / "config_access.log"),
                "security_breaches": str(self.log_dir / "security_breaches.log"),
            },
        }


# Global instance
_audit_logger = None


def get_audit_logger() -> AuditLogger:
    """Get the global audit logger instance."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger
