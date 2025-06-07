"""
FIX Client - Core FIX Protocol Session Management.

This module provides the main FIX client implementation for Interactive Brokers
integration, handling session establishment, authentication, heartbeat management,
and message processing.

Features:
- SSL/TLS FIX session management
- Automatic reconnection with exponential backoff
- Heartbeat and test request handling
- Sequence number management
- Message persistence and recovery
- Comprehensive error handling and logging
"""

import asyncio
import ssl
import socket
import time
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import yaml
import structlog

from .fix_message_utils import FIXMessageUtils, FIXMessage, FIXVersion


class FIXSessionState(Enum):
    """FIX session states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    LOGON_SENT = "logon_sent"
    LOGGED_ON = "logged_on"
    LOGOUT_SENT = "logout_sent"
    RECONNECTING = "reconnecting"
    ERROR = "error"


@dataclass
class FIXSessionConfig:
    """FIX session configuration."""
    # Connection settings
    host: str
    port: int
    use_ssl: bool = True
    ssl_verify: bool = True
    
    # FIX protocol settings
    fix_version: str = "FIX.4.2"
    sender_comp_id: str = "CLIENT"
    target_comp_id: str = "IB"
    sender_sub_id: Optional[str] = None
    target_sub_id: Optional[str] = None
    
    # Authentication
    username: Optional[str] = None
    password: Optional[str] = None
    
    # Session management
    heartbeat_interval: int = 30
    logon_timeout: int = 30
    logout_timeout: int = 10
    
    # Sequence numbers
    reset_on_logon: bool = False
    reset_on_logout: bool = False
    reset_on_disconnect: bool = False
    
    # Reconnection
    auto_reconnect: bool = True
    max_reconnect_attempts: int = 10
    reconnect_interval: int = 5
    reconnect_backoff: float = 2.0
    max_reconnect_interval: int = 300
    
    # Message validation
    validate_length: bool = True
    validate_checksum: bool = True
    validate_sequence: bool = True
    
    # Logging and persistence
    log_incoming: bool = True
    log_outgoing: bool = True
    persist_messages: bool = True
    message_store_path: str = "data/fix_messages"
    
    # Performance
    send_buffer_size: int = 8192
    recv_buffer_size: int = 8192
    socket_timeout: int = 30


@dataclass
class FIXSessionStats:
    """FIX session statistics."""
    session_start_time: Optional[datetime] = None
    total_messages_sent: int = 0
    total_messages_received: int = 0
    heartbeats_sent: int = 0
    heartbeats_received: int = 0
    test_requests_sent: int = 0
    test_requests_received: int = 0
    reconnection_attempts: int = 0
    last_heartbeat_sent: Optional[datetime] = None
    last_heartbeat_received: Optional[datetime] = None
    last_message_sent: Optional[datetime] = None
    last_message_received: Optional[datetime] = None


class FIXClient:
    """
    FIX Client for Interactive Brokers integration.
    
    Provides comprehensive FIX protocol session management including connection
    establishment, authentication, heartbeat handling, and message processing.
    """
    
    def __init__(
        self,
        config: FIXSessionConfig,
        logger: Optional[structlog.stdlib.BoundLogger] = None
    ):
        """Initialize FIX client."""
        self.config = config
        self.logger = logger or structlog.get_logger(__name__)
        
        # Session state
        self.state = FIXSessionState.DISCONNECTED
        self.socket: Optional[socket.socket] = None
        self.ssl_socket: Optional[ssl.SSLSocket] = None
        
        # Sequence numbers
        self.outgoing_seq_num = 1
        self.incoming_seq_num = 1
        self.expected_seq_num = 1
        
        # Message utilities
        fix_version = FIXVersion.FIX_4_2 if config.fix_version == "FIX.4.2" else FIXVersion.FIX_4_4
        self.message_utils = FIXMessageUtils(fix_version)
        
        # Session management
        self.session_id = None
        self.logon_time = None
        self.last_heartbeat_time = None
        self.last_test_request_time = None
        self.pending_test_request_id = None
        
        # Threading and async
        self._running = False
        self._heartbeat_task = None
        self._receive_task = None
        self._reconnect_task = None
        self._lock = threading.Lock()
        
        # Message handlers
        self.message_handlers: Dict[str, Callable[[FIXMessage], None]] = {}
        self.admin_message_handlers: Dict[str, Callable[[FIXMessage], None]] = {}
        
        # Statistics
        self.stats = FIXSessionStats()
        
        # Message persistence
        self.message_store_path = Path(config.message_store_path)
        if config.persist_messages:
            self.message_store_path.mkdir(parents=True, exist_ok=True)
        
        # Setup default admin message handlers
        self._setup_admin_handlers()
        
        self.logger.info(
            "FIX client initialized",
            sender_comp_id=config.sender_comp_id,
            target_comp_id=config.target_comp_id,
            host=config.host,
            port=config.port
        )
    
    def _setup_admin_handlers(self):
        """Setup default administrative message handlers."""
        self.admin_message_handlers = {
            "0": self._handle_heartbeat,
            "1": self._handle_test_request,
            "2": self._handle_resend_request,
            "3": self._handle_reject,
            "4": self._handle_sequence_reset,
            "5": self._handle_logout,
            "A": self._handle_logon,
        }
    
    async def connect(self) -> bool:
        """Connect to FIX server and establish session."""
        try:
            self.logger.info("Connecting to FIX server", host=self.config.host, port=self.config.port)
            self.state = FIXSessionState.CONNECTING
            
            # Create socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.config.socket_timeout)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, self.config.send_buffer_size)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.config.recv_buffer_size)
            
            # Connect to server
            await asyncio.get_event_loop().run_in_executor(
                None, self.socket.connect, (self.config.host, self.config.port)
            )
            
            # Setup SSL if required
            if self.config.use_ssl:
                ssl_context = ssl.create_default_context()
                if not self.config.ssl_verify:
                    ssl_context.check_hostname = False
                    ssl_context.verify_mode = ssl.CERT_NONE
                
                self.ssl_socket = ssl_context.wrap_socket(
                    self.socket,
                    server_hostname=self.config.host if self.config.ssl_verify else None
                )
            
            self.logger.info("Socket connected successfully")
            
            # Start background tasks
            self._running = True
            self._receive_task = asyncio.create_task(self._receive_loop())
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            
            # Send logon message
            await self._send_logon()
            
            # Wait for logon response
            logon_timeout = time.time() + self.config.logon_timeout
            while self.state == FIXSessionState.LOGON_SENT and time.time() < logon_timeout:
                await asyncio.sleep(0.1)
            
            if self.state == FIXSessionState.LOGGED_ON:
                self.stats.session_start_time = datetime.now(timezone.utc)
                self.logger.info("FIX session established successfully")
                return True
            else:
                self.logger.error("Logon failed or timed out", state=self.state.value)
                await self.disconnect()
                return False
                
        except Exception as e:
            self.logger.error("Connection failed", error=str(e))
            self.state = FIXSessionState.ERROR
            await self.disconnect()
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from FIX server."""
        try:
            self.logger.info("Disconnecting from FIX server")
            
            # Send logout if logged on
            if self.state == FIXSessionState.LOGGED_ON:
                await self._send_logout()
                
                # Wait for logout response
                logout_timeout = time.time() + self.config.logout_timeout
                while self.state == FIXSessionState.LOGOUT_SENT and time.time() < logout_timeout:
                    await asyncio.sleep(0.1)
            
            # Stop background tasks
            self._running = False
            
            if self._receive_task:
                self._receive_task.cancel()
            if self._heartbeat_task:
                self._heartbeat_task.cancel()
            if self._reconnect_task:
                self._reconnect_task.cancel()
            
            # Close sockets
            if self.ssl_socket:
                self.ssl_socket.close()
                self.ssl_socket = None
            
            if self.socket:
                self.socket.close()
                self.socket = None
            
            self.state = FIXSessionState.DISCONNECTED
            self.logger.info("FIX session disconnected")
            
        except Exception as e:
            self.logger.error("Error during disconnect", error=str(e))
    
    async def send_message(self, message: str) -> bool:
        """Send a FIX message."""
        try:
            if self.state not in [FIXSessionState.LOGGED_ON, FIXSessionState.LOGON_SENT]:
                self.logger.warning("Cannot send message - not logged on", state=self.state.value)
                return False
            
            # Get active socket
            active_socket = self.ssl_socket if self.ssl_socket else self.socket
            if not active_socket:
                self.logger.error("No active socket connection")
                return False
            
            # Send message
            message_bytes = message.encode('utf-8')
            await asyncio.get_event_loop().run_in_executor(
                None, active_socket.send, message_bytes
            )
            
            # Update statistics
            self.stats.total_messages_sent += 1
            self.stats.last_message_sent = datetime.now(timezone.utc)
            
            # Log outgoing message
            if self.config.log_outgoing:
                self.logger.debug("Sent FIX message", message=message.replace('\x01', '|'))
            
            # Persist message
            if self.config.persist_messages:
                await self._persist_message(message, "outgoing")
            
            return True
            
        except Exception as e:
            self.logger.error("Error sending message", error=str(e))
            return False
    
    async def _receive_loop(self):
        """Background task to receive messages."""
        buffer = b""
        
        while self._running:
            try:
                # Get active socket
                active_socket = self.ssl_socket if self.ssl_socket else self.socket
                if not active_socket:
                    await asyncio.sleep(0.1)
                    continue
                
                # Receive data
                data = await asyncio.get_event_loop().run_in_executor(
                    None, active_socket.recv, 4096
                )
                
                if not data:
                    self.logger.warning("Connection closed by server")
                    break
                
                buffer += data
                
                # Process complete messages
                while b'\x01' in buffer:
                    # Find message boundary (SOH after checksum)
                    soh_pos = buffer.find(b'\x01')
                    if soh_pos == -1:
                        break
                    
                    # Look for checksum field (10=xxx)
                    checksum_start = buffer.find(b'10=')
                    if checksum_start == -1 or checksum_start > soh_pos:
                        # Need more data
                        break
                    
                    # Find end of checksum field
                    checksum_end = buffer.find(b'\x01', checksum_start)
                    if checksum_end == -1:
                        # Need more data
                        break
                    
                    # Extract complete message
                    message_bytes = buffer[:checksum_end + 1]
                    buffer = buffer[checksum_end + 1:]
                    
                    # Process message
                    message_str = message_bytes.decode('utf-8')
                    await self._process_received_message(message_str)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Error in receive loop", error=str(e))
                if self.config.auto_reconnect:
                    await self._schedule_reconnect()
                break
    
    async def _process_received_message(self, message_str: str):
        """Process a received FIX message."""
        try:
            # Parse message
            message = self.message_utils.parse_message(message_str)
            
            # Update statistics
            self.stats.total_messages_received += 1
            self.stats.last_message_received = datetime.now(timezone.utc)
            
            # Log incoming message
            if self.config.log_incoming:
                self.logger.debug("Received FIX message", message=message_str.replace('\x01', '|'))
            
            # Persist message
            if self.config.persist_messages:
                await self._persist_message(message_str, "incoming")
            
            # Validate sequence number
            if self.config.validate_sequence and message.sequence_number:
                if message.sequence_number != self.expected_seq_num:
                    self.logger.warning(
                        "Sequence number gap detected",
                        expected=self.expected_seq_num,
                        received=message.sequence_number
                    )
                    # TODO: Send resend request
                
                self.expected_seq_num = message.sequence_number + 1
            
            # Route message to appropriate handler
            if self.message_utils.is_admin_message(message.msg_type):
                handler = self.admin_message_handlers.get(message.msg_type)
                if handler:
                    await asyncio.get_event_loop().run_in_executor(None, handler, message)
                else:
                    self.logger.warning("No handler for admin message", msg_type=message.msg_type)
            else:
                handler = self.message_handlers.get(message.msg_type)
                if handler:
                    await asyncio.get_event_loop().run_in_executor(None, handler, message)
                else:
                    self.logger.warning("No handler for application message", msg_type=message.msg_type)
            
        except Exception as e:
            self.logger.error("Error processing received message", error=str(e))
    
    async def _heartbeat_loop(self):
        """Background task to send heartbeats."""
        while self._running:
            try:
                await asyncio.sleep(self.config.heartbeat_interval)
                
                if self.state == FIXSessionState.LOGGED_ON:
                    # Check if we need to send a heartbeat
                    now = time.time()
                    if (not self.last_heartbeat_time or 
                        now - self.last_heartbeat_time >= self.config.heartbeat_interval):
                        
                        await self._send_heartbeat()
                    
                    # Check for missed heartbeats
                    if (self.stats.last_message_received and
                        now - self.stats.last_message_received.timestamp() > self.config.heartbeat_interval * 2):
                        
                        self.logger.warning("Missed heartbeat from server")
                        await self._send_test_request()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Error in heartbeat loop", error=str(e))
    
    async def _send_logon(self):
        """Send logon message."""
        try:
            self.state = FIXSessionState.LOGON_SENT
            
            message = self.message_utils.build_logon_message(
                sender_comp_id=self.config.sender_comp_id,
                target_comp_id=self.config.target_comp_id,
                sequence_number=self.outgoing_seq_num,
                heartbeat_interval=self.config.heartbeat_interval,
                username=self.config.username,
                password=self.config.password,
                reset_seq_num=self.config.reset_on_logon,
                sender_sub_id=self.config.sender_sub_id,
                target_sub_id=self.config.target_sub_id
            )
            
            await self.send_message(message)
            self.outgoing_seq_num += 1
            
            self.logger.info("Logon message sent")
            
        except Exception as e:
            self.logger.error("Error sending logon", error=str(e))
            self.state = FIXSessionState.ERROR
    
    async def _send_logout(self, text: Optional[str] = None):
        """Send logout message."""
        try:
            self.state = FIXSessionState.LOGOUT_SENT
            
            message = self.message_utils.build_logout_message(
                sender_comp_id=self.config.sender_comp_id,
                target_comp_id=self.config.target_comp_id,
                sequence_number=self.outgoing_seq_num,
                text=text,
                sender_sub_id=self.config.sender_sub_id,
                target_sub_id=self.config.target_sub_id
            )
            
            await self.send_message(message)
            self.outgoing_seq_num += 1
            
            self.logger.info("Logout message sent")
            
        except Exception as e:
            self.logger.error("Error sending logout", error=str(e))
    
    async def _send_heartbeat(self, test_req_id: Optional[str] = None):
        """Send heartbeat message."""
        try:
            message = self.message_utils.build_heartbeat_message(
                sender_comp_id=self.config.sender_comp_id,
                target_comp_id=self.config.target_comp_id,
                sequence_number=self.outgoing_seq_num,
                test_req_id=test_req_id,
                sender_sub_id=self.config.sender_sub_id,
                target_sub_id=self.config.target_sub_id
            )
            
            await self.send_message(message)
            self.outgoing_seq_num += 1
            self.last_heartbeat_time = time.time()
            self.stats.heartbeats_sent += 1
            
        except Exception as e:
            self.logger.error("Error sending heartbeat", error=str(e))
    
    async def _send_test_request(self):
        """Send test request message."""
        try:
            test_req_id = f"TEST-{int(time.time())}"
            self.pending_test_request_id = test_req_id
            self.last_test_request_time = time.time()
            
            message = self.message_utils.build_test_request_message(
                sender_comp_id=self.config.sender_comp_id,
                target_comp_id=self.config.target_comp_id,
                sequence_number=self.outgoing_seq_num,
                test_req_id=test_req_id,
                sender_sub_id=self.config.sender_sub_id,
                target_sub_id=self.config.target_sub_id
            )
            
            await self.send_message(message)
            self.outgoing_seq_num += 1
            self.stats.test_requests_sent += 1
            
            self.logger.debug("Test request sent", test_req_id=test_req_id)
            
        except Exception as e:
            self.logger.error("Error sending test request", error=str(e))
    
    # Administrative message handlers
    def _handle_logon(self, message: FIXMessage):
        """Handle logon response."""
        self.logger.info("Received logon response")
        self.state = FIXSessionState.LOGGED_ON
        self.logon_time = datetime.now(timezone.utc)
    
    def _handle_logout(self, message: FIXMessage):
        """Handle logout message."""
        text = message.fields.get(58, "")
        self.logger.info("Received logout", text=text)
        self.state = FIXSessionState.DISCONNECTED
    
    def _handle_heartbeat(self, message: FIXMessage):
        """Handle heartbeat message."""
        self.stats.heartbeats_received += 1
        self.stats.last_heartbeat_received = datetime.now(timezone.utc)
        
        # Check if this is a response to our test request
        test_req_id = message.fields.get(112)
        if test_req_id and test_req_id == self.pending_test_request_id:
            self.pending_test_request_id = None
            self.logger.debug("Test request response received", test_req_id=test_req_id)
    
    def _handle_test_request(self, message: FIXMessage):
        """Handle test request message."""
        test_req_id = message.fields.get(112)
        self.stats.test_requests_received += 1
        
        # Send heartbeat response
        asyncio.create_task(self._send_heartbeat(test_req_id))
        
        self.logger.debug("Test request received, sending heartbeat", test_req_id=test_req_id)
    
    def _handle_resend_request(self, message: FIXMessage):
        """Handle resend request message."""
        begin_seq_no = message.fields.get(7)
        end_seq_no = message.fields.get(16, "0")
        
        self.logger.warning(
            "Resend request received",
            begin_seq_no=begin_seq_no,
            end_seq_no=end_seq_no
        )
        
        # TODO: Implement message resend logic
    
    def _handle_reject(self, message: FIXMessage):
        """Handle reject message."""
        ref_seq_num = message.fields.get(45)
        text = message.fields.get(58, "")
        
        self.logger.error("Message rejected", ref_seq_num=ref_seq_num, text=text)
    
    def _handle_sequence_reset(self, message: FIXMessage):
        """Handle sequence reset message."""
        new_seq_no = message.fields.get(36)
        gap_fill_flag = message.fields.get(123, "N")
        
        if new_seq_no:
            self.expected_seq_num = int(new_seq_no)
            self.logger.info(
                "Sequence reset received",
                new_seq_no=new_seq_no,
                gap_fill=gap_fill_flag == "Y"
            )
    
    async def _persist_message(self, message: str, direction: str):
        """Persist message to disk."""
        try:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{direction}_{timestamp}.fix"
            filepath = self.message_store_path / filename
            
            with open(filepath, 'w') as f:
                f.write(message)
                
        except Exception as e:
            self.logger.error("Error persisting message", error=str(e))
    
    async def _schedule_reconnect(self):
        """Schedule reconnection attempt."""
        if not self.config.auto_reconnect:
            return
        
        if self.stats.reconnection_attempts >= self.config.max_reconnect_attempts:
            self.logger.error("Max reconnection attempts reached")
            self.state = FIXSessionState.ERROR
            return
        
        self.state = FIXSessionState.RECONNECTING
        self.stats.reconnection_attempts += 1
        
        # Calculate backoff delay
        delay = min(
            self.config.reconnect_interval * (self.config.reconnect_backoff ** (self.stats.reconnection_attempts - 1)),
            self.config.max_reconnect_interval
        )
        
        self.logger.info(
            "Scheduling reconnection",
            attempt=self.stats.reconnection_attempts,
            delay=delay
        )
        
        self._reconnect_task = asyncio.create_task(self._reconnect_after_delay(delay))
    
    async def _reconnect_after_delay(self, delay: float):
        """Reconnect after delay."""
        try:
            await asyncio.sleep(delay)
            
            # Cleanup current connection
            await self.disconnect()
            
            # Reset sequence numbers if configured
            if self.config.reset_on_disconnect:
                self.outgoing_seq_num = 1
                self.incoming_seq_num = 1
                self.expected_seq_num = 1
            
            # Attempt reconnection
            success = await self.connect()
            if success:
                self.stats.reconnection_attempts = 0
                self.logger.info("Reconnection successful")
            else:
                await self._schedule_reconnect()
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error("Error during reconnection", error=str(e))
            await self._schedule_reconnect()
    
    def register_message_handler(self, msg_type: str, handler: Callable[[FIXMessage], None]):
        """Register a message handler for application messages."""
        self.message_handlers[msg_type] = handler
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        return {
            "state": self.state.value,
            "session_start_time": self.stats.session_start_time.isoformat() if self.stats.session_start_time else None,
            "total_messages_sent": self.stats.total_messages_sent,
            "total_messages_received": self.stats.total_messages_received,
            "heartbeats_sent": self.stats.heartbeats_sent,
            "heartbeats_received": self.stats.heartbeats_received,
            "test_requests_sent": self.stats.test_requests_sent,
            "test_requests_received": self.stats.test_requests_received,
            "reconnection_attempts": self.stats.reconnection_attempts,
            "outgoing_seq_num": self.outgoing_seq_num,
            "incoming_seq_num": self.incoming_seq_num,
            "expected_seq_num": self.expected_seq_num,
        }
    
    def is_connected(self) -> bool:
        """Check if session is connected and logged on."""
        return self.state == FIXSessionState.LOGGED_ON


def load_fix_config(config_path: str = "services/execution/fix_connector/fix_session_config.yaml") -> FIXSessionConfig:
    """Load FIX session configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        fix_session_data = config_data.get('fix_session', {})
        return FIXSessionConfig(**fix_session_data)
        
    except Exception as e:
        logger = structlog.get_logger(__name__)
        logger.error("Error loading FIX config", error=str(e))
        raise 