"""
FIX Message Utilities - Core FIX Protocol Message Handling.

This module provides comprehensive utilities for FIX protocol message processing,
including parsing, validation, generation, and formatting of FIX messages.

Features:
- FIX message parsing and validation
- Checksum calculation and verification
- Message formatting and serialization
- Tag/value pair handling
- Sequence number management
- Time formatting utilities
"""

import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

import structlog

logger = structlog.get_logger(__name__)


class FIXVersion(Enum):
    """Supported FIX versions."""

    FIX_4_2 = "FIX.4.2"
    FIX_4_4 = "FIX.4.4"


@dataclass
class FIXField:
    """FIX field definition."""

    tag: int
    name: str
    required: bool = False
    data_type: str = "STRING"
    description: str = ""


@dataclass
class FIXMessage:
    """FIX message representation."""

    msg_type: str
    fields: dict[int, str] = field(default_factory=dict)
    raw_message: str = ""
    sequence_number: int | None = None
    timestamp: datetime | None = None
    valid: bool = True
    errors: list[str] = field(default_factory=list)


class FIXMessageUtils:
    """
    Utilities for FIX message processing.

    Provides comprehensive FIX protocol message handling including parsing,
    validation, generation, and formatting.
    """

    # FIX field separator
    SOH = "\x01"  # Start of Header (ASCII 1)

    # Standard FIX fields
    STANDARD_FIELDS = {
        8: FIXField(8, "BeginString", True, "STRING", "FIX version"),
        9: FIXField(9, "BodyLength", True, "LENGTH", "Message body length"),
        10: FIXField(10, "CheckSum", True, "STRING", "Message checksum"),
        11: FIXField(11, "ClOrdID", False, "STRING", "Client order ID"),
        14: FIXField(14, "CumQty", False, "QTY", "Cumulative quantity"),
        15: FIXField(15, "Currency", False, "CURRENCY", "Currency"),
        17: FIXField(17, "ExecID", False, "STRING", "Execution ID"),
        20: FIXField(20, "ExecTransType", False, "CHAR", "Execution transaction type"),
        31: FIXField(31, "LastPx", False, "PRICE", "Last price"),
        32: FIXField(32, "LastQty", False, "QTY", "Last quantity"),
        34: FIXField(34, "MsgSeqNum", True, "SEQNUM", "Message sequence number"),
        35: FIXField(35, "MsgType", True, "STRING", "Message type"),
        37: FIXField(37, "OrderID", False, "STRING", "Order ID"),
        38: FIXField(38, "OrderQty", False, "QTY", "Order quantity"),
        39: FIXField(39, "OrdStatus", False, "CHAR", "Order status"),
        40: FIXField(40, "OrdType", False, "CHAR", "Order type"),
        41: FIXField(41, "OrigClOrdID", False, "STRING", "Original client order ID"),
        44: FIXField(44, "Price", False, "PRICE", "Price"),
        49: FIXField(49, "SenderCompID", True, "STRING", "Sender company ID"),
        52: FIXField(52, "SendingTime", True, "UTCTIMESTAMP", "Sending time"),
        54: FIXField(54, "Side", False, "CHAR", "Side"),
        55: FIXField(55, "Symbol", False, "STRING", "Symbol"),
        56: FIXField(56, "TargetCompID", True, "STRING", "Target company ID"),
        58: FIXField(58, "Text", False, "STRING", "Free format text"),
        59: FIXField(59, "TimeInForce", False, "CHAR", "Time in force"),
        60: FIXField(60, "TransactTime", False, "UTCTIMESTAMP", "Transaction time"),
        99: FIXField(99, "StopPx", False, "PRICE", "Stop price"),
        108: FIXField(108, "HeartBtInt", False, "INT", "Heartbeat interval"),
        112: FIXField(112, "TestReqID", False, "STRING", "Test request ID"),
        141: FIXField(
            141, "ResetSeqNumFlag", False, "BOOLEAN", "Reset sequence number flag"
        ),
        150: FIXField(150, "ExecType", False, "CHAR", "Execution type"),
        151: FIXField(151, "LeavesQty", False, "QTY", "Leaves quantity"),
        207: FIXField(207, "SecurityExchange", False, "EXCHANGE", "Security exchange"),
        553: FIXField(553, "Username", False, "STRING", "Username"),
        554: FIXField(554, "Password", False, "STRING", "Password"),
    }

    def __init__(self, fix_version: FIXVersion = FIXVersion.FIX_4_2):
        """Initialize FIX message utilities."""
        self.fix_version = fix_version
        self.begin_string = fix_version.value

    def parse_message(self, raw_message: str) -> FIXMessage:
        """
        Parse a raw FIX message into structured format.

        Args:
            raw_message: Raw FIX message string

        Returns:
            FIXMessage object with parsed fields
        """
        try:
            # Remove any trailing newlines or whitespace
            raw_message = raw_message.strip()

            # Split message into tag=value pairs
            pairs = raw_message.split(self.SOH)
            fields = {}
            errors = []

            for pair in pairs:
                if not pair:
                    continue

                if "=" not in pair:
                    errors.append(f"Invalid field format: {pair}")
                    continue

                tag_str, value = pair.split("=", 1)

                try:
                    tag = int(tag_str)
                    fields[tag] = value
                except ValueError:
                    errors.append(f"Invalid tag: {tag_str}")
                    continue

            # Extract message type
            msg_type = fields.get(35, "")
            if not msg_type:
                errors.append("Missing MsgType (35)")

            # Validate required header fields
            if 8 not in fields:
                errors.append("Missing BeginString (8)")
            if 9 not in fields:
                errors.append("Missing BodyLength (9)")
            if 10 not in fields:
                errors.append("Missing CheckSum (10)")
            if 34 not in fields:
                errors.append("Missing MsgSeqNum (34)")
            if 49 not in fields:
                errors.append("Missing SenderCompID (49)")
            if 52 not in fields:
                errors.append("Missing SendingTime (52)")
            if 56 not in fields:
                errors.append("Missing TargetCompID (56)")

            # Validate checksum
            if not self._validate_checksum(raw_message):
                errors.append("Invalid checksum")

            # Validate body length
            if not self._validate_body_length(raw_message):
                errors.append("Invalid body length")

            # Extract sequence number
            sequence_number = None
            if 34 in fields:
                try:
                    sequence_number = int(fields[34])
                except ValueError:
                    errors.append("Invalid sequence number")

            # Parse timestamp
            timestamp = None
            if 52 in fields:
                timestamp = self._parse_timestamp(fields[52])

            return FIXMessage(
                msg_type=msg_type,
                fields=fields,
                raw_message=raw_message,
                sequence_number=sequence_number,
                timestamp=timestamp,
                valid=len(errors) == 0,
                errors=errors,
            )

        except Exception as e:
            logger.error("Error parsing FIX message", error=str(e), message=raw_message)
            return FIXMessage(
                msg_type="",
                fields={},
                raw_message=raw_message,
                valid=False,
                errors=[f"Parse error: {str(e)}"],
            )

    def build_message(
        self,
        msg_type: str,
        fields: dict[int, str | int | float],
        sender_comp_id: str,
        target_comp_id: str,
        sequence_number: int,
        sender_sub_id: str | None = None,
        target_sub_id: str | None = None,
    ) -> str:
        """
        Build a FIX message from fields.

        Args:
            msg_type: Message type
            fields: Dictionary of tag -> value pairs
            sender_comp_id: Sender company ID
            target_comp_id: Target company ID
            sequence_number: Message sequence number
            sender_sub_id: Optional sender sub ID
            target_sub_id: Optional target sub ID

        Returns:
            Complete FIX message string
        """
        try:
            # Start with header fields
            message_fields = {
                8: self.begin_string,  # BeginString
                35: msg_type,  # MsgType
                49: sender_comp_id,  # SenderCompID
                56: target_comp_id,  # TargetCompID
                34: str(sequence_number),  # MsgSeqNum
                52: self._format_timestamp(datetime.now(timezone.utc)),  # SendingTime
            }

            # Add optional sub IDs
            if sender_sub_id:
                message_fields[50] = sender_sub_id  # SenderSubID
            if target_sub_id:
                message_fields[57] = target_sub_id  # TargetSubID

            # Add application fields
            for tag, value in fields.items():
                if tag not in [8, 9, 10]:  # Don't override header/trailer fields
                    message_fields[tag] = str(value)

            # Build message without body length and checksum
            message_parts = []

            # Sort fields by tag (header fields first, then others)
            header_tags = [8, 9, 35, 49, 56, 50, 57, 34, 52]
            sorted_tags = []

            # Add header tags in order
            for tag in header_tags:
                if tag in message_fields and tag != 9:  # Skip BodyLength for now
                    sorted_tags.append(tag)

            # Add remaining tags in ascending order
            remaining_tags = sorted(
                [
                    tag
                    for tag in message_fields.keys()
                    if tag not in header_tags and tag not in [9, 10]
                ]
            )
            sorted_tags.extend(remaining_tags)

            # Build message parts
            for tag in sorted_tags:
                if tag in message_fields:
                    message_parts.append(f"{tag}={message_fields[tag]}")

            # Calculate body length (everything after BeginString and BodyLength)
            body = self.SOH.join(message_parts[1:])  # Skip BeginString
            body_length = len(body) + len(self.SOH)  # Add SOH after body length

            # Insert body length
            message_parts.insert(1, f"9={body_length}")

            # Build complete message without checksum
            message_without_checksum = self.SOH.join(message_parts) + self.SOH

            # Calculate and append checksum
            checksum = self._calculate_checksum(message_without_checksum)
            complete_message = (
                message_without_checksum + f"10={checksum:03d}" + self.SOH
            )

            return complete_message

        except Exception as e:
            logger.error("Error building FIX message", error=str(e))
            raise

    def build_logon_message(
        self,
        sender_comp_id: str,
        target_comp_id: str,
        sequence_number: int,
        heartbeat_interval: int,
        username: str | None = None,
        password: str | None = None,
        reset_seq_num: bool = False,
        sender_sub_id: str | None = None,
        target_sub_id: str | None = None,
    ) -> str:
        """Build a logon message."""
        fields = {
            108: heartbeat_interval,  # HeartBtInt
        }

        if username:
            fields[553] = username  # Username
        if password:
            fields[554] = password  # Password
        if reset_seq_num:
            fields[141] = "Y"  # ResetSeqNumFlag

        return self.build_message(
            msg_type="A",
            fields=fields,
            sender_comp_id=sender_comp_id,
            target_comp_id=target_comp_id,
            sequence_number=sequence_number,
            sender_sub_id=sender_sub_id,
            target_sub_id=target_sub_id,
        )

    def build_logout_message(
        self,
        sender_comp_id: str,
        target_comp_id: str,
        sequence_number: int,
        text: str | None = None,
        sender_sub_id: str | None = None,
        target_sub_id: str | None = None,
    ) -> str:
        """Build a logout message."""
        fields = {}
        if text:
            fields[58] = text  # Text

        return self.build_message(
            msg_type="5",
            fields=fields,
            sender_comp_id=sender_comp_id,
            target_comp_id=target_comp_id,
            sequence_number=sequence_number,
            sender_sub_id=sender_sub_id,
            target_sub_id=target_sub_id,
        )

    def build_heartbeat_message(
        self,
        sender_comp_id: str,
        target_comp_id: str,
        sequence_number: int,
        test_req_id: str | None = None,
        sender_sub_id: str | None = None,
        target_sub_id: str | None = None,
    ) -> str:
        """Build a heartbeat message."""
        fields = {}
        if test_req_id:
            fields[112] = test_req_id  # TestReqID

        return self.build_message(
            msg_type="0",
            fields=fields,
            sender_comp_id=sender_comp_id,
            target_comp_id=target_comp_id,
            sequence_number=sequence_number,
            sender_sub_id=sender_sub_id,
            target_sub_id=target_sub_id,
        )

    def build_test_request_message(
        self,
        sender_comp_id: str,
        target_comp_id: str,
        sequence_number: int,
        test_req_id: str,
        sender_sub_id: str | None = None,
        target_sub_id: str | None = None,
    ) -> str:
        """Build a test request message."""
        fields = {112: test_req_id}  # TestReqID

        return self.build_message(
            msg_type="1",
            fields=fields,
            sender_comp_id=sender_comp_id,
            target_comp_id=target_comp_id,
            sequence_number=sequence_number,
            sender_sub_id=sender_sub_id,
            target_sub_id=target_sub_id,
        )

    def build_order_single_message(
        self,
        sender_comp_id: str,
        target_comp_id: str,
        sequence_number: int,
        cl_ord_id: str,
        symbol: str,
        side: str,
        order_qty: float,
        ord_type: str,
        price: float | None = None,
        stop_px: float | None = None,
        time_in_force: str = "0",  # Day
        currency: str = "USD",
        exchange: str | None = None,
        sender_sub_id: str | None = None,
        target_sub_id: str | None = None,
    ) -> str:
        """Build a new order single message."""
        fields = {
            11: cl_ord_id,  # ClOrdID
            55: symbol,  # Symbol
            54: side,  # Side
            38: str(order_qty),  # OrderQty
            40: ord_type,  # OrdType
            59: time_in_force,  # TimeInForce
            15: currency,  # Currency
            60: self._format_timestamp(datetime.now(timezone.utc)),  # TransactTime
        }

        if price is not None:
            fields[44] = str(price)  # Price
        if stop_px is not None:
            fields[99] = str(stop_px)  # StopPx
        if exchange:
            fields[207] = exchange  # SecurityExchange

        return self.build_message(
            msg_type="D",
            fields=fields,
            sender_comp_id=sender_comp_id,
            target_comp_id=target_comp_id,
            sequence_number=sequence_number,
            sender_sub_id=sender_sub_id,
            target_sub_id=target_sub_id,
        )

    def build_order_cancel_request_message(
        self,
        sender_comp_id: str,
        target_comp_id: str,
        sequence_number: int,
        orig_cl_ord_id: str,
        cl_ord_id: str,
        symbol: str,
        side: str,
        order_qty: float,
        sender_sub_id: str | None = None,
        target_sub_id: str | None = None,
    ) -> str:
        """Build an order cancel request message."""
        fields = {
            41: orig_cl_ord_id,  # OrigClOrdID
            11: cl_ord_id,  # ClOrdID
            55: symbol,  # Symbol
            54: side,  # Side
            38: str(order_qty),  # OrderQty
            60: self._format_timestamp(datetime.now(timezone.utc)),  # TransactTime
        }

        return self.build_message(
            msg_type="F",
            fields=fields,
            sender_comp_id=sender_comp_id,
            target_comp_id=target_comp_id,
            sequence_number=sequence_number,
            sender_sub_id=sender_sub_id,
            target_sub_id=target_sub_id,
        )

    def _calculate_checksum(self, message: str) -> int:
        """Calculate FIX message checksum."""
        checksum = sum(ord(char) for char in message) % 256
        return checksum

    def _validate_checksum(self, message: str) -> bool:
        """Validate FIX message checksum."""
        try:
            # Find checksum field
            checksum_match = re.search(r"10=(\d{3})", message)
            if not checksum_match:
                return False

            expected_checksum = int(checksum_match.group(1))

            # Calculate checksum for message without checksum field
            checksum_pos = message.rfind("10=")
            if checksum_pos == -1:
                return False

            message_without_checksum = message[:checksum_pos]
            calculated_checksum = self._calculate_checksum(message_without_checksum)

            return calculated_checksum == expected_checksum

        except Exception:
            return False

    def _validate_body_length(self, message: str) -> bool:
        """Validate FIX message body length."""
        try:
            # Find body length field
            body_length_match = re.search(r"9=(\d+)", message)
            if not body_length_match:
                return False

            expected_length = int(body_length_match.group(1))

            # Calculate actual body length
            body_start = message.find("35=")  # Start after BodyLength
            checksum_start = message.rfind("10=")  # End before CheckSum

            if body_start == -1 or checksum_start == -1:
                return False

            actual_length = checksum_start - body_start

            return actual_length == expected_length

        except Exception:
            return False

    def _format_timestamp(self, dt: datetime) -> str:
        """Format datetime as FIX timestamp (YYYYMMDD-HH:MM:SS.sss)."""
        return dt.strftime("%Y%m%d-%H:%M:%S.%f")[:-3]

    def _parse_timestamp(self, timestamp_str: str) -> datetime | None:
        """Parse FIX timestamp string to datetime."""
        try:
            # Handle different timestamp formats
            formats = [
                "%Y%m%d-%H:%M:%S.%f",
                "%Y%m%d-%H:%M:%S",
                "%Y%m%d-%H:%M:%S.%f",
            ]

            for fmt in formats:
                try:
                    return datetime.strptime(timestamp_str, fmt).replace(
                        tzinfo=timezone.utc
                    )
                except ValueError:
                    continue

            return None

        except Exception:
            return None

    def generate_client_order_id(self) -> str:
        """Generate a unique client order ID."""
        timestamp = int(time.time() * 1000)  # milliseconds
        random_part = str(uuid.uuid4())[:8]
        return f"AI-{timestamp}-{random_part}"

    def get_field_name(self, tag: int) -> str:
        """Get field name for a tag."""
        field = self.STANDARD_FIELDS.get(tag)
        return field.name if field else f"Tag{tag}"

    def is_admin_message(self, msg_type: str) -> bool:
        """Check if message type is administrative."""
        admin_types = ["0", "1", "2", "3", "4", "5", "A"]
        return msg_type in admin_types

    def format_message_for_display(self, message: FIXMessage) -> str:
        """Format FIX message for human-readable display."""
        lines = [f"Message Type: {message.msg_type}"]

        if message.sequence_number:
            lines.append(f"Sequence: {message.sequence_number}")

        if message.timestamp:
            lines.append(f"Timestamp: {message.timestamp}")

        lines.append("Fields:")

        for tag in sorted(message.fields.keys()):
            field_name = self.get_field_name(tag)
            value = message.fields[tag]
            lines.append(f"  {tag} ({field_name}): {value}")

        if message.errors:
            lines.append("Errors:")
            for error in message.errors:
                lines.append(f"  - {error}")

        return "\n".join(lines)


# Convenience functions
def parse_fix_message(
    raw_message: str, fix_version: FIXVersion = FIXVersion.FIX_4_2
) -> FIXMessage:
    """Parse a FIX message."""
    utils = FIXMessageUtils(fix_version)
    return utils.parse_message(raw_message)


def build_fix_message(
    msg_type: str,
    fields: dict[int, str | int | float],
    sender_comp_id: str,
    target_comp_id: str,
    sequence_number: int,
    fix_version: FIXVersion = FIXVersion.FIX_4_2,
) -> str:
    """Build a FIX message."""
    utils = FIXMessageUtils(fix_version)
    return utils.build_message(
        msg_type, fields, sender_comp_id, target_comp_id, sequence_number
    )
