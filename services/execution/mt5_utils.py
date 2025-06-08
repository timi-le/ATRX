"""
MetaTrader 5 Utilities - Helper Functions.

This module provides utility functions for MT5 integration including:
- Symbol information and validation
- Time synchronization and market hours
- Margin and lot size calculations
- Price formatting and conversion
- Market data helpers
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

import MetaTrader5 as mt5
import structlog


class MarketSession(Enum):
    """Market trading sessions."""

    SYDNEY = "sydney"
    TOKYO = "tokyo"
    LONDON = "london"
    NEW_YORK = "new_york"
    CLOSED = "closed"


@dataclass
class SymbolInfo:
    """Enhanced symbol information."""

    name: str
    description: str
    currency_base: str
    currency_profit: str
    currency_margin: str
    digits: int
    point: float
    spread: int
    bid: float
    ask: float
    volume_min: float
    volume_max: float
    volume_step: float
    contract_size: float
    margin_initial: float
    margin_maintenance: float
    swap_long: float
    swap_short: float
    trade_mode: int
    trade_allowed: bool
    session_deals: int
    session_buy_orders: int
    session_sell_orders: int


@dataclass
class MarketHours:
    """Market hours information."""

    symbol: str
    session: MarketSession
    is_open: bool
    next_open: datetime | None
    next_close: datetime | None
    time_to_open: timedelta | None
    time_to_close: timedelta | None


class MT5Utils:
    """Utility functions for MT5 operations."""

    def __init__(self, logger: structlog.stdlib.BoundLogger | None = None):
        """Initialize MT5 utilities."""
        self.logger = logger or structlog.get_logger(__name__)

        # Market session times (UTC)
        self.session_times = {
            MarketSession.SYDNEY: {"open": "21:00", "close": "06:00"},
            MarketSession.TOKYO: {"open": "00:00", "close": "09:00"},
            MarketSession.LONDON: {"open": "07:00", "close": "16:00"},
            MarketSession.NEW_YORK: {"open": "12:00", "close": "21:00"},
        }

        # Major currency pairs
        self.major_pairs = [
            "EURUSD",
            "GBPUSD",
            "USDJPY",
            "USDCHF",
            "AUDUSD",
            "USDCAD",
            "NZDUSD",
        ]

        # Minor pairs
        self.minor_pairs = [
            "EURJPY",
            "EURGBP",
            "EURCHF",
            "EURAUD",
            "EURCAD",
            "EURNZD",
            "GBPJPY",
            "GBPCHF",
            "GBPAUD",
            "GBPCAD",
            "GBPNZD",
            "AUDJPY",
            "AUDCHF",
            "AUDCAD",
            "AUDNZD",
            "CADJPY",
            "CADCHF",
            "NZDJPY",
            "NZDCHF",
            "NZDCAD",
            "CHFJPY",
        ]

    def get_symbol_info(self, symbol: str) -> SymbolInfo | None:
        """Get comprehensive symbol information."""
        try:
            info = mt5.symbol_info(symbol)
            if info is None:
                self.logger.warning("Symbol not found", symbol=symbol)
                return None

            return SymbolInfo(
                name=info.name,
                description=info.description,
                currency_base=info.currency_base,
                currency_profit=info.currency_profit,
                currency_margin=info.currency_margin,
                digits=info.digits,
                point=info.point,
                spread=info.spread,
                bid=info.bid,
                ask=info.ask,
                volume_min=info.volume_min,
                volume_max=info.volume_max,
                volume_step=info.volume_step,
                contract_size=info.contract_size,
                margin_initial=info.margin_initial,
                margin_maintenance=info.margin_maintenance,
                swap_long=info.swap_long,
                swap_short=info.swap_short,
                trade_mode=info.trade_mode,
                trade_allowed=info.visible,
                session_deals=info.session_deals,
                session_buy_orders=info.session_buy_orders,
                session_sell_orders=info.session_sell_orders,
            )

        except Exception as e:
            self.logger.error("Error getting symbol info", symbol=symbol, error=str(e))
            return None

    def validate_symbol(self, symbol: str) -> bool:
        """Validate if symbol is available for trading."""
        try:
            info = mt5.symbol_info(symbol)
            if info is None:
                return False

            # Check if symbol is visible and trading is allowed
            return info.visible and info.trade_mode != mt5.SYMBOL_TRADE_MODE_DISABLED

        except Exception as e:
            self.logger.error("Error validating symbol", symbol=symbol, error=str(e))
            return False

    def get_available_symbols(self, group: str = "*") -> list[str]:
        """Get list of available symbols."""
        try:
            symbols = mt5.symbols_get(group)
            if symbols is None:
                return []

            return [symbol.name for symbol in symbols if symbol.visible]

        except Exception as e:
            self.logger.error("Error getting symbols", error=str(e))
            return []

    def get_major_pairs(self) -> list[str]:
        """Get available major currency pairs."""
        available_symbols = self.get_available_symbols()
        return [pair for pair in self.major_pairs if pair in available_symbols]

    def get_minor_pairs(self) -> list[str]:
        """Get available minor currency pairs."""
        available_symbols = self.get_available_symbols()
        return [pair for pair in self.minor_pairs if pair in available_symbols]

    def calculate_lot_size(
        self,
        symbol: str,
        risk_amount: float,
        stop_loss_pips: float,
        account_currency: str = "USD",
    ) -> float:
        """Calculate optimal lot size based on risk management."""
        try:
            symbol_info = self.get_symbol_info(symbol)
            if symbol_info is None:
                return 0.0

            # Calculate pip value
            pip_value = self.calculate_pip_value(symbol, 1.0, account_currency)
            if pip_value == 0:
                return 0.0

            # Calculate lot size
            risk_per_pip = risk_amount / stop_loss_pips
            lot_size = risk_per_pip / pip_value

            # Round to valid lot size
            lot_size = self.round_lot_size(symbol, lot_size)

            return lot_size

        except Exception as e:
            self.logger.error("Error calculating lot size", symbol=symbol, error=str(e))
            return 0.0

    def calculate_pip_value(
        self, symbol: str, lot_size: float, account_currency: str = "USD"
    ) -> float:
        """Calculate pip value for a given lot size."""
        try:
            symbol_info = self.get_symbol_info(symbol)
            if symbol_info is None:
                return 0.0

            # For most forex pairs, pip value = (pip in decimal) * lot size * contract size
            pip_decimal = symbol_info.point
            if symbol_info.digits == 5 or symbol_info.digits == 3:
                pip_decimal *= 10  # Account for fractional pips

            pip_value = pip_decimal * lot_size * symbol_info.contract_size

            # Convert to account currency if needed
            if symbol_info.currency_profit != account_currency:
                conversion_rate = self.get_conversion_rate(
                    symbol_info.currency_profit, account_currency
                )
                pip_value *= conversion_rate

            return pip_value

        except Exception as e:
            self.logger.error(
                "Error calculating pip value", symbol=symbol, error=str(e)
            )
            return 0.0

    def get_conversion_rate(self, from_currency: str, to_currency: str) -> float:
        """Get conversion rate between currencies."""
        try:
            if from_currency == to_currency:
                return 1.0

            # Try direct conversion
            symbol = f"{from_currency}{to_currency}"
            info = mt5.symbol_info(symbol)
            if info is not None:
                return info.bid

            # Try inverse conversion
            symbol = f"{to_currency}{from_currency}"
            info = mt5.symbol_info(symbol)
            if info is not None:
                return 1.0 / info.ask

            # Try USD cross rates
            if from_currency != "USD" and to_currency != "USD":
                usd_from = self.get_conversion_rate(from_currency, "USD")
                usd_to = self.get_conversion_rate("USD", to_currency)
                return usd_from * usd_to

            self.logger.warning(
                "Cannot find conversion rate",
                from_currency=from_currency,
                to_currency=to_currency,
            )
            return 1.0

        except Exception as e:
            self.logger.error("Error getting conversion rate", error=str(e))
            return 1.0

    def round_lot_size(self, symbol: str, lot_size: float) -> float:
        """Round lot size to valid increment."""
        try:
            symbol_info = self.get_symbol_info(symbol)
            if symbol_info is None:
                return 0.0

            # Ensure lot size is within bounds
            lot_size = max(
                symbol_info.volume_min, min(symbol_info.volume_max, lot_size)
            )

            # Round to valid step
            steps = round(lot_size / symbol_info.volume_step)
            return steps * symbol_info.volume_step

        except Exception as e:
            self.logger.error("Error rounding lot size", symbol=symbol, error=str(e))
            return 0.0

    def calculate_margin_required(self, symbol: str, lot_size: float) -> float:
        """Calculate margin required for a position."""
        try:
            symbol_info = self.get_symbol_info(symbol)
            if symbol_info is None:
                return 0.0

            # Get account info for leverage
            account_info = mt5.account_info()
            if account_info is None:
                return 0.0

            # Calculate margin
            contract_size = lot_size * symbol_info.contract_size
            margin = (contract_size * symbol_info.bid) / account_info.leverage

            return margin

        except Exception as e:
            self.logger.error("Error calculating margin", symbol=symbol, error=str(e))
            return 0.0

    def get_market_hours(self, symbol: str) -> MarketHours:
        """Get market hours information for symbol."""
        try:
            now_utc = datetime.now(timezone.utc)
            current_session = self.get_current_session()

            # Check if market is currently open
            is_open = self.is_market_open(symbol)

            # Calculate next open/close times
            next_open = None
            next_close = None
            time_to_open = None
            time_to_close = None

            if is_open:
                # Market is open, find next close
                next_close = self._get_next_session_time("close", now_utc)
                if next_close:
                    time_to_close = next_close - now_utc
            else:
                # Market is closed, find next open
                next_open = self._get_next_session_time("open", now_utc)
                if next_open:
                    time_to_open = next_open - now_utc

            return MarketHours(
                symbol=symbol,
                session=current_session,
                is_open=is_open,
                next_open=next_open,
                next_close=next_close,
                time_to_open=time_to_open,
                time_to_close=time_to_close,
            )

        except Exception as e:
            self.logger.error("Error getting market hours", symbol=symbol, error=str(e))
            return MarketHours(
                symbol=symbol,
                session=MarketSession.CLOSED,
                is_open=False,
                next_open=None,
                next_close=None,
                time_to_open=None,
                time_to_close=None,
            )

    def get_current_session(self) -> MarketSession:
        """Get current market session."""
        try:
            now_utc = datetime.now(timezone.utc)
            current_time = now_utc.strftime("%H:%M")

            # Check each session
            for session, times in self.session_times.items():
                if self._is_time_in_session(
                    current_time, times["open"], times["close"]
                ):
                    return session

            return MarketSession.CLOSED

        except Exception as e:
            self.logger.error("Error getting current session", error=str(e))
            return MarketSession.CLOSED

    def is_market_open(self, symbol: str) -> bool:
        """Check if market is open for trading."""
        try:
            # Check if it's weekend
            now_utc = datetime.now(timezone.utc)
            if now_utc.weekday() >= 5:  # Saturday = 5, Sunday = 6
                # Check if it's Friday after 21:00 UTC or before Monday 21:00 UTC
                if now_utc.weekday() == 5 or (
                    now_utc.weekday() == 6 and now_utc.hour < 21
                ):
                    return False

            # Check symbol-specific trading hours
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return False

            return symbol_info.trade_mode != mt5.SYMBOL_TRADE_MODE_DISABLED

        except Exception as e:
            self.logger.error(
                "Error checking market status", symbol=symbol, error=str(e)
            )
            return False

    def _is_time_in_session(
        self, current_time: str, open_time: str, close_time: str
    ) -> bool:
        """Check if current time is within session hours."""
        try:
            current = datetime.strptime(current_time, "%H:%M").time()
            open_t = datetime.strptime(open_time, "%H:%M").time()
            close_t = datetime.strptime(close_time, "%H:%M").time()

            if open_t <= close_t:
                # Same day session
                return open_t <= current <= close_t
            else:
                # Overnight session
                return current >= open_t or current <= close_t

        except Exception:
            return False

    def _get_next_session_time(
        self, time_type: str, from_time: datetime
    ) -> datetime | None:
        """Get next session open or close time."""
        try:
            # This is a simplified implementation
            # In practice, you'd want to consider actual market holidays and specific symbol hours

            if time_type == "open":
                # Next Monday 21:00 UTC (Sydney open)
                days_until_monday = (7 - from_time.weekday()) % 7
                if days_until_monday == 0 and from_time.hour >= 21:
                    days_until_monday = 7

                next_monday = from_time + timedelta(days=days_until_monday)
                return next_monday.replace(hour=21, minute=0, second=0, microsecond=0)

            else:  # close
                # Next Friday 21:00 UTC (New York close)
                days_until_friday = (4 - from_time.weekday()) % 7
                if days_until_friday == 0 and from_time.hour >= 21:
                    days_until_friday = 7

                next_friday = from_time + timedelta(days=days_until_friday)
                return next_friday.replace(hour=21, minute=0, second=0, microsecond=0)

        except Exception as e:
            self.logger.error("Error calculating next session time", error=str(e))
            return None

    def format_price(self, symbol: str, price: float) -> str:
        """Format price according to symbol digits."""
        try:
            symbol_info = self.get_symbol_info(symbol)
            if symbol_info is None:
                return f"{price:.5f}"

            return f"{price:.{symbol_info.digits}f}"

        except Exception as e:
            self.logger.error("Error formatting price", symbol=symbol, error=str(e))
            return f"{price:.5f}"

    def calculate_pips(self, symbol: str, price1: float, price2: float) -> float:
        """Calculate pips between two prices."""
        try:
            symbol_info = self.get_symbol_info(symbol)
            if symbol_info is None:
                return 0.0

            pip_size = symbol_info.point
            if symbol_info.digits == 5 or symbol_info.digits == 3:
                pip_size *= 10  # Account for fractional pips

            return abs(price2 - price1) / pip_size

        except Exception as e:
            self.logger.error("Error calculating pips", symbol=symbol, error=str(e))
            return 0.0

    def get_spread_info(self, symbol: str) -> dict[str, Any]:
        """Get spread information for symbol."""
        try:
            symbol_info = self.get_symbol_info(symbol)
            if symbol_info is None:
                return {}

            spread_points = symbol_info.spread
            spread_pips = spread_points * symbol_info.point

            if symbol_info.digits == 5 or symbol_info.digits == 3:
                spread_pips /= 10  # Convert to pips from fractional pips

            spread_percentage = (
                (spread_pips / symbol_info.bid) * 100 if symbol_info.bid > 0 else 0
            )

            return {
                "symbol": symbol,
                "spread_points": spread_points,
                "spread_pips": spread_pips,
                "spread_percentage": spread_percentage,
                "bid": symbol_info.bid,
                "ask": symbol_info.ask,
                "point_value": symbol_info.point,
            }

        except Exception as e:
            self.logger.error("Error getting spread info", symbol=symbol, error=str(e))
            return {}

    def get_server_time(self) -> datetime | None:
        """Get MT5 server time."""
        try:
            # Get last tick to determine server time
            tick = mt5.symbol_info_tick("EURUSD")  # Use EURUSD as reference
            if tick is None:
                return None

            return datetime.fromtimestamp(tick.time)

        except Exception as e:
            self.logger.error("Error getting server time", error=str(e))
            return None

    def synchronize_time(self) -> bool:
        """Check if local time is synchronized with server."""
        try:
            server_time = self.get_server_time()
            if server_time is None:
                return False

            local_time = datetime.now()
            time_diff = abs((server_time - local_time).total_seconds())

            # Allow up to 5 seconds difference
            return time_diff <= 5.0

        except Exception as e:
            self.logger.error("Error checking time synchronization", error=str(e))
            return False

    def get_trading_statistics(self, symbol: str) -> dict[str, Any]:
        """Get trading statistics for symbol."""
        try:
            symbol_info = self.get_symbol_info(symbol)
            if symbol_info is None:
                return {}

            return {
                "symbol": symbol,
                "session_deals": symbol_info.session_deals,
                "session_buy_orders": symbol_info.session_buy_orders,
                "session_sell_orders": symbol_info.session_sell_orders,
                "total_session_orders": symbol_info.session_buy_orders
                + symbol_info.session_sell_orders,
                "buy_sell_ratio": (
                    symbol_info.session_buy_orders / symbol_info.session_sell_orders
                    if symbol_info.session_sell_orders > 0
                    else 0
                ),
                "current_spread": symbol_info.spread,
                "current_bid": symbol_info.bid,
                "current_ask": symbol_info.ask,
            }

        except Exception as e:
            self.logger.error(
                "Error getting trading statistics", symbol=symbol, error=str(e)
            )
            return {}


def create_mt5_utils(logger: structlog.stdlib.BoundLogger | None = None) -> MT5Utils:
    """Factory function to create MT5 utilities."""
    return MT5Utils(logger=logger)
