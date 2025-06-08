"""
Time-Based Scalping Strategy for Choppy Market Regimes.

This strategy implements an edge-enhanced scalping system for choppy/sideways market conditions.
Features include:
- Multi-timeframe analysis (1H bias, 5M entry)
- Session-based trading (London/Tokyo only)
- Pattern detection (double top/bottom, flags, pennants)
- Volatility window filtering
- RSI reversal signals
- Time-based kill switches
"""

from dataclasses import dataclass
from datetime import datetime, time
from enum import Enum
from typing import Any

import numpy as np
import structlog

from core.interfaces import Signal, Strategy
from core.interfaces.trading_interfaces import OrderSide


class ScalpingPattern(Enum):
    """Scalping pattern types."""

    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    FLAG = "flag"
    PENNANT = "pennant"


class TradingSession(Enum):
    """Trading session types."""

    LONDON = "london"
    TOKYO = "tokyo"
    NEW_YORK = "new_york"
    SYDNEY = "sydney"


@dataclass
class ScalpingPosition:
    """Scalping position tracking."""

    symbol: str
    entry_price: float
    entry_side: OrderSide
    stop_loss: float
    take_profit: float
    entry_time: datetime
    pattern_used: ScalpingPattern
    session: TradingSession
    unrealized_pnl: float = 0.0
    last_update: datetime = None


@dataclass
class SessionInfo:
    """Trading session information."""

    name: TradingSession
    start_time: time
    end_time: time
    is_active: bool = False


class TimeScalpingStrategy(Strategy):
    """
    Time-Based Scalping Strategy for choppy regimes.

    Key Features:
    - Multi-timeframe analysis (1H macro bias, 5M entry signals)
    - Session filtering (London 08:00-17:00 GMT, Tokyo 00:00-09:00 GMT)
    - Pattern detection (double tops/bottoms, flags, pennants)
    - Volatility window (min/max ATR thresholds)
    - RSI reversal signals with multi-timeframe alignment
    - Time-based kill switches (15-minute stall protection)
    - Maximum trades per session limit
    """

    def __init__(
        self,
        config: dict[str, Any],
        logger: structlog.stdlib.BoundLogger | None = None,
    ):
        self.config = config
        self.logger = logger or structlog.get_logger(__name__)

        # Strategy parameters
        self.macro_timeframe = config.get("macro_timeframe", "1H")
        self.entry_timeframe = config.get("entry_timeframe", "5M")
        self.active_sessions = [
            TradingSession(s)
            for s in config.get("active_sessions", ["london", "tokyo"])
        ]
        self.patterns = [
            ScalpingPattern(p)
            for p in config.get(
                "patterns", ["double_top", "double_bottom", "flag", "pennant"]
            )
        ]
        self.min_volatility_atr = config.get("min_volatility_atr", 0.0001)
        self.max_volatility_atr = config.get("max_volatility_atr", 0.0005)
        self.rsi_period = config.get("rsi_period", 14)
        self.rsi_entry_threshold = config.get("rsi_entry_threshold", 20)
        self.time_kill_minutes = config.get("time_kill_minutes", 15)
        self.max_trades_per_session = config.get("max_trades_per_session", 3)
        self.require_htf_alignment = config.get("require_htf_alignment", True)
        self.htf_ma_period = config.get("htf_ma_period", 50)

        # State tracking
        self.active_positions: dict[str, ScalpingPosition] = {}
        self.session_trade_count: dict[TradingSession, int] = {
            session: 0 for session in TradingSession
        }
        self.price_history: dict[
            str, list[tuple[datetime, float, float, float, float]]
        ] = {}
        self.last_atr: dict[str, float] = {}
        self.last_rsi: dict[str, float] = {}
        self.htf_bias: dict[str, OrderSide] = {}  # Higher timeframe bias

        # Session definitions
        self.sessions = {
            TradingSession.LONDON: SessionInfo(
                TradingSession.LONDON, time(8, 0), time(17, 0)
            ),
            TradingSession.TOKYO: SessionInfo(
                TradingSession.TOKYO, time(0, 0), time(9, 0)
            ),
            TradingSession.NEW_YORK: SessionInfo(
                TradingSession.NEW_YORK, time(13, 0), time(22, 0)
            ),
            TradingSession.SYDNEY: SessionInfo(
                TradingSession.SYDNEY, time(22, 0), time(7, 0)
            ),
        }

        self.logger.info(
            "TimeScalpingStrategy initialized",
            active_sessions=[s.value for s in self.active_sessions],
            patterns=[p.value for p in self.patterns],
            min_volatility_atr=self.min_volatility_atr,
            max_volatility_atr=self.max_volatility_atr,
            time_kill_minutes=self.time_kill_minutes,
        )

    async def generate_signal(
        self,
        market_data: Any,
        features: dict[str, float] | None = None,
        regime: str | None = None,
    ) -> Signal | None:
        """Generate time-based scalping signal."""

        if not market_data or not features:
            return None

        symbol = getattr(market_data, "symbol", "EURUSD")

        # Extract OHLC data
        open_price = getattr(market_data, "open", features.get("open", 0))
        high_price = getattr(market_data, "high", features.get("high", 0))
        low_price = getattr(market_data, "low", features.get("low", 0))
        close_price = getattr(market_data, "close", features.get("close", 0))

        if any(p <= 0 for p in [open_price, high_price, low_price, close_price]):
            return None

        try:
            # Update market state
            self._update_market_state(
                symbol, features, open_price, high_price, low_price, close_price
            )

            # Check session filter
            current_session = self._get_current_session()
            if not current_session or current_session not in self.active_sessions:
                return None

            # Check existing positions
            if symbol in self.active_positions:
                return await self._manage_existing_position(
                    symbol, close_price, features
                )

            # Check session trade limit
            if self.session_trade_count[current_session] >= self.max_trades_per_session:
                return None

            # Check for new scalping entry
            entry_signal = await self._check_scalping_entry(
                symbol, features, current_session
            )

            if entry_signal:
                # Increment session trade count
                self.session_trade_count[current_session] += 1

            return entry_signal

        except Exception as e:
            self.logger.error(f"Time scalping signal generation failed: {e}")
            return None

    def _update_market_state(
        self,
        symbol: str,
        features: dict[str, float],
        open_price: float,
        high_price: float,
        low_price: float,
        close_price: float,
    ) -> None:
        """Update market state tracking."""

        # Update price history
        if symbol not in self.price_history:
            self.price_history[symbol] = []

        self.price_history[symbol].append(
            (datetime.now(), open_price, high_price, low_price, close_price)
        )

        # Keep only last 100 bars for pattern detection
        if len(self.price_history[symbol]) > 100:
            self.price_history[symbol] = self.price_history[symbol][-100:]

        # Update ATR
        atr = features.get("atr_14", features.get("atr", 0.0001))
        self.last_atr[symbol] = atr

        # Update RSI
        rsi = features.get("rsi_14", features.get("rsi", 50))
        self.last_rsi[symbol] = rsi

        # Update higher timeframe bias
        self._update_htf_bias(symbol, features)

    def _update_htf_bias(self, symbol: str, features: dict[str, float]) -> None:
        """Update higher timeframe bias."""

        # Use higher timeframe moving average for bias
        htf_ma_key = f"sma_{self.htf_ma_period}"
        htf_ma = features.get(htf_ma_key, features.get("sma_50", 0))
        current_price = features.get("close", 0)

        if htf_ma > 0 and current_price > 0:
            if current_price > htf_ma:
                self.htf_bias[symbol] = OrderSide.BUY
            else:
                self.htf_bias[symbol] = OrderSide.SELL

    def _get_current_session(self) -> TradingSession | None:
        """Get current trading session."""

        current_time = datetime.now().time()

        for session_type, session_info in self.sessions.items():
            if self._is_time_in_session(current_time, session_info):
                return session_type

        return None

    def _is_time_in_session(self, current_time: time, session: SessionInfo) -> bool:
        """Check if current time is within session hours."""

        if session.start_time <= session.end_time:
            # Normal session (e.g., London 08:00-17:00)
            return session.start_time <= current_time <= session.end_time
        else:
            # Overnight session (e.g., Sydney 22:00-07:00)
            return (
                current_time >= session.start_time or current_time <= session.end_time
            )

    async def _check_scalping_entry(
        self, symbol: str, features: dict[str, float], session: TradingSession
    ) -> Signal | None:
        """Check for scalping entry conditions."""

        # Check volatility window
        atr = self.last_atr.get(symbol, 0.0001)
        if not (self.min_volatility_atr <= atr <= self.max_volatility_atr):
            return None

        # Check RSI reversal conditions
        rsi_signal = self._check_rsi_reversal(symbol, features)
        if not rsi_signal:
            return None

        side, rsi_strength = rsi_signal

        # Check higher timeframe alignment if required
        if self.require_htf_alignment:
            if not self._check_htf_alignment(symbol, side):
                return None

        # Check for pattern confirmation
        pattern_signal = self._detect_scalping_patterns(symbol, features, side)
        if not pattern_signal:
            return None

        pattern_type, pattern_strength = pattern_signal

        # Calculate overall signal strength and confidence
        strength = (rsi_strength + pattern_strength) / 2
        confidence = self._calculate_scalping_confidence(
            symbol, features, pattern_type, strength
        )

        return Signal(
            symbol=symbol,
            side=side,
            strength=strength,
            confidence=confidence,
            strategy_name="time_scalping",
            timestamp=datetime.now(),
            features=features,
        )

    def _check_rsi_reversal(
        self, symbol: str, features: dict[str, float]
    ) -> tuple[OrderSide, float] | None:
        """Check for RSI reversal signals."""

        rsi = self.last_rsi.get(symbol, 50)

        # Check for oversold reversal (buy signal)
        if rsi <= (50 - self.rsi_entry_threshold):
            strength = max(
                0.1,
                (50 - self.rsi_entry_threshold - rsi) / (50 - self.rsi_entry_threshold),
            )
            return OrderSide.BUY, strength

        # Check for overbought reversal (sell signal)
        elif rsi >= (50 + self.rsi_entry_threshold):
            strength = max(
                0.1,
                (rsi - (50 + self.rsi_entry_threshold))
                / (50 - self.rsi_entry_threshold),
            )
            return OrderSide.SELL, strength

        return None

    def _check_htf_alignment(self, symbol: str, side: OrderSide) -> bool:
        """Check higher timeframe alignment."""

        htf_bias = self.htf_bias.get(symbol)
        if not htf_bias:
            return True  # Allow if no HTF bias available

        return htf_bias == side

    def _detect_scalping_patterns(
        self, symbol: str, features: dict[str, float], expected_side: OrderSide
    ) -> tuple[ScalpingPattern, float] | None:
        """Detect scalping patterns."""

        if symbol not in self.price_history or len(self.price_history[symbol]) < 10:
            return None

        history = self.price_history[symbol]

        # Double Top/Bottom patterns
        if (
            ScalpingPattern.DOUBLE_TOP in self.patterns
            or ScalpingPattern.DOUBLE_BOTTOM in self.patterns
        ):
            double_pattern = self._detect_double_patterns(history, expected_side)
            if double_pattern:
                return double_pattern

        # Flag patterns
        if ScalpingPattern.FLAG in self.patterns:
            flag_pattern = self._detect_flag_pattern(history, expected_side)
            if flag_pattern:
                return flag_pattern

        # Pennant patterns
        if ScalpingPattern.PENNANT in self.patterns:
            pennant_pattern = self._detect_pennant_pattern(history, expected_side)
            if pennant_pattern:
                return pennant_pattern

        return None

    def _detect_double_patterns(
        self,
        history: list[tuple[datetime, float, float, float, float]],
        expected_side: OrderSide,
    ) -> tuple[ScalpingPattern, float] | None:
        """Detect double top/bottom patterns."""

        if len(history) < 20:
            return None

        # Extract highs and lows from recent history
        recent_highs = [bar[2] for bar in history[-20:]]  # High prices
        recent_lows = [bar[3] for bar in history[-20:]]  # Low prices

        # Simple double top detection
        if (
            expected_side == OrderSide.SELL
            and ScalpingPattern.DOUBLE_TOP in self.patterns
        ):
            max_high = max(recent_highs)
            high_indices = [
                i
                for i, h in enumerate(recent_highs)
                if abs(h - max_high) < max_high * 0.001
            ]

            if len(high_indices) >= 2 and (high_indices[-1] - high_indices[0]) >= 5:
                return ScalpingPattern.DOUBLE_TOP, 0.8

        # Simple double bottom detection
        if (
            expected_side == OrderSide.BUY
            and ScalpingPattern.DOUBLE_BOTTOM in self.patterns
        ):
            min_low = min(recent_lows)
            low_indices = [
                i
                for i, l in enumerate(recent_lows)
                if abs(l - min_low) < min_low * 0.001
            ]

            if len(low_indices) >= 2 and (low_indices[-1] - low_indices[0]) >= 5:
                return ScalpingPattern.DOUBLE_BOTTOM, 0.8

        return None

    def _detect_flag_pattern(
        self,
        history: list[tuple[datetime, float, float, float, float]],
        expected_side: OrderSide,
    ) -> tuple[ScalpingPattern, float] | None:
        """Detect flag patterns."""

        if len(history) < 15:
            return None

        # Simplified flag detection
        # Look for consolidation after strong move

        recent_closes = [bar[4] for bar in history[-15:]]

        # Check for consolidation (low volatility in recent bars)
        recent_range = max(recent_closes[-5:]) - min(recent_closes[-5:])
        earlier_range = max(recent_closes[-15:-10]) - min(recent_closes[-15:-10])

        if recent_range < earlier_range * 0.5:  # Consolidation detected
            # Check if there was a strong move before consolidation
            move_strength = (
                abs(recent_closes[-10] - recent_closes[-15]) / recent_closes[-15]
            )

            if move_strength > 0.002:  # Significant move (0.2% for FX)
                return ScalpingPattern.FLAG, 0.7

        return None

    def _detect_pennant_pattern(
        self,
        history: list[tuple[datetime, float, float, float, float]],
        expected_side: OrderSide,
    ) -> tuple[ScalpingPattern, float] | None:
        """Detect pennant patterns."""

        if len(history) < 15:
            return None

        # Simplified pennant detection
        # Look for converging highs and lows

        recent_highs = [bar[2] for bar in history[-10:]]
        recent_lows = [bar[3] for bar in history[-10:]]

        # Check if highs are trending down and lows are trending up
        high_slope = np.polyfit(range(len(recent_highs)), recent_highs, 1)[0]
        low_slope = np.polyfit(range(len(recent_lows)), recent_lows, 1)[0]

        if high_slope < 0 and low_slope > 0:  # Converging pattern
            return ScalpingPattern.PENNANT, 0.7

        return None

    def _calculate_scalping_confidence(
        self,
        symbol: str,
        features: dict[str, float],
        pattern_type: ScalpingPattern,
        signal_strength: float,
    ) -> float:
        """Calculate scalping signal confidence."""

        # Base confidence from signal strength
        confidence = signal_strength

        # Session boost (London and Tokyo are preferred)
        current_session = self._get_current_session()
        if current_session in [TradingSession.LONDON, TradingSession.TOKYO]:
            confidence = min(1.0, confidence * 1.1)

        # Volatility boost (mid-range volatility is preferred)
        atr = self.last_atr.get(symbol, 0.0001)
        optimal_atr = (self.min_volatility_atr + self.max_volatility_atr) / 2
        vol_factor = 1 - abs(atr - optimal_atr) / optimal_atr
        confidence = confidence * (0.8 + 0.2 * vol_factor)

        # Pattern type boost
        if pattern_type in [ScalpingPattern.DOUBLE_TOP, ScalpingPattern.DOUBLE_BOTTOM]:
            confidence = min(1.0, confidence * 1.1)  # Double patterns are more reliable

        return confidence

    async def _manage_existing_position(
        self, symbol: str, current_price: float, features: dict[str, float]
    ) -> Signal | None:
        """Manage existing scalping position."""

        position = self.active_positions[symbol]

        # Update unrealized PnL
        if position.entry_side == OrderSide.BUY:
            position.unrealized_pnl = (current_price - position.entry_price) * 10000
        else:
            position.unrealized_pnl = (position.entry_price - current_price) * 10000

        position.last_update = datetime.now()

        # Check time kill switch
        if self._should_time_kill(position):
            return await self._close_position(symbol, "time_kill")

        # Check stop loss
        if self._should_stop_loss(position, current_price):
            return await self._close_position(symbol, "stop_loss")

        # Check take profit
        if self._should_take_profit(position, current_price):
            return await self._close_position(symbol, "take_profit")

        return None

    def _should_time_kill(self, position: ScalpingPosition) -> bool:
        """Check if time kill should trigger."""
        elapsed = datetime.now() - position.entry_time
        return elapsed.total_seconds() > (self.time_kill_minutes * 60)

    def _should_stop_loss(
        self, position: ScalpingPosition, current_price: float
    ) -> bool:
        """Check if stop loss should trigger."""
        if position.entry_side == OrderSide.BUY:
            return current_price <= position.stop_loss
        else:
            return current_price >= position.stop_loss

    def _should_take_profit(
        self, position: ScalpingPosition, current_price: float
    ) -> bool:
        """Check if take profit should trigger."""
        if position.entry_side == OrderSide.BUY:
            return current_price >= position.take_profit
        else:
            return current_price <= position.take_profit

    async def _close_position(self, symbol: str, reason: str) -> Signal:
        """Close scalping position."""

        position = self.active_positions[symbol]

        # Determine close side (opposite of entry)
        close_side = (
            OrderSide.SELL if position.entry_side == OrderSide.BUY else OrderSide.BUY
        )

        # Remove position from tracking
        del self.active_positions[symbol]

        self.logger.info(
            f"Closing scalping position for {symbol}",
            reason=reason,
            entry_side=position.entry_side.value,
            close_side=close_side.value,
            unrealized_pnl=position.unrealized_pnl,
            pattern_used=position.pattern_used.value,
        )

        return Signal(
            symbol=symbol,
            side=close_side,
            strength=1.0,
            confidence=1.0,
            strategy_name="time_scalping_close",
            timestamp=datetime.now(),
            features={"close_reason": reason},
        )

    def reset_session_counters(self) -> None:
        """Reset session trade counters (called at session start)."""
        current_session = self._get_current_session()
        if current_session:
            self.session_trade_count[current_session] = 0
            self.logger.info(f"Reset trade counter for {current_session.value} session")

    async def update_parameters(self, params: dict[str, Any]) -> None:
        """Update strategy parameters."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
                self.logger.info(f"Updated parameter {key} = {value}")

    def get_parameters(self) -> dict[str, Any]:
        """Get current strategy parameters."""
        return {
            "macro_timeframe": self.macro_timeframe,
            "entry_timeframe": self.entry_timeframe,
            "active_sessions": [s.value for s in self.active_sessions],
            "patterns": [p.value for p in self.patterns],
            "min_volatility_atr": self.min_volatility_atr,
            "max_volatility_atr": self.max_volatility_atr,
            "rsi_period": self.rsi_period,
            "rsi_entry_threshold": self.rsi_entry_threshold,
            "time_kill_minutes": self.time_kill_minutes,
            "max_trades_per_session": self.max_trades_per_session,
            "require_htf_alignment": self.require_htf_alignment,
            "htf_ma_period": self.htf_ma_period,
        }

    def get_name(self) -> str:
        """Get strategy name."""
        return "time_scalping"

    def get_active_positions(self) -> dict[str, ScalpingPosition]:
        """Get active scalping positions."""
        return self.active_positions.copy()

    def get_session_status(self) -> dict[str, Any]:
        """Get current session status."""
        current_session = self._get_current_session()
        return {
            "current_session": current_session.value if current_session else None,
            "session_trade_counts": {
                s.value: count for s, count in self.session_trade_count.items()
            },
            "max_trades_per_session": self.max_trades_per_session,
            "active_sessions": [s.value for s in self.active_sessions],
        }
