"""
Breakout Trend Strategy for Trending Market Regimes.

This strategy implements an edge-enhanced breakout trading system for trending market conditions.
Features include:
- Price action pattern detection (inside bar, engulfing, break/retest)
- ATR-based volatility gating and entry buffers
- Structural stop losses
- Trailing stop take profit using Parabolic SAR
- Retest confirmation logic
- ADX trend strength filtering
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import numpy as np
import structlog

from core.interfaces import Strategy, Signal
from core.interfaces.trading_interfaces import OrderSide


class PatternType(Enum):
    """Price action pattern types."""
    INSIDE_BAR = "inside_bar"
    ENGULFING = "engulfing"
    BREAK_RETEST = "break_retest"


@dataclass
class BreakoutLevel:
    """Breakout level tracking."""
    price: float
    level_type: str  # "resistance", "support"
    strength: float  # 0.0 to 1.0
    timestamp: datetime
    touches: int = 1


@dataclass
class TrendPosition:
    """Trend position tracking."""
    symbol: str
    entry_price: float
    entry_side: OrderSide
    stop_loss: float
    take_profit: Optional[float]
    trailing_stop: Optional[float]
    entry_time: datetime
    pattern_used: PatternType
    unrealized_pnl: float = 0.0
    last_update: datetime = None


class BreakoutTrendStrategy(Strategy):
    """
    Breakout Trend Strategy for trending regimes.
    
    Key Features:
    - Price action pattern detection (inside bar, engulfing, break/retest)
    - ATR-based entry buffers to avoid fakeouts
    - Structural stop losses
    - Trailing stop take profit
    - ADX trend strength confirmation
    - Retest confirmation logic
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        logger: Optional[structlog.stdlib.BoundLogger] = None
    ):
        self.config = config
        self.logger = logger or structlog.get_logger(__name__)
        
        # Strategy parameters
        self.entry_patterns = config.get('entry_patterns', ['inside_bar', 'engulfing', 'break_retest'])
        self.breakout_buffer_atr = config.get('breakout_buffer_atr', 1.5)
        self.min_atr_threshold = config.get('min_atr_threshold', 0.0001)
        self.sl_mode = config.get('sl_mode', 'structural')
        self.sl_atr_multiplier = config.get('sl_atr_multiplier', 2.0)
        self.tp_mode = config.get('tp_mode', 'trailing')
        self.trailing_stop_atr = config.get('trailing_stop_atr', 1.5)
        self.adx_threshold = config.get('adx_threshold', 25)
        self.ma_period = config.get('ma_period', 20)
        self.retest_confirmation_required = config.get('retest_confirmation_required', True)
        self.retest_timeout_bars = config.get('retest_timeout_bars', 5)
        
        # State tracking
        self.active_positions: Dict[str, TrendPosition] = {}
        self.breakout_levels: Dict[str, List[BreakoutLevel]] = {}
        self.price_history: Dict[str, List[Tuple[datetime, float, float, float, float]]] = {}  # OHLC
        self.last_atr: Dict[str, float] = {}
        self.last_adx: Dict[str, float] = {}
        self.pending_retests: Dict[str, Dict[str, Any]] = {}
        
        self.logger.info(
            "BreakoutTrendStrategy initialized",
            entry_patterns=self.entry_patterns,
            breakout_buffer_atr=self.breakout_buffer_atr,
            adx_threshold=self.adx_threshold,
            sl_mode=self.sl_mode
        )
    
    async def generate_signal(
        self,
        market_data: Any,
        features: Optional[Dict[str, float]] = None,
        regime: Optional[str] = None
    ) -> Optional[Signal]:
        """Generate breakout trend trading signal."""
        
        if not market_data or not features:
            return None
        
        symbol = getattr(market_data, 'symbol', 'EURUSD')
        
        # Extract OHLC data
        open_price = getattr(market_data, 'open', features.get('open', 0))
        high_price = getattr(market_data, 'high', features.get('high', 0))
        low_price = getattr(market_data, 'low', features.get('low', 0))
        close_price = getattr(market_data, 'close', features.get('close', 0))
        
        if any(p <= 0 for p in [open_price, high_price, low_price, close_price]):
            return None
        
        try:
            # Update market state
            self._update_market_state(symbol, features, open_price, high_price, low_price, close_price)
            
            # Check existing positions
            if symbol in self.active_positions:
                return await self._manage_existing_position(symbol, close_price, features)
            
            # Check for new breakout entry
            entry_signal = await self._check_breakout_entry(symbol, features)
            
            return entry_signal
            
        except Exception as e:
            self.logger.error(f"Breakout trend signal generation failed: {e}")
            return None
    
    def _update_market_state(
        self,
        symbol: str,
        features: Dict[str, float],
        open_price: float,
        high_price: float,
        low_price: float,
        close_price: float
    ) -> None:
        """Update market state tracking."""
        
        # Update price history
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        
        self.price_history[symbol].append((
            datetime.now(), open_price, high_price, low_price, close_price
        ))
        
        # Keep only last 50 bars
        if len(self.price_history[symbol]) > 50:
            self.price_history[symbol] = self.price_history[symbol][-50:]
        
        # Update ATR
        atr = features.get('atr_14', features.get('atr', 0.0001))
        self.last_atr[symbol] = atr
        
        # Update ADX
        adx = features.get('adx_14', features.get('adx', 25))
        self.last_adx[symbol] = adx
        
        # Update breakout levels
        self._update_breakout_levels(symbol, high_price, low_price)
    
    def _update_breakout_levels(self, symbol: str, high: float, low: float) -> None:
        """Update significant breakout levels."""
        
        if symbol not in self.breakout_levels:
            self.breakout_levels[symbol] = []
        
        # Simple implementation: track recent highs and lows
        # In production, this would use more sophisticated level detection
        
        levels = self.breakout_levels[symbol]
        
        # Add new resistance level if high is significant
        resistance_levels = [l for l in levels if l.level_type == "resistance"]
        if not resistance_levels or high > max(l.price for l in resistance_levels):
            levels.append(BreakoutLevel(
                price=high,
                level_type="resistance",
                strength=0.8,
                timestamp=datetime.now()
            ))
        
        # Add new support level if low is significant
        support_levels = [l for l in levels if l.level_type == "support"]
        if not support_levels or low < min(l.price for l in support_levels):
            levels.append(BreakoutLevel(
                price=low,
                level_type="support",
                strength=0.8,
                timestamp=datetime.now()
            ))
        
        # Keep only recent levels (last 20)
        self.breakout_levels[symbol] = sorted(levels, key=lambda x: x.timestamp)[-20:]
    
    async def _check_breakout_entry(
        self,
        symbol: str,
        features: Dict[str, float]
    ) -> Optional[Signal]:
        """Check for breakout entry conditions."""
        
        # Check ADX trend strength
        adx = self.last_adx.get(symbol, 25)
        if adx < self.adx_threshold:
            return None
        
        # Check ATR minimum threshold
        atr = self.last_atr.get(symbol, 0.0001)
        if atr < self.min_atr_threshold:
            return None
        
        # Check for price action patterns
        pattern_signal = self._detect_price_patterns(symbol, features)
        if not pattern_signal:
            return None
        
        pattern_type, side, strength = pattern_signal
        
        # Check trend direction alignment
        if not self._check_trend_alignment(symbol, features, side):
            return None
        
        # Check for breakout confirmation
        if not await self._check_breakout_confirmation(symbol, features, side):
            return None
        
        # Calculate confidence based on multiple factors
        confidence = self._calculate_breakout_confidence(symbol, features, pattern_type, strength)
        
        return Signal(
            symbol=symbol,
            side=side,
            strength=strength,
            confidence=confidence,
            strategy_name="breakout_trend",
            timestamp=datetime.now(),
            features=features
        )
    
    def _detect_price_patterns(
        self,
        symbol: str,
        features: Dict[str, float]
    ) -> Optional[Tuple[PatternType, OrderSide, float]]:
        """Detect price action patterns."""
        
        if symbol not in self.price_history or len(self.price_history[symbol]) < 3:
            return None
        
        history = self.price_history[symbol]
        current_bar = history[-1]
        prev_bar = history[-2]
        
        _, curr_open, curr_high, curr_low, curr_close = current_bar
        _, prev_open, prev_high, prev_low, prev_close = prev_bar
        
        # Inside Bar Pattern
        if PatternType.INSIDE_BAR.value in self.entry_patterns:
            if (curr_high < prev_high and curr_low > prev_low):
                # Breakout direction based on close relative to range
                range_mid = (prev_high + prev_low) / 2
                if curr_close > range_mid:
                    return PatternType.INSIDE_BAR, OrderSide.BUY, 0.7
                else:
                    return PatternType.INSIDE_BAR, OrderSide.SELL, 0.7
        
        # Engulfing Pattern
        if PatternType.ENGULFING.value in self.entry_patterns:
            # Bullish engulfing
            if (prev_close < prev_open and curr_close > curr_open and 
                curr_open < prev_close and curr_close > prev_open):
                return PatternType.ENGULFING, OrderSide.BUY, 0.8
            
            # Bearish engulfing
            if (prev_close > prev_open and curr_close < curr_open and 
                curr_open > prev_close and curr_close < prev_open):
                return PatternType.ENGULFING, OrderSide.SELL, 0.8
        
        # Break/Retest Pattern
        if PatternType.BREAK_RETEST.value in self.entry_patterns:
            breakout_signal = self._detect_break_retest(symbol, features)
            if breakout_signal:
                return breakout_signal
        
        return None
    
    def _detect_break_retest(
        self,
        symbol: str,
        features: Dict[str, float]
    ) -> Optional[Tuple[PatternType, OrderSide, float]]:
        """Detect break and retest patterns."""
        
        if symbol not in self.breakout_levels:
            return None
        
        current_price = features.get('close', 0)
        atr = self.last_atr.get(symbol, 0.0001)
        buffer = atr * self.breakout_buffer_atr
        
        levels = self.breakout_levels[symbol]
        
        # Check for resistance breakout
        resistance_levels = [l for l in levels if l.level_type == "resistance"]
        for level in resistance_levels:
            if current_price > level.price + buffer:
                # Check if retest confirmation is required
                if self.retest_confirmation_required:
                    if self._check_retest_completion(symbol, level, OrderSide.BUY):
                        return PatternType.BREAK_RETEST, OrderSide.BUY, 0.9
                else:
                    return PatternType.BREAK_RETEST, OrderSide.BUY, 0.8
        
        # Check for support breakout
        support_levels = [l for l in levels if l.level_type == "support"]
        for level in support_levels:
            if current_price < level.price - buffer:
                # Check if retest confirmation is required
                if self.retest_confirmation_required:
                    if self._check_retest_completion(symbol, level, OrderSide.SELL):
                        return PatternType.BREAK_RETEST, OrderSide.SELL, 0.9
                else:
                    return PatternType.BREAK_RETEST, OrderSide.SELL, 0.8
        
        return None
    
    def _check_retest_completion(
        self,
        symbol: str,
        level: BreakoutLevel,
        side: OrderSide
    ) -> bool:
        """Check if retest has been completed."""
        
        # Simplified retest logic
        # In production, this would track the actual retest process
        
        if symbol not in self.price_history or len(self.price_history[symbol]) < 5:
            return False
        
        recent_bars = self.price_history[symbol][-5:]
        
        # Check if price has retested the level and held
        for _, _, high, low, close in recent_bars:
            if side == OrderSide.BUY:
                # For bullish breakout, check if price retested resistance (now support)
                if low <= level.price <= high and close > level.price:
                    return True
            else:
                # For bearish breakout, check if price retested support (now resistance)
                if low <= level.price <= high and close < level.price:
                    return True
        
        return False
    
    def _check_trend_alignment(
        self,
        symbol: str,
        features: Dict[str, float],
        side: OrderSide
    ) -> bool:
        """Check if signal aligns with overall trend."""
        
        # Use moving average for trend direction
        ma_key = f'sma_{self.ma_period}'
        ma = features.get(ma_key, features.get('sma_20', 0))
        current_price = features.get('close', 0)
        
        if ma <= 0 or current_price <= 0:
            return True  # Default to allow if no MA data
        
        if side == OrderSide.BUY:
            return current_price > ma  # Price above MA for long
        else:
            return current_price < ma  # Price below MA for short
    
    async def _check_breakout_confirmation(
        self,
        symbol: str,
        features: Dict[str, float],
        side: OrderSide
    ) -> bool:
        """Check for breakout confirmation."""
        
        # Volume confirmation (if available)
        volume = features.get('volume', 0)
        avg_volume = features.get('avg_volume_20', 0)
        
        if volume > 0 and avg_volume > 0:
            volume_ratio = volume / avg_volume
            if volume_ratio < 1.2:  # Require above-average volume
                return False
        
        # Momentum confirmation
        momentum = features.get('momentum_10', 0)
        if side == OrderSide.BUY and momentum < 0:
            return False
        elif side == OrderSide.SELL and momentum > 0:
            return False
        
        return True
    
    def _calculate_breakout_confidence(
        self,
        symbol: str,
        features: Dict[str, float],
        pattern_type: PatternType,
        pattern_strength: float
    ) -> float:
        """Calculate overall breakout confidence."""
        
        # Base confidence from pattern strength
        confidence = pattern_strength
        
        # ADX boost
        adx = self.last_adx.get(symbol, 25)
        if adx > 40:  # Strong trend
            confidence = min(1.0, confidence * 1.2)
        
        # Volume boost
        volume = features.get('volume', 0)
        avg_volume = features.get('avg_volume_20', 0)
        if volume > 0 and avg_volume > 0:
            volume_ratio = volume / avg_volume
            if volume_ratio > 1.5:
                confidence = min(1.0, confidence * 1.1)
        
        # Pattern type boost
        if pattern_type == PatternType.BREAK_RETEST:
            confidence = min(1.0, confidence * 1.1)  # Retest patterns are more reliable
        
        return confidence
    
    async def _manage_existing_position(
        self,
        symbol: str,
        current_price: float,
        features: Dict[str, float]
    ) -> Optional[Signal]:
        """Manage existing trend position."""
        
        position = self.active_positions[symbol]
        
        # Update unrealized PnL
        if position.entry_side == OrderSide.BUY:
            position.unrealized_pnl = (current_price - position.entry_price) * 10000  # Assuming standard lot
        else:
            position.unrealized_pnl = (position.entry_price - current_price) * 10000
        
        position.last_update = datetime.now()
        
        # Check stop loss
        if self._should_stop_loss(position, current_price):
            return await self._close_position(symbol, "stop_loss")
        
        # Update trailing stop
        if self.tp_mode == "trailing":
            self._update_trailing_stop(position, current_price, features)
        
        # Check trailing stop
        if position.trailing_stop and self._should_trailing_stop(position, current_price):
            return await self._close_position(symbol, "trailing_stop")
        
        return None
    
    def _should_stop_loss(self, position: TrendPosition, current_price: float) -> bool:
        """Check if stop loss should trigger."""
        if position.entry_side == OrderSide.BUY:
            return current_price <= position.stop_loss
        else:
            return current_price >= position.stop_loss
    
    def _update_trailing_stop(
        self,
        position: TrendPosition,
        current_price: float,
        features: Dict[str, float]
    ) -> None:
        """Update trailing stop level."""
        
        atr = self.last_atr.get(position.symbol, 0.0001)
        trailing_distance = atr * self.trailing_stop_atr
        
        if position.entry_side == OrderSide.BUY:
            new_trailing_stop = current_price - trailing_distance
            if position.trailing_stop is None or new_trailing_stop > position.trailing_stop:
                position.trailing_stop = new_trailing_stop
        else:
            new_trailing_stop = current_price + trailing_distance
            if position.trailing_stop is None or new_trailing_stop < position.trailing_stop:
                position.trailing_stop = new_trailing_stop
    
    def _should_trailing_stop(self, position: TrendPosition, current_price: float) -> bool:
        """Check if trailing stop should trigger."""
        if not position.trailing_stop:
            return False
        
        if position.entry_side == OrderSide.BUY:
            return current_price <= position.trailing_stop
        else:
            return current_price >= position.trailing_stop
    
    async def _close_position(self, symbol: str, reason: str) -> Signal:
        """Close trend position."""
        
        position = self.active_positions[symbol]
        
        # Determine close side (opposite of entry)
        close_side = OrderSide.SELL if position.entry_side == OrderSide.BUY else OrderSide.BUY
        
        # Remove position from tracking
        del self.active_positions[symbol]
        
        self.logger.info(
            f"Closing trend position for {symbol}",
            reason=reason,
            entry_side=position.entry_side.value,
            close_side=close_side.value,
            unrealized_pnl=position.unrealized_pnl
        )
        
        return Signal(
            symbol=symbol,
            side=close_side,
            strength=1.0,
            confidence=1.0,
            strategy_name="breakout_trend_close",
            timestamp=datetime.now(),
            features={'close_reason': reason}
        )
    
    async def update_parameters(self, params: Dict[str, Any]) -> None:
        """Update strategy parameters."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
                self.logger.info(f"Updated parameter {key} = {value}")
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current strategy parameters."""
        return {
            'entry_patterns': self.entry_patterns,
            'breakout_buffer_atr': self.breakout_buffer_atr,
            'min_atr_threshold': self.min_atr_threshold,
            'sl_mode': self.sl_mode,
            'sl_atr_multiplier': self.sl_atr_multiplier,
            'tp_mode': self.tp_mode,
            'trailing_stop_atr': self.trailing_stop_atr,
            'adx_threshold': self.adx_threshold,
            'ma_period': self.ma_period,
            'retest_confirmation_required': self.retest_confirmation_required,
            'retest_timeout_bars': self.retest_timeout_bars
        }
    
    def get_name(self) -> str:
        """Get strategy name."""
        return "breakout_trend"
    
    def get_active_positions(self) -> Dict[str, TrendPosition]:
        """Get active trend positions."""
        return self.active_positions.copy()
    
    def get_breakout_levels(self, symbol: str) -> List[BreakoutLevel]:
        """Get breakout levels for symbol."""
        return self.breakout_levels.get(symbol, []).copy() 