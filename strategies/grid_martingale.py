"""
Grid Martingale Strategy for Mean-Reverting Market Regimes.

This strategy implements an edge-enhanced grid trading system with martingale position sizing
for mean-reverting market conditions. Features include:
- ATR-based dynamic grid spacing
- VWAP/Bollinger Band take profit logic
- RSI divergence entry confirmation
- Time-based stop loss
- Dynamic grid expansion during high volatility
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import structlog

from core.interfaces import Signal, Strategy
from core.interfaces.trading_interfaces import OrderSide


@dataclass
class GridLevel:
    """Individual grid level configuration."""

    price: float
    size: float
    level: int
    side: OrderSide
    is_active: bool = True
    entry_time: datetime | None = None


@dataclass
class GridPosition:
    """Grid position tracking."""

    symbol: str
    grid_levels: list[GridLevel]
    total_size: float
    unrealized_pnl: float
    entry_time: datetime
    last_update: datetime


class GridMartingaleStrategy(Strategy):
    """
    Grid Martingale Strategy for mean-reverting regimes.

    Key Features:
    - Bidirectional grid with ATR-based spacing
    - Martingale position sizing (1.3-1.7x multiplier)
    - VWAP/Bollinger mid-point take profit
    - 30-minute time stop
    - RSI divergence confirmation
    - Dynamic grid expansion in volatile conditions
    """

    def __init__(
        self,
        config: dict[str, Any],
        logger: structlog.stdlib.BoundLogger | None = None,
    ):
        self.config = config
        self.logger = logger or structlog.get_logger(__name__)

        # Strategy parameters
        self.entry_mode = config.get("entry_mode", "bidirectional")
        self.grid_spacing_atr_multiplier = config.get(
            "grid_spacing_atr_multiplier", 0.5
        )
        self.martingale_multiplier = config.get("martingale_multiplier", 1.5)
        self.max_grid_levels = config.get("max_grid_levels", 5)
        self.tp_mode = config.get("tp_mode", "vwap")
        self.tp_atr_multiplier = config.get("tp_atr_multiplier", 1.0)
        self.time_stop_minutes = config.get("time_stop_minutes", 30)
        self.max_unrealized_loss_pct = config.get("max_unrealized_loss_pct", 2.0)
        self.rsi_divergence_required = config.get("rsi_divergence_required", True)
        self.rsi_oversold = config.get("rsi_oversold", 30)
        self.rsi_overbought = config.get("rsi_overbought", 70)
        self.volatility_expansion_threshold = config.get(
            "volatility_expansion_threshold", 1.5
        )
        self.grid_expansion_factor = config.get("grid_expansion_factor", 1.3)

        # State tracking
        self.active_grids: dict[str, GridPosition] = {}
        self.last_atr: dict[str, float] = {}
        self.last_rsi: dict[str, float] = {}
        self.avg_volatility: dict[str, float] = {}

        self.logger.info(
            "GridMartingaleStrategy initialized",
            entry_mode=self.entry_mode,
            grid_spacing_atr=self.grid_spacing_atr_multiplier,
            martingale_multiplier=self.martingale_multiplier,
            max_levels=self.max_grid_levels,
        )

    async def generate_signal(
        self,
        market_data: Any,
        features: dict[str, float] | None = None,
        regime: str | None = None,
    ) -> Signal | None:
        """Generate grid martingale trading signal."""

        if not market_data or not features:
            return None

        symbol = getattr(market_data, "symbol", "EURUSD")
        current_price = getattr(market_data, "close", features.get("close", 0))

        if current_price <= 0:
            return None

        try:
            # Update market state
            self._update_market_state(symbol, features)

            # Check existing grid positions
            if symbol in self.active_grids:
                return await self._manage_existing_grid(symbol, current_price, features)

            # Check for new grid entry
            entry_signal = await self._check_grid_entry(symbol, current_price, features)

            if entry_signal:
                # Initialize new grid
                await self._initialize_grid(
                    symbol, current_price, features, entry_signal.side
                )
                return entry_signal

            return None

        except Exception as e:
            self.logger.error(f"Grid martingale signal generation failed: {e}")
            return None

    def _update_market_state(self, symbol: str, features: dict[str, float]) -> None:
        """Update market state tracking."""

        # Update ATR
        atr = features.get("atr_14", features.get("atr", 0.0001))
        self.last_atr[symbol] = atr

        # Update RSI
        rsi = features.get("rsi_14", features.get("rsi", 50))
        self.last_rsi[symbol] = rsi

        # Update average volatility (simple moving average)
        if symbol not in self.avg_volatility:
            self.avg_volatility[symbol] = atr
        else:
            # Exponential moving average with alpha = 0.1
            self.avg_volatility[symbol] = 0.1 * atr + 0.9 * self.avg_volatility[symbol]

    async def _check_grid_entry(
        self, symbol: str, current_price: float, features: dict[str, float]
    ) -> Signal | None:
        """Check for grid entry conditions."""

        # Check RSI divergence if required
        if self.rsi_divergence_required:
            rsi = self.last_rsi.get(symbol, 50)

            # Entry conditions based on RSI extremes
            if rsi <= self.rsi_oversold:
                # Oversold - potential long entry
                side = OrderSide.BUY
            elif rsi >= self.rsi_overbought:
                # Overbought - potential short entry
                side = OrderSide.SELL
            else:
                # No clear RSI signal
                return None
        else:
            # Default to bidirectional entry
            side = OrderSide.BUY  # Will create both sides in bidirectional mode

        # Check volatility conditions
        atr = self.last_atr.get(symbol, 0.0001)
        if atr <= 0:
            return None

        # Calculate signal strength based on RSI extremity
        rsi = self.last_rsi.get(symbol, 50)
        if side == OrderSide.BUY:
            strength = max(0.1, (self.rsi_oversold - rsi) / self.rsi_oversold)
        else:
            strength = max(
                0.1, (rsi - self.rsi_overbought) / (100 - self.rsi_overbought)
            )

        # Calculate confidence based on volatility and RSI
        vol_ratio = atr / self.avg_volatility.get(symbol, atr)
        confidence = min(1.0, strength * 0.7 + (1 / vol_ratio) * 0.3)

        # Estimate TP/SL in pips for the position sizer
        # ATR is in price units, convert to pips (assuming 1 pip = 0.0001 for EURUSD-like pairs)
        pips_per_point = 10000
        atr_pips = atr * pips_per_point

        take_profit_pips = atr_pips * self.tp_atr_multiplier
        # Base SL on a multiple of ATR, can be refined
        stop_loss_pips = atr_pips * 2.0

        return Signal(
            symbol=symbol,
            side=side,
            strength=strength,
            confidence=confidence,
            strategy_name="grid_martingale",
            timestamp=datetime.now(),
            features=features,
            price=current_price,
            take_profit_pips=take_profit_pips,
            stop_loss_pips=stop_loss_pips,
            win_probability=0.55,  # Placeholder: This should be derived from historical performance
        )

    async def _initialize_grid(
        self,
        symbol: str,
        entry_price: float,
        features: dict[str, float],
        primary_side: OrderSide,
    ) -> None:
        """Initialize new grid position."""

        atr = self.last_atr.get(symbol, 0.0001)

        # Calculate grid spacing
        base_spacing = atr * self.grid_spacing_atr_multiplier

        # Adjust for volatility
        vol_ratio = atr / self.avg_volatility.get(symbol, atr)
        if vol_ratio > self.volatility_expansion_threshold:
            spacing = base_spacing * self.grid_expansion_factor
        else:
            spacing = base_spacing

        grid_levels = []

        if self.entry_mode == "bidirectional":
            # Create both long and short grids
            grid_levels.extend(
                self._create_grid_levels(
                    entry_price, spacing, OrderSide.BUY, self.max_grid_levels // 2
                )
            )
            grid_levels.extend(
                self._create_grid_levels(
                    entry_price, spacing, OrderSide.SELL, self.max_grid_levels // 2
                )
            )
        else:
            # Create single-direction grid
            grid_levels = self._create_grid_levels(
                entry_price, spacing, primary_side, self.max_grid_levels
            )

        # Create grid position
        self.active_grids[symbol] = GridPosition(
            symbol=symbol,
            grid_levels=grid_levels,
            total_size=sum(level.size for level in grid_levels),
            unrealized_pnl=0.0,
            entry_time=datetime.now(),
            last_update=datetime.now(),
        )

        self.logger.info(
            f"Initialized grid for {symbol}",
            entry_price=entry_price,
            spacing=spacing,
            levels=len(grid_levels),
            mode=self.entry_mode,
        )

    def _create_grid_levels(
        self, entry_price: float, spacing: float, side: OrderSide, num_levels: int
    ) -> list[GridLevel]:
        """Create grid levels for one side."""

        levels = []
        base_size = 0.01  # Base position size

        for i in range(num_levels):
            # Calculate price level
            if side == OrderSide.BUY:
                price = entry_price - (spacing * (i + 1))
            else:
                price = entry_price + (spacing * (i + 1))

            # Calculate martingale size
            size = base_size * (self.martingale_multiplier**i)

            levels.append(
                GridLevel(
                    price=price, size=size, level=i + 1, side=side, is_active=True
                )
            )

        return levels

    async def _manage_existing_grid(
        self, symbol: str, current_price: float, features: dict[str, float]
    ) -> Signal | None:
        """Manage existing grid position."""

        grid = self.active_grids[symbol]

        # Check time stop
        if self._should_time_stop(grid):
            return await self._close_grid(symbol, "time_stop")

        # Check unrealized loss limit
        if self._should_kill_switch(grid, current_price):
            return await self._close_grid(symbol, "kill_switch")

        # Check take profit conditions
        if await self._should_take_profit(symbol, current_price, features):
            return await self._close_grid(symbol, "take_profit")

        # Check for grid level hits (would be handled by execution engine)
        # This is just for monitoring
        self._update_grid_status(grid, current_price)

        return None

    def _should_time_stop(self, grid: GridPosition) -> bool:
        """Check if time stop should trigger."""
        elapsed = datetime.now() - grid.entry_time
        return elapsed.total_seconds() > (self.time_stop_minutes * 60)

    def _should_kill_switch(self, grid: GridPosition, current_price: float) -> bool:
        """Check if kill switch should trigger based on unrealized loss."""
        # This would need account balance to calculate percentage
        # For now, use a simple threshold
        return abs(grid.unrealized_pnl) > 1000  # Placeholder

    async def _should_take_profit(
        self, symbol: str, current_price: float, features: dict[str, float]
    ) -> bool:
        """Check take profit conditions."""

        if self.tp_mode == "vwap":
            vwap = features.get("vwap", current_price)
            atr = self.last_atr.get(symbol, 0.0001)
            tp_distance = atr * self.tp_atr_multiplier

            # Check if price is near VWAP within TP distance
            return abs(current_price - vwap) <= tp_distance

        elif self.tp_mode == "bollinger_mid":
            bb_mid = features.get("bb_middle_20", features.get("sma_20", current_price))
            atr = self.last_atr.get(symbol, 0.0001)
            tp_distance = atr * self.tp_atr_multiplier

            return abs(current_price - bb_mid) <= tp_distance

        return False

    async def _close_grid(self, symbol: str, reason: str) -> Signal:
        """Close grid position."""

        grid = self.active_grids[symbol]

        # Determine close side (opposite of net position)
        net_long_size = sum(
            level.size for level in grid.grid_levels if level.side == OrderSide.BUY
        )
        net_short_size = sum(
            level.size for level in grid.grid_levels if level.side == OrderSide.SELL
        )

        if net_long_size > net_short_size:
            close_side = OrderSide.SELL
        else:
            close_side = OrderSide.BUY

        close_size = abs(net_long_size - net_short_size)

        # Remove grid from active tracking
        del self.active_grids[symbol]

        self.logger.info(
            f"Closing grid for {symbol}",
            reason=reason,
            close_side=close_side.value,
            close_size=close_size,
        )

        return Signal(
            symbol=symbol,
            side=close_side,
            strength=1.0,  # Full strength for close
            confidence=1.0,
            strategy_name="grid_martingale_close",
            timestamp=datetime.now(),
            features={"close_reason": reason},
        )

    def _update_grid_status(self, grid: GridPosition, current_price: float) -> None:
        """Update grid status and unrealized PnL."""

        # Simple unrealized PnL calculation (placeholder)
        # In real implementation, this would track actual fills
        total_pnl = 0.0

        for level in grid.grid_levels:
            if level.is_active:
                if level.side == OrderSide.BUY:
                    pnl = (current_price - level.price) * level.size
                else:
                    pnl = (level.price - current_price) * level.size
                total_pnl += pnl

        grid.unrealized_pnl = total_pnl
        grid.last_update = datetime.now()

    async def update_parameters(self, params: dict[str, Any]) -> None:
        """Update strategy parameters."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
                self.logger.info(f"Updated parameter {key} = {value}")

    def get_parameters(self) -> dict[str, Any]:
        """Get current strategy parameters."""
        return {
            "entry_mode": self.entry_mode,
            "grid_spacing_atr_multiplier": self.grid_spacing_atr_multiplier,
            "martingale_multiplier": self.martingale_multiplier,
            "max_grid_levels": self.max_grid_levels,
            "tp_mode": self.tp_mode,
            "tp_atr_multiplier": self.tp_atr_multiplier,
            "time_stop_minutes": self.time_stop_minutes,
            "max_unrealized_loss_pct": self.max_unrealized_loss_pct,
            "rsi_divergence_required": self.rsi_divergence_required,
            "rsi_oversold": self.rsi_oversold,
            "rsi_overbought": self.rsi_overbought,
            "volatility_expansion_threshold": self.volatility_expansion_threshold,
            "grid_expansion_factor": self.grid_expansion_factor,
        }

    def get_name(self) -> str:
        """Get strategy name."""
        return "grid_martingale"

    def get_active_grids(self) -> dict[str, GridPosition]:
        """Get active grid positions."""
        return self.active_grids.copy()

    def get_grid_status(self, symbol: str) -> dict[str, Any] | None:
        """Get status of grid for specific symbol."""
        if symbol not in self.active_grids:
            return None

        grid = self.active_grids[symbol]
        return {
            "symbol": grid.symbol,
            "total_levels": len(grid.grid_levels),
            "active_levels": sum(1 for level in grid.grid_levels if level.is_active),
            "total_size": grid.total_size,
            "unrealized_pnl": grid.unrealized_pnl,
            "entry_time": grid.entry_time,
            "elapsed_minutes": (datetime.now() - grid.entry_time).total_seconds() / 60,
        }
