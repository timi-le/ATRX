"""
Performance Metrics Module - Comprehensive Backtesting Analytics.

This module provides detailed performance analysis for backtesting results including:
- PnL calculation and tracking
- Risk-adjusted returns (Sharpe, Sortino, Calmar)
- Drawdown analysis
- Win/loss statistics
- Strategy comparison metrics
- Regime-based performance breakdown
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import numpy as np
import pandas as pd
import structlog

from backtester.execution_simulator import Fill
from core.interfaces.trading_interfaces import OrderSide, Position


class MetricType(Enum):
    """Types of performance metrics."""

    RETURN = "return"
    RISK = "risk"
    RATIO = "ratio"
    DRAWDOWN = "drawdown"
    TRADE = "trade"
    REGIME = "regime"


@dataclass
class TradeMetrics:
    """Metrics for individual trades."""

    trade_id: str
    symbol: str
    entry_time: datetime
    exit_time: datetime | None
    entry_price: float
    exit_price: float | None
    quantity: float
    side: OrderSide
    pnl: float
    commission: float
    slippage: float
    duration_hours: float | None
    max_favorable_excursion: float = 0.0  # MFE
    max_adverse_excursion: float = 0.0  # MAE
    is_winner: bool = False


@dataclass
class PeriodMetrics:
    """Performance metrics for a specific time period."""

    start_date: datetime
    end_date: datetime
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: float
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    payoff_ratio: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_commission: float
    total_slippage: float


@dataclass
class PerformanceConfig:
    """Configuration for performance calculation."""

    initial_capital: float = 100000.0
    risk_free_rate: float = 0.02  # 2% annual risk-free rate
    benchmark_return: float = 0.0  # Benchmark return for comparison
    trading_days_per_year: int = 252
    confidence_level: float = 0.95  # For VaR calculation

    # Drawdown settings
    drawdown_threshold: float = 0.05  # 5% drawdown threshold

    # Trade analysis settings
    min_trade_duration_hours: float = 1.0
    max_trade_duration_hours: float = 168.0  # 1 week

    # Regime analysis
    regime_labels: list[str] = field(
        default_factory=lambda: ["trending", "ranging", "volatile"]
    )


class PerformanceAnalyzer:
    """
    Comprehensive performance analysis for backtesting results.

    Calculates and tracks various performance metrics including:
    - Returns and risk metrics
    - Drawdown analysis
    - Trade-level statistics
    - Regime-based performance
    """

    def __init__(
        self,
        config: PerformanceConfig,
        logger: structlog.stdlib.BoundLogger | None = None,
    ):
        self.config = config
        self.logger = logger or structlog.get_logger(__name__)

        # Performance tracking
        self.equity_curve: list[tuple[datetime, float]] = []
        self.trades: list[TradeMetrics] = []
        self.daily_returns: list[tuple[datetime, float]] = []
        self.positions_history: list[tuple[datetime, dict[str, Position]]] = []

        # Current state
        self.current_equity = config.initial_capital
        self.peak_equity = config.initial_capital
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        self.drawdown_start: datetime | None = None
        self.max_drawdown_duration = 0.0

        # Trade tracking
        self.open_trades: dict[str, TradeMetrics] = {}
        self.trade_counter = 0
        self.fill_counter = 0  # Track total fills processed

        # Trade statistics tracking (ADDED)
        self.total_commission = 0.0
        self.total_slippage = 0.0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0.0
        self.total_loss = 0.0

        # Regime tracking
        self.regime_performance: dict[str, list[float]] = {
            regime: [] for regime in config.regime_labels
        }
        self.current_regime: str | None = None

        self.logger.info(
            "PerformanceAnalyzer initialized", initial_capital=config.initial_capital
        )

    def update_equity(self, timestamp: datetime, new_equity: float) -> None:
        """Update equity curve and calculate drawdown."""
        self.current_equity = new_equity
        self.equity_curve.append((timestamp, new_equity))

        # Update peak equity
        if new_equity > self.peak_equity:
            self.peak_equity = new_equity
            # End drawdown period
            if self.drawdown_start:
                duration = (
                    timestamp - self.drawdown_start
                ).total_seconds() / 3600  # hours
                self.max_drawdown_duration = max(self.max_drawdown_duration, duration)
                self.drawdown_start = None
            self.current_drawdown = 0.0
        else:
            # Calculate current drawdown
            self.current_drawdown = (self.peak_equity - new_equity) / self.peak_equity

            # Track maximum drawdown
            if self.current_drawdown > self.max_drawdown:
                self.max_drawdown = self.current_drawdown
                if not self.drawdown_start:
                    self.drawdown_start = timestamp

        # Calculate daily return if we have previous equity
        if len(self.equity_curve) > 1:
            prev_equity = self.equity_curve[-2][1]
            daily_return = (new_equity - prev_equity) / prev_equity
            self.daily_returns.append((timestamp, daily_return))

            # Add to regime performance if regime is set
            if self.current_regime:
                self.regime_performance[self.current_regime].append(daily_return)

    def update_positions(
        self, timestamp: datetime, positions: dict[str, Position]
    ) -> None:
        """Update positions history."""
        self.positions_history.append((timestamp, positions.copy()))

    def set_regime(self, regime: str) -> None:
        """Set current market regime for regime-based analysis."""
        if regime in self.config.regime_labels:
            self.current_regime = regime
        else:
            self.logger.warning("Unknown regime", regime=regime)

    def record_fill(self, fill: Fill, current_positions: dict[str, Position]) -> None:
        """Record a fill and update trade tracking with proper position-based P&L calculation."""
        symbol = fill.symbol
        self.logger.debug(
            "record_fill called",
            fill=fill,
            current_positions_from_exec_sim=current_positions,
            internal_tracked_positions_before=getattr(
                self, "_internal_positions", {}
            ).copy(),
        )

        # Increment fill counter
        self.fill_counter += 1

        # Always track commission and slippage
        self.total_commission += fill.commission
        self.total_slippage += abs(fill.slippage)

        # Get the current position for this symbol BEFORE this fill (from our internal tracking)
        previous_tracked_position = self._get_tracked_position(symbol)
        self.logger.debug(
            "Got previous_tracked_position",
            previous_tracked_position=previous_tracked_position,
        )

        # Get the position AFTER this fill from execution simulator
        # This represents the 'true' state of the position after the fill according to the simulator
        current_sim_position = current_positions.get(symbol)
        self.logger.debug(
            "Got current_sim_position from ExecutionSimulator",
            current_sim_position=current_sim_position,
        )

        # Track position changes and generate trades on position closures/reversals
        if fill.side == OrderSide.BUY:
            self._handle_buy_fill(fill, previous_tracked_position, current_sim_position)
        else:
            self._handle_sell_fill(
                fill, previous_tracked_position, current_sim_position
            )

        # Update our internal position tracking to reflect the new state from the simulator
        self._update_tracked_position(symbol, current_sim_position, fill.timestamp)
        self.logger.debug(
            "record_fill finished",
            internal_tracked_positions_after=getattr(
                self, "_internal_positions", {}
            ).copy(),
            current_equity=self.current_equity,
        )

        # Update equity curve (for visualization)
        self.equity_curve.append((fill.timestamp, self.current_equity))

        # Log progress every 1000 fills
        if self.fill_counter % 1000 == 0:
            self.logger.info(
                f"Processed {self.fill_counter} fills -> {len(self.trades)} completed trades, equity: ${self.current_equity:.2f}"
            )

    def _get_tracked_position(self, symbol: str) -> dict | None:
        """Get our internally tracked position for a symbol."""
        pos = getattr(self, "_internal_positions", {}).get(symbol)
        self.logger.debug(
            "_get_tracked_position called", symbol=symbol, position_found=pos
        )
        return pos

    def _update_tracked_position(
        self, symbol: str, current_sim_position: Position | None, timestamp: datetime
    ) -> None:
        """Update our internal position tracking based on the state from ExecutionSimulator."""
        if not hasattr(self, "_internal_positions"):
            self._internal_positions = {}

        self.logger.debug(
            "_update_tracked_position called",
            symbol=symbol,
            current_sim_position=current_sim_position,
            timestamp=timestamp,
            internal_pos_before_update=self._internal_positions.get(symbol),
        )

        existing_tracked_pos = self._internal_positions.get(symbol)

        if (
            current_sim_position and abs(current_sim_position.quantity) > 1e-9
        ):  # Use epsilon for float comparison
            if (
                not existing_tracked_pos
                or abs(existing_tracked_pos.get("quantity", 0)) < 1e-9
            ):  # Was new or zero
                entry_time = timestamp  # This fill opened/reopened the position
                self.logger.debug(
                    "Position is new or was zero. Setting new entry_time",
                    symbol=symbol,
                    entry_time=entry_time,
                )
            else:
                entry_time = existing_tracked_pos.get(
                    "entry_time", timestamp
                )  # Preserve existing entry time
                self.logger.debug(
                    "Position exists. Preserving entry_time",
                    symbol=symbol,
                    entry_time=entry_time,
                )

            self._internal_positions[symbol] = {
                "quantity": current_sim_position.quantity,
                "avg_price": current_sim_position.avg_price,
                "entry_time": entry_time,
            }
            self.logger.debug(
                "Updated/set internal position",
                symbol=symbol,
                new_tracked_pos=self._internal_positions[symbol],
            )
        else:
            # Position is closed (or effectively zero), remove from tracking
            if symbol in self._internal_positions:
                self.logger.debug(
                    "Position closed or zero. Removing from internal tracking",
                    symbol=symbol,
                    last_tracked_pos=self._internal_positions[symbol],
                )
                self._internal_positions.pop(symbol, None)
            else:
                self.logger.debug(
                    "Position closed or zero, and was not in internal tracking.",
                    symbol=symbol,
                )

    def _handle_buy_fill(
        self,
        fill: Fill,
        previous_tracked_position: dict | None,
        current_sim_position: Position | None,
    ) -> None:
        """Handle a BUY fill - opening long or closing short."""
        symbol = fill.symbol
        self.logger.debug(
            "_handle_buy_fill called",
            fill=fill,
            previous_tracked_position=previous_tracked_position,
            current_sim_position=current_sim_position,
        )

        # Check if we had a short position (in our internal tracking) that's being closed by this BUY fill
        if (
            previous_tracked_position
            and previous_tracked_position.get("quantity", 0) < -1e-9
        ):  # Epsilon for float comparison
            self.logger.debug(
                "Identified closing of a short position",
                symbol=symbol,
                prev_qty=previous_tracked_position["quantity"],
                fill_qty=fill.quantity,
            )
            prev_short_quantity = abs(previous_tracked_position["quantity"])
            close_quantity = min(fill.quantity, prev_short_quantity)

            if close_quantity > 1e-9:  # Ensure meaningful quantity is being closed
                self.logger.debug(
                    "Calculating P&L for closed short portion",
                    symbol=symbol,
                    close_qty=close_quantity,
                    prev_avg_price=previous_tracked_position["avg_price"],
                    fill_price=fill.price,
                )
                pnl = close_quantity * (
                    previous_tracked_position["avg_price"] - fill.price
                )
                pnl -= fill.commission
                self.logger.debug(
                    "P&L calculated for short close",
                    pnl=pnl,
                    commission=fill.commission,
                )

                self._create_completed_trade(
                    symbol=symbol,
                    side=OrderSide.SELL,  # Original side was SELL (short)
                    entry_price=previous_tracked_position["avg_price"],
                    exit_price=fill.price,
                    quantity=close_quantity,
                    entry_time=previous_tracked_position.get(
                        "entry_time", fill.timestamp
                    ),
                    exit_time=fill.timestamp,
                    pnl=pnl,
                    commission=fill.commission,
                )
            else:
                self.logger.debug(
                    "Close quantity too small, not creating trade for short close",
                    symbol=symbol,
                    close_qty=close_quantity,
                )
        else:
            self.logger.debug(
                "BUY fill did not close a previous short position or no previous short position existed.",
                symbol=symbol,
                prev_pos=previous_tracked_position,
            )

    def _handle_sell_fill(
        self,
        fill: Fill,
        previous_tracked_position: dict | None,
        current_sim_position: Position | None,
    ) -> None:
        """Handle a SELL fill - closing long or opening short."""
        symbol = fill.symbol
        self.logger.debug(
            "_handle_sell_fill called",
            fill=fill,
            previous_tracked_position=previous_tracked_position,
            current_sim_position=current_sim_position,
        )

        # Check if we had a long position (in our internal tracking) that's being closed by this SELL fill
        if (
            previous_tracked_position
            and previous_tracked_position.get("quantity", 0) > 1e-9
        ):  # Epsilon for float comparison
            self.logger.debug(
                "Identified closing of a long position",
                symbol=symbol,
                prev_qty=previous_tracked_position["quantity"],
                fill_qty=fill.quantity,
            )
            prev_long_quantity = previous_tracked_position["quantity"]
            close_quantity = min(fill.quantity, prev_long_quantity)

            if close_quantity > 1e-9:  # Ensure meaningful quantity is being closed
                self.logger.debug(
                    "Calculating P&L for closed long portion",
                    symbol=symbol,
                    close_qty=close_quantity,
                    prev_avg_price=previous_tracked_position["avg_price"],
                    fill_price=fill.price,
                )
                pnl = close_quantity * (
                    fill.price - previous_tracked_position["avg_price"]
                )
                pnl -= fill.commission
                self.logger.debug(
                    "P&L calculated for long close", pnl=pnl, commission=fill.commission
                )

                self._create_completed_trade(
                    symbol=symbol,
                    side=OrderSide.BUY,  # Original side was BUY (long)
                    entry_price=previous_tracked_position["avg_price"],
                    exit_price=fill.price,
                    quantity=close_quantity,
                    entry_time=previous_tracked_position.get(
                        "entry_time", fill.timestamp
                    ),
                    exit_time=fill.timestamp,
                    pnl=pnl,
                    commission=fill.commission,
                )
            else:
                self.logger.debug(
                    "Close quantity too small, not creating trade for long close",
                    symbol=symbol,
                    close_qty=close_quantity,
                )
        else:
            self.logger.debug(
                "SELL fill did not close a previous long position or no previous long position existed.",
                symbol=symbol,
                prev_pos=previous_tracked_position,
            )

    def _create_completed_trade(
        self,
        symbol: str,
        side: OrderSide,
        entry_price: float,
        exit_price: float,
        quantity: float,
        entry_time: datetime,
        exit_time: datetime,
        pnl: float,
        commission: float,
    ) -> None:
        """Create a TradeMetrics record for a completed round-trip trade."""
        self.trade_counter += 1

        trade = TradeMetrics(
            trade_id=f"trade_{self.trade_counter}",
            symbol=symbol,
            entry_time=entry_time,
            exit_time=exit_time,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
            side=side,
            pnl=pnl,
            commission=commission,
            slippage=0.0,  # Could be calculated if needed
            duration_hours=(exit_time - entry_time).total_seconds() / 3600,
            max_favorable_excursion=0.0,  # Could be tracked if needed
            max_adverse_excursion=0.0,  # Could be tracked if needed
            is_winner=(pnl > 0),
        )

        # Add to trades list
        self.trades.append(trade)

        # Update equity with realized P&L
        self.current_equity += pnl

        # Update running statistics
        if pnl > 0:
            self.winning_trades += 1
            self.total_profit += pnl
        else:
            self.losing_trades += 1
            self.total_loss += abs(pnl)

        # Update peak equity and drawdown tracking
        self._update_drawdown_tracking(exit_time)

        # Log completed trade (every 50 trades to see progress)
        if len(self.trades) % 50 == 0:
            self.logger.info(
                f"Completed trade #{len(self.trades)}: {symbol} {side.value} "
                f"P&L=${pnl:.2f}, Equity=${self.current_equity:.2f}"
            )

    def _update_drawdown_tracking(self, timestamp: datetime) -> None:
        """Update drawdown tracking after equity change."""
        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity
            self.current_drawdown = 0.0
            if self.drawdown_start:
                # End of drawdown period
                duration = (
                    timestamp - self.drawdown_start
                ).total_seconds() / 86400  # days
                if duration > self.max_drawdown_duration:
                    self.max_drawdown_duration = duration
                self.drawdown_start = None
        else:
            # In drawdown
            self.current_drawdown = (
                self.peak_equity - self.current_equity
            ) / self.peak_equity
            if self.current_drawdown > self.max_drawdown:
                self.max_drawdown = self.current_drawdown
            if not self.drawdown_start:
                self.drawdown_start = timestamp

    def calculate_metrics(
        self, start_date: datetime | None = None, end_date: datetime | None = None
    ) -> PeriodMetrics:
        """Calculate comprehensive performance metrics for a period."""
        # Filter data by date range if specified
        if start_date or end_date:
            equity_data = self._filter_equity_data(start_date, end_date)
            returns_data = self._filter_returns_data(start_date, end_date)
            trades_data = self._filter_trades_data(start_date, end_date)
        else:
            equity_data = self.equity_curve
            returns_data = self.daily_returns
            trades_data = self.trades

        if not equity_data or len(equity_data) < 2:
            return self._empty_metrics(start_date, end_date)

        # Basic return calculations
        initial_equity = equity_data[0][1]
        final_equity = equity_data[-1][1]
        total_return = (final_equity - initial_equity) / initial_equity

        # Time period
        period_start = equity_data[0][0]
        period_end = equity_data[-1][0]
        period_days = (period_end - period_start).days

        # Annualized return
        if period_days > 0:
            years = period_days / 365.25
            annualized_return = (1 + total_return) ** (1 / years) - 1
        else:
            annualized_return = 0.0

        # Volatility calculation
        if returns_data:
            returns = [r[1] for r in returns_data]
            volatility = np.std(returns) * np.sqrt(self.config.trading_days_per_year)
        else:
            volatility = 0.0

        # Risk-adjusted ratios
        excess_return = annualized_return - self.config.risk_free_rate
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0.0

        # Sortino ratio (using downside deviation)
        if returns_data:
            negative_returns = [r for r in [r[1] for r in returns_data] if r < 0]
            downside_deviation = (
                np.std(negative_returns) * np.sqrt(self.config.trading_days_per_year)
                if negative_returns
                else 0.0
            )
            sortino_ratio = (
                excess_return / downside_deviation if downside_deviation > 0 else 0.0
            )
        else:
            sortino_ratio = 0.0

        # Calmar ratio
        calmar_ratio = (
            annualized_return / self.max_drawdown if self.max_drawdown > 0 else 0.0
        )

        # Trade statistics
        trade_stats = self._calculate_trade_statistics(trades_data)

        return PeriodMetrics(
            start_date=period_start,
            end_date=period_end,
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=self.max_drawdown,
            max_drawdown_duration=self.max_drawdown_duration,
            win_rate=trade_stats["win_rate"],
            profit_factor=trade_stats["profit_factor"],
            avg_win=trade_stats["avg_win"],
            avg_loss=trade_stats["avg_loss"],
            payoff_ratio=trade_stats["payoff_ratio"],
            total_trades=trade_stats["total_trades"],
            winning_trades=trade_stats["winning_trades"],
            losing_trades=trade_stats["losing_trades"],
            total_commission=trade_stats["total_commission"],
            total_slippage=trade_stats["total_slippage"],
        )

    def _calculate_trade_statistics(
        self, trades: list[TradeMetrics]
    ) -> dict[str, float]:
        """Calculate trade-level statistics."""
        if not trades:
            return {
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "payoff_ratio": 0.0,
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "total_commission": 0.0,
                "total_slippage": 0.0,
            }

        winning_trades = [t for t in trades if t.is_winner]
        losing_trades = [t for t in trades if not t.is_winner]

        total_trades = len(trades)
        num_winning = len(winning_trades)
        num_losing = len(losing_trades)

        win_rate = num_winning / total_trades if total_trades > 0 else 0.0

        # Profit factor
        gross_profit = sum(t.pnl for t in winning_trades) if winning_trades else 0.0
        gross_loss = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 0.0
        profit_factor = (
            gross_profit / gross_loss
            if gross_loss > 0
            else float("inf")
            if gross_profit > 0
            else 0.0
        )

        # Average win/loss
        avg_win = gross_profit / num_winning if num_winning > 0 else 0.0
        avg_loss = gross_loss / num_losing if num_losing > 0 else 0.0

        # Payoff ratio
        payoff_ratio = (
            avg_win / avg_loss if avg_loss > 0 else float("inf") if avg_win > 0 else 0.0
        )

        # Costs
        total_commission = sum(t.commission for t in trades)
        total_slippage = sum(t.slippage for t in trades)

        return {
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "payoff_ratio": payoff_ratio,
            "total_trades": total_trades,
            "winning_trades": num_winning,
            "losing_trades": num_losing,
            "total_commission": total_commission,
            "total_slippage": total_slippage,
        }

    def calculate_regime_performance(self) -> dict[str, dict[str, float]]:
        """Calculate performance metrics by market regime."""
        regime_metrics = {}

        for regime, returns in self.regime_performance.items():
            if not returns:
                regime_metrics[regime] = {
                    "total_return": 0.0,
                    "volatility": 0.0,
                    "sharpe_ratio": 0.0,
                    "max_drawdown": 0.0,
                    "win_rate": 0.0,
                    "num_periods": 0,
                }
                continue

            # Calculate metrics for this regime
            total_return = np.prod([1 + r for r in returns]) - 1
            volatility = np.std(returns) * np.sqrt(self.config.trading_days_per_year)

            # Sharpe ratio
            excess_return = (
                np.mean(returns) * self.config.trading_days_per_year
                - self.config.risk_free_rate
            )
            sharpe_ratio = excess_return / volatility if volatility > 0 else 0.0

            # Drawdown calculation for regime
            cumulative = np.cumprod([1 + r for r in returns])
            running_max = np.maximum.accumulate(cumulative)
            drawdowns = (running_max - cumulative) / running_max
            max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0.0

            # Win rate
            positive_returns = [r for r in returns if r > 0]
            win_rate = len(positive_returns) / len(returns) if returns else 0.0

            regime_metrics[regime] = {
                "total_return": total_return,
                "volatility": volatility,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "win_rate": win_rate,
                "num_periods": len(returns),
            }

        return regime_metrics

    def calculate_rolling_metrics(self, window_days: int = 30) -> pd.DataFrame:
        """Calculate rolling performance metrics."""
        if len(self.daily_returns) < window_days:
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(self.daily_returns, columns=["date", "return"])
        df.set_index("date", inplace=True)

        # Calculate rolling metrics
        rolling_return = (
            df["return"].rolling(window=window_days).mean()
            * self.config.trading_days_per_year
        )
        rolling_volatility = df["return"].rolling(window=window_days).std() * np.sqrt(
            self.config.trading_days_per_year
        )
        rolling_sharpe = (
            rolling_return - self.config.risk_free_rate
        ) / rolling_volatility

        # Rolling drawdown
        cumulative_returns = (1 + df["return"]).cumprod()
        rolling_max = cumulative_returns.rolling(
            window=window_days, min_periods=1
        ).max()
        rolling_drawdown = (rolling_max - cumulative_returns) / rolling_max

        result = pd.DataFrame(
            {
                "rolling_return": rolling_return,
                "rolling_volatility": rolling_volatility,
                "rolling_sharpe": rolling_sharpe,
                "rolling_drawdown": rolling_drawdown,
            }
        )

        return result

    def calculate_var(self, confidence_level: float | None = None) -> float:
        """Calculate Value at Risk."""
        if not self.daily_returns:
            return 0.0

        confidence = confidence_level or self.config.confidence_level
        returns = [r[1] for r in self.daily_returns]

        # Historical VaR
        var = np.percentile(returns, (1 - confidence) * 100)

        return abs(var)

    def calculate_expected_shortfall(
        self, confidence_level: float | None = None
    ) -> float:
        """Calculate Expected Shortfall (Conditional VaR)."""
        if not self.daily_returns:
            return 0.0

        confidence = confidence_level or self.config.confidence_level
        returns = [r[1] for r in self.daily_returns]

        var = self.calculate_var(confidence)
        tail_returns = [r for r in returns if r <= -var]

        if not tail_returns:
            return 0.0

        return abs(np.mean(tail_returns))

    def get_equity_curve_df(self) -> pd.DataFrame:
        """Get equity curve as DataFrame."""
        if not self.equity_curve:
            return pd.DataFrame()

        df = pd.DataFrame(self.equity_curve, columns=["timestamp", "equity"])
        df.set_index("timestamp", inplace=True)

        # Add additional columns
        df["returns"] = df["equity"].pct_change()
        df["cumulative_returns"] = (1 + df["returns"]).cumprod() - 1

        # Drawdown calculation
        df["running_max"] = df["equity"].expanding().max()
        df["drawdown"] = (df["running_max"] - df["equity"]) / df["running_max"]

        return df

    def get_trades_df(self) -> pd.DataFrame:
        """Get trades as DataFrame."""
        if not self.trades:
            return pd.DataFrame()

        trades_data = []
        for trade in self.trades:
            trades_data.append(
                {
                    "trade_id": trade.trade_id,
                    "symbol": trade.symbol,
                    "entry_time": trade.entry_time,
                    "exit_time": trade.exit_time,
                    "entry_price": trade.entry_price,
                    "exit_price": trade.exit_price,
                    "quantity": trade.quantity,
                    "side": trade.side.value,
                    "pnl": trade.pnl,
                    "commission": trade.commission,
                    "slippage": trade.slippage,
                    "duration_hours": trade.duration_hours,
                    "is_winner": trade.is_winner,
                }
            )

        return pd.DataFrame(trades_data)

    def _filter_equity_data(
        self, start_date: datetime | None, end_date: datetime | None
    ) -> list[tuple[datetime, float]]:
        """Filter equity curve by date range."""
        filtered = []
        for timestamp, equity in self.equity_curve:
            if start_date and timestamp < start_date:
                continue
            if end_date and timestamp > end_date:
                continue
            filtered.append((timestamp, equity))
        return filtered

    def _filter_returns_data(
        self, start_date: datetime | None, end_date: datetime | None
    ) -> list[tuple[datetime, float]]:
        """Filter returns by date range."""
        filtered = []
        for timestamp, return_val in self.daily_returns:
            if start_date and timestamp < start_date:
                continue
            if end_date and timestamp > end_date:
                continue
            filtered.append((timestamp, return_val))
        return filtered

    def _filter_trades_data(
        self, start_date: datetime | None, end_date: datetime | None
    ) -> list[TradeMetrics]:
        """Filter trades by date range."""
        filtered = []
        for trade in self.trades:
            if start_date and trade.entry_time < start_date:
                continue
            if end_date and trade.entry_time > end_date:
                continue
            filtered.append(trade)
        return filtered

    def _empty_metrics(
        self, start_date: datetime | None, end_date: datetime | None
    ) -> PeriodMetrics:
        """Return empty metrics for periods with no data."""
        return PeriodMetrics(
            start_date=start_date or datetime.now(),
            end_date=end_date or datetime.now(),
            total_return=0.0,
            annualized_return=0.0,
            volatility=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            max_drawdown=0.0,
            max_drawdown_duration=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            payoff_ratio=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            total_commission=0.0,
            total_slippage=0.0,
        )

    def reset(self) -> None:
        """Reset all performance tracking."""
        self.equity_curve.clear()
        self.trades.clear()
        self.daily_returns.clear()
        self.positions_history.clear()
        self.open_trades.clear()

        self.current_equity = self.config.initial_capital
        self.peak_equity = self.config.initial_capital
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        self.drawdown_start = None
        self.max_drawdown_duration = 0.0
        self.trade_counter = 0
        self.fill_counter = 0  # Reset fill counter

        # Reset trade statistics tracking (ADDED)
        self.total_commission = 0.0
        self.total_slippage = 0.0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0.0
        self.total_loss = 0.0

        for regime in self.regime_performance:
            self.regime_performance[regime].clear()

        self.logger.info("PerformanceAnalyzer reset")


# Utility functions
def create_performance_config(
    initial_capital: float = 100000.0,
    risk_free_rate: float = 0.02,
    trading_days_per_year: int = 252,
) -> PerformanceConfig:
    """Create performance configuration with common defaults."""
    return PerformanceConfig(
        initial_capital=initial_capital,
        risk_free_rate=risk_free_rate,
        trading_days_per_year=trading_days_per_year,
    )


def compare_strategies(
    analyzers: dict[str, PerformanceAnalyzer],
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> pd.DataFrame:
    """Compare performance metrics across multiple strategies."""
    comparison_data = []

    for strategy_name, analyzer in analyzers.items():
        metrics = analyzer.calculate_metrics(start_date, end_date)

        comparison_data.append(
            {
                "strategy": strategy_name,
                "total_return": metrics.total_return,
                "annualized_return": metrics.annualized_return,
                "volatility": metrics.volatility,
                "sharpe_ratio": metrics.sharpe_ratio,
                "sortino_ratio": metrics.sortino_ratio,
                "calmar_ratio": metrics.calmar_ratio,
                "max_drawdown": metrics.max_drawdown,
                "win_rate": metrics.win_rate,
                "profit_factor": metrics.profit_factor,
                "total_trades": metrics.total_trades,
                "total_commission": metrics.total_commission,
            }
        )

    return pd.DataFrame(comparison_data).set_index("strategy")
