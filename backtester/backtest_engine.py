"""
Backtest Engine - Core Backtesting Framework.

This module provides the main backtesting engine that orchestrates the entire
backtesting process by integrating:
- Historical data replay
- Feature computation
- Regime detection
- ML prediction
- Strategy execution
- Order simulation
- Performance analysis

The engine simulates the complete live trading pipeline using historical data.
"""

print("[MODULE IMPORT DEBUG] backtester/backtest_engine.py is being imported!")

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from collections.abc import Callable

import pandas as pd
import structlog

from backtester.execution_simulator import ExecutionConfig, ExecutionSimulator
from backtester.market_replay import (
    DataPoint,
    MarketReplay,
    create_replay_config,
)
from backtester.performance_metrics import PerformanceAnalyzer, PerformanceConfig
from core.feature_engine import HighPerformanceFeatureEngine
from core.interfaces.data_interfaces import OHLCV, MarketData
from core.interfaces.trading_interfaces import (
    Order,
    OrderType,
    Position,
    Signal,
)
from core.ml_predictor import MLPredictor, Prediction
from core.position_sizer import KellyPositionSizer
from core.regime_detector import (
    RegimeFeatures,
    RegimeOutput,
    RegimeType,
    RuleBasedRegimeDetector,
)
from core.risk_manager import CoreRiskManager
from core.strategy_switcher import StrategySwitcher


class BacktestMode(Enum):
    """Backtesting modes."""

    FULL_PIPELINE = "full_pipeline"  # Complete trading pipeline
    STRATEGY_ONLY = "strategy_only"  # Strategy signals only
    EXECUTION_ONLY = "execution_only"  # Execution simulation only


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""

    # Data configuration
    symbols: list[str] = field(default_factory=lambda: ["EURUSD", "GBPUSD", "USDJPY"])
    timeframe: str = "1m"
    start_date: datetime | None = None
    end_date: datetime | None = None
    data_path: str = "data/historical"

    # Backtesting mode
    mode: BacktestMode = BacktestMode.FULL_PIPELINE

    # Initial capital and account settings
    initial_capital: float = 100000.0
    base_currency: str = "USD"
    leverage: float = 1.0

    # Strategy configuration
    strategies: list[str] = field(
        default_factory=lambda: ["grid_martingale", "breakout_trend"]
    )
    strategy_weights: dict[str, float] = field(default_factory=dict)

    # Risk management
    max_position_size: float = 0.1  # 10% of capital per position
    max_total_exposure: float = 0.5  # 50% total exposure
    stop_loss_pct: float = 0.02  # 2% stop loss
    take_profit_pct: float = 0.04  # 4% take profit

    # Execution settings
    enable_slippage: bool = True
    enable_commission: bool = True
    enable_latency: bool = True

    # Performance tracking
    benchmark_symbol: str | None = None
    save_results: bool = True
    results_path: str = "outputs/backtest_results"

    # Logging and debugging
    log_level: str = "INFO"
    save_trades: bool = True
    save_equity_curve: bool = True

    def __post_init__(self):
        if not self.strategy_weights:
            # Equal weights for all strategies
            weight = 1.0 / len(self.strategies) if self.strategies else 1.0
            self.strategy_weights = {strategy: weight for strategy in self.strategies}


@dataclass
class BacktestState:
    """Current state of the backtest."""

    current_time: datetime
    current_equity: float
    current_positions: dict[str, Position]
    pending_orders: dict[str, Order]
    last_market_data: dict[str, MarketData | OHLCV]
    data_history: dict[str, list[DataPoint]] = field(
        default_factory=dict
    )  # History for feature calculation
    current_regime: RegimeType | None = None
    current_features: RegimeFeatures | None = None
    current_regime_output: RegimeOutput | None = None
    current_ml_prediction: Prediction | None = None


class BacktestEngine:
    """
    Main backtesting engine that orchestrates the complete backtesting process.

    Integrates all components of the trading system:
    - Market data replay
    - Feature computation
    - Regime detection
    - Strategy execution
    - Order simulation
    - Performance analysis
    """

    def __init__(
        self,
        config: BacktestConfig,
        feature_engine: HighPerformanceFeatureEngine | None = None,
        regime_detector: RuleBasedRegimeDetector | None = None,
        strategy_switcher: StrategySwitcher | None = None,
        position_sizer: KellyPositionSizer | None = None,
        risk_manager: CoreRiskManager | None = None,
        ml_predictor: MLPredictor | None = None,
        logger: structlog.stdlib.BoundLogger | None = None,
    ):
        self.config = config
        self.logger = logger or structlog.get_logger(__name__)

        # Core components
        self.feature_engine = feature_engine
        self.regime_detector = regime_detector
        self.strategy_switcher = strategy_switcher
        self.position_sizer = position_sizer
        self.risk_manager = risk_manager
        self.ml_predictor = ml_predictor

        # Backtesting components
        self.market_replay: MarketReplay | None = None
        self.execution_simulator: ExecutionSimulator | None = None
        self.performance_analyzer: PerformanceAnalyzer | None = None

        # State tracking
        self.state: BacktestState | None = None
        self.is_running = False
        self.results: dict[str, Any] = {}

        # Event callbacks
        self.on_data_callbacks: list[Callable] = []
        self.on_signal_callbacks: list[Callable] = []
        self.on_order_callbacks: list[Callable] = []
        self.on_fill_callbacks: list[Callable] = []

        self.logger.info(
            "BacktestEngine initialized",
            symbols=config.symbols,
            timeframe=config.timeframe,
            mode=config.mode.value,
            initial_capital=config.initial_capital,
        )

    async def initialize(self) -> None:
        """Initialize all backtesting components."""
        try:
            # Initialize market replay
            replay_config = create_replay_config(
                symbols=self.config.symbols,
                timeframe=self.config.timeframe,
                start_date=self.config.start_date,
                end_date=self.config.end_date,
                data_path=self.config.data_path,
                replay_speed=0.0,  # As fast as possible
            )
            self.market_replay = MarketReplay(replay_config, self.logger)

            # Initialize execution simulator
            execution_config = ExecutionConfig(
                min_latency_ms=10 if self.config.enable_latency else 0,
                max_latency_ms=100 if self.config.enable_latency else 0,
                base_slippage_bps=0.5 if self.config.enable_slippage else 0.0,
                commission_per_lot=7.0 if self.config.enable_commission else 0.0,
                rejection_rate=0.01,
                partial_fill_probability=0.05,
            )
            self.execution_simulator = ExecutionSimulator(execution_config, self.logger)

            # Initialize performance analyzer
            performance_config = PerformanceConfig(
                initial_capital=self.config.initial_capital,
                risk_free_rate=0.02,
                trading_days_per_year=252,
            )
            self.performance_analyzer = PerformanceAnalyzer(
                performance_config, self.logger
            )

            # Initialize state
            self.state = BacktestState(
                current_time=datetime.now(),
                current_equity=self.config.initial_capital,
                current_positions={},
                pending_orders={},
                last_market_data={},
            )

            self.logger.info("BacktestEngine initialization completed")

        except Exception as e:
            self.logger.error("Error initializing BacktestEngine", error=str(e))
            raise

    async def run(self) -> dict[str, Any]:
        """Run the complete backtest."""
        print("[DEBUG] BacktestEngine.run() method called!")
        self.logger.info(
            f"Starting backtest run: {self.config.start_date} to {self.config.end_date}, symbols={self.config.symbols}, mode={self.config.mode}"
        )

        if (
            not self.market_replay
            or not self.execution_simulator
            or not self.performance_analyzer
        ):
            await self.initialize()

        self.is_running = True
        self.logger.info("Starting backtest")

        try:
            # Load historical data
            await self.market_replay.load_data()

            # Initialize equity tracking
            self.performance_analyzer.update_equity(
                self.state.current_time, self.state.current_equity
            )

            # Main backtesting loop
            data_point_count = 0
            async for data_point in self.market_replay.stream():
                if not self.is_running:
                    break

                data_point_count += 1
                if data_point_count % 1000 == 0:
                    self.logger.info(
                        f"Processing data point {data_point_count}",
                        symbol=data_point.symbol,
                        timestamp=data_point.timestamp,
                    )

                await self._process_data_point(data_point)

            self.logger.info(
                f"Backtest loop completed. Total data points processed: {data_point_count}"
            )

            # Finalize results
            await self._finalize_backtest()

            self.logger.info("Backtest completed successfully")
            return self.results

        except Exception as e:
            self.logger.error("Error during backtest", error=str(e))
            raise
        finally:
            self.is_running = False

    async def _process_data_point(self, data_point: DataPoint) -> None:
        """Process a single data point through the full pipeline."""
        self.logger.debug(
            "BacktestEngine._process_data_point entered",
            data_point_symbol=data_point.symbol,
            data_point_timestamp=data_point.timestamp,
        )

        # 1. Update data history for feature calculation
        self._update_data_history(data_point)

        # 2. Update market data for execution and PnL
        await self.execution_simulator.update_market_data(data_point)
        self.state.last_market_data[data_point.symbol] = data_point.data

        # 3. Process signals and orders based on the configured mode
        if self.config.mode == BacktestMode.FULL_PIPELINE:
            await self._process_full_pipeline(data_point)
        elif self.config.mode == BacktestMode.STRATEGY_ONLY:
            await self._process_strategy_only(data_point)
        elif self.config.mode == BacktestMode.EXECUTION_ONLY:
            await self._process_execution_only(data_point)

        # 4. Wait for all submitted orders to be executed
        await self.execution_simulator.wait_for_pending_orders(timeout=5.0)

        # 5. Update performance with any new fills
        await self._update_performance(data_point.timestamp)

    def _update_data_history(self, data_point: DataPoint):
        """Update the rolling window of historical data."""
        symbol = data_point.symbol
        if symbol not in self.state.data_history:
            self.state.data_history[symbol] = []

        self.state.data_history[symbol].append(data_point)

        # Keep buffer size manageable, e.g., 500 points, which should be enough for most indicators
        max_history = 500
        if len(self.state.data_history[symbol]) > max_history:
            self.state.data_history[symbol] = self.state.data_history[symbol][
                -max_history:
            ]

    async def _process_full_pipeline(self, data_point: DataPoint) -> None:
        """Process data through the complete trading pipeline."""
        self.logger.debug(
            "BacktestEngine._process_full_pipeline entered",
            data_point_symbol=data_point.symbol,
            data_point_timestamp=data_point.timestamp,
        )
        # 1. Feature computation
        if self.feature_engine:
            features = await self._compute_features(data_point)
            if features:
                self.state.current_features = features

        # 2. Regime detection
        if self.regime_detector:
            regime_output = await self._detect_regime(data_point)
            if regime_output:
                self.state.current_regime_output = regime_output
                self.state.current_regime = regime_output.regime
                self.performance_analyzer.set_regime(regime_output.regime)

        # 3. ML prediction
        if self.ml_predictor:
            ml_prediction = await self._predict_signal(data_point)
            if ml_prediction:
                self.state.current_ml_prediction = ml_prediction

        # 4. Strategy signal generation
        if self.strategy_switcher:
            signals = await self._generate_signals(data_point)

            # Process each signal
            for signal in signals:
                await self._process_signal(signal)

    async def _process_strategy_only(self, data_point: DataPoint) -> None:
        """Process strategy signals only (no execution simulation)."""
        # Generate signals and track performance without actual execution
        if self.strategy_switcher:
            signals = await self._generate_signals(data_point)

            for signal in signals:
                # Call signal callbacks
                for callback in self.on_signal_callbacks:
                    await callback(signal)

                # Track theoretical performance
                await self._track_theoretical_performance(signal, data_point)

    async def _process_execution_only(self, data_point: DataPoint) -> None:
        """Process execution simulation only (with predefined orders)."""
        # This mode would be used to test execution quality with known orders
        # Implementation would depend on having a predefined order list

    async def _compute_features(
        self, data_point: DataPoint
    ) -> RegimeFeatures | None:
        """Compute features for the current data point."""
        try:
            if not self.feature_engine:
                return None

            # For now, we will create a DataFrame from the recent data points
            recent_data_points = self._get_recent_market_data(
                data_point.symbol, lookback=200
            )  # Need enough data for indicators
            if not recent_data_points:
                return None

            # Convert list of DataPoint data to DataFrame
            data_list = [dp.data for dp in recent_data_points]
            df = pd.DataFrame(data_list)
            df["time"] = pd.to_datetime([dp.timestamp for dp in recent_data_points])
            df = df.set_index("time")

            features = await self.feature_engine.compute_features(df, data_point.symbol)

            if features:
                self.logger.debug(
                    "Features computed",
                    symbol=data_point.symbol,
                    features=features.to_dict(),
                )

            return features

        except Exception as e:
            self.logger.warning("Error computing features", error=str(e))
            return None

    async def _detect_regime(self, data_point: DataPoint) -> RegimeOutput | None:
        """Detect market regime using the feature set."""
        if not self.regime_detector or not self.state.current_features:
            return None

        try:
            regime_output = await self.regime_detector.predict(
                self.state.current_features
            )
            self.state.current_regime_output = regime_output
            self.state.current_regime = regime_output.regime
            self.logger.debug(
                "Regime detected",
                regime=self.state.current_regime.value,
                confidence=regime_output.confidence,
            )
            return regime_output
        except Exception as e:
            self.logger.error(
                "Error during regime detection",
                error=str(e),
                features=self.state.current_features.to_dict(),
            )
            return None

    async def _predict_signal(self, data_point: DataPoint) -> Prediction | None:
        """Generate a signal from the ML predictor."""
        if not self.ml_predictor or not self.state.current_features:
            return None

        try:
            # The features are already computed and stored in the state
            features_df = pd.DataFrame([self.state.current_features.features])

            prediction = await self.ml_predictor.predict(features_df)
            self.logger.info("ML prediction generated", prediction=prediction)
            return prediction

        except Exception as e:
            self.logger.error("Error generating ML prediction", error=str(e))
            return None

    async def _generate_signals(self, data_point: DataPoint) -> list[Signal]:
        """Generate trading signals based on the current regime and features."""
        signals = []

        if self.state.current_regime_output and self.strategy_switcher:
            decision = await self.strategy_switcher.choose_strategy(
                regime=self.state.current_regime_output,
                ml_prediction=self.state.current_ml_prediction,
                features=self.state.current_features.features,
                market_data=data_point,
            )

            if decision and decision.signal:
                self.logger.info("Signal generated", signal=decision.signal)
                # Fire signal callbacks
                for callback in self.on_signal_callbacks:
                    await callback(decision.signal)
                return [decision.signal]

        return signals

    async def _process_signal(self, signal: Signal) -> None:
        """Process a generated signal to create and submit an order."""

        # 1. Calculate position size
        position_size = await self._calculate_position_size(signal)
        if position_size <= 0:
            self.logger.info(
                "Position size is zero or less, skipping order.", signal=signal
            )
            return

        # 2. Check pre-trade risk limits
        risk_ok = await self._check_risk_limits(signal, position_size)
        if not risk_ok:
            self.logger.warning(
                "Trade rejected due to pre-trade risk checks.",
                signal=signal,
                size=position_size,
            )
            return

        # 3. Create order
        order = await self._create_order(signal, position_size)

        # 4. Submit order to execution simulator
        if order:
            await self._submit_order(order)

    async def _calculate_position_size(self, signal: Signal) -> float:
        """Calculate position size using the position sizer component."""
        if not self.position_sizer:
            # Fallback to a fixed fractional size if no sizer is present
            return self.state.current_equity * 0.01

        # Build the inputs for the real KellyPositionSizer
        trade_signal_input = TradeSignalInput(
            symbol=signal.symbol,
            side=signal.side,
            signal_confidence=signal.confidence,
            take_profit_pips=signal.take_profit_pips,
            stop_loss_pips=signal.stop_loss_pips,
            win_probability=signal.win_probability,
            reward_risk_ratio=signal.take_profit_pips / signal.stop_loss_pips
            if signal.stop_loss_pips > 0
            else None,
            current_price=signal.price,
            volatility_atr=self.state.current_features.atr
            if self.state.current_features
            else 0,
            timestamp=signal.timestamp,
            strategy_name=signal.strategy_name,
            features=signal.features,
        )

        portfolio_state = PortfolioState(
            total_capital=self.state.current_equity,
            current_drawdown=self.performance_analyzer.get_current_drawdown(),
            daily_pnl=0,  # This would require more detailed tracking
            open_positions=list(self.state.current_positions.values()),
            volatility_history=[],  # This would require more detailed tracking
            performance_history=[],  # This would require more detailed tracking
        )

        try:
            sizing_result = await self.position_sizer.calculate_position_size(
                signal=trade_signal_input, portfolio_state=portfolio_state
            )
            # Convert the fractional size to units
            position_value = self.state.current_equity * sizing_result.position_size
            unit_size = int(position_value / signal.price)
            return max(1000, unit_size)  # Enforce minimum
        except Exception as e:
            self.logger.error("Error during position sizing", error=str(e))
            return 0.0

    async def _check_risk_limits(self, signal: Signal, position_size: float) -> bool:
        """Check pre-trade risk limits."""
        if not self.risk_manager:
            # Basic risk checks
            total_exposure = sum(
                abs(pos.quantity * pos.avg_price)
                for pos in self.state.current_positions.values()
            )
            max_exposure = self.state.current_equity * self.config.max_total_exposure

            return total_exposure + position_size <= max_exposure

        # Create a dummy order for risk checking
        order = Order(
            order_id=str(uuid.uuid4()),
            symbol=signal.symbol,
            side=signal.side,
            order_type=OrderType.MARKET,
            quantity=position_size,
            timestamp=signal.timestamp,
        )

        return await self.risk_manager.check_pre_trade_risk(
            order, self.state.current_positions, self.state.current_equity
        )

    async def _create_order(
        self, signal: Signal, position_size: float
    ) -> Order | None:
        """Create an order from a signal."""
        try:
            order_id = f"order_{uuid.uuid4().hex[:8]}"

            order = Order(
                order_id=order_id,
                symbol=signal.symbol,
                side=signal.side,
                order_type=OrderType.MARKET,  # Default to market orders
                quantity=position_size,
                timestamp=signal.timestamp,
            )

            return order

        except Exception as e:
            self.logger.error("Error creating order", error=str(e))
            return None

    async def _submit_order(self, order: Order) -> None:
        """Submit an order for execution."""
        try:
            # Submit to execution simulator
            order_id = await self.execution_simulator.submit_order(order)

            # Track pending order
            self.state.pending_orders[order_id] = order

            # Call order callbacks
            for callback in self.on_order_callbacks:
                await callback(order)

            self.logger.debug(
                "Order submitted",
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side.value,
                quantity=order.quantity,
            )

        except Exception as e:
            self.logger.error("Error submitting order", error=str(e))

    async def _update_performance(self, timestamp: datetime) -> None:
        """Update performance tracking."""
        self.logger.debug("_update_performance called", current_time=timestamp)
        try:
            # Get current positions from execution simulator
            current_positions = self.execution_simulator.get_positions()
            self.state.current_positions = current_positions

            # Process any new fills (only unprocessed ones)
            new_fills = self.execution_simulator.get_fills()
            fill_count = 0
            for fill in new_fills:
                self.performance_analyzer.record_fill(fill, current_positions)
                fill_count += 1

                # Call fill callbacks
                for callback in self.on_fill_callbacks:
                    await callback(fill)

            # Mark fills as processed in the execution simulator
            if fill_count > 0:
                self.execution_simulator.mark_fills_processed(fill_count)
                # Log progress every 1000 fills to track performance
                if self.execution_simulator.processed_fill_count % 1000 == 0:
                    self.logger.info(
                        f"Processed {self.execution_simulator.processed_fill_count} fills, current equity: ${self.performance_analyzer.current_equity:.2f}"
                    )

            # Get equity from performance analyzer (which tracks fills)
            equity = self.performance_analyzer.current_equity

            # Add unrealized PnL from current positions
            for symbol, position in current_positions.items():
                if symbol in self.state.last_market_data:
                    market_data = self.state.last_market_data[symbol]

                    if isinstance(market_data, MarketData):
                        current_price = market_data.mid
                    else:
                        current_price = market_data.close

                    unrealized_pnl = (
                        current_price - position.avg_price
                    ) * position.quantity
                    equity += unrealized_pnl

            self.state.current_equity = equity

            # Update performance analyzer
            self.performance_analyzer.update_equity(timestamp, equity)
            self.performance_analyzer.update_positions(timestamp, current_positions)

        except Exception as e:
            self.logger.error("Error updating performance", error=str(e))

    def _get_recent_market_data(
        self, symbol: str, lookback: int = 200
    ) -> list[DataPoint] | None:
        """Get recent market data for a symbol from the historical buffer."""
        if symbol not in self.state.data_history:
            return None

        # Ensure we have enough data for the lookback period
        if len(self.state.data_history[symbol]) < lookback:
            return None

        return self.state.data_history[symbol][-lookback:]

    async def _track_theoretical_performance(
        self, signal: Signal, data_point: DataPoint
    ) -> None:
        """Track theoretical performance for strategy-only mode."""
        # This would track what would have happened if the signal was executed
        # Implementation depends on specific requirements

    async def _finalize_backtest(self) -> None:
        """Finalize backtest and compile results."""
        try:
            # Wait for all pending orders to complete
            if self.execution_simulator:
                await self.execution_simulator.wait_for_pending_orders(timeout=30.0)
                self.logger.info("All pending orders processed before finalization")

            # Calculate final performance metrics
            final_metrics = self.performance_analyzer.calculate_metrics()

            # Get execution statistics
            execution_stats = self.execution_simulator.get_statistics()

            # Get regime performance
            regime_performance = (
                self.performance_analyzer.calculate_regime_performance()
            )

            # Compile results
            self.results = {
                "config": {
                    "symbols": self.config.symbols,
                    "timeframe": self.config.timeframe,
                    "start_date": self.config.start_date.isoformat()
                    if self.config.start_date
                    else None,
                    "end_date": self.config.end_date.isoformat()
                    if self.config.end_date
                    else None,
                    "initial_capital": self.config.initial_capital,
                    "mode": self.config.mode.value,
                },
                "performance": {
                    "total_return": final_metrics.total_return,
                    "annualized_return": final_metrics.annualized_return,
                    "volatility": final_metrics.volatility,
                    "sharpe_ratio": final_metrics.sharpe_ratio,
                    "sortino_ratio": final_metrics.sortino_ratio,
                    "calmar_ratio": final_metrics.calmar_ratio,
                    "max_drawdown": final_metrics.max_drawdown,
                    "max_drawdown_duration": final_metrics.max_drawdown_duration,
                    "win_rate": final_metrics.win_rate,
                    "profit_factor": final_metrics.profit_factor,
                    "total_trades": final_metrics.total_trades,
                    "total_commission": final_metrics.total_commission,
                    "total_slippage": final_metrics.total_slippage,
                },
                "execution": execution_stats,
                "regime_performance": regime_performance,
                "final_equity": self.state.current_equity,
                "final_positions": {
                    symbol: {
                        "quantity": pos.quantity,
                        "avg_price": pos.avg_price,
                        "unrealized_pnl": pos.unrealized_pnl,
                    }
                    for symbol, pos in self.state.current_positions.items()
                },
            }

            # Save results if configured
            if self.config.save_results:
                await self._save_results()

            self.logger.info(
                "Backtest finalized",
                total_return=final_metrics.total_return,
                sharpe_ratio=final_metrics.sharpe_ratio,
                max_drawdown=final_metrics.max_drawdown,
                total_trades=final_metrics.total_trades,
            )

        except Exception as e:
            self.logger.error("Error finalizing backtest", error=str(e))

    async def _save_results(self) -> None:
        """Save backtest results to files."""
        try:
            import json
            from pathlib import Path

            # Create results directory
            results_dir = Path(self.config.results_path)
            results_dir.mkdir(parents=True, exist_ok=True)

            # Generate timestamp for unique filenames
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save main results
            results_file = results_dir / f"backtest_results_{timestamp}.json"
            with open(results_file, "w") as f:
                json.dump(self.results, f, indent=2, default=str)

            # Save equity curve
            if self.config.save_equity_curve:
                equity_df = self.performance_analyzer.get_equity_curve_df()
                if not equity_df.empty:
                    equity_file = results_dir / f"equity_curve_{timestamp}.csv"
                    equity_df.to_csv(equity_file)

            # Save trades
            if self.config.save_trades:
                trades_df = self.performance_analyzer.get_trades_df()
                if not trades_df.empty:
                    trades_file = results_dir / f"trades_{timestamp}.csv"
                    trades_df.to_csv(trades_file, index=False)

            self.logger.info("Results saved", results_dir=str(results_dir))

        except Exception as e:
            self.logger.error("Error saving results", error=str(e))

    def stop(self) -> None:
        """Stop the backtest."""
        self.is_running = False
        if self.market_replay:
            self.market_replay.stop()
        self.logger.info("Backtest stopped")

    def add_data_callback(self, callback: Callable) -> None:
        """Add callback for data events."""
        self.on_data_callbacks.append(callback)

    def add_signal_callback(self, callback: Callable) -> None:
        """Add callback for signal events."""
        self.on_signal_callbacks.append(callback)

    def add_order_callback(self, callback: Callable) -> None:
        """Add callback for order events."""
        self.on_order_callbacks.append(callback)

    def add_fill_callback(self, callback: Callable) -> None:
        """Add callback for fill events."""
        self.on_fill_callbacks.append(callback)


# Utility functions
def create_backtest_config(
    symbols: list[str],
    timeframe: str = "1m",
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    initial_capital: float = 100000.0,
    mode: BacktestMode = BacktestMode.FULL_PIPELINE,
) -> BacktestConfig:
    """Create backtest configuration with common defaults."""
    return BacktestConfig(
        symbols=symbols,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        mode=mode,
    )


async def run_simple_backtest(
    symbols: list[str],
    start_date: datetime,
    end_date: datetime,
    initial_capital: float = 100000.0,
    strategies: list[str] | None = None,
) -> dict[str, Any]:
    """Run a simple backtest with minimal configuration."""
    config = create_backtest_config(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
    )

    if strategies:
        config.strategies = strategies

    engine = BacktestEngine(config)
    await engine.initialize()

    return await engine.run()
