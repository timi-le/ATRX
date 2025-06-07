"""
Backtester Module - High-Fidelity Backtesting Framework.

This module provides a comprehensive backtesting framework for FX trading strategies
that simulates live execution using historical data and the full strategy pipeline.

Key Components:
- MarketReplay: Historical data streaming and replay
- ExecutionSimulator: Realistic order execution simulation
- PerformanceMetrics: Comprehensive performance analysis
- BacktestEngine: Main orchestration engine

Features:
- Support for multiple timeframes (M1, H1, D1, etc.)
- Realistic execution simulation with slippage, latency, and commission
- Comprehensive performance metrics and risk analysis
- Regime-based performance breakdown
- Strategy comparison capabilities
- Full pipeline integration (features → regime → ML → strategy → execution)
"""

from .market_replay import (
    MarketReplay,
    ReplayConfig,
    DataPoint,
    TimeFrame,
    create_replay_config
)

from .execution_simulator import (
    ExecutionSimulator,
    ExecutionConfig,
    Fill,
    FillType,
    RealisticExecutionSimulator,
    create_execution_config
)

from .performance_metrics import (
    PerformanceAnalyzer,
    PerformanceConfig,
    TradeMetrics,
    PeriodMetrics,
    MetricType,
    create_performance_config,
    compare_strategies
)

from .backtest_engine import (
    BacktestEngine,
    BacktestConfig,
    BacktestMode,
    BacktestState,
    create_backtest_config,
    run_simple_backtest
)

__all__ = [
    # Market Replay
    "MarketReplay",
    "ReplayConfig", 
    "DataPoint",
    "TimeFrame",
    "create_replay_config",
    
    # Execution Simulation
    "ExecutionSimulator",
    "ExecutionConfig",
    "Fill",
    "FillType",
    "RealisticExecutionSimulator",
    "create_execution_config",
    
    # Performance Metrics
    "PerformanceAnalyzer",
    "PerformanceConfig",
    "TradeMetrics",
    "PeriodMetrics",
    "MetricType",
    "create_performance_config",
    "compare_strategies",
    
    # Backtest Engine
    "BacktestEngine",
    "BacktestConfig",
    "BacktestMode",
    "BacktestState",
    "create_backtest_config",
    "run_simple_backtest"
]

__version__ = "1.0.0"
__author__ = "FX Quant System"
__description__ = "High-fidelity backtesting framework for FX trading strategies" 