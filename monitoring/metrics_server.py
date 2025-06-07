"""
Prometheus Metrics Server for FX Quant Trading System

This module provides comprehensive monitoring infrastructure using Prometheus
to track trading performance, execution quality, system health, and more.

Features:
- Real-time PnL and equity tracking
- Trade execution metrics
- Strategy and regime monitoring  
- System health and error tracking
- Position exposure monitoring
- Latency and performance metrics
"""

import time
import threading
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from prometheus_client import (
    start_http_server, 
    Gauge, 
    Counter, 
    Histogram, 
    Info,
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST
)
from dataclasses import dataclass
import json

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class MetricsConfig:
    """Configuration for metrics server."""
    port: int = 9000
    host: str = '0.0.0.0'  # Listen on all interfaces to allow Docker access
    registry: Optional[CollectorRegistry] = None
    update_interval: float = 1.0  # seconds

class FXTradingMetrics:
    """
    Comprehensive metrics collector for FX Trading System.
    
    This class manages all Prometheus metrics for the trading system,
    providing real-time observability into system performance.
    """
    
    def __init__(self, config: MetricsConfig = None):
        """Initialize the metrics collector."""
        self.config = config or MetricsConfig()
        self.registry = self.config.registry or CollectorRegistry()
        self._initialize_metrics()
        self._running = False
        self._update_thread = None
        
        logger.info("FX Trading Metrics initialized")
    
    def _initialize_metrics(self):
        """Initialize all Prometheus metrics."""
        
        # === TRADING PERFORMANCE METRICS ===
        
        # PnL and Equity
        self.pnl_gauge = Gauge(
            'fxai_pnl_equity', 
            'Current PnL (equity curve) in USD',
            registry=self.registry
        )
        
        self.daily_pnl = Gauge(
            'fxai_daily_pnl',
            'Daily PnL in USD',
            registry=self.registry
        )
        
        self.unrealized_pnl = Gauge(
            'fxai_unrealized_pnl',
            'Current unrealized PnL in USD',
            registry=self.registry
        )
        
        # Trade Counts and Success Rates
        self.trade_count = Counter(
            'fxai_trades_total', 
            'Total trades executed',
            ['symbol', 'side', 'status'],
            registry=self.registry
        )
        
        self.winning_trades = Counter(
            'fxai_winning_trades_total',
            'Total winning trades',
            ['symbol'],
            registry=self.registry
        )
        
        self.losing_trades = Counter(
            'fxai_losing_trades_total', 
            'Total losing trades',
            ['symbol'],
            registry=self.registry
        )
        
        # === EXECUTION METRICS ===
        
        # Latency Metrics
        self.execution_latency = Histogram(
            'fxai_execution_latency_seconds',
            'Order execution latency in seconds',
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
            registry=self.registry
        )
        
        self.fill_latency = Histogram(
            'fxai_fill_latency_seconds',
            'Time from order to fill in seconds',
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
            registry=self.registry
        )
        
        # Slippage and Execution Quality
        self.order_slippage = Histogram(
            'fxai_order_slippage_pips',
            'Order slippage in pips',
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0],
            registry=self.registry
        )
        
        self.fill_rate = Gauge(
            'fxai_fill_rate_ratio',
            'Percentage of orders that get filled',
            ['symbol'],
            registry=self.registry
        )
        
        # === POSITION AND EXPOSURE METRICS ===
        
        self.position_exposure = Gauge(
            'fxai_position_exposure_usd',
            'Current position exposure in USD',
            ['symbol', 'side'],
            registry=self.registry
        )
        
        self.total_exposure = Gauge(
            'fxai_total_exposure_usd',
            'Total position exposure across all symbols',
            registry=self.registry
        )
        
        self.position_count = Gauge(
            'fxai_open_positions',
            'Number of open positions',
            ['symbol'],
            registry=self.registry
        )
        
        # === STRATEGY AND REGIME METRICS ===
        
        self.regime_distribution = Gauge(
            'fxai_regime_ratio',
            'Proportion of regime classification',
            ['regime', 'symbol'],
            registry=self.registry
        )
        
        self.strategy_signals = Counter(
            'fxai_strategy_signals_total',
            'Total strategy signals generated',
            ['strategy', 'signal_type', 'symbol'],
            registry=self.registry
        )
        
        self.strategy_performance = Gauge(
            'fxai_strategy_pnl',
            'PnL attributed to each strategy',
            ['strategy'],
            registry=self.registry
        )
        
        # === SYSTEM HEALTH METRICS ===
        
        # Error Tracking
        self.error_counter = Counter(
            'fxai_errors_total',
            'Total system errors',
            ['error_type', 'component'],
            registry=self.registry
        )
        
        self.exception_counter = Counter(
            'fxai_exceptions_total',
            'Total exceptions by type',
            ['exception_type', 'component'],
            registry=self.registry
        )
        
        # System Performance
        self.memory_usage = Gauge(
            'fxai_memory_usage_mb',
            'Memory usage in MB',
            registry=self.registry
        )
        
        self.cpu_usage = Gauge(
            'fxai_cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )
        
        # Market Data Health
        self.market_data_lag = Gauge(
            'fxai_market_data_lag_seconds',
            'Market data lag in seconds',
            ['symbol'],
            registry=self.registry
        )
        
        self.market_data_updates = Counter(
            'fxai_market_data_updates_total',
            'Total market data updates received',
            ['symbol', 'data_type'],
            registry=self.registry
        )
        
        # === BUSINESS METRICS ===
        
        # Risk Metrics
        self.current_drawdown = Gauge(
            'fxai_current_drawdown_ratio',
            'Current drawdown as ratio of peak equity',
            registry=self.registry
        )
        
        self.max_drawdown = Gauge(
            'fxai_max_drawdown_ratio',
            'Maximum drawdown experienced',
            registry=self.registry
        )
        
        self.var_estimate = Gauge(
            'fxai_var_estimate',
            'Value at Risk estimate',
            ['confidence_level'],
            registry=self.registry
        )
        
        # Performance Ratios
        self.sharpe_ratio = Gauge(
            'fxai_sharpe_ratio',
            'Current Sharpe ratio',
            registry=self.registry
        )
        
        self.win_rate = Gauge(
            'fxai_win_rate_ratio',
            'Win rate as percentage',
            ['symbol'],
            registry=self.registry
        )
        
        # === SYSTEM INFO ===
        
        self.system_info = Info(
            'fxai_system_info',
            'System information',
            registry=self.registry
        )
        
        # Set static system info
        self.system_info.info({
            'version': '1.0.0',
            'component': 'fx_quant_trading_system',
            'build_date': datetime.now().isoformat(),
            'python_version': '3.9+'
        })
        
        logger.info("All metrics initialized successfully")
    
    def start_server(self) -> None:
        """Start the Prometheus metrics HTTP server."""
        try:
            start_http_server(
                self.config.port, 
                addr=self.config.host,
                registry=self.registry
            )
            self._running = True
            
            logger.info(f"🚀 Metrics server running on http://{self.config.host}:{self.config.port}/metrics")
            
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
            raise
    
    def stop_server(self) -> None:
        """Stop the metrics server and background updates."""
        self._running = False
        if self._update_thread and self._update_thread.is_alive():
            self._update_thread.join(timeout=5.0)
        
        logger.info("Metrics server stopped")
    
    # === TRADING METRICS UPDATE METHODS ===
    
    def update_pnl(self, equity: float, daily_pnl: float = None, unrealized_pnl: float = None) -> None:
        """Update PnL and equity metrics."""
        self.pnl_gauge.set(equity)
        
        if daily_pnl is not None:
            self.daily_pnl.set(daily_pnl)
        
        if unrealized_pnl is not None:
            self.unrealized_pnl.set(unrealized_pnl)
    
    def record_trade(self, symbol: str, side: str, status: str, pnl: float = None) -> None:
        """Record a completed trade."""
        self.trade_count.labels(symbol=symbol, side=side, status=status).inc()
        
        if pnl is not None and status == 'filled':
            if pnl > 0:
                self.winning_trades.labels(symbol=symbol).inc()
            else:
                self.losing_trades.labels(symbol=symbol).inc()
    
    def record_execution_latency(self, latency_seconds: float) -> None:
        """Record order execution latency."""
        self.execution_latency.observe(latency_seconds)
    
    def record_fill_latency(self, latency_seconds: float) -> None:
        """Record time from order to fill."""
        self.fill_latency.observe(latency_seconds)
    
    def record_slippage(self, slippage_pips: float) -> None:
        """Record order slippage in pips."""
        self.order_slippage.observe(slippage_pips)
    
    def update_fill_rate(self, symbol: str, fill_rate: float) -> None:
        """Update fill rate for a symbol."""
        self.fill_rate.labels(symbol=symbol).set(fill_rate)
    
    def update_position_exposure(self, symbol: str, side: str, exposure_usd: float) -> None:
        """Update position exposure metrics."""
        self.position_exposure.labels(symbol=symbol, side=side).set(exposure_usd)
    
    def update_total_exposure(self, total_exposure: float) -> None:
        """Update total exposure across all positions."""
        self.total_exposure.set(total_exposure)
    
    def update_position_count(self, symbol: str, count: int) -> None:
        """Update number of open positions."""
        self.position_count.labels(symbol=symbol).set(count)
    
    def update_regime(self, symbol: str, regime: str, ratio: float) -> None:
        """Update regime classification metrics."""
        self.regime_distribution.labels(regime=regime, symbol=symbol).set(ratio)
    
    def record_strategy_signal(self, strategy: str, signal_type: str, symbol: str) -> None:
        """Record a strategy signal."""
        self.strategy_signals.labels(
            strategy=strategy, 
            signal_type=signal_type, 
            symbol=symbol
        ).inc()
    
    def update_strategy_performance(self, strategy: str, pnl: float) -> None:
        """Update strategy-specific performance."""
        self.strategy_performance.labels(strategy=strategy).set(pnl)
    
    def record_error(self, error_type: str, component: str) -> None:
        """Record a system error."""
        self.error_counter.labels(error_type=error_type, component=component).inc()
    
    def record_exception(self, exception_type: str, component: str) -> None:
        """Record an exception."""
        self.exception_counter.labels(
            exception_type=exception_type, 
            component=component
        ).inc()
    
    def update_system_resources(self, memory_mb: float, cpu_percent: float) -> None:
        """Update system resource usage."""
        self.memory_usage.set(memory_mb)
        self.cpu_usage.set(cpu_percent)
    
    def update_market_data_lag(self, symbol: str, lag_seconds: float) -> None:
        """Update market data lag metrics."""
        self.market_data_lag.labels(symbol=symbol).set(lag_seconds)
    
    def record_market_data_update(self, symbol: str, data_type: str) -> None:
        """Record a market data update."""
        self.market_data_updates.labels(symbol=symbol, data_type=data_type).inc()
    
    def update_risk_metrics(self, 
                          current_drawdown: float, 
                          max_drawdown: float,
                          var_95: float = None,
                          var_99: float = None) -> None:
        """Update risk metrics."""
        self.current_drawdown.set(current_drawdown)
        self.max_drawdown.set(max_drawdown)
        
        if var_95 is not None:
            self.var_estimate.labels(confidence_level='95').set(var_95)
        
        if var_99 is not None:
            self.var_estimate.labels(confidence_level='99').set(var_99)
    
    def update_performance_ratios(self, sharpe_ratio: float) -> None:
        """Update performance ratio metrics."""
        self.sharpe_ratio.set(sharpe_ratio)
    
    def update_win_rate(self, symbol: str, win_rate: float) -> None:
        """Update win rate for a symbol."""
        self.win_rate.labels(symbol=symbol).set(win_rate)
    
    # === BULK UPDATE METHODS ===
    
    def update_from_backtest_results(self, results_df: Any) -> None:
        """Update metrics from backtest results DataFrame."""
        if hasattr(results_df, 'empty') and not results_df.empty:
            # Calculate and update key metrics from backtest
            total_pnl = results_df.get('pnl', pd.Series()).sum()
            win_rate = (results_df.get('pnl', pd.Series()) > 0).mean()
            
            self.update_pnl(total_pnl)
            
            # Update per-symbol metrics if available
            if 'symbol' in results_df.columns:
                for symbol in results_df['symbol'].unique():
                    symbol_data = results_df[results_df['symbol'] == symbol]
                    symbol_win_rate = (symbol_data['pnl'] > 0).mean()
                    self.update_win_rate(symbol, symbol_win_rate)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of current metrics values."""
        try:
            # This would typically query the registry for current values
            # For now, return a basic structure
            return {
                'server_running': self._running,
                'metrics_count': len(list(self.registry._collector_to_names.keys())),
                'endpoint': f"http://{self.config.host}:{self.config.port}/metrics"
            }
        except Exception as e:
            logger.error(f"Error getting metrics summary: {e}")
            return {'error': str(e)}

# Global metrics instance
_metrics_instance = None

def get_metrics() -> FXTradingMetrics:
    """Get the global metrics instance."""
    global _metrics_instance
    if _metrics_instance is None:
        _metrics_instance = FXTradingMetrics()
    return _metrics_instance

def start_metrics_server(port: int = 9000, host: str = '0.0.0.0') -> FXTradingMetrics:
    """Start the metrics server and return the metrics instance."""
    config = MetricsConfig(port=port, host=host)
    metrics = FXTradingMetrics(config)
    metrics.start_server()
    
    # Set global instance
    global _metrics_instance
    _metrics_instance = metrics
    
    return metrics

# Convenience functions for common operations
def record_trade_execution(symbol: str, side: str, status: str, 
                         latency_seconds: float = None, 
                         slippage_pips: float = None,
                         pnl: float = None) -> None:
    """Convenience function to record trade execution with all related metrics."""
    metrics = get_metrics()
    
    metrics.record_trade(symbol, side, status, pnl)
    
    if latency_seconds is not None:
        metrics.record_execution_latency(latency_seconds)
    
    if slippage_pips is not None:
        metrics.record_slippage(slippage_pips)

def update_trading_session_metrics(equity: float, 
                                 daily_pnl: float,
                                 total_exposure: float,
                                 sharpe_ratio: float,
                                 current_drawdown: float) -> None:
    """Convenience function to update key trading session metrics."""
    metrics = get_metrics()
    
    metrics.update_pnl(equity, daily_pnl)
    metrics.update_total_exposure(total_exposure)
    metrics.update_performance_ratios(sharpe_ratio)
    metrics.update_risk_metrics(current_drawdown, current_drawdown)  # Assuming current is also max for simplicity 