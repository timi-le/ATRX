"""
Configuration settings for the FX AI-Quant Trading System.
"""

from pathlib import Path

from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    """Data source configuration."""

    # Data providers
    dukascopy_enabled: bool = True
    oanda_enabled: bool = True
    oanda_api_key: str | None = None
    oanda_account_id: str | None = None
    oanda_environment: str = "practice"  # practice or live

    # Data paths
    raw_data_path: Path = Path("data/raw")
    processed_data_path: Path = Path("data/processed")
    historical_data_path: Path = Path("data/historical")

    # Symbols to trade
    fx_pairs: list[str] = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]

    # Data retention
    tick_data_retention_days: int = 30
    bar_data_retention_days: int = 365


class TradingConfig(BaseModel):
    """Trading strategy configuration."""

    # Strategy weights
    scalping_weight: float = Field(0.3, ge=0.0, le=1.0)
    breakout_weight: float = Field(0.4, ge=0.0, le=1.0)
    arbitrage_weight: float = Field(0.3, ge=0.0, le=1.0)

    # Signal thresholds
    min_signal_strength: float = Field(0.6, ge=0.0, le=1.0)
    min_confidence: float = Field(0.7, ge=0.0, le=1.0)

    # Position sizing
    kelly_multiplier: float = Field(0.25, ge=0.0, le=1.0)  # Conservative Kelly
    max_position_size: float = Field(0.1, ge=0.0, le=1.0)  # 10% max per position
    max_total_exposure: float = Field(0.8, ge=0.0, le=1.0)  # 80% max total

    # Execution
    order_slice_size: float = Field(10000.0, gt=0)  # USD
    twap_interval_seconds: int = Field(30, gt=0)


class RiskConfig(BaseModel):
    """Risk management configuration."""

    # Drawdown limits
    max_daily_drawdown: float = Field(0.02, ge=0.0, le=1.0)  # 2%
    max_total_drawdown: float = Field(0.1, ge=0.0, le=1.0)  # 10%

    # Position limits
    max_position_per_symbol: float = Field(100000.0, gt=0)  # USD
    max_correlation_exposure: float = Field(0.5, ge=0.0, le=1.0)

    # VaR settings
    var_confidence_level: float = Field(0.95, ge=0.0, le=1.0)
    var_horizon_days: int = Field(1, gt=0)
    var_limit: float = Field(0.05, ge=0.0, le=1.0)  # 5% of account

    # Emergency stops
    emergency_stop_enabled: bool = True
    max_consecutive_losses: int = Field(5, gt=0)
    circuit_breaker_threshold: float = Field(0.05, ge=0.0, le=1.0)  # 5%


class MessagingConfig(BaseModel):
    """Messaging system configuration."""

    # ZeroMQ settings
    zmq_enabled: bool = True
    zmq_publish_port: int = 5555
    zmq_subscribe_port: int = 5556
    zmq_bind_address: str = "tcp://*"
    zmq_connect_address: str = "tcp://localhost"

    # Redis settings
    redis_enabled: bool = True
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str | None = None

    # Message queue settings
    max_queue_size: int = 10000
    message_timeout_seconds: int = 30


class MLConfig(BaseModel):
    """Machine learning configuration."""

    # Model paths
    model_path: Path = Path("models")
    lstm_model_path: Path = Path("models/lstm")
    cnn_model_path: Path = Path("models/cnn")
    ensemble_model_path: Path = Path("models/ensemble")

    # Training settings
    training_data_window: int = 10000  # bars
    validation_split: float = Field(0.2, ge=0.0, le=1.0)
    walk_forward_steps: int = 252  # trading days

    # Model parameters
    lstm_units: int = 50
    cnn_filters: int = 32
    ensemble_models: list[str] = ["lstm", "cnn", "xgboost"]

    # Regime detection
    n_regimes: int = 3
    regime_features: list[str] = ["volatility", "momentum", "macro_surprise"]


class MonitoringConfig(BaseModel):
    """Monitoring and metrics configuration."""

    # Prometheus
    prometheus_enabled: bool = True
    prometheus_port: int = 8000
    prometheus_host: str = "0.0.0.0"

    # Grafana
    grafana_enabled: bool = True
    grafana_port: int = 3000

    # Logging
    log_level: str = "INFO"
    log_file: Path = Path("logs/trading_system.log")
    max_log_size_mb: int = 100
    log_retention_days: int = 30

    # Alerts
    alert_email_enabled: bool = False
    alert_email_recipients: list[str] = []
    alert_slack_enabled: bool = False
    alert_slack_webhook: str | None = None


class SystemConfig(BaseModel):
    """Main system configuration combining all subsystems."""

    # Environment
    environment: str = "development"  # development, staging, production
    debug: bool = True

    # Subsystem configurations
    data: DataConfig = DataConfig()
    trading: TradingConfig = TradingConfig()
    risk: RiskConfig = RiskConfig()
    messaging: MessagingConfig = MessagingConfig()
    ml: MLConfig = MLConfig()
    monitoring: MonitoringConfig = MonitoringConfig()

    # System settings
    system_timezone: str = "UTC"
    heartbeat_interval_seconds: int = 30
    performance_target_latency_ms: int = 100
    performance_target_throughput: int = 1000  # ticks/sec

    class Config:
        """Pydantic configuration."""

        env_prefix = "FX_TRADING_"
        case_sensitive = False
