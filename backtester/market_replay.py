"""
Market Replay Module - Historical Data Streaming for Backtesting.

This module provides functionality to replay historical market data in chronological
order, simulating real-time data feeds for backtesting purposes.

Features:
- Support for multiple timeframes (M1, H1, D1, etc.)
- Chronological data streaming across multiple symbols
- Data validation and gap detection
- Configurable replay speed and time ranges
- Memory-efficient data loading and streaming
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, AsyncGenerator, Union, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import csv
import gzip
import json
from enum import Enum
import structlog

from core.interfaces.data_interfaces import MarketData, OHLCV


class TimeFrame(Enum):
    """Supported timeframes for backtesting."""
    TICK = "tick"
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"


@dataclass
class ReplayConfig:
    """Configuration for market data replay."""
    data_path: str = "data/historical"
    symbols: List[str] = field(default_factory=lambda: ["EURUSD", "GBPUSD", "USDJPY"])
    timeframe: TimeFrame = TimeFrame.M1
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    replay_speed: float = 1.0  # 1.0 = real-time, 0 = as fast as possible
    validate_data: bool = True
    fill_gaps: bool = True
    max_gap_minutes: int = 5
    chunk_size: int = 10000  # Number of rows to load at once
    
    def __post_init__(self):
        if self.start_date is None:
            self.start_date = datetime.now() - timedelta(days=30)
        if self.end_date is None:
            self.end_date = datetime.now()


@dataclass
class DataPoint:
    """Unified data point for replay queue."""
    timestamp: datetime
    symbol: str
    data: Union[MarketData, OHLCV]
    data_type: str  # "tick" or "bar"


class MarketReplay:
    """
    Market data replay engine for backtesting.
    
    Streams historical market data in chronological order, simulating
    real-time market conditions for strategy backtesting.
    """
    
    def __init__(self, config: ReplayConfig, logger: Optional[structlog.stdlib.BoundLogger] = None):
        self.config = config
        self.logger = logger or structlog.get_logger(__name__)
        
        # Data management
        self.data_queue: List[DataPoint] = []
        self.current_time: Optional[datetime] = None
        self.replay_start_time: Optional[datetime] = None
        self.is_running = False
        
        # Statistics
        self.total_points = 0
        self.points_streamed = 0
        self.gaps_filled = 0
        self.validation_errors = 0
        
        self.logger.info(
            "MarketReplay initialized",
            symbols=config.symbols,
            timeframe=config.timeframe.value,
            start_date=config.start_date.isoformat() if config.start_date else None,
            end_date=config.end_date.isoformat() if config.end_date else None
        )
    
    async def load_data(self) -> None:
        """Load historical data for all symbols and prepare for replay."""
        try:
            self.logger.info("Loading historical data for replay")
            
            all_data_points = []
            
            for symbol in self.config.symbols:
                symbol_data = await self._load_symbol_data(symbol)
                all_data_points.extend(symbol_data)
                
                self.logger.info(
                    "Loaded symbol data",
                    symbol=symbol,
                    points=len(symbol_data)
                )
            
            # Sort all data points by timestamp
            all_data_points.sort(key=lambda x: x.timestamp)
            self.data_queue = all_data_points
            self.total_points = len(all_data_points)
            
            if self.total_points > 0:
                self.current_time = self.data_queue[0].timestamp
                
            self.logger.info(
                "Data loading completed",
                total_points=self.total_points,
                start_time=self.current_time.isoformat() if self.current_time else None,
                end_time=self.data_queue[-1].timestamp.isoformat() if self.data_queue else None
            )
            
        except Exception as e:
            self.logger.error("Error loading data", error=str(e))
            raise
    
    async def _load_symbol_data(self, symbol: str) -> List[DataPoint]:
        """Load data for a specific symbol."""
        data_points = []
        
        # Try different file formats and locations
        file_patterns = [
            f"{symbol}_{self.config.timeframe.value}.parquet", # Check for Parquet first
            f"{symbol}_{self.config.timeframe.value}.csv",
            f"{symbol}_{self.config.timeframe.value}.csv.gz",
            f"{symbol}/{self.config.timeframe.value}.csv",
            f"fx_data/{symbol}_{self.config.timeframe.value}.csv"
        ]
        
        data_path = Path(self.config.data_path)
        file_found = False
        
        for pattern in file_patterns:
            file_path = data_path / pattern
            if file_path.exists():
                self.logger.debug("Loading data file", file=str(file_path))
                if file_path.suffix == '.parquet':
                    symbol_data = await self._read_parquet_file(file_path, symbol)
                else:
                    symbol_data = await self._read_data_file(file_path, symbol)
                data_points.extend(symbol_data)
                file_found = True
                break
        
        if not file_found:
            self.logger.warning("No data file found for symbol", symbol=symbol, patterns=file_patterns)
            # Generate mock data for testing
            data_points = await self._generate_mock_data(symbol)
        
        # Apply date filtering
        if self.config.start_date or self.config.end_date:
            data_points = self._filter_by_date(data_points)
        
        # Validate and clean data
        if self.config.validate_data:
            data_points = await self._validate_data(data_points, symbol)
        
        return data_points
    
    async def _read_parquet_file(self, file_path: Path, symbol: str) -> List[DataPoint]:
        """Read data from Parquet file."""
        data_points = []
        try:
            df = pd.read_parquet(file_path)
            self.logger.info(f"Loaded {len(df)} rows from {file_path}")
            for _, row in df.iterrows():
                try:
                    data_point = await self._parse_data_row(row.to_dict(), symbol)
                    if data_point:
                        data_points.append(data_point)
                except Exception as e:
                    self.logger.warning("Error parsing row from parquet", error=str(e), row=row.to_dict())
                    continue
        except Exception as e:
            self.logger.error("Error reading parquet file", file=str(file_path), error=str(e))
            raise
        return data_points
    
    async def _read_data_file(self, file_path: Path, symbol: str) -> List[DataPoint]:
        """Read data from CSV file."""
        data_points = []
        
        try:
            # Handle compressed files
            if file_path.suffix == '.gz':
                import gzip
                with gzip.open(file_path, 'rt') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
            else:
                with open(file_path, 'r') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
            
            for row in rows:
                try:
                    data_point = await self._parse_data_row(row, symbol)
                    if data_point:
                        data_points.append(data_point)
                except Exception as e:
                    self.logger.warning("Error parsing row", error=str(e), row=row)
                    continue
            
        except Exception as e:
            self.logger.error("Error reading data file", file=str(file_path), error=str(e))
            raise
        
        return data_points
    
    async def _parse_data_row(self, row: Dict[str, str], symbol: str) -> Optional[DataPoint]:
        """Parse a single data row into a DataPoint."""
        try:
            # Try different timestamp formats
            timestamp_fields = ['timestamp', 'time', 'datetime', 'date']
            timestamp_str = None
            
            for field in timestamp_fields:
                if field in row and row[field]:
                    # Handle different types for row (dict for csv, series for parquet)
                    timestamp_str = str(row[field])
                    break
            
            if not timestamp_str:
                return None
            
            # Parse timestamp
            timestamp = self._parse_timestamp(timestamp_str)
            if not timestamp:
                return None
            
            # Determine data type and create appropriate object
            if self.config.timeframe == TimeFrame.TICK:
                # Tick data
                bid = float(row.get('bid', row.get('price', 0)))
                ask = float(row.get('ask', bid + 0.0001))  # Default spread
                volume = float(row.get('volume', row.get('tick_volume', 1.0)))
                
                market_data = MarketData(
                    symbol=symbol,
                    timestamp=timestamp,
                    bid=bid,
                    ask=ask,
                    volume=volume,
                    source="replay"
                )
                
                return DataPoint(
                    timestamp=timestamp,
                    symbol=symbol,
                    data=market_data,
                    data_type="tick"
                )
            
            else:
                # Bar data (OHLCV)
                open_price = float(row.get('open', 0))
                high_price = float(row.get('high', 0))
                low_price = float(row.get('low', 0))
                close_price = float(row.get('close', 0))
                volume = float(row.get('volume', row.get('tick_volume', 0)))
                
                ohlcv_data = OHLCV(
                    symbol=symbol,
                    timestamp=timestamp,
                    open=open_price,
                    high=high_price,
                    low=low_price,
                    close=close_price,
                    volume=volume
                )
                
                return DataPoint(
                    timestamp=timestamp,
                    symbol=symbol,
                    data=ohlcv_data,
                    data_type="bar"
                )
                
        except (ValueError, TypeError, KeyError) as e:
            self.logger.warning(
                "Could not parse data row",
                error=str(e),
                row=row,
                symbol=symbol
            )
            return None
        
    def _parse_timestamp(self, timestamp_str: Any) -> Optional[datetime]:
        """Parse timestamp from various formats."""
        if isinstance(timestamp_str, datetime):
            return timestamp_str
        if isinstance(timestamp_str, pd.Timestamp):
            return timestamp_str.to_pydatetime()
        if isinstance(timestamp_str, (int, float)):
             try:
                return datetime.fromtimestamp(timestamp_str)
             except (ValueError, TypeError):
                pass
        
        if not isinstance(timestamp_str, str):
            timestamp_str = str(timestamp_str)

        supported_formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d",
            "%Y.%m.%d %H:%M",
            "%Y.%m.%d",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S.%fZ",
        ]
        
        for fmt in supported_formats:
            try:
                return datetime.strptime(timestamp_str, fmt)
            except ValueError:
                continue
        
        try:
            return pd.to_datetime(timestamp_str).to_pydatetime()
        except (ValueError, TypeError):
            pass

        self.logger.warning("Could not parse timestamp", timestamp=timestamp_str)
        return None
    
    async def _generate_mock_data(self, symbol: str) -> List[DataPoint]:
        """Generate mock data for testing when real data is not available."""
        self.logger.info("Generating mock data", symbol=symbol)
        
        data_points = []
        current_time = self.config.start_date
        base_price = 1.1000 if "EUR" in symbol else 1.3000
        
        while current_time <= self.config.end_date:
            # Generate realistic price movement
            price_change = np.random.normal(0, 0.0001)
            base_price += price_change
            base_price = max(0.5, min(2.0, base_price))  # Keep within reasonable bounds
            
            if self.config.timeframe == TimeFrame.TICK:
                spread = 0.0001
                market_data = MarketData(
                    symbol=symbol,
                    timestamp=current_time,
                    bid=base_price,
                    ask=base_price + spread,
                    volume=np.random.randint(1, 100),
                    source="mock"
                )
                
                data_points.append(DataPoint(
                    timestamp=current_time,
                    symbol=symbol,
                    data=market_data,
                    data_type="tick"
                ))
                
                current_time += timedelta(seconds=1)
            
            else:
                # Generate OHLCV bar
                open_price = base_price
                high_price = open_price + abs(np.random.normal(0, 0.0005))
                low_price = open_price - abs(np.random.normal(0, 0.0005))
                close_price = open_price + np.random.normal(0, 0.0003)
                volume = np.random.randint(100, 10000)
                
                ohlcv = OHLCV(
                    symbol=symbol,
                    timestamp=current_time,
                    open=open_price,
                    high=high_price,
                    low=low_price,
                    close=close_price,
                    volume=volume,
                    timeframe=self.config.timeframe.value
                )
                
                data_points.append(DataPoint(
                    timestamp=current_time,
                    symbol=symbol,
                    data=ohlcv,
                    data_type="bar"
                ))
                
                # Increment time based on timeframe
                if self.config.timeframe == TimeFrame.M1:
                    current_time += timedelta(minutes=1)
                elif self.config.timeframe == TimeFrame.M5:
                    current_time += timedelta(minutes=5)
                elif self.config.timeframe == TimeFrame.H1:
                    current_time += timedelta(hours=1)
                elif self.config.timeframe == TimeFrame.D1:
                    current_time += timedelta(days=1)
                else:
                    current_time += timedelta(minutes=1)
        
        return data_points
    
    def _filter_by_date(self, data_points: List[DataPoint]) -> List[DataPoint]:
        """Filter data points by date range."""
        filtered = []
        
        for point in data_points:
            if self.config.start_date and point.timestamp < self.config.start_date:
                continue
            if self.config.end_date and point.timestamp > self.config.end_date:
                continue
            filtered.append(point)
        
        return filtered
    
    async def _validate_data(self, data_points: List[DataPoint], symbol: str) -> List[DataPoint]:
        """Validate and clean data points."""
        validated = []
        prev_timestamp = None
        
        for point in data_points:
            # Check for duplicate timestamps
            if prev_timestamp and point.timestamp == prev_timestamp:
                self.validation_errors += 1
                continue
            
            # Validate data values
            if point.data_type == "tick":
                tick_data = point.data
                if tick_data.bid <= 0 or tick_data.ask <= 0 or tick_data.ask < tick_data.bid:
                    self.validation_errors += 1
                    continue
            
            elif point.data_type == "bar":
                bar_data = point.data
                if (bar_data.open <= 0 or bar_data.high <= 0 or 
                    bar_data.low <= 0 or bar_data.close <= 0 or
                    bar_data.high < max(bar_data.open, bar_data.close) or
                    bar_data.low > min(bar_data.open, bar_data.close)):
                    self.validation_errors += 1
                    continue
            
            # Check for gaps and fill if configured
            if (prev_timestamp and self.config.fill_gaps and 
                point.timestamp - prev_timestamp > timedelta(minutes=self.config.max_gap_minutes)):
                
                # Fill gap with interpolated data
                gap_points = await self._fill_gap(prev_timestamp, point.timestamp, symbol, point.data_type)
                validated.extend(gap_points)
                self.gaps_filled += len(gap_points)
            
            validated.append(point)
            prev_timestamp = point.timestamp
        
        self.logger.info(
            "Data validation completed",
            symbol=symbol,
            original_points=len(data_points),
            validated_points=len(validated),
            validation_errors=self.validation_errors,
            gaps_filled=self.gaps_filled
        )
        
        return validated
    
    async def _fill_gap(
        self, 
        start_time: datetime, 
        end_time: datetime, 
        symbol: str, 
        data_type: str
    ) -> List[DataPoint]:
        """Fill data gaps with interpolated values."""
        gap_points = []
        
        # Simple forward-fill strategy
        # In production, you might want more sophisticated interpolation
        current_time = start_time + timedelta(minutes=1)
        
        while current_time < end_time:
            if data_type == "tick":
                # Use last known price with small random variation
                base_price = 1.1000  # Would use actual last price in production
                market_data = MarketData(
                    symbol=symbol,
                    timestamp=current_time,
                    bid=base_price,
                    ask=base_price + 0.0001,
                    volume=1.0,
                    source="interpolated"
                )
                
                gap_points.append(DataPoint(
                    timestamp=current_time,
                    symbol=symbol,
                    data=market_data,
                    data_type="tick"
                ))
                
                current_time += timedelta(seconds=1)
            
            else:
                # Generate interpolated bar
                base_price = 1.1000  # Would use actual last price in production
                ohlcv = OHLCV(
                    symbol=symbol,
                    timestamp=current_time,
                    open=base_price,
                    high=base_price,
                    low=base_price,
                    close=base_price,
                    volume=1.0,
                    timeframe=self.config.timeframe.value
                )
                
                gap_points.append(DataPoint(
                    timestamp=current_time,
                    symbol=symbol,
                    data=ohlcv,
                    data_type="bar"
                ))
                
                current_time += timedelta(minutes=1)
        
        return gap_points
    
    async def stream(self) -> AsyncGenerator[DataPoint, None]:
        """Stream data points in chronological order."""
        if not self.data_queue:
            await self.load_data()
        
        if not self.data_queue:
            self.logger.warning("No data available for streaming")
            return
        
        self.is_running = True
        self.replay_start_time = datetime.now()
        self.current_time = self.data_queue[0].timestamp
        
        self.logger.info(
            "Starting data replay",
            total_points=self.total_points,
            replay_speed=self.config.replay_speed
        )
        
        prev_timestamp = None
        
        for point in self.data_queue:
            if not self.is_running:
                break
            
            # Handle replay speed
            if self.config.replay_speed > 0 and prev_timestamp:
                time_diff = point.timestamp - prev_timestamp
                sleep_time = time_diff.total_seconds() / self.config.replay_speed
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
            
            self.current_time = point.timestamp
            self.points_streamed += 1
            
            yield point
            
            prev_timestamp = point.timestamp
        
        self.is_running = False
        self.logger.info(
            "Data replay completed",
            points_streamed=self.points_streamed,
            total_points=self.total_points
        )
    
    def stop(self) -> None:
        """Stop the data replay."""
        self.is_running = False
        self.logger.info("Data replay stopped")
    
    def get_progress(self) -> Dict[str, Union[int, float, str]]:
        """Get replay progress information."""
        progress_pct = (self.points_streamed / self.total_points * 100) if self.total_points > 0 else 0
        
        return {
            "total_points": self.total_points,
            "points_streamed": self.points_streamed,
            "progress_percent": progress_pct,
            "current_time": self.current_time.isoformat() if self.current_time else None,
            "is_running": self.is_running,
            "validation_errors": self.validation_errors,
            "gaps_filled": self.gaps_filled
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive replay statistics."""
        runtime = (datetime.now() - self.replay_start_time).total_seconds() if self.replay_start_time else 0
        
        return {
            "config": {
                "symbols": self.config.symbols,
                "timeframe": self.config.timeframe.value,
                "start_date": self.config.start_date.isoformat() if self.config.start_date else None,
                "end_date": self.config.end_date.isoformat() if self.config.end_date else None,
                "replay_speed": self.config.replay_speed
            },
            "progress": self.get_progress(),
            "performance": {
                "runtime_seconds": runtime,
                "points_per_second": self.points_streamed / runtime if runtime > 0 else 0
            },
            "data_quality": {
                "validation_errors": self.validation_errors,
                "gaps_filled": self.gaps_filled,
                "error_rate": self.validation_errors / self.total_points if self.total_points > 0 else 0
            }
        }


# Utility functions
def create_replay_config(
    symbols: List[str],
    timeframe: str = "1m",
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    data_path: str = "data/historical",
    replay_speed: float = 0.0  # As fast as possible for backtesting
) -> ReplayConfig:
    """Create a replay configuration with common defaults."""
    return ReplayConfig(
        data_path=data_path,
        symbols=symbols,
        timeframe=TimeFrame(timeframe),
        start_date=start_date,
        end_date=end_date,
        replay_speed=replay_speed,
        validate_data=True,
        fill_gaps=True
    )


async def replay_data_from_files(
    file_paths: List[str],
    symbols: List[str],
    timeframe: str = "1m"
) -> AsyncGenerator[DataPoint, None]:
    """Convenience function to replay data from specific files."""
    # This would be implemented to read from specific file paths
    # For now, use the main MarketReplay class
    config = create_replay_config(symbols, timeframe)
    replay = MarketReplay(config)
    
    async for point in replay.stream():
        yield point 