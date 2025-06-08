# MT5 Live Data Directory

This directory contains real forex data downloaded directly from your MetaTrader 5 broker connection.

## Purpose

- **Real broker data**: Unlike synthetic data, this is actual historical price data from your broker
- **Manual Colab upload**: Files here are intended for manual upload to Google Colab
- **Multi-timeframe coverage**: M1, M5, M15, M30, H1, H4, D1 timeframes
- **Multiple currencies**: EURUSD, GBPUSD, USDJPY

## Data Source

Data is downloaded using:
- **Script**: `scripts/download_mt5_live_data.py`
- **Configuration**: Uses existing broker config from `services/execution/config_mt5.yaml`
- **Date Range**: 2018-01-01 to 2025-03-31 (7+ years)

## Usage

1. **Download data**: Run `python scripts/download_mt5_live_data.py`
2. **Check files**: Verify CSV files are created in this directory
3. **Upload to Colab**: Manually upload files to your Google Colab environment
4. **Train models**: Use with `scripts/colab_training.py` for training

## File Naming Convention

Files follow the pattern: `{SYMBOL}_{TIMEFRAME}_mt5_live.csv`

Examples:
- `EURUSD_M1_mt5_live.csv`
- `GBPUSD_H1_mt5_live.csv`
- `USDJPY_D1_mt5_live.csv`

## Notes

- Requires active MT5 terminal connection
- Uses your existing demo account credentials
- Data quality depends on broker's historical data availability
- Files include download summary (`mt5_download_summary.json`)
