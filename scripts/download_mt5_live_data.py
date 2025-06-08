#!/usr/bin/env python3
"""
MT5 Live Data Download Script
Downloads real forex data from MetaTrader 5 broker connection
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import MetaTrader5 as mt5
import pandas as pd
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("mt5_download.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class MT5DataDownloader:
    """Download real forex data from MT5 broker connection"""

    def __init__(self, config_path: str = "services/execution/config_mt5.yaml"):
        self.data_dir = "data/forex/mt5_live"
        self.ensure_data_dir()

        # Load existing broker configuration
        self.config = self._load_config(config_path)

        # Currency pairs to download
        self.symbols = ["EURUSD", "GBPUSD", "USDJPY"]

        # Timeframe mapping (MT5 timeframe -> string name)
        self.timeframes = {
            mt5.TIMEFRAME_M1: "M1",
            mt5.TIMEFRAME_M5: "M5",
            mt5.TIMEFRAME_M15: "M15",
            mt5.TIMEFRAME_M30: "M30",
            mt5.TIMEFRAME_H1: "H1",
            mt5.TIMEFRAME_H4: "H4",
            mt5.TIMEFRAME_D1: "D1",
        }

        # Download parameters
        self.start_date = datetime(2018, 1, 1)
        self.end_date = datetime(2025, 3, 31)

    def ensure_data_dir(self):
        """Create data directory if it doesn't exist"""
        os.makedirs(self.data_dir, exist_ok=True)

    def _load_config(self, config_path: str) -> dict:
        """Load MT5 configuration from existing config file"""
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                raise FileNotFoundError(f"Configuration file not found: {config_path}")

            with open(config_file) as f:
                config_data = yaml.safe_load(f)

            mt5_config = config_data.get("mt5", {})

            # Validate required fields
            required_fields = ["login", "password", "server"]
            for field in required_fields:
                if field not in mt5_config or not mt5_config[field]:
                    raise ValueError(
                        f"Missing required MT5 configuration field: {field}"
                    )

            logger.info(f"Loaded MT5 configuration from {config_path}")
            logger.info(f"Login: {mt5_config['login']}")
            logger.info(f"Server: {mt5_config['server']}")

            return mt5_config

        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise

    def connect_to_mt5(self) -> bool:
        """
        Connect to MT5 terminal using existing broker configuration

        Returns:
            bool: True if connection successful
        """
        try:
            # Initialize MT5
            init_params = {}
            if self.config.get("path"):
                init_params["path"] = self.config["path"]
            if self.config.get("portable", False):
                init_params["portable"] = self.config["portable"]

            if not mt5.initialize(**init_params):
                logger.error(
                    f"MT5 initialization failed, error code: {mt5.last_error()}"
                )
                return False

            logger.info("MT5 initialized successfully")

            # Login using configuration
            login_params = {
                "login": int(self.config["login"]),
                "password": self.config["password"],
                "server": self.config["server"],
            }

            if self.config.get("timeout"):
                login_params["timeout"] = self.config["timeout"]

            if not mt5.login(**login_params):
                logger.error(f"MT5 login failed, error code: {mt5.last_error()}")
                return False

            # Verify connection
            account_info = mt5.account_info()
            if account_info is None:
                logger.error("Failed to get account info")
                return False

            logger.info(f"Connected to MT5 successfully!")
            logger.info(f"Account: {account_info.login}")
            logger.info(f"Server: {account_info.server}")
            logger.info(f"Company: {account_info.company}")
            logger.info(f"Currency: {account_info.currency}")
            logger.info(f"Balance: {account_info.balance}")

            return True

        except Exception as e:
            logger.error(f"Error connecting to MT5: {e}")
            return False

    def get_symbol_info(self, symbol: str) -> dict | None:
        """Get symbol information and ensure it's available"""
        try:
            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                logger.warning(f"Symbol {symbol} not found, trying to select...")

                # Try to select the symbol
                if not mt5.symbol_select(symbol, True):
                    logger.error(f"Failed to select symbol {symbol}")
                    return None

                # Try again
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info is None:
                    logger.error(f"Symbol {symbol} still not available")
                    return None

            logger.info(
                f"Symbol {symbol} info: "
                f"spread={symbol_info.spread}, "
                f"digits={symbol_info.digits}, "
                f"point={symbol_info.point}"
            )

            return {
                "name": symbol_info.name,
                "spread": symbol_info.spread,
                "digits": symbol_info.digits,
                "point": symbol_info.point,
                "trade_mode": symbol_info.trade_mode,
            }

        except Exception as e:
            logger.error(f"Error getting symbol info for {symbol}: {e}")
            return None

    def download_symbol_data(
        self, symbol: str, timeframe: int, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame | None:
        """
        Download historical data for a specific symbol and timeframe

        Args:
            symbol: Currency pair symbol
            timeframe: MT5 timeframe constant
            start_date: Start date for data
            end_date: End date for data

        Returns:
            DataFrame with OHLCV data or None if failed
        """
        try:
            logger.info(
                f"Downloading {symbol} {self.timeframes[timeframe]} data from {start_date} to {end_date}"
            )

            # Get rates
            rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)

            if rates is None or len(rates) == 0:
                logger.warning(
                    f"No data received for {symbol} {self.timeframes[timeframe]}"
                )
                return None

            # Convert to DataFrame
            df = pd.DataFrame(rates)

            # Convert time to datetime
            df["time"] = pd.to_datetime(df["time"], unit="s")

            # Rename columns to standard format
            df = df.rename(
                columns={
                    "time": "Date",
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "tick_volume": "Volume",
                }
            )

            # Set Date as index
            df.set_index("Date", inplace=True)

            # Remove any duplicate timestamps
            df = df[~df.index.duplicated(keep="first")]

            # Sort by date
            df.sort_index(inplace=True)

            logger.info(
                f"Downloaded {len(df)} bars for {symbol} {self.timeframes[timeframe]}"
            )
            logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")

            return df

        except Exception as e:
            logger.error(
                f"Error downloading data for {symbol} {self.timeframes[timeframe]}: {e}"
            )
            return None

    def save_data(self, df: pd.DataFrame, symbol: str, timeframe_name: str) -> str:
        """Save DataFrame to CSV file"""
        try:
            filename = f"{symbol}_{timeframe_name}_mt5_live.csv"
            filepath = os.path.join(self.data_dir, filename)

            # Save to CSV with Date column (reset index)
            df_save = df.reset_index()
            df_save.to_csv(filepath, index=False)

            file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
            logger.info(f"Saved {filename} ({file_size:.2f} MB, {len(df)} rows)")

            return filepath

        except Exception as e:
            logger.error(f"Error saving data for {symbol} {timeframe_name}: {e}")
            return ""

    def download_all_data(self) -> dict[str, dict[str, str]]:
        """
        Download all forex data for all symbols and timeframes

        Returns:
            Dictionary with download results
        """
        results = {}

        try:
            # Connect to MT5 using existing configuration
            if not self.connect_to_mt5():
                logger.error("Failed to connect to MT5")
                return results

            # Download data for each symbol
            for symbol in self.symbols:
                logger.info(f"\n{'='*50}")
                logger.info(f"Processing symbol: {symbol}")
                logger.info(f"{'='*50}")

                # Check symbol availability
                symbol_info = self.get_symbol_info(symbol)
                if symbol_info is None:
                    logger.error(f"Skipping {symbol} - not available")
                    continue

                results[symbol] = {}

                # Download each timeframe
                for timeframe, timeframe_name in self.timeframes.items():
                    logger.info(f"\nDownloading {symbol} {timeframe_name}...")

                    # Download data
                    df = self.download_symbol_data(
                        symbol, timeframe, self.start_date, self.end_date
                    )

                    if df is not None and len(df) > 0:
                        # Save data
                        filepath = self.save_data(df, symbol, timeframe_name)
                        results[symbol][timeframe_name] = {
                            "filepath": filepath,
                            "rows": len(df),
                            "start_date": str(df.index[0]),
                            "end_date": str(df.index[-1]),
                            "success": True,
                        }
                    else:
                        logger.warning(f"No data for {symbol} {timeframe_name}")
                        results[symbol][timeframe_name] = {
                            "filepath": "",
                            "rows": 0,
                            "success": False,
                        }

                    # Small delay between requests
                    time.sleep(0.5)

            # Save summary
            self.save_download_summary(results)

        except Exception as e:
            logger.error(f"Error in download_all_data: {e}")

        finally:
            # Shutdown MT5
            mt5.shutdown()
            logger.info("MT5 connection closed")

        return results

    def save_download_summary(self, results: dict) -> None:
        """Save download summary to JSON file"""
        try:
            summary_file = os.path.join(self.data_dir, "mt5_download_summary.json")

            # Add metadata
            summary = {
                "download_time": datetime.now().isoformat(),
                "broker_config": {
                    "login": self.config["login"],
                    "server": self.config["server"],
                    "company": "N/A",  # Will be filled during connection
                },
                "total_symbols": len(self.symbols),
                "total_timeframes": len(self.timeframes),
                "date_range": {
                    "start": self.start_date.isoformat(),
                    "end": self.end_date.isoformat(),
                },
                "results": results,
            }

            with open(summary_file, "w") as f:
                json.dump(summary, f, indent=2)

            logger.info(f"Download summary saved to {summary_file}")

            # Print summary
            self.print_summary(results)

        except Exception as e:
            logger.error(f"Error saving summary: {e}")

    def print_summary(self, results: dict) -> None:
        """Print download summary"""
        logger.info("\n" + "=" * 60)
        logger.info("DOWNLOAD SUMMARY")
        logger.info("=" * 60)

        total_files = 0
        total_rows = 0
        successful = 0

        for symbol, timeframes in results.items():
            logger.info(f"\n{symbol}:")
            for timeframe, data in timeframes.items():
                if data["success"]:
                    logger.info(f"  ✓ {timeframe}: {data['rows']:,} rows")
                    total_files += 1
                    total_rows += data["rows"]
                    successful += 1
                else:
                    logger.info(f"  ✗ {timeframe}: Failed")

        logger.info(
            f"\nTotal successful downloads: {successful}/{len(self.symbols) * len(self.timeframes)}"
        )
        logger.info(f"Total files created: {total_files}")
        logger.info(f"Total data rows: {total_rows:,}")
        logger.info(f"Data saved to: {self.data_dir}")


def main():
    """Main function to run the MT5 data download"""
    print("=" * 60)
    print("MT5 LIVE DATA DOWNLOADER")
    print("=" * 60)
    print("This script will download real forex data from your MT5 broker")
    print("Using existing broker configuration from project settings")
    print("=" * 60)

    try:
        # Create downloader (will load existing config automatically)
        downloader = MT5DataDownloader()

        print(f"\nUsing broker configuration:")
        print(f"  Login: {downloader.config['login']}")
        print(f"  Server: {downloader.config['server']}")
        print(f"  Symbols: {', '.join(downloader.symbols)}")
        print(f"  Timeframes: {', '.join(downloader.timeframes.values())}")
        print(
            f"  Date range: {downloader.start_date.date()} to {downloader.end_date.date()}"
        )

        # Download all data
        print("\nStarting download process...")
        results = downloader.download_all_data()

        if results:
            print(
                f"\n✅ Download completed! Check the '{downloader.data_dir}' folder for your data."
            )
            print("Files are ready to be uploaded to Colab manually.")
            print(f"\nNext steps:")
            print(f"1. Check the generated CSV files in {downloader.data_dir}/")
            print(f"2. Upload the files to your Google Colab environment")
            print(
                f"3. Use the colab_training.py script to train models with this real data"
            )
        else:
            print("\n❌ Download failed. Check the logs for details.")
            print("Common issues:")
            print("- MT5 terminal not running")
            print("- Demo account expired")
            print("- Network connectivity issues")
            print("- Broker server maintenance")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Make sure:")
        print("- MT5 terminal is installed and running")
        print("- Configuration file exists: services/execution/config_mt5.yaml")
        print("- Your demo account credentials are valid")


if __name__ == "__main__":
    main()
