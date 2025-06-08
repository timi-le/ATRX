#!/usr/bin/env python3
"""
Forex Data Combination Script
Combines synthetic and real MT5 forex data for enhanced training datasets
"""

import json
import logging
import os
from datetime import datetime

import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ForexDataCombiner:
    """Combines synthetic and real MT5 forex data for enhanced training"""

    def __init__(self):
        self.synthetic_dir = "data/forex"
        self.real_dir = "data/forex/mt5_live"
        self.output_dir = "data/forex/combined"
        self.ensure_output_dir()

        # Currency pairs and timeframes
        self.symbols = ["EURUSD", "GBPUSD", "USDJPY"]
        self.timeframes = ["M5", "M15", "M30", "H1", "H4", "D1"]

        # Timeframes available in real MT5 data
        self.real_timeframes = ["H1", "H4", "D1"]
        self.synthetic_only_timeframes = ["M5", "M15", "M30"]

    def ensure_output_dir(self):
        """Create output directory if it doesn't exist"""
        os.makedirs(self.output_dir, exist_ok=True)

    def load_synthetic_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Load synthetic data file"""
        try:
            filename = f"{symbol}_{timeframe}_2018_2025.csv"
            filepath = os.path.join(self.synthetic_dir, filename)

            if not os.path.exists(filepath):
                logger.warning(f"Synthetic file not found: {filepath}")
                return pd.DataFrame()

            df = pd.read_csv(filepath)

            # Handle different date column names - synthetic data has 'Unnamed: 0' as date index
            if "Unnamed: 0" in df.columns:
                df["Date"] = pd.to_datetime(df["Unnamed: 0"])
                df = df.drop(columns=["Unnamed: 0"])
            elif "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"])
            elif "time" in df.columns:
                df["Date"] = pd.to_datetime(df["time"])
                df = df.drop(columns=["time"])
            else:
                # Try to use index as date if no date column found
                if df.index.dtype == "object":
                    df["Date"] = pd.to_datetime(df.index)
                    df = df.reset_index(drop=True)
                else:
                    logger.error(
                        f"No date column found in {filepath}. Columns: {df.columns.tolist()}"
                    )
                    return pd.DataFrame()

            # Standardize column names to match expected format
            column_mapping = {
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }

            for old_col, new_col in column_mapping.items():
                if old_col in df.columns:
                    df = df.rename(columns={old_col: new_col})

            # Ensure we have required columns
            required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
            for col in required_cols:
                if col not in df.columns:
                    logger.error(f"Missing required column '{col}' in {filepath}")
                    return pd.DataFrame()

            # Select only required columns and add source marker
            df = df[required_cols].copy()
            df["source"] = "synthetic"

            logger.info(f"Loaded synthetic {symbol} {timeframe}: {len(df)} rows")
            return df

        except Exception as e:
            logger.error(f"Error loading synthetic data for {symbol} {timeframe}: {e}")
            return pd.DataFrame()

    def load_real_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Load real MT5 data file"""
        try:
            filename = f"{symbol}_{timeframe}_mt5_live.csv"
            filepath = os.path.join(self.real_dir, filename)

            if not os.path.exists(filepath):
                logger.warning(f"Real file not found: {filepath}")
                return pd.DataFrame()

            df = pd.read_csv(filepath)

            # Handle different date column names
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"])
            elif "time" in df.columns:
                df["Date"] = pd.to_datetime(df["time"])
                df = df.drop(columns=["time"])
            else:
                logger.error(
                    f"No date column found in {filepath}. Columns: {df.columns.tolist()}"
                )
                return pd.DataFrame()

            # Remove MT5-specific columns
            columns_to_remove = ["spread", "real_volume"]
            for col in columns_to_remove:
                if col in df.columns:
                    df = df.drop(columns=[col])

            # Standardize column names to match expected format
            column_mapping = {
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
                "tick_volume": "Volume",  # MT5 uses tick_volume
            }

            for old_col, new_col in column_mapping.items():
                if old_col in df.columns:
                    df = df.rename(columns={old_col: new_col})

            # Ensure we have required columns
            required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
            for col in required_cols:
                if col not in df.columns:
                    logger.error(f"Missing required column '{col}' in {filepath}")
                    return pd.DataFrame()

            # Select only required columns and add source marker
            df = df[required_cols].copy()
            df["source"] = "real_mt5"

            logger.info(f"Loaded real {symbol} {timeframe}: {len(df)} rows")
            return df

        except Exception as e:
            logger.error(f"Error loading real data for {symbol} {timeframe}: {e}")
            return pd.DataFrame()

    def combine_datasets(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Combine synthetic and real data for a specific symbol/timeframe"""
        try:
            # Load synthetic data (always available)
            synthetic_df = self.load_synthetic_data(symbol, timeframe)

            # For timeframes where we have real data, combine both
            if timeframe in self.real_timeframes:
                real_df = self.load_real_data(symbol, timeframe)

                if not real_df.empty and not synthetic_df.empty:
                    # Combine both datasets
                    combined_df = pd.concat([synthetic_df, real_df], ignore_index=True)

                    # Sort by date
                    combined_df = combined_df.sort_values("Date").reset_index(drop=True)

                    # Remove duplicate dates (prefer real data over synthetic)
                    combined_df = combined_df.drop_duplicates(
                        subset=["Date"], keep="last"
                    )

                    logger.info(
                        f"Combined {symbol} {timeframe}: {len(synthetic_df)} synthetic + {len(real_df)} real = {len(combined_df)} total"
                    )
                    return combined_df

                elif not real_df.empty:
                    # Only real data available
                    logger.info(
                        f"Using only real data for {symbol} {timeframe}: {len(real_df)} rows"
                    )
                    return real_df

                elif not synthetic_df.empty:
                    # Only synthetic data available
                    logger.info(
                        f"Using only synthetic data for {symbol} {timeframe}: {len(synthetic_df)} rows"
                    )
                    return synthetic_df
                else:
                    logger.warning(f"No data available for {symbol} {timeframe}")
                    return pd.DataFrame()
            else:
                # For M5, M15, M30 - only synthetic data available
                if not synthetic_df.empty:
                    logger.info(
                        f"Using synthetic data for {symbol} {timeframe}: {len(synthetic_df)} rows"
                    )
                    return synthetic_df
                else:
                    logger.warning(
                        f"No synthetic data available for {symbol} {timeframe}"
                    )
                    return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error combining datasets for {symbol} {timeframe}: {e}")
            return pd.DataFrame()

    def save_combined_data(self, df: pd.DataFrame, symbol: str, timeframe: str) -> str:
        """Save combined dataset to file"""
        try:
            if df.empty:
                logger.warning(f"No data to save for {symbol} {timeframe}")
                return ""

            filename = f"{symbol}_{timeframe}_combined.csv"
            filepath = os.path.join(self.output_dir, filename)

            # Remove source column before saving
            df_save = df.drop(columns=["source"] if "source" in df.columns else [])
            df_save.to_csv(filepath, index=False)

            file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
            logger.info(f"Saved {filename} ({file_size:.2f} MB, {len(df)} rows)")

            return filepath

        except Exception as e:
            logger.error(f"Error saving combined data for {symbol} {timeframe}: {e}")
            return ""

    def generate_summary_report(self, results: dict) -> dict:
        """Generate comprehensive summary report"""
        try:
            summary = {
                "generation_time": datetime.now().isoformat(),
                "approach": "Data Augmentation - Synthetic + Real MT5 Combined",
                "source_directories": {
                    "synthetic": self.synthetic_dir,
                    "real_mt5": self.real_dir,
                    "output": self.output_dir,
                },
                "symbols": self.symbols,
                "timeframes": self.timeframes,
                "combination_strategy": {
                    "real_enhanced": self.real_timeframes,
                    "synthetic_only": self.synthetic_only_timeframes,
                },
                "results": results,
                "statistics": {},
            }

            # Calculate statistics
            total_files = 0
            total_rows = 0
            synthetic_only_files = 0
            enhanced_files = 0
            failed_files = 0

            for symbol, timeframes in results.items():
                for timeframe, data in timeframes.items():
                    if data["success"]:
                        total_files += 1
                        total_rows += data["rows"]

                        if timeframe in self.real_timeframes:
                            enhanced_files += 1
                        else:
                            synthetic_only_files += 1
                    else:
                        failed_files += 1

            summary["statistics"] = {
                "total_files_created": total_files,
                "total_data_rows": total_rows,
                "enhanced_files_with_real_data": enhanced_files,
                "synthetic_only_files": synthetic_only_files,
                "failed_combinations": failed_files,
                "success_rate": f"{(total_files / (total_files + failed_files) * 100):.1f}%"
                if (total_files + failed_files) > 0
                else "0%",
            }

            return summary

        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return {}

    def combine_all_data(self) -> dict:
        """Combine all forex data for all symbols and timeframes"""
        results = {}

        logger.info("=" * 60)
        logger.info("FOREX DATA COMBINATION - AUGMENTATION APPROACH")
        logger.info("=" * 60)
        logger.info("Combining synthetic and real MT5 data for enhanced training")
        logger.info(f"Output directory: {self.output_dir}")

        try:
            for symbol in self.symbols:
                logger.info(f"\n{'='*50}")
                logger.info(f"Processing symbol: {symbol}")
                logger.info(f"{'='*50}")

                results[symbol] = {}

                for timeframe in self.timeframes:
                    logger.info(f"\nCombining {symbol} {timeframe}...")

                    # Combine datasets
                    combined_df = self.combine_datasets(symbol, timeframe)

                    if not combined_df.empty:
                        # Save combined data
                        filepath = self.save_combined_data(
                            combined_df, symbol, timeframe
                        )

                        # Count data sources
                        synthetic_count = (
                            len(combined_df[combined_df["source"] == "synthetic"])
                            if "source" in combined_df.columns
                            else 0
                        )
                        real_count = (
                            len(combined_df[combined_df["source"] == "real_mt5"])
                            if "source" in combined_df.columns
                            else 0
                        )

                        results[symbol][timeframe] = {
                            "filepath": filepath,
                            "rows": len(combined_df),
                            "synthetic_rows": synthetic_count,
                            "real_rows": real_count,
                            "start_date": str(combined_df["Date"].min()),
                            "end_date": str(combined_df["Date"].max()),
                            "success": True,
                        }
                    else:
                        logger.warning(f"No data combined for {symbol} {timeframe}")
                        results[symbol][timeframe] = {
                            "filepath": "",
                            "rows": 0,
                            "synthetic_rows": 0,
                            "real_rows": 0,
                            "success": False,
                        }

            # Generate and save summary
            summary = self.generate_summary_report(results)
            self.save_summary(summary)

            # Print final summary
            self.print_final_summary(summary)

        except Exception as e:
            logger.error(f"Error in combine_all_data: {e}")

        return results

    def save_summary(self, summary: dict) -> None:
        """Save combination summary to JSON file"""
        try:
            summary_file = os.path.join(self.output_dir, "combination_summary.json")

            with open(summary_file, "w") as f:
                json.dump(summary, f, indent=2)

            logger.info(f"Combination summary saved to {summary_file}")

        except Exception as e:
            logger.error(f"Error saving summary: {e}")

    def print_final_summary(self, summary: dict) -> None:
        """Print final combination summary"""
        try:
            logger.info("\n" + "=" * 60)
            logger.info("COMBINATION SUMMARY")
            logger.info("=" * 60)

            stats = summary.get("statistics", {})

            logger.info(f"Total files created: {stats.get('total_files_created', 0)}")
            logger.info(f"Total data rows: {stats.get('total_data_rows', 0):,}")
            logger.info(
                f"Enhanced files (real+synthetic): {stats.get('enhanced_files_with_real_data', 0)}"
            )
            logger.info(f"Synthetic-only files: {stats.get('synthetic_only_files', 0)}")
            logger.info(f"Success rate: {stats.get('success_rate', '0%')}")

            logger.info(f"\nData breakdown by symbol:")
            results = summary.get("results", {})
            for symbol, timeframes in results.items():
                logger.info(f"\n{symbol}:")
                for timeframe, data in timeframes.items():
                    if data["success"]:
                        synthetic_rows = data.get("synthetic_rows", 0)
                        real_rows = data.get("real_rows", 0)
                        total_rows = data["rows"]

                        if real_rows > 0:
                            logger.info(
                                f"  ✓ {timeframe}: {total_rows:,} rows ({synthetic_rows:,} synthetic + {real_rows:,} real)"
                            )
                        else:
                            logger.info(
                                f"  ✓ {timeframe}: {total_rows:,} rows (synthetic only)"
                            )
                    else:
                        logger.info(f"  ✗ {timeframe}: Failed")

            logger.info(
                f"\nCombined data ready for Colab upload: {summary['source_directories']['output']}"
            )

        except Exception as e:
            logger.error(f"Error printing summary: {e}")


def main():
    """Main function to run the data combination"""
    print("=" * 60)
    print("FOREX DATA COMBINATION TOOL")
    print("=" * 60)
    print("Option 2: Data Augmentation Approach")
    print("Combining synthetic + real MT5 data for maximum training samples")
    print("=" * 60)

    try:
        # Create combiner
        combiner = ForexDataCombiner()

        print("\nData sources:")
        print(f"  Synthetic data: {combiner.synthetic_dir}")
        print(f"  Real MT5 data: {combiner.real_dir}")
        print(f"  Combined output: {combiner.output_dir}")

        print(f"\nStrategy:")
        print(f"  Enhanced timeframes (synthetic + real): {combiner.real_timeframes}")
        print(f"  Synthetic-only timeframes: {combiner.synthetic_only_timeframes}")

        # Combine all data
        print("\nStarting data combination process...")
        results = combiner.combine_all_data()

        if results:
            print(f"\n✅ Data combination completed!")
            print(f"Check the '{combiner.output_dir}' folder for combined datasets.")
            print(
                "Files are ready for enhanced Colab training with maximum data coverage."
            )
        else:
            print("\n❌ Data combination failed. Check the logs for details.")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Make sure both synthetic and real MT5 data directories exist.")


if __name__ == "__main__":
    main()
