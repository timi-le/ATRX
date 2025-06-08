"""
Multi-Currency Forex Data Downloader
Downloads EURUSD, GBPUSD, USDJPY data and saves to CSV files for GitHub storage
"""

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Try to import data sources
try:
    import MetaTrader5 as mt5

    MT5_AVAILABLE = True
    print("✅ MetaTrader5 available")
except ImportError:
    MT5_AVAILABLE = False
    print("⚠️ MetaTrader5 not available")

try:
    import yfinance as yf

    YF_AVAILABLE = True
    print("✅ Yahoo Finance available")
except ImportError:
    YF_AVAILABLE = False
    print("⚠️ Yahoo Finance not available")


def download_yahoo_fx_data(symbol, start_date="2018-01-01", end_date="2025-03-31"):
    """Download forex data from Yahoo Finance"""
    if not YF_AVAILABLE:
        return None

    try:
        symbol_yf = f"{symbol}=X"
        print(f"📈 Downloading {symbol} from Yahoo Finance...")

        ticker = yf.Ticker(symbol_yf)

        # Try different time periods
        data = None
        for period in ["max", "5y", "2y"]:
            try:
                print(f"🔄 Trying period: {period}")
                data = ticker.history(period=period, interval="1h")
                if len(data) > 1000:
                    break
            except Exception as e:
                print(f"❌ Period {period} failed: {e}")
                continue

        if data is None or len(data) < 1000:
            print(f"❌ Failed to download sufficient {symbol} data")
            return None

        # Clean column names
        data.columns = data.columns.str.lower()

        # Filter to date range if possible
        try:
            data = data[(data.index >= start_date) & (data.index <= end_date)]
        except:
            pass  # Use all available data if filtering fails

        print(f"✅ Downloaded {len(data):,} {symbol} data points")
        print(f"📅 Date range: {data.index.min()} to {data.index.max()}")

        return data

    except Exception as e:
        print(f"❌ Yahoo Finance download failed for {symbol}: {e}")
        return None


def download_mt5_fx_data(symbol, start_date="2018-01-01", end_date="2025-03-31"):
    """Download forex data from MT5"""
    if not MT5_AVAILABLE:
        return None

    try:
        print(f"📊 Attempting MT5 download for {symbol}...")

        if not mt5.initialize():
            print("❌ MT5 initialization failed")
            return None

        # Convert dates
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        # Get data
        rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_H1, start_dt, end_dt)

        if rates is None or len(rates) == 0:
            print(f"❌ No MT5 data for {symbol}")
            return None

        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)

        # Clean columns
        df.columns = [
            "open",
            "high",
            "low",
            "close",
            "tick_volume",
            "spread",
            "real_volume",
        ]
        df = df[["open", "high", "low", "close", "tick_volume"]].copy()
        df.rename(columns={"tick_volume": "volume"}, inplace=True)

        print(f"✅ Downloaded {len(df):,} {symbol} MT5 data points")
        return df

    except Exception as e:
        print(f"❌ MT5 download failed for {symbol}: {e}")
        return None
    finally:
        if MT5_AVAILABLE:
            mt5.shutdown()


def generate_synthetic_fx_data(symbol, start_date="2018-01-01", end_date="2025-03-31"):
    """Generate realistic synthetic forex data"""
    print(f"🎲 Generating synthetic {symbol} data...")

    # Date range
    dates = pd.date_range(start=start_date, end=end_date, freq="1H")
    n_samples = len(dates)

    # Currency-specific parameters
    params = {
        "EURUSD": {"base_price": 1.1000, "volatility": 0.0003, "trend": 0.10},
        "GBPUSD": {"base_price": 1.2500, "volatility": 0.0004, "trend": -0.05},
        "USDJPY": {"base_price": 110.00, "volatility": 0.0003, "trend": 0.15},
    }

    param = params.get(symbol, params["EURUSD"])

    np.random.seed(hash(symbol) % 1000)  # Different seed per currency

    # Generate regime-based returns
    regimes = np.random.choice([0, 1, 2], n_samples, p=[0.6, 0.25, 0.15])

    # Add regime persistence
    for i in range(1, n_samples):
        if np.random.random() < 0.85:
            regimes[i] = regimes[i - 1]

    returns = np.zeros(n_samples)
    base_vol = param["volatility"]

    for i in range(n_samples):
        if regimes[i] == 0:  # Normal
            returns[i] = np.random.normal(0, base_vol)
        elif regimes[i] == 1:  # Trending
            momentum = (
                base_vol * 0.5 if i > 0 and returns[i - 1] > 0 else -base_vol * 0.5
            )
            returns[i] = np.random.normal(momentum, base_vol * 0.7)
        else:  # Volatile
            returns[i] = np.random.normal(0, base_vol * 3)

    # Add volatility clustering
    for i in range(1, n_samples):
        if abs(returns[i - 1]) > base_vol * 2:
            returns[i] *= 1.4

    # Add long-term trend and cycles
    trend_component = np.linspace(0, param["trend"], n_samples)
    cycle_component = 0.03 * np.sin(np.linspace(0, 12 * np.pi, n_samples))

    # Generate prices
    prices = param["base_price"] * np.exp(
        np.cumsum(returns) + (trend_component + cycle_component) * 0.1
    )

    # Create OHLC data
    noise = np.random.normal(0, base_vol * 0.1, n_samples)
    spread = np.random.normal(0, base_vol * 0.3, n_samples)

    data = pd.DataFrame(
        {
            "open": prices + noise,
            "high": prices + np.abs(spread) + np.abs(noise),
            "low": prices - np.abs(spread) - np.abs(noise),
            "close": prices,
            "volume": np.random.lognormal(10, 0.5, n_samples),
        },
        index=dates,
    )

    # Ensure OHLC relationships
    data["high"] = np.maximum(data["high"], np.maximum(data["open"], data["close"]))
    data["low"] = np.minimum(data["low"], np.minimum(data["open"], data["close"]))

    print(f"✅ Generated {len(data):,} {symbol} synthetic data points")
    return data


def download_currency_pair(symbol):
    """Download data for a single currency pair with fallbacks"""
    print(f"\n{'='*50}")
    print(f"📊 DOWNLOADING {symbol} DATA")
    print(f"{'='*50}")

    # Try MT5 first
    data = download_mt5_fx_data(symbol)
    if data is not None and len(data) > 1000:
        data["source"] = "MT5"
        return data

    # Try Yahoo Finance
    data = download_yahoo_fx_data(symbol)
    if data is not None and len(data) > 1000:
        data["source"] = "YahooFinance"
        return data

    # Generate synthetic as last resort
    data = generate_synthetic_fx_data(symbol)
    data["source"] = "Synthetic"
    return data


def main():
    """Download multi-currency forex data"""
    print("🚀 MULTI-CURRENCY FOREX DATA DOWNLOADER")
    print("=" * 60)

    # Currency pairs to download
    currency_pairs = ["EURUSD", "GBPUSD", "USDJPY"]

    # Create data directory
    data_dir = "data/forex"
    os.makedirs(data_dir, exist_ok=True)

    download_summary = {}

    # Download each currency pair
    for symbol in currency_pairs:
        try:
            data = download_currency_pair(symbol)

            if data is not None:
                # Save to CSV
                filename = f"{data_dir}/{symbol}_2018_2025.csv"
                data.to_csv(filename)

                download_summary[symbol] = {
                    "status": "SUCCESS",
                    "source": data["source"].iloc[0]
                    if "source" in data.columns
                    else "Unknown",
                    "samples": len(data),
                    "date_range": f"{data.index.min()} to {data.index.max()}",
                    "file": filename,
                }

                print(f"💾 Saved {symbol} data to: {filename}")

            else:
                download_summary[symbol] = {
                    "status": "FAILED",
                    "source": "None",
                    "samples": 0,
                    "date_range": "N/A",
                    "file": "N/A",
                }

        except Exception as e:
            print(f"❌ Error downloading {symbol}: {e}")
            download_summary[symbol] = {
                "status": "ERROR",
                "source": "Error",
                "samples": 0,
                "date_range": str(e),
                "file": "N/A",
            }

    # Print summary
    print(f"\n{'='*60}")
    print("📊 DOWNLOAD SUMMARY")
    print(f"{'='*60}")

    for symbol, info in download_summary.items():
        print(f"\n🔹 {symbol}:")
        print(f"   Status: {info['status']}")
        print(f"   Source: {info['source']}")
        print(f"   Samples: {info['samples']:,}")
        print(f"   Range: {info['date_range']}")
        print(f"   File: {info['file']}")

    # Save summary to JSON
    import json

    summary_file = f"{data_dir}/download_summary.json"
    with open(summary_file, "w") as f:
        json.dump(download_summary, f, indent=2, default=str)

    print(f"\n💾 Summary saved to: {summary_file}")
    print(f"\n✅ Data download complete! Files ready for GitHub commit.")

    return download_summary


if __name__ == "__main__":
    summary = main()
