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


def download_mt5_fx_data(symbol, timeframe='H1', start_date="2018-01-01", end_date="2025-03-31"):
    """Download data from MetaTrader 5 for specified timeframe"""
    if not MT5_AVAILABLE:
        return None

    print(f"📊 Attempting MT5 download: {symbol} {timeframe} ({start_date} to {end_date})")

    try:
        # Initialize MT5
        if not mt5.initialize():
            print(f"❌ MT5 initialization failed")
            return None

        # Timeframe mapping
        timeframe_map = {
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1,
        }

        if timeframe not in timeframe_map:
            print(f"❌ Unsupported timeframe: {timeframe}")
            return None

        tf = timeframe_map[timeframe]

        # Convert dates to datetime
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        # Get data
        rates = mt5.copy_rates_range(symbol, tf, start_dt, end_dt)

        if rates is None or len(rates) == 0:
            print(f"❌ No {symbol} {timeframe} data returned from MT5")
            return None

        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)

        # Filter by date range to ensure we're within bounds
        df = df[(df.index >= start_dt) & (df.index <= end_dt)]

        # Keep only OHLCV columns
        df = df[['open', 'high', 'low', 'close', 'tick_volume']].copy()
        df.rename(columns={'tick_volume': 'volume'}, inplace=True)

        print(f"✅ Downloaded {len(df):,} {symbol} {timeframe} MT5 data points")
        return df

    except Exception as e:
        print(f"❌ MT5 download failed for {symbol} {timeframe}: {e}")
        return None
    finally:
        if MT5_AVAILABLE:
            mt5.shutdown()


def download_currency_pair(symbol, timeframe='H1'):
    """Download data for a single currency pair and timeframe with fallbacks"""
    print(f"\n📊 DOWNLOADING {symbol} {timeframe} DATA")
    print(f"{'='*40}")

    # Try MT5 first
    data = download_mt5_fx_data(symbol, timeframe)
    if data is not None and len(data) > 1000:
        data["source"] = "MT5"
        data["timeframe"] = timeframe
        return data

    # For non-H1 timeframes, try to resample from H1 if Yahoo Finance
    if timeframe != 'H1':
        print(f"⏬ Trying to get {symbol} H1 data for resampling...")
        h1_data = download_yahoo_fx_data(symbol)
        if h1_data is not None and len(h1_data) > 1000:
            resampled = resample_to_timeframe(h1_data, timeframe)
            if resampled is not None:
                resampled["source"] = "YahooFinance_Resampled"
                resampled["timeframe"] = timeframe
                return resampled
    else:
        # Try Yahoo Finance for H1
        data = download_yahoo_fx_data(symbol)
        if data is not None and len(data) > 1000:
            data["source"] = "YahooFinance"
            data["timeframe"] = timeframe
            return data

    # Generate synthetic as last resort
    data = generate_synthetic_fx_data(symbol, timeframe)
    data["source"] = "Synthetic"
    data["timeframe"] = timeframe
    return data


def resample_to_timeframe(df, target_timeframe):
    """Resample hourly data to target timeframe"""
    try:
        timeframe_rules = {
            'M5': '5T',    # 5 minutes
            'M15': '15T',  # 15 minutes  
            'M30': '30T',  # 30 minutes
            'H1': '1H',    # 1 hour
            'H4': '4H',    # 4 hours
            'D1': '1D',    # 1 day
        }
        
        if target_timeframe not in timeframe_rules:
            return None
            
        rule = timeframe_rules[target_timeframe]
        
        # For upsampling (M5, M15, M30), we can't create more granular data
        if target_timeframe in ['M5', 'M15', 'M30']:
            print(f"⚠️ Cannot upsample to {target_timeframe} from H1 data")
            return None
        
        # Resample OHLC data
        resampled = df.resample(rule).agg({
            'open': 'first',
            'high': 'max', 
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        print(f"✅ Resampled to {target_timeframe}: {len(resampled):,} samples")
        return resampled
        
    except Exception as e:
        print(f"❌ Resampling failed: {e}")
        return None


def generate_synthetic_fx_data(symbol, timeframe='H1', start_date="2018-01-01", end_date="2025-03-31"):
    """Generate realistic synthetic forex data for specified timeframe"""
    print(f"🎲 Generating synthetic {symbol} {timeframe} data...")

    # Timeframe frequency mapping
    freq_map = {
        'M5': '5T',
        'M15': '15T', 
        'M30': '30T',
        'H1': '1H',
        'H4': '4H',
        'D1': '1D'
    }
    
    freq = freq_map.get(timeframe, '1H')
    dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    n_samples = len(dates)

    # Currency-specific parameters (adjusted by timeframe)
    params = {
        "EURUSD": {"base_price": 1.1000, "volatility": 0.0003, "trend": 0.10},
        "GBPUSD": {"base_price": 1.2500, "volatility": 0.0004, "trend": -0.05},
        "USDJPY": {"base_price": 110.00, "volatility": 0.0003, "trend": 0.15},
    }

    param = params.get(symbol, params["EURUSD"])
    
    # Adjust volatility by timeframe
    vol_multiplier = {
        'M5': 0.3, 'M15': 0.5, 'M30': 0.7, 
        'H1': 1.0, 'H4': 1.8, 'D1': 3.0
    }
    
    base_vol = param["volatility"] * vol_multiplier.get(timeframe, 1.0)

    np.random.seed(hash(f"{symbol}_{timeframe}") % 1000)

    # Generate regime-based returns
    regimes = np.random.choice([0, 1, 2], n_samples, p=[0.6, 0.25, 0.15])

    # Add regime persistence
    for i in range(1, n_samples):
        if np.random.random() < 0.85:
            regimes[i] = regimes[i - 1]

    returns = np.zeros(n_samples)

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

    # Create OHLC data with timeframe-adjusted spread
    noise = np.random.normal(0, base_vol * 0.1, n_samples)
    spread = np.random.normal(0, base_vol * 0.3, n_samples)

    data = pd.DataFrame(
        {
            "open": prices + noise,
            "high": prices + np.abs(spread) + np.abs(noise),
            "low": prices - np.abs(spread) - np.abs(noise),
            "close": prices,
            "volume": np.random.lognormal(8 + np.log(vol_multiplier.get(timeframe, 1.0)), 0.5, n_samples),
        },
        index=dates,
    )

    # Ensure OHLC relationships
    data["high"] = np.maximum(data["high"], np.maximum(data["open"], data["close"]))
    data["low"] = np.minimum(data["low"], np.minimum(data["open"], data["close"]))

    print(f"✅ Generated {len(data):,} {symbol} {timeframe} synthetic data points")
    return data


def main():
    """Download multi-currency, multi-timeframe forex data"""
    print("🚀 MULTI-CURRENCY MULTI-TIMEFRAME FOREX DATA DOWNLOADER")
    print("=" * 70)

    # Currency pairs and timeframes
    currency_pairs = ["EURUSD", "GBPUSD", "USDJPY"]
    timeframes = ["M5", "M15", "M30", "H1", "H4", "D1"]

    # Create data directory
    data_dir = "data/forex"
    os.makedirs(data_dir, exist_ok=True)

    download_summary = {}

    # Download each currency pair and timeframe
    for symbol in currency_pairs:
        download_summary[symbol] = {}
        
        for timeframe in timeframes:
            try:
                data = download_currency_pair(symbol, timeframe)

                if data is not None:
                    # Save to CSV
                    filename = f"{data_dir}/{symbol}_{timeframe}_2018_2025.csv"
                    data.to_csv(filename)

                    download_summary[symbol][timeframe] = {
                        "status": "SUCCESS",
                        "source": data["source"].iloc[0] if "source" in data.columns else "Unknown",
                        "samples": len(data),
                        "date_range": f"{data.index.min()} to {data.index.max()}",
                        "file": filename,
                    }

                    print(f"💾 Saved {symbol} {timeframe} data to: {filename}")

                else:
                    download_summary[symbol][timeframe] = {
                        "status": "FAILED",
                        "source": "None",
                        "samples": 0,
                        "date_range": "N/A", 
                        "file": "N/A",
                    }

            except Exception as e:
                print(f"❌ Error downloading {symbol} {timeframe}: {e}")
                download_summary[symbol][timeframe] = {
                    "status": "ERROR",
                    "source": "Error",
                    "samples": 0,
                    "date_range": str(e),
                    "file": "N/A",
                }

    # Print comprehensive summary
    print(f"\n{'='*70}")
    print("📊 MULTI-TIMEFRAME DOWNLOAD SUMMARY")
    print(f"{'='*70}")

    total_files = 0
    total_samples = 0

    for symbol in currency_pairs:
        print(f"\n🔹 {symbol}:")
        for timeframe in timeframes:
            info = download_summary[symbol][timeframe]
            status_icon = "✅" if info['status'] == "SUCCESS" else "❌"
            print(f"   {status_icon} {timeframe}: {info['samples']:,} samples ({info['source']})")
            
            if info['status'] == "SUCCESS":
                total_files += 1
                total_samples += info['samples']

    print(f"\n{'='*70}")
    print(f"📈 TOTAL: {total_files} files, {total_samples:,} total samples")
    print(f"🎯 Coverage: {len(currency_pairs)} currencies × {len(timeframes)} timeframes")
    print(f"{'='*70}")

    # Save detailed summary to JSON
    import json

    summary_file = f"{data_dir}/multi_timeframe_summary.json"
    with open(summary_file, "w") as f:
        json.dump(download_summary, f, indent=2, default=str)

    print(f"\n💾 Detailed summary saved to: {summary_file}")
    print(f"\n✅ Multi-timeframe data download complete! Ready for robust model training.")

    return download_summary


if __name__ == "__main__":
    summary = main()
