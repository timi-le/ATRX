"""
Train and Save Base Predictor Models (Multi-Timeframe Version)

This script implements a professional-grade training pipeline.
It performs the following steps:
1.  Downloads historical data for specified symbols across multiple timeframes (M15, H1, D1).
2.  Generates a rich, multi-timeframe feature set, aligning all features to the primary M15 timeframe.
3.  Applies the Triple-Barrier Method to label the M15 data for supervised learning.
4.  Trains three distinct base models: XGBoost, LSTM, and CNN, on the rich MTF feature set.
5.  Saves the trained models to the 'models/' directory for the meta-learner.
"""
import asyncio
import os
import sys
from pathlib import Path

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# --- Setup Project Root ---
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
dotenv_path = project_root / ".env"
load_dotenv(dotenv_path=dotenv_path)

from core.feature_engine import FeatureConfig, HighPerformanceFeatureEngine
from core.labeling import generate_triple_barrier_labels

# --- Import Core Modules ---
from core.ml_predictor import CNNPredictor, LSTMPredictor, XGBoostPredictor
from data.mt5_data_downloader import MT5DataDownloader

# --- MTF Configuration ---
START_DATE = "2018-01-01"
END_DATE = "2025-03-31"
SYMBOLS = ["EURUSD", "GBPUSD"]
PRIMARY_TIMEFRAME = "M15"
TIMEFRAMES = [
    mt5.TIMEFRAME_M15,
    mt5.TIMEFRAME_H1,
    mt5.TIMEFRAME_D1,
]

# --- Triple Barrier Configuration ---
LOOK_AHEAD_PERIODS = 8
PT_SL_MULT = [1.5, 1.5]  # Symmetrical profit-take/stop-loss ATR multipliers
MIN_VOLATILITY_LOOKBACK = 20  # Look at last 20 bars of M15 for vol


async def main():
    """Main function to run the MTF training pipeline."""
    print("🚀 Starting Professional Multi-Timeframe Model Training Pipeline 🚀")
    Path("models").mkdir(exist_ok=True)

    # --- 1. Download Data ---
    mt5_login = os.getenv("MT5_LOGIN")
    mt5_password = os.getenv("MT5_PASSWORD")
    mt5_server = os.getenv("MT5_SERVER")

    if not all([mt5_login, mt5_password, mt5_server]):
        print("FATAL: MT5 credentials not found in .env file. Cannot download data.")
        return

    try:
        downloader = MT5DataDownloader(
            login=int(mt5_login), password=mt5_password, server=mt5_server
        )
    except Exception as e:
        print(f"FATAL: Failed to connect to MetaTrader 5. Error: {e}")
        return

    # Download MTF data for all symbols and combine them
    all_symbols_mtf_data: dict[str, pd.DataFrame] = {}
    for symbol in SYMBOLS:
        print(f"\nDownloading data for {symbol}...")
        mtf_data = downloader.download_mtf_data(
            symbol=symbol,
            timeframes=TIMEFRAMES,
            start_date=pd.to_datetime(START_DATE),
            end_date=pd.to_datetime(END_DATE),
        )

        # Add to combined dictionary, handling potential overlaps
        for tf_str, df in mtf_data.items():
            if tf_str not in all_symbols_mtf_data:
                all_symbols_mtf_data[tf_str] = []
            all_symbols_mtf_data[tf_str].append(df)

    downloader.disconnect()

    # Concatenate data from all symbols for each timeframe
    combined_mtf_data = {}
    for tf_str, df_list in all_symbols_mtf_data.items():
        if df_list:
            df = (
                pd.concat(df_list)
                .sort_values("time")
                .drop_duplicates("time")
                .set_index("time")
            )
            combined_mtf_data[tf_str] = df
            print(f"Combined {tf_str} data: {len(df)} unique bars.")

    if not combined_mtf_data:
        print("FATAL: No data was downloaded. Cannot proceed.")
        return

    # --- 2. Feature Engineering ---
    print("\n🔬 Generating Multi-Timeframe Features...")
    feature_config = FeatureConfig(use_numba=True, max_workers=os.cpu_count() or 1)
    feature_engine = HighPerformanceFeatureEngine(config=feature_config)

    # Generate the rich MTF feature set
    feature_df = await feature_engine.compute_mtf_features(
        mtf_data=combined_mtf_data, primary_tf=PRIMARY_TIMEFRAME
    )

    print(
        f"Generated {len(feature_df.columns)} features for {len(feature_df)} data points."
    )
    if feature_df.empty:
        print("FATAL: No features were generated. Check data and feature engine logic.")
        return

    # --- 3. Labeling ---
    print("\n🏷️ Applying Triple-Barrier Labeling...")

    # The labeling function needs 'close' and 'atr' from the primary timeframe
    primary_tf_atr_col = f"atr_{PRIMARY_TIMEFRAME}"
    if primary_tf_atr_col not in feature_df.columns:
        # As a fallback, compute ATR on the primary data if not present in features
        feature_df[primary_tf_atr_col] = _compute_atr(
            feature_df["high"].values,
            feature_df["low"].values,
            feature_df["close"].values,
            period=14,
        )

    labeling_input_df = feature_df[["close", primary_tf_atr_col]].rename(
        columns={primary_tf_atr_col: "atr"}
    )

    labels = generate_triple_barrier_labels(
        data=labeling_input_df,
        upper_mult=PT_SL_MULT[0],
        lower_mult=PT_SL_MULT[1],
        vertical_barrier_bars=LOOK_AHEAD_PERIODS,
    )

    # Align final features and labels
    final_df = feature_df.join(labels).dropna()
    final_features = final_df.drop(
        columns=[
            "open",
            "high",
            "low",
            "close",
            "tick_volume",
            "spread",
            "real_volume",
            "triple_barrier_label",
        ]
    )
    final_labels = final_df["triple_barrier_label"]

    print(
        f"Generated {len(final_labels)} labels. Distribution:\n{final_labels.value_counts(normalize=True)}"
    )

    if final_features.empty or final_labels.empty:
        print(
            "FATAL: No data remaining after feature generation and labeling. Cannot train."
        )
        return

    # --- 4. Train & Save Models ---
    print("\n🤖 Training and Saving Models...")

    # XGBoost
    print("\n--- Training XGBoost Model ---")
    xgb_predictor = XGBoostPredictor()
    xgb_predictor.train(final_features, final_labels)
    xgb_predictor.save("models/xgboost_model_mtf.json")
    print("XGBoost MTF model saved successfully.")

    # LSTM
    print("\n--- Training LSTM Model ---")
    lstm_predictor = LSTMPredictor(n_features=len(final_features.columns))
    lstm_predictor.train(final_features, final_labels)
    lstm_predictor.save("models/lstm_model_mtf.h5")
    print("LSTM MTF model saved successfully.")

    # CNN
    print("\n--- Training CNN Model ---")
    cnn_predictor = CNNPredictor(
        n_features=final_features.shape[1],
        image_size=int(np.sqrt(final_features.shape[1])),
    )
    cnn_predictor.train(final_features, final_labels)
    cnn_predictor.save("models/cnn_model_mtf.h5")
    print("CNN MTF model saved successfully.")

    print(
        "\n✅ Professional Multi-Timeframe Training Pipeline Completed Successfully! ✅"
    )


if __name__ == "__main__":
    # Windows compatibility for asyncio
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # This import is needed here for the script to run standalone for the labeling part.
    from core.feature_engine import _compute_atr

    asyncio.run(main())
