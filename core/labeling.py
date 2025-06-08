"""
Triple-Barrier Method for Financial Machine Learning Labeling.

This module provides functions to generate labels for model training based on the
Triple-Barrier Method, a technique popularized by Dr. Marcos Lopez de Prado.
"""

import numpy as np
import pandas as pd
import structlog
from numba import jit


@jit(nopython=True, cache=True)
def _apply_triple_barrier(
    prices: np.ndarray,
    atr: np.ndarray,
    upper_mult: float,
    lower_mult: float,
    vertical_barrier_bars: int,
) -> np.ndarray:
    """
    Numba-optimized core logic for applying the triple-barrier method.

    Args:
        prices (np.ndarray): Array of closing prices.
        atr (np.ndarray): Array of Average True Range values.
        upper_mult (float): Multiplier for ATR to set the upper barrier.
        lower_mult (float): Multiplier for ATR to set the lower barrier.
        vertical_barrier_bars (int): Maximum number of bars to wait for a barrier touch.

    Returns:
        np.ndarray: An array of labels (+1, -1, 0).
    """
    n_points = len(prices)
    labels = np.zeros(n_points)

    for i in range(n_points - vertical_barrier_bars):
        entry_price = prices[i]

        # Define barriers for the current point
        upper_barrier = entry_price + (atr[i] * upper_mult)
        lower_barrier = entry_price - (atr[i] * lower_mult)

        # Look ahead for barrier touches
        for j in range(1, vertical_barrier_bars + 1):
            future_price = prices[i + j]

            # Check if upper barrier is hit
            if future_price >= upper_barrier:
                labels[i] = 1
                break  # Exit inner loop once a barrier is hit

            # Check if lower barrier is hit
            if future_price <= lower_barrier:
                labels[i] = -1
                break  # Exit inner loop

        # If the inner loop completes without a break, the vertical barrier was hit.
        # The label is already 0, so no action is needed.

    return labels


def generate_triple_barrier_labels(
    data: pd.DataFrame,
    upper_mult: float = 2.0,
    lower_mult: float = 1.5,
    vertical_barrier_bars: int = 10,
    logger: structlog.stdlib.BoundLogger = None,
) -> pd.Series:
    """
    Generates trading labels for a given OHLCV dataset using the Triple-Barrier Method.

    Args:
        data (pd.DataFrame): DataFrame with 'close', and 'atr' columns.
        upper_mult (float): Multiplier for ATR to set the profit-take barrier.
        lower_mult (float): Multiplier for ATR to set the stop-loss barrier.
        vertical_barrier_bars (int): Maximum holding period in bars.
        logger: Structured logger.

    Returns:
        pd.Series: A series of labels (+1 for profit-take, -1 for stop-loss, 0 for timeout).
    """
    _logger = logger or structlog.get_logger(__name__)

    if not all(col in data.columns for col in ["close", "atr"]):
        raise ValueError("Input DataFrame must contain 'close' and 'atr' columns.")

    if data["atr"].isnull().any():
        _logger.warning(
            "ATR column contains NaNs. Filling with a forward fill strategy."
        )
        data["atr"] = data["atr"].ffill().bfill()

    _logger.info(
        "Generating Triple-Barrier labels",
        upper_multiplier=upper_mult,
        lower_multiplier=lower_mult,
        vertical_barrier=f"{vertical_barrier_bars} bars",
    )

    prices_np = data["close"].to_numpy()
    atr_np = data["atr"].to_numpy()

    labels_np = _apply_triple_barrier(
        prices_np, atr_np, upper_mult, lower_mult, vertical_barrier_bars
    )

    labels = pd.Series(labels_np, index=data.index, name="triple_barrier_label")

    # The last `vertical_barrier_bars` will be 0 as they cannot be computed.
    # We should explicitly mark them as NaN to indicate they are not valid labels.
    labels.iloc[-vertical_barrier_bars:] = np.nan

    label_dist = {
        str(k): v for k, v in labels.value_counts(normalize=True).to_dict().items()
    }
    _logger.info("Label generation complete.", label_distribution=label_dist)

    return labels
