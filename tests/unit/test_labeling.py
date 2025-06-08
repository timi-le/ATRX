"""
Unit tests for the Triple-Barrier Method labeling module.
"""

import numpy as np
import pandas as pd
import pytest

from core.labeling import generate_triple_barrier_labels


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    close = np.array(
        [100, 102, 105, 103, 106, 104, 108, 110, 107, 109, 112, 110, 108, 105, 103, 100]
    )
    atr = np.full_like(close, 2.0)  # Constant ATR for predictability
    data = pd.DataFrame({"close": close, "atr": atr})
    return data


def test_upper_barrier_hit(sample_data):
    """Test the scenario where the upper barrier is hit first."""
    # At index 0, price is 100, ATR is 2. Upper barrier = 100 + 2*2 = 104.
    # Price hits 105 at index 2.
    labels = generate_triple_barrier_labels(
        sample_data, upper_mult=2.0, lower_mult=2.0, vertical_barrier_bars=5
    )
    assert labels[0] == 1


def test_lower_barrier_hit():
    """Test the scenario where the lower barrier is hit first."""
    # At index 0, price is 110. Lower barrier = 110 - 1.5*2 = 107.
    # Price hits 105 at index 2.
    close = np.array([110, 109, 105, 108, 112, 115])
    atr = np.full_like(close, 2.0)
    data = pd.DataFrame({"close": close, "atr": atr})

    labels = generate_triple_barrier_labels(
        data, upper_mult=3.0, lower_mult=1.5, vertical_barrier_bars=4
    )
    assert labels[0] == -1


def test_vertical_barrier_hit():
    """Test the scenario where the vertical barrier is hit first."""
    # Price moves sideways, never hitting the upper/lower barriers.
    close = np.array([100, 100.1, 99.9, 100.2, 99.8, 100.3])
    atr = np.full_like(close, 1.0)
    data = pd.DataFrame({"close": close, "atr": atr})

    labels = generate_triple_barrier_labels(
        data, upper_mult=2.0, lower_mult=2.0, vertical_barrier_bars=4
    )
    assert labels[0] == 0


def test_last_values_are_nan(sample_data):
    """Test that the last n values are NaN as they cannot be computed."""
    vertical_barrier_bars = 5
    labels = generate_triple_barrier_labels(
        sample_data, vertical_barrier_bars=vertical_barrier_bars
    )
    assert labels.iloc[-vertical_barrier_bars:].isnull().all()


def test_with_missing_atr(sample_data):
    """Test that the function handles missing ATR values gracefully."""
    sample_data.loc[2, "atr"] = np.nan

    # Should run without error due to ffill/bfill logic
    labels = generate_triple_barrier_labels(sample_data)
    assert not labels.isnull().all()  # Check that it produced some valid labels


def test_input_validation():
    """Test that the function raises error for missing columns."""
    with pytest.raises(ValueError, match="must contain 'close' and 'atr'"):
        generate_triple_barrier_labels(pd.DataFrame({"price": [1, 2, 3]}))
