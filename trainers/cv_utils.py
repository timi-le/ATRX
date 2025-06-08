"""
Cross-Validation Utilities for FX AI-Quant Trading System.

This module provides time-series aware cross-validation methods including:
- Walk-forward optimization
- Time-series k-fold cross-validation
- Proper data leakage prevention
- Out-of-sample testing
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from collections.abc import Iterator

import numpy as np
import structlog
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import BaseCrossValidator

logger = structlog.get_logger(__name__)


@dataclass
class CVResult:
    """Cross-validation result container."""

    fold: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    train_score: float
    test_score: float
    train_accuracy: float | None = None
    test_accuracy: float | None = None
    model_params: dict[str, Any] | None = None
    training_time: float | None = None


class TimeSeriesKFold(BaseCrossValidator):
    """
    Time Series K-Fold cross-validator.

    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals, in train/test sets.
    In each split, test indices must be higher than before, and thus shuffling
    in cross validator is inappropriate.
    """

    def __init__(
        self,
        n_splits: int = 5,
        max_train_size: int | None = None,
        test_size: int | None = None,
        gap: int = 0,
    ):
        """
        Initialize TimeSeriesKFold.

        Args:
            n_splits: Number of splits
            max_train_size: Maximum size for training set
            test_size: Size of test set
            gap: Gap between train and test sets to prevent data leakage
        """
        self.n_splits = n_splits
        self.max_train_size = max_train_size
        self.test_size = test_size
        self.gap = gap

    def split(self, X, y=None, groups=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Generate indices to split data into training and test set."""
        n_samples = len(X)
        n_splits = self.n_splits

        if self.test_size is None:
            test_size = n_samples // (n_splits + 1)
        else:
            test_size = self.test_size

        indices = np.arange(n_samples)

        for i in range(n_splits):
            # Calculate test indices
            test_start = (i + 1) * test_size + i * self.gap
            test_end = test_start + test_size

            if test_end > n_samples:
                break

            test_indices = indices[test_start:test_end]

            # Calculate train indices
            train_end = test_start - self.gap

            if self.max_train_size is not None:
                train_start = max(0, train_end - self.max_train_size)
            else:
                train_start = 0

            train_indices = indices[train_start:train_end]

            if len(train_indices) == 0:
                continue

            yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Returns the number of splitting iterations in the cross-validator."""
        return self.n_splits


class WalkForwardOptimizer:
    """
    Walk-Forward Optimization for time series models.

    Implements rolling window training and testing with proper
    time-series constraints and out-of-sample validation.
    """

    def __init__(
        self,
        train_window: int,
        test_window: int,
        step_size: int = 1,
        gap: int = 0,
        min_train_size: int | None = None,
        expanding_window: bool = False,
    ):
        """
        Initialize Walk-Forward Optimizer.

        Args:
            train_window: Size of training window
            test_window: Size of test window
            step_size: Step size for rolling window
            gap: Gap between train and test to prevent leakage
            min_train_size: Minimum training size
            expanding_window: If True, use expanding window instead of rolling
        """
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size
        self.gap = gap
        self.min_train_size = min_train_size or train_window
        self.expanding_window = expanding_window
        self.logger = logger.bind(component="WalkForwardOptimizer")

    def split(self, X, y=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Generate train/test splits for walk-forward optimization."""
        n_samples = len(X)

        # Start position for first test window
        current_pos = self.train_window + self.gap

        while current_pos + self.test_window <= n_samples:
            # Test indices
            test_start = current_pos
            test_end = current_pos + self.test_window
            test_indices = np.arange(test_start, test_end)

            # Train indices
            train_end = current_pos - self.gap

            if self.expanding_window:
                train_start = 0
            else:
                train_start = max(0, train_end - self.train_window)

            # Ensure minimum training size
            if train_end - train_start < self.min_train_size:
                current_pos += self.step_size
                continue

            train_indices = np.arange(train_start, train_end)

            self.logger.debug(
                "Walk-forward split",
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_size=len(train_indices),
                test_size=len(test_indices),
            )

            yield train_indices, test_indices

            # Move to next position
            current_pos += self.step_size

    def optimize(
        self,
        model_class,
        X: np.ndarray,
        y: np.ndarray,
        param_grid: dict[str, list[Any]],
        scoring: str = "mse",
        n_jobs: int = 1,
    ) -> tuple[dict[str, Any], list[CVResult]]:
        """
        Perform walk-forward optimization with parameter grid search.

        Args:
            model_class: Model class to optimize
            X: Feature data
            y: Target data
            param_grid: Parameter grid for optimization
            scoring: Scoring metric
            n_jobs: Number of parallel jobs

        Returns:
            Tuple of (best_params, cv_results)
        """
        from itertools import product

        # Generate parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(product(*param_values))

        best_score = float("inf") if scoring in ["mse", "mae"] else float("-inf")
        best_params = None
        all_results = []

        self.logger.info(
            "Starting walk-forward optimization",
            n_param_combinations=len(param_combinations),
            scoring=scoring,
        )

        for param_combo in param_combinations:
            params = dict(zip(param_names, param_combo))

            # Perform cross-validation for this parameter combination
            cv_scores = []
            fold_results = []

            for fold, (train_idx, test_idx) in enumerate(self.split(X, y)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # Train model with current parameters
                model = model_class(**params)

                start_time = datetime.now()
                model.fit(X_train, y_train)
                training_time = (datetime.now() - start_time).total_seconds()

                # Evaluate
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                if scoring == "mse":
                    train_score = mean_squared_error(y_train, y_train_pred)
                    test_score = mean_squared_error(y_test, y_test_pred)
                elif scoring == "mae":
                    train_score = mean_absolute_error(y_train, y_train_pred)
                    test_score = mean_absolute_error(y_test, y_test_pred)
                elif scoring == "accuracy":
                    train_score = accuracy_score(
                        y_train, (y_train_pred > 0.5).astype(int)
                    )
                    test_score = accuracy_score(y_test, (y_test_pred > 0.5).astype(int))

                cv_scores.append(test_score)

                result = CVResult(
                    fold=fold,
                    train_start=train_idx[0],
                    train_end=train_idx[-1],
                    test_start=test_idx[0],
                    test_end=test_idx[-1],
                    train_score=train_score,
                    test_score=test_score,
                    model_params=params,
                    training_time=training_time,
                )
                fold_results.append(result)

            # Calculate mean CV score
            mean_cv_score = np.mean(cv_scores)

            # Update best parameters
            is_better = (scoring in ["mse", "mae"] and mean_cv_score < best_score) or (
                scoring == "accuracy" and mean_cv_score > best_score
            )

            if is_better:
                best_score = mean_cv_score
                best_params = params

            all_results.extend(fold_results)

            self.logger.debug(
                "Parameter combination evaluated",
                params=params,
                mean_cv_score=mean_cv_score,
                is_best=is_better,
            )

        self.logger.info(
            "Walk-forward optimization completed",
            best_params=best_params,
            best_score=best_score,
        )

        return best_params, all_results


class PurgedKFold(BaseCrossValidator):
    """
    Purged K-Fold cross-validator for financial time series.

    Implements purging to prevent data leakage in overlapping samples
    and embargo to prevent look-ahead bias.
    """

    def __init__(
        self, n_splits: int = 5, purge_length: int = 0, embargo_length: int = 0
    ):
        """
        Initialize PurgedKFold.

        Args:
            n_splits: Number of splits
            purge_length: Number of samples to purge before test set
            embargo_length: Number of samples to embargo after test set
        """
        self.n_splits = n_splits
        self.purge_length = purge_length
        self.embargo_length = embargo_length

    def split(self, X, y=None, groups=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Generate purged train/test splits."""
        n_samples = len(X)
        test_size = n_samples // self.n_splits

        for i in range(self.n_splits):
            # Test indices
            test_start = i * test_size
            test_end = (i + 1) * test_size if i < self.n_splits - 1 else n_samples
            test_indices = np.arange(test_start, test_end)

            # Train indices with purging and embargo
            train_indices = []

            # Before test set (with purging)
            if test_start > self.purge_length:
                train_indices.extend(range(0, test_start - self.purge_length))

            # After test set (with embargo)
            if test_end + self.embargo_length < n_samples:
                train_indices.extend(range(test_end + self.embargo_length, n_samples))

            train_indices = np.array(train_indices)

            if len(train_indices) > 0:
                yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Returns the number of splitting iterations in the cross-validator."""
        return self.n_splits


def sharpe_ratio_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Sharpe ratio as a scoring metric.

    Args:
        y_true: True returns
        y_pred: Predicted returns

    Returns:
        Sharpe ratio
    """
    # Calculate strategy returns (assuming we trade in direction of prediction)
    strategy_returns = y_true * np.sign(y_pred)

    if len(strategy_returns) == 0 or np.std(strategy_returns) == 0:
        return 0.0

    return np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)


def information_ratio_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Information Ratio as a scoring metric.

    Args:
        y_true: True returns
        y_pred: Predicted returns

    Returns:
        Information ratio
    """
    # Calculate tracking error (difference between predicted and actual)
    tracking_error = y_pred - y_true

    if len(tracking_error) == 0 or np.std(tracking_error) == 0:
        return 0.0

    return np.mean(tracking_error) / np.std(tracking_error)


def calmar_ratio_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Calmar Ratio as a scoring metric.

    Args:
        y_true: True returns
        y_pred: Predicted returns

    Returns:
        Calmar ratio
    """
    # Calculate strategy returns
    strategy_returns = y_true * np.sign(y_pred)

    if len(strategy_returns) == 0:
        return 0.0

    # Calculate cumulative returns
    cumulative_returns = np.cumprod(1 + strategy_returns)

    # Calculate maximum drawdown
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = np.min(drawdown)

    if max_drawdown == 0:
        return 0.0

    # Annualized return
    total_return = cumulative_returns[-1] - 1
    annualized_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1

    return annualized_return / abs(max_drawdown)


def validate_cv_setup(
    X: np.ndarray,
    y: np.ndarray,
    cv_method: BaseCrossValidator,
    min_samples_per_fold: int = 100,
) -> bool:
    """
    Validate cross-validation setup for time series data.

    Args:
        X: Feature data
        y: Target data
        cv_method: Cross-validation method
        min_samples_per_fold: Minimum samples required per fold

    Returns:
        True if setup is valid
    """
    try:
        splits = list(cv_method.split(X, y))

        if len(splits) == 0:
            logger.error("No valid splits generated")
            return False

        for i, (train_idx, test_idx) in enumerate(splits):
            if len(train_idx) < min_samples_per_fold:
                logger.error(
                    "Insufficient training samples",
                    fold=i,
                    train_samples=len(train_idx),
                    min_required=min_samples_per_fold,
                )
                return False

            if len(test_idx) == 0:
                logger.error("Empty test set", fold=i)
                return False

            # Check for data leakage (test indices should be after train indices)
            if hasattr(cv_method, "gap") or isinstance(
                cv_method, (TimeSeriesKFold, PurgedKFold)
            ):
                if len(train_idx) > 0 and len(test_idx) > 0:
                    if np.max(train_idx) >= np.min(test_idx):
                        logger.warning(
                            "Potential data leakage detected",
                            fold=i,
                            max_train_idx=np.max(train_idx),
                            min_test_idx=np.min(test_idx),
                        )

        logger.info(
            "Cross-validation setup validated",
            n_splits=len(splits),
            cv_method=cv_method.__class__.__name__,
        )
        return True

    except Exception as e:
        logger.error("Cross-validation validation failed", error=str(e))
        return False
