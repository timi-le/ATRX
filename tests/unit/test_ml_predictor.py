"""
Unit tests for the ML Predictor module.
"""


import numpy as np
import pandas as pd
import pytest

from core.ml_predictor import CNNPredictor, LSTMPredictor, Prediction, XGBoostPredictor


@pytest.fixture
def sample_training_data():
    """Create sample features and labels for training."""
    X = pd.DataFrame(
        {
            "feature1": np.random.rand(100),
            "feature2": np.random.rand(100),
            "feature3": np.random.rand(100),
        }
    )
    y = pd.Series(np.random.choice([-1, 0, 1], 100))
    return X, y


@pytest.fixture
def sample_prediction_data():
    """Create a single sample for prediction."""
    return pd.DataFrame([{"feature1": 0.5, "feature2": 0.3, "feature3": 0.8}])


class TestXGBoostPredictor:
    """Tests for the XGBoostPredictor."""

    def test_initialization(self):
        """Test that the model initializes correctly."""
        predictor = XGBoostPredictor()
        assert predictor.model is not None
        assert predictor.model.get_params()["objective"] == "multi:softprob"
        assert predictor.model.get_params()["num_class"] == 3

    def test_training(self, sample_training_data):
        """Test the model training process."""
        X, y = sample_training_data
        predictor = XGBoostPredictor()

        # Should train without errors
        predictor.train(X, y)

        # Check if the model has been fitted
        assert predictor.model._Booster is not None

    def test_prediction(self, sample_training_data, sample_prediction_data):
        """Test the prediction process."""
        X_train, y_train = sample_training_data
        X_pred = sample_prediction_data

        predictor = XGBoostPredictor()
        predictor.train(X_train, y_train)

        prediction = predictor.predict(X_pred)

        assert isinstance(prediction, Prediction)
        assert prediction.signal in [-1, 0, 1]
        assert 0.0 <= prediction.probability <= 1.0

    def test_save_and_load(self, tmp_path, sample_training_data):
        """Test saving and loading the model."""
        X, y = sample_training_data
        predictor = XGBoostPredictor()
        predictor.train(X, y)

        model_path = tmp_path / "test_model.joblib"
        predictor.save(model_path)

        assert model_path.exists()

        new_predictor = XGBoostPredictor()
        new_predictor.load(model_path)

        assert new_predictor.model is not None

        # Check if the loaded model can predict
        prediction = new_predictor.predict(X.head(1))
        assert isinstance(prediction, Prediction)
        assert prediction.signal in [-1, 0, 1]


class TestLSTMPredictor:
    """Tests for the LSTMPredictor."""

    def test_initialization(self):
        """Test that the LSTM model initializes correctly."""
        predictor = LSTMPredictor(n_features=3)
        assert predictor.model is not None
        assert len(predictor.model.layers) == 5  # LSTM, Dropout, LSTM, Dropout, Dense

    def test_training(self, sample_training_data):
        """Test the LSTM model training process."""
        X, y = sample_training_data
        predictor = LSTMPredictor(n_features=3, timesteps=5)

        # Should train without errors
        predictor.train(X, y)

        # Check if weights have been updated (simple check)
        initial_weights = LSTMPredictor(n_features=3, timesteps=5).model.get_weights()[
            0
        ]
        trained_weights = predictor.model.get_weights()[0]
        assert not np.array_equal(initial_weights, trained_weights)

    def test_prediction(self, sample_training_data):
        """Test the LSTM prediction process."""
        X_train, y_train = sample_training_data

        predictor = LSTMPredictor(n_features=3, timesteps=5)
        predictor.train(X_train, y_train)

        # Prediction requires at least `timesteps` data points
        X_pred = X_train.head(10)
        prediction = predictor.predict(X_pred)

        assert isinstance(prediction, Prediction)
        assert prediction.signal in [-1, 0, 1]
        assert 0.0 <= prediction.probability <= 1.0

    def test_save_and_load(self, tmp_path, sample_training_data):
        """Test saving and loading the LSTM model."""
        X, y = sample_training_data
        predictor = LSTMPredictor(n_features=3, timesteps=5)
        predictor.train(X, y)

        model_path = tmp_path / "lstm_model.h5"
        predictor.save(model_path)

        assert model_path.exists()

        new_predictor = LSTMPredictor(n_features=3, timesteps=5)
        new_predictor.load(model_path)

        assert new_predictor.model is not None

        # Check if the loaded model can predict
        prediction = new_predictor.predict(X.head(10))
        assert isinstance(prediction, Prediction)
        assert prediction.signal in [-1, 0, 1]


class TestCNNPredictor:
    """Tests for the CNNPredictor."""

    def test_initialization(self):
        """Test that the CNN model initializes correctly."""
        predictor = CNNPredictor(image_size=5, n_features=25)
        assert predictor.model is not None
        assert len(predictor.model.layers) == 5  # Conv, Pool, Flatten, Dense, Dense

    def test_training(self):
        """Test the CNN model training process."""
        X = pd.DataFrame(np.random.rand(50, 25))
        y = pd.Series(np.random.choice([-1, 0, 1], 50))

        predictor = CNNPredictor(image_size=5, n_features=25)
        predictor.train(X, y)

        initial_weights = CNNPredictor(image_size=5, n_features=25).model.get_weights()[
            0
        ]
        trained_weights = predictor.model.get_weights()[0]
        assert not np.array_equal(initial_weights, trained_weights)

    def test_prediction(self):
        """Test the CNN prediction process."""
        X_train = pd.DataFrame(np.random.rand(50, 25))
        y_train = pd.Series(np.random.choice([-1, 0, 1], 50))

        predictor = CNNPredictor(image_size=5, n_features=25)
        predictor.train(X_train, y_train)

        X_pred = pd.DataFrame(np.random.rand(1, 25))
        prediction = predictor.predict(X_pred)

        assert isinstance(prediction, Prediction)
        assert prediction.signal in [-1, 0, 1]
        assert 0.0 <= prediction.probability <= 1.0

    def test_save_and_load(self, tmp_path):
        """Test saving and loading the CNN model."""
        X = pd.DataFrame(np.random.rand(50, 25))
        y = pd.Series(np.random.choice([-1, 0, 1], 50))
        predictor = CNNPredictor(image_size=5, n_features=25)
        predictor.train(X, y)

        model_path = tmp_path / "cnn_model.h5"
        predictor.save(model_path)

        assert model_path.exists()

        new_predictor = CNNPredictor(image_size=5, n_features=25)
        new_predictor.load(model_path)

        assert new_predictor.model is not None

        prediction = new_predictor.predict(pd.DataFrame(np.random.rand(1, 25)))
        assert isinstance(prediction, Prediction)
        assert prediction.signal in [-1, 0, 1]
