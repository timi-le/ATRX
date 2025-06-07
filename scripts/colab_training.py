import os
import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Data Generation ---
def generate_sample_forex_data(n_samples=10000):
    """
    Generate realistic forex sample data for training.
    """
    print("Generating sample forex data...")
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='5T')
    
    np.random.seed(42)
    price_changes = np.random.normal(0, 0.0001, n_samples)
    for i in range(1, n_samples):
        if np.random.random() < 0.1:
            price_changes[i] *= 3
            
    prices = 1.1000 + np.cumsum(price_changes)
    
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': prices + np.abs(np.random.normal(0, 0.0002, n_samples)),
        'low': prices - np.abs(np.random.normal(0, 0.0002, n_samples)),
        'close': prices + np.random.normal(0, 0.00005, n_samples),
        'volume': np.random.randint(100, 1000, n_samples)
    })
    return data

# --- Feature Engineering ---
def create_features_and_labels(data):
    """
    Create features and labels for ML training.
    """
    print("Creating features and labels...")
    data['returns'] = data['close'].pct_change()
    data['sma_10'] = data['close'].rolling(10).mean()
    data['sma_20'] = data['close'].rolling(20).mean()
    data['rsi'] = 50 + np.random.normal(0, 15, len(data))
    data['volatility'] = data['returns'].rolling(20).std()
    data['trend_strength'] = np.abs(data['sma_10'] - data['sma_20']) / data['close']
    data['volatility_regime'] = pd.qcut(data['volatility'].fillna(0), 3, labels=[0, 1, 2])
    data['trend_regime'] = np.where(data['sma_10'] > data['sma_20'], 1, 0)
    
    feature_columns = [
        'returns', 'rsi', 'trend_strength', 'volatility',
        'volatility_regime', 'trend_regime'
    ]
    
    for i in range(1, 6):
        data[f'returns_lag_{i}'] = data['returns'].shift(i)
        feature_columns.append(f'returns_lag_{i}')
        
    data['future_return'] = data['returns'].shift(-1)
    data['label'] = np.where(data['future_return'] > 0.0001, 2,
                   np.where(data['future_return'] < -0.0001, 0, 1))
    
    data = data.dropna()
    X = data[feature_columns].values
    y = data['label'].values
    
    print(f"Features shape: {X.shape}, Labels shape: {y.shape}")
    return X, y, feature_columns

# --- Model Training ---
def train_xgboost(X_train, y_train, X_test, y_test, scaler, feature_names, models_dir):
    """
    Train and save an XGBoost model.
    """
    print("\n--- Training XGBoost Model ---")
    xgb_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    
    y_pred = xgb_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"XGBoost Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))
    
    model_path = os.path.join(models_dir, "xgboost_model_trained.pkl")
    model_data = {
        'model': xgb_model, 'scaler': scaler,
        'feature_names': feature_names, 'accuracy': accuracy
    }
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"XGBoost model saved to: {model_path}")
    return accuracy

def train_tensorflow_model(X_train, y_train, X_test, y_test, build_fn, model_name, models_dir):
    """
    Train and save a TensorFlow model (LSTM or CNN).
    """
    print(f"\n--- Training {model_name} Model ---")
    import tensorflow as tf
    from tensorflow.keras.utils import to_categorical

    time_steps = 10
    
    def create_sequences(X, y, time_steps):
        X_seq, y_seq = [], []
        for i in range(time_steps, len(X)):
            X_seq.append(X[i-time_steps:i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)

    X_train_seq, y_train_seq = create_sequences(X_train, y_train, time_steps)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, time_steps)
    
    y_train_cat = to_categorical(y_train_seq, num_classes=3)
    y_test_cat = to_categorical(y_test_seq, num_classes=3)
    
    model = build_fn(input_shape=(time_steps, X_train.shape[1]))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    
    model.fit(X_train_seq, y_train_cat, validation_data=(X_test_seq, y_test_cat),
              epochs=30, batch_size=32, verbose=1)
              
    loss, accuracy = model.evaluate(X_test_seq, y_test_cat, verbose=0)
    print(f"{model_name} Accuracy: {accuracy:.4f}")
    
    model_path = os.path.join(models_dir, f"{model_name.lower()}_model_trained.h5")
    model.save(model_path)
    print(f"{model_name} model saved to: {model_path}")
    return accuracy

def build_lstm(input_shape):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape), Dropout(0.2),
        LSTM(50), Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(3, activation='softmax')
    ])
    return model

def build_cnn(input_shape):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
    model = Sequential([
        Conv1D(64, 3, activation='relu', input_shape=input_shape), MaxPooling1D(2),
        Conv1D(32, 3, activation='relu'), Dropout(0.2),
        Flatten(),
        Dense(50, activation='relu'),
        Dense(3, activation='softmax')
    ])
    return model

# --- Main Execution ---
if __name__ == "__main__":
    # Define directory to save models, assuming this script is run from repository root in Colab
    MODELS_DIR = '/content/drive/MyDrive/FX_Models'
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # 1. Generate Data
    data = generate_sample_forex_data()
    
    # 2. Create Features
    X, y, feature_names = create_features_and_labels(data)
    
    # 3. Split and Scale Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Train Models
    xgb_acc = train_xgboost(X_train_scaled, y_train, X_test_scaled, y_test, scaler, feature_names, MODELS_DIR)
    lstm_acc = train_tensorflow_model(X_train_scaled, y_train, X_test_scaled, y_test, build_lstm, "LSTM", MODELS_DIR)
    cnn_acc = train_tensorflow_model(X_train_scaled, y_train, X_test_scaled, y_test, build_cnn, "CNN", MODELS_DIR)

    # 5. Summary
    print("\n" + "="*60)
    print("TRAINING SCRIPT COMPLETE!")
    print(f"Models are saved in: {MODELS_DIR}")
    print("\nPerformance Summary:")
    print(f"  - XGBoost Accuracy: {xgb_acc:.4f}")
    print(f"  - LSTM Accuracy: {lstm_acc:.4f}")
    print(f"  - CNN Accuracy: {cnn_acc:.4f}")
    print("="*60) 