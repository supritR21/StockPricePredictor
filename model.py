# model.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
import os

def preprocess_data_for_lstm(series, lookback=60):
    """
    series: pd.Series (index: date) of the target column
    returns: X, y, scaler
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(series.values.reshape(-1, 1))
    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback:i, 0])
        y.append(scaled[i, 0])
    X = np.array(X)
    y = np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y, scaler

def build_lstm_model(input_shape, units=50, dropout=0.2):
    model = Sequential()
    model.add(LSTM(units, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout))
    model.add(LSTM(units // 1, return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_lstm(X_train, y_train, X_val=None, y_val=None,
               epochs=20, batch_size=32, model_path=None):
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm_model(input_shape)
    callbacks = [EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val) if X_val is not None else None,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    if model_path:
        model.save(model_path)
    return model, history

def load_lstm_model(path):
    if os.path.exists(path):
        return load_model(path)
    else:
        raise FileNotFoundError(f"No model file at {path}")

def predict_future_lstm(model, last_window, days, scaler):
    """last_window: 1D array of length lookback representing most recent values (scaled or raw?)
       Expect last_window to be already scaled to the same scaler used in training."""
    preds = []
    curr = last_window.copy()
    for _ in range(days):
        x = curr.reshape(1, curr.shape[0], 1)
        p = model.predict(x)[0, 0]
        preds.append(p)
        curr = np.roll(curr, -1)
        curr[-1] = p
    # inverse transform
    preds = np.array(preds).reshape(-1, 1)
    inv = scaler.inverse_transform(preds).flatten()
    return inv

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
