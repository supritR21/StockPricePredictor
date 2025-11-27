# models.py
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, GRU, Dense, Dropout, BatchNormalization
import numpy as np

def build_lstm_gru_model(seq_len, n_features, units=64, dropout=0.2):
    inp = Input(shape=(seq_len, n_features))
    x = LSTM(units, return_sequences=True)(inp)
    x = Dropout(dropout)(x)
    x = GRU(units//2, return_sequences=False)(x)
    x = Dropout(dropout)(x)
    x = Dense(units//2, activation='relu')(x)
    out = Dense(1, activation='linear')(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='mse')
    return model

def train_model(model, X_train, y_train, X_val=None, y_val=None, epochs=25, batch_size=32):
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    ]
    if X_val is not None:
        history = model.fit(X_train, y_train, validation_data=(X_val,y_val), epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=1)
    else:
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=1)
    return model, history

def predict_with_uncertainty(model, X, n_iter=50, dropout_rate=0.2):
    """
    Monte Carlo dropout: run model in training mode multiple times to get predictive distribution.
    We assume the model has Dropout layers.
    Returns mean and std of predictions.
    """
    preds = []
    # Ensure inputs are numpy arrays
    X = np.array(X)
    for i in range(n_iter):
        # Keras functional API: call model with training=True to enable dropout
        preds.append(model(X, training=True).numpy().reshape(-1))
    preds = np.array(preds)  # shape (n_iter, n_samples)
    mean = preds.mean(axis=0)
    std = preds.std(axis=0)
    return mean, std

def ensemble_predict(models, X):
    """Simple ensemble across multiple models (optional)."""
    pred_list = [m.predict(X).reshape(-1) for m in models]
    preds = np.vstack(pred_list)
    return preds.mean(axis=0), preds.std(axis=0)
