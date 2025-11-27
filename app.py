# app.py
import streamlit as st
from datetime import timedelta
import numpy as np
import pandas as pd
import time

from data import fetch_history, make_features, create_sequences, last_sequence_from_df
from models import build_lstm_gru_model, train_model, predict_with_uncertainty
from utils import plot_candlestick, plot_history_with_forecast
from backtest import simple_moving_average_signals, evaluate_strategy

st.set_page_config(layout="wide", page_title="Stock Predict Pro", initial_sidebar_state="expanded")

# --- Sidebar: inputs ---
st.sidebar.header("Configuration")
symbol = st.sidebar.text_input("Ticker (yfinance)", value="AAPL")
period = st.sidebar.selectbox("History period", options=['6mo','1y','2y','5y'], index=2)
interval = st.sidebar.selectbox("Interval", options=['1d','1wk','1mo'], index=0)

seq_len = st.sidebar.slider("Sequence length (days)", 20, 120, 60)
future_days = st.sidebar.slider("Forecast horizon (days)", 1, 60, 7)
train_epochs = st.sidebar.slider("Train epochs", 5, 100, 30)
units = st.sidebar.slider("Model units", 32, 256, 64)
retrain = st.sidebar.button("Retrain model now")

# Auto refresh controls (streamlit built-in)
auto_refresh = st.sidebar.checkbox("Auto-refresh data every 60s", value=False)
if auto_refresh:
    st.experimental_rerun() if st.experimental_get_query_params().get("autorefresh") else st.experimental_set_query_params(autorefresh="1")
# Note: Streamlit has limited background; autorefresh implemented via page reload (simple approach)

# --- Main page ---
st.title("📈 Stock Predict Pro — Full Upgrade (LSTM + GRU)")
col1, col2 = st.columns([2,1])

with col2:
    st.subheader("Quick controls")
    st.write(f"Ticker: **{symbol.upper()}**")
    st.write(f"Sequence length: **{seq_len}**")
    st.write(f"Forecast: **{future_days} days**")
    st.write("Change settings on the left and press *Retrain model now* if needed.")

# Fetch data
@st.cache_data(ttl=60)
def load(symbol, period, interval):
    df = fetch_history(symbol, period=period, interval=interval)
    return df

try:
    df = load(symbol, period, interval)
except Exception as e:
    st.error(f"Failed to load data for {symbol}: {e}")
    st.stop()

# Prepare data
feat_df = make_features(df)
feature_cols = ['log_close','ma7','ma21','vol_ma7']
feat_df = feat_df.dropna()
seq_data = feat_df[feature_cols].values
X_all, y_all = create_sequences(seq_data, seq_len)
# normalize target scale to last price (we will predict log returns or log price depending on training target).
# For simplicity use Adj_Close directly scaled per dataset:
y_price = feat_df['Adj_Close'].values[seq_len:]

# Split train/val
split = int(len(X_all) * 0.8)
X_train, X_val = X_all[:split], X_all[split:]
y_train, y_val = y_price[:split], y_price[split:]

# Build or load model (simple in-memory model)
if 'model' not in st.session_state or retrain:
    with st.spinner("Building and training model..."):
        model = build_lstm_gru_model(seq_len, len(feature_cols), units=units, dropout=0.2)
        model, history = train_model(model, X_train, y_train, X_val, y_val, epochs=train_epochs, batch_size=32)
        st.session_state['model'] = model
        st.success("Model trained.")
else:
    model = st.session_state['model']

# KPI cards
with col1:
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Last Close", f"${df['Adj_Close'].iloc[-1]:.2f}", delta=f"{(df['Adj_Close'].iloc[-1]/df['Adj_Close'].iloc[-2]-1)*100:.2f}%")
    # Simple validation RMSE
    val_preds = model.predict(X_val).reshape(-1)
    rmse = np.sqrt(((val_preds - y_val)**2).mean())
    kpi2.metric("Validation RMSE", f"{rmse:.4f}")
    kpi3.metric("Data points", f"{len(df)}")

# Candlestick
st.subheader(f"{symbol.upper()} price")
st.plotly_chart(plot_candlestick(df.tail(240), title=f"{symbol.upper()} — Recent Candles"), use_container_width=True)

# Forecast
st.subheader("Forecast & Uncertainty")
# create last sequence
last_seq = last_sequence_from_df(feat_df, feature_cols, seq_len)
# generate iterative forecasts: feed predicted value back into features (simple approach)
forecast_dates = [df.index[-1] + timedelta(days=i+1) for i in range(future_days)]
X_input = last_seq.copy()
preds_mean = []
preds_std = []

# We'll iteratively predict; for each step we append predicted log/price back into feature vector with naive updates.
# Simpler approach: predict using rolling window of features (use last sequence repeatedly if features cannot be updated reliably).
X_step = X_input.copy()
for i in range(future_days):
    X_step_reshaped = X_step.reshape(1, seq_len, len(feature_cols))
    mean_pred, std_pred = predict_with_uncertainty(model, X_step_reshaped, n_iter=50)
    preds_mean.append(float(mean_pred[0]))
    preds_std.append(float(std_pred[0]))
    # shift features: we create naive next-step feature row by copying last row but updating log_close to predicted log price
    # A more sophisticated approach simulates volume and MA; here we approximate by shifting previous features.
    next_row = X_step[-1].copy()
    # replace log_close with log(predicted) if positive; to match training features -> we use price space
    next_row[0] = np.log(max(preds_mean[-1], 1e-6))
    # simple moving averages: use previous ma7 and ma21 as-is (placeholder)
    X_step = np.vstack([X_step[1:], next_row])

preds_mean = np.array(preds_mean)
preds_std = np.array(preds_std)

# Show forecast plot (historical plus forecast)
hist_for_plot = df[['Adj_Close']].loc[feat_df.index[0]:]
fig = plot_history_with_forecast(hist_for_plot, forecast_dates, preds_mean, preds_std, title=f"{symbol.upper()} Forecast")
st.plotly_chart(fig, use_container_width=True)

# Forecast table
fc_df = pd.DataFrame({
    "date": forecast_dates,
    "pred_mean": preds_mean,
    "pred_std": preds_std,
    "lower_95": preds_mean - 1.96*preds_std,
    "upper_95": preds_mean + 1.96*preds_std
})
fc_df['date'] = pd.to_datetime(fc_df['date'])
st.dataframe(fc_df.set_index('date').style.format({"pred_mean":"{:.2f}", "pred_std":"{:.2f}", "lower_95":"{:.2f}", "upper_95":"{:.2f}"}), height=240)

# Backtester
st.subheader("Backtesting — MA Crossover")
bt_df = simple_moving_average_signals(df.copy(), short=7, long=21)
st.line_chart(pd.DataFrame({
    'strategy': bt_df['cum_strategy'],
    'buy_hold': bt_df['cum_buy_hold']
}).dropna())

kp = evaluate_strategy(bt_df.dropna())
st.metric("Strategy total return", f"{kp['total_return_strategy']*100:.2f}%")
st.metric("Buy & Hold total return", f"{kp['total_return_buyhold']*100:.2f}%")

st.markdown("---")
st.write("Built modularly: `data.py`, `models.py`, `backtest.py`, `utils.py`, `app.py`.")
st.write("Suggestions: extend features in `data.make_features`, add more models to `models.py`, save/serialize with joblib or model.save(), and add scheduler or webhook for production auto-updates.")
