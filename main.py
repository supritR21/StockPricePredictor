# main.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
import os

# Try to import yfinance (for live data)
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except Exception:
    YFINANCE_AVAILABLE = False

# LSTM helpers
from model import preprocess_data_for_lstm, train_lstm, load_lstm_model, predict_future_lstm, rmse as rmse_fn

st.set_page_config(layout="wide", page_title="Stock Forecast with Live API / LSTM / Synthesis")

st.title("Stock Forecast — CSV + Live API + LSTM + Synthesis")

# --- Load CSV
DATA_PATH = "stock_data.csv"
if not os.path.exists(DATA_PATH):
    st.error(f"CSV file not found at {DATA_PATH}. Please put stock_data.csv in the app folder.")
    st.stop()

data = pd.read_csv(DATA_PATH, parse_dates=["Date"])
data.set_index("Date", inplace=True)
data = data.sort_index()

# --- UI: Data source and controls
col1, col2 = st.columns([1, 3])
with col1:
    st.sidebar.header("Data & Forecast Options")
    data_source = st.sidebar.radio("Data source", options=["CSV", "Live (yfinance)"])
    if data_source == "Live (yfinance)" and not YFINANCE_AVAILABLE:
        st.sidebar.warning("yfinance not installed or no internet — using CSV only.")
        data_source = "CSV"

    auto_update = st.sidebar.button("Fetch & Append Latest (Live)") if data_source == "Live (yfinance)" else None

    forecast_method = st.sidebar.selectbox("Forecast method", ["RandomForest (fast)", "LSTM (trainable)"])
    mode = st.sidebar.radio("Mode", ["Backtest (predict last N days)", "Forecast (predict future N days)"])
    future_days = st.sidebar.slider("Days to predict", min_value=1, max_value=30, value=7)
    synth = st.sidebar.checkbox("Enable Future Date Synthesis (create synthetic actuals)", value=False)

with col2:
    st.markdown("## Dataset overview")
    st.write(f"Data range: **{data.index.min().date()}** → **{data.index.max().date()}**")
    st.dataframe(data.tail(5))

# --- LIVE FETCH & APPEND (if requested)
def fetch_latest_yfinance(symbol, period="1mo", interval="1d"):
    # fetch daily OHLC for the symbol; user uses column names like 'AMZN', 'DPZ', etc.
    # We assume the CSV has columns AMZN, DPZ, BTC, NFLX (close prices). yfinance returns 'Close' column.
    df = yf.download(symbol, period=period, interval=interval, progress=False, threads=False)
    if df.empty:
        return None
    df = df[['Close']].rename(columns={'Close': symbol})
    df.index = pd.to_datetime(df.index).normalize()
    return df

if data_source == "Live (yfinance)" and 'auto_update' in locals() and auto_update:
    st.info("Fetching latest data via yfinance...")
    # For each symbol present in CSV columns, try fetching latest close and append
    symbols = [c for c in data.columns if c in ["AMZN", "DPZ", "BTC-USD", "BTC", "NFLX"]]
    # map BTC to BTC-USD for yfinance
    sym_map = {"BTC": "BTC-USD"}
    appended = False
    for sym in symbols:
        yf_sym = sym_map.get(sym, sym)
        try:
            new_df = fetch_latest_yfinance(yf_sym, period="7d", interval="1d")
        except Exception as e:
            st.error(f"yfinance fetch failed for {yf_sym}: {e}")
            new_df = None
        if new_df is not None:
            # integrate into data
            for date, row in new_df.iterrows():
                d = date.normalize()
                v = row[yf_sym]
                if d in data.index:
                    # update value
                    data.loc[d, sym] = v
                else:
                    # create new row if needed
                    new_row = pd.Series({sym: v})
                    # ensure all columns exist
                    for col in data.columns:
                        if col not in new_row.index:
                            new_row[col] = np.nan
                    new_row.name = d
                    data = data.append(new_row)
                    appended = True
    if appended:
        data = data.sort_index()
        data.to_csv(DATA_PATH)
        st.success("Latest data fetched and appended to CSV.")
    else:
        st.info("No new rows were added from API.")

# --- Select target
stock_options = [c for c in data.columns if c in ["AMZN", "DPZ", "BTC", "NFLX", "BTC-USD"]]
if len(stock_options) == 0:
    st.error("No recognized stock columns in CSV. Expected columns: AMZN, DPZ, BTC, NFLX")
    st.stop()

target_column = st.selectbox("Select target column for prediction", options=stock_options)

# Build features: simple approach — use other columns as features (fill NaN forward)
features = [c for c in stock_options if c != target_column]
if len(features) == 0:
    st.error("Not enough feature columns to train. Need at least one other column.")
    st.stop()

X = data[features].ffill().bfill()
y = data[target_column].ffill().bfill()

# --- Train/test split for RandomForest baseline
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

# Random Forest training (fast)
rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)
val_pred = rf_model.predict(X_val)
val_mse = mean_squared_error(y_val, val_pred)
st.write(f"Validation MSE (RandomForest): {val_mse:.4f}")

# --- LSTM training option
lstm_model = None
LSTM_LOOKBACK = 60
LSTM_MODEL_PATH = "saved_lstm.h5"

if forecast_method == "LSTM (trainable)":
    st.info("Preparing data for LSTM (this may take a bit)...")
    # For LSTM we only use target column (univariate)
    series = y.dropna()
    if len(series) < LSTM_LOOKBACK + 2:
        st.error("Not enough data for LSTM lookback. Reduce lookback or add more rows.")
    else:
        # Train/test split for LSTM (80/20)
        split_idx = int(len(series) * 0.8)
        train_series = series.iloc[:split_idx]
        val_series = series.iloc[split_idx - LSTM_LOOKBACK:]  # overlap for windows

        X_train_l, y_train_l, scaler = preprocess_data_for_lstm(train_series, lookback=LSTM_LOOKBACK)
        X_val_l, y_val_l, _ = preprocess_data_for_lstm(val_series, lookback=LSTM_LOOKBACK)

        # Train or load existing model
        if st.sidebar.button("Train LSTM (may take time)"):
            with st.spinner("Training LSTM..."):
                lstm_model, _ = train_lstm(X_train_l, y_train_l, X_val_l, y_val_l,
                                           epochs=20, batch_size=32, model_path=LSTM_MODEL_PATH)
                st.success("LSTM trained and saved.")
        else:
            # try to load existing
            if os.path.exists(LSTM_MODEL_PATH):
                try:
                    lstm_model = load_lstm_model(LSTM_MODEL_PATH)
                    st.info("Loaded existing LSTM model from disk.")
                except Exception:
                    st.warning("Failed to load saved LSTM model. You can press 'Train LSTM' to train a new one.")

# --- Prediction logic
last_date = data.index.max()
st.write(f"Last available date in data: {last_date.date()}")

if mode == "Backtest (predict last N days)":
    # backtesting: predict last N actual dates
    test_dates = data.index[-future_days:]
    X_test = X.loc[test_dates]
    rf_preds = rf_model.predict(X_test)
    pred_df = pd.DataFrame({"Date": test_dates, "Predicted_RF": rf_preds})
    pred_df = pred_df.set_index("Date")
    pred_df["Actual"] = y.loc[test_dates]

    # If LSTM selected and available, predict using LSTM for the same dates (sliding window)
    if forecast_method == "LSTM (trainable)" and lstm_model is not None:
        series = y.dropna()
        # Build windows that target the test_dates
        # We will predict by using windows that end at each test_date
        lstm_preds = []
        for dt in test_dates:
            # find position of dt in series
            idx = series.index.get_loc(dt)
            if idx - LSTM_LOOKBACK + 1 < 0:
                lstm_preds.append(np.nan)
                continue
            window = series.iloc[idx - LSTM_LOOKBACK + 1: idx + 1].values
            # scale using scaler from earlier training (recreate scaler)
            # Simpler: scale with min/max of train_series used above. We'll re-run preprocess to obtain scaler.
            _, _, scaler_local = preprocess_data_for_lstm(series[:idx+1], lookback=LSTM_LOOKBACK)
            scaled_window = scaler_local.transform(window.reshape(-1, 1)).flatten()[-LSTM_LOOKBACK:]
            pred_scaled = lstm_model.predict(scaled_window.reshape(1, LSTM_LOOKBACK, 1))[0, 0]
            pred_unscaled = scaler_local.inverse_transform(np.array(pred_scaled).reshape(-1,1)).flatten()[0]
            lstm_preds.append(pred_unscaled)
        pred_df["Predicted_LSTM"] = lstm_preds

    # RMSEs
    rf_rmse = np.sqrt(mean_squared_error(pred_df["Actual"].dropna(), pred_df["Predicted_RF"].loc[pred_df["Actual"].dropna().index]))
    st.success(f"Backtest RMSE (RandomForest): {rf_rmse:.4f}")
    if "Predicted_LSTM" in pred_df.columns and pred_df["Predicted_LSTM"].notna().any():
        lstm_rmse = np.sqrt(mean_squared_error(pred_df["Actual"].dropna(), pred_df["Predicted_LSTM"].dropna()))
        st.success(f"Backtest RMSE (LSTM): {lstm_rmse:.4f}")

    plot_pred = pred_df

else:
    # Forecast mode: predict future N days after last_date
    future_dates = [last_date + timedelta(days=i) for i in range(1, future_days + 1)]
    # For RF: create features for each future day by shifting last known row forward (naive)
    last_row = X.iloc[-1].values.reshape(1, -1)
    rf_preds = []
    curr_row = last_row.copy()
    for i in range(future_days):
        p = rf_model.predict(curr_row)[0]
        rf_preds.append(p)
        # naive: shift curr_row values to keep same features (or implement feature forecast)
        # Here we keep features constant, which is simplistic; you can improve with more complex feature forecasting
    pred_df = pd.DataFrame({"Date": future_dates, "Predicted_RF": rf_preds}).set_index("Date")

    # LSTM forecasting
    if forecast_method == "LSTM (trainable)" and lstm_model is not None:
        # we need the last lookback values from series
        series = y.dropna()
        last_window = series.values[-LSTM_LOOKBACK:]
        # scale with scaler built from whole series for safety
        _, _, scaler_local = preprocess_data_for_lstm(series, lookback=LSTM_LOOKBACK)
        scaled_last = scaler_local.transform(last_window.reshape(-1,1)).flatten()
        lstm_preds_unscaled = predict_future_lstm(lstm_model, scaled_last, days=future_days, scaler=scaler_local)
        pred_df["Predicted_LSTM"] = lstm_preds_unscaled

    # find actual future values in dataset if any
    actual_future = data[target_column].loc[(data.index > last_date) & (data.index <= future_dates[-1])]
    if synth and len(actual_future) == 0:
        # produce synthetic actuals using linear extrap + noise
        # derive simple linear trend from last 30 days
        look = min(30, len(y))
        recent = y.dropna().iloc[-look:]
        x = np.arange(len(recent))
        coef = np.polyfit(x, recent.values, 1)  # linear fit
        slope, intercept = coef[0], coef[1]
        synth_vals = []
        for i in range(1, future_days+1):
            base = intercept + slope*(len(recent) + i)
            noise = np.random.normal(scale=np.std(recent.values) * 0.02)  # small noise ~2%
            synth_vals.append(base + noise)
        synth_df = pd.Series(synth_vals, index=pred_df.index)
        actual_future = synth_df
        st.info("Using synthetic actual values for comparison (synthesis enabled).")

    plot_pred = pred_df.copy()
    # attach actuals to plot_pred for overlapping dates
    if len(actual_future) > 0:
        # align by date index if overlap
        common_idx = plot_pred.index.intersection(actual_future.index)
        if len(common_idx) > 0:
            overlap_rmse = np.sqrt(mean_squared_error(actual_future.loc[common_idx], plot_pred.loc[common_idx, "Predicted_RF"]))
            st.success(f"Forecast RMSE (RandomForest) for overlapping actuals: {overlap_rmse:.4f}")

# --- Plotting
fig = go.Figure()

# Historical trace (last 300 rows or full if less)
hist = data[target_column].iloc[-600:] if len(data) > 600 else data[target_column]
fig.add_trace(go.Scatter(x=hist.index, y=hist.values, mode='lines', name='Historical', line=dict(color='blue')))

# Predicted traces
if "Predicted_RF" in plot_pred.columns:
    fig.add_trace(go.Scatter(x=plot_pred.index, y=plot_pred["Predicted_RF"], mode='lines+markers', name='Predicted (RF)', line=dict(color='red', dash='dash')))

if "Predicted_LSTM" in plot_pred.columns:
    fig.add_trace(go.Scatter(x=plot_pred.index, y=plot_pred["Predicted_LSTM"], mode='lines+markers', name='Predicted (LSTM)', line=dict(color='orange', dash='dash')))

# Actual trace (green) — if present
actual_plot_series = None
if mode == "Backtest (predict last N days)":
    actual_plot_series = plot_pred["Actual"]
else:
    # forecast mode: check if data contains actuals in the predicted date range or if we synthesized
    if 'actual_future' in locals() and len(actual_future) > 0:
        actual_plot_series = actual_future

if actual_plot_series is not None and len(actual_plot_series.dropna()) > 0:
    fig.add_trace(go.Scatter(x=actual_plot_series.index, y=actual_plot_series.values, mode='lines+markers', name='Actual', line=dict(color='green', width=3)))

fig.update_layout(title=f"{target_column} Prediction — Method: {forecast_method} — Mode: {mode}",
                  xaxis_title='Date', yaxis_title='Price', template='plotly_white')

st.plotly_chart(fig, use_container_width=True)

# --- Optionally show table
with st.expander("Show prediction table"):
    if 'plot_pred' in locals():
        st.dataframe(plot_pred.head(50))
