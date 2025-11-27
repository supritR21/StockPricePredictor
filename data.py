# data.py
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import timedelta

def fetch_history(symbol: str, period='2y', interval='1d'):
    """Fetch historical OHLCV data using yfinance."""
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {symbol}")
    df = df[['Open','High','Low','Close','Adj Close','Volume']].rename(columns={'Adj Close':'Adj_Close'})
    df.index = pd.to_datetime(df.index)
    return df

def make_features(df: pd.DataFrame):
    """Basic features: returns DataFrame with normalized price and volume features.
       Add more technical indicators as needed."""
    df = df.copy()
    df['ret'] = df['Adj_Close'].pct_change().fillna(0)
    df['log_close'] = np.log(df['Adj_Close'])
    df['ma7'] = df['Adj_Close'].rolling(7).mean()
    df['ma21'] = df['Adj_Close'].rolling(21).mean()
    df['vol_ma7'] = df['Volume'].rolling(7).mean()
    df = df.dropna()
    return df

def create_sequences(series: np.ndarray, seq_len: int):
    X, y = [], []
    for i in range(len(series)-seq_len):
        X.append(series[i:i+seq_len])
        y.append(series[i+seq_len])
    X = np.array(X)
    y = np.array(y)
    return X, y

def last_sequence_from_df(df: pd.DataFrame, feature_cols, seq_len: int):
    arr = df[feature_cols].values
    return arr[-seq_len:]
