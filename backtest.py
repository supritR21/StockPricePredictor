# backtest.py
import pandas as pd
import numpy as np

def simple_moving_average_signals(df, short=7, long=21):
    df = df.copy()
    df['ma_short'] = df['Adj_Close'].rolling(short).mean()
    df['ma_long'] = df['Adj_Close'].rolling(long).mean()
    df['signal'] = 0
    df.loc[df['ma_short'] > df['ma_long'], 'signal'] = 1
    df['position'] = df['signal'].shift(1).fillna(0)
    df['returns'] = df['Adj_Close'].pct_change().fillna(0)
    df['strategy_returns'] = df['position'] * df['returns']
    df['cum_strategy'] = (1 + df['strategy_returns']).cumprod()
    df['cum_buy_hold'] = (1 + df['returns']).cumprod()
    return df

def evaluate_strategy(df):
    """Return some KPIs for strategy vs buy and hold."""
    total_return_strategy = df['cum_strategy'].iloc[-1] - 1
    total_return_buyhold = df['cum_buy_hold'].iloc[-1] - 1
    annualized_strategy = (1 + total_return_strategy) ** (252/len(df)) - 1
    annualized_bh = (1 + total_return_buyhold) ** (252/len(df)) - 1
    drawdown = (df['cum_strategy'].cummax() - df['cum_strategy']).max()
    sharpe = (df['strategy_returns'].mean() / (df['strategy_returns'].std() + 1e-9)) * np.sqrt(252)
    return {
        'total_return_strategy': total_return_strategy,
        'total_return_buyhold': total_return_buyhold,
        'annualized_strategy': annualized_strategy,
        'annualized_buyhold': annualized_bh,
        'max_drawdown': drawdown,
        'sharpe': sharpe
    }
