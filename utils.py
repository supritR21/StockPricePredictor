# utils.py
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def plot_candlestick(df, title="Price"):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Adj_Close'],
        name='Candlestick'
    ))
    fig.update_layout(title=title, xaxis_title='Date', yaxis_title='Price', template='plotly_white')
    return fig

def plot_history_with_forecast(df, forecast_dates, forecast_mean, forecast_std=None, title="Forecast"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Adj_Close'], name='Historical', mode='lines'))
    fig.add_trace(go.Scatter(x=forecast_dates, y=forecast_mean, name='Forecast Mean', mode='lines+markers'))
    if forecast_std is not None:
        upper = forecast_mean + 1.96 * forecast_std
        lower = forecast_mean - 1.96 * forecast_std
        fig.add_trace(go.Scatter(x=forecast_dates, y=upper, mode='lines', name='Upper 95%', line=dict(dash='dash')))
        fig.add_trace(go.Scatter(x=forecast_dates, y=lower, mode='lines', name='Lower 95%', line=dict(dash='dash')))
    fig.update_layout(title=title, xaxis_title='Date', yaxis_title='Price', template='plotly_white')
    return fig
