# StockPricePredictor

A Streamlit-based stock forecasting dashboard that combines:

- live market data from Yahoo Finance (`yfinance`)
- an LSTM + GRU deep learning model for short-horizon forecasts
- uncertainty estimation via Monte Carlo dropout
- a simple moving-average crossover backtest for baseline strategy comparison

The app is modular and split across `data.py`, `models.py`, `backtest.py`, `utils.py`, and `app.py`.

## Features

- Interactive Streamlit UI with configurable ticker, history window, interval, model size, and forecast horizon
- OHLC candlestick visualization with Plotly
- Sequence-based deep learning model (LSTM + GRU)
- Forecast confidence bands (95% interval) using repeated stochastic forward passes
- Built-in strategy backtest (7/21 SMA crossover vs. buy-and-hold)
- Basic performance KPIs: validation RMSE, strategy return, buy-and-hold return

## Tech Stack

- Python
- Streamlit
- TensorFlow / Keras
- Pandas / NumPy / Scikit-learn
- Plotly
- yfinance

## Project Structure

```text
StockPricePredictor/
	app.py               # Streamlit entrypoint and UI flow
	data.py              # Data download, feature engineering, sequence creation
	models.py            # LSTM+GRU model, training, uncertainty prediction helpers
	backtest.py          # SMA strategy signals and KPI evaluation
	utils.py             # Plotly chart builders
	requirements.txt     # Python dependencies
	stock_data.csv       # Sample historical dataset (not required by main app flow)
```

## Setup

### 1. Clone and enter the project

```bash
git clone <your-repo-url>
cd StockPricePredictor
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Run the App

```bash
streamlit run app.py
```

After launch, open the local Streamlit URL shown in the terminal (typically `http://localhost:8501`).

## Usage

1. Enter a ticker symbol (for example: `AAPL`, `MSFT`, `TSLA`).
2. Choose history period and interval.
3. Tune sequence length, epochs, model units, and forecast horizon.
4. Click **Retrain model now** to rebuild and retrain with current settings.
5. Review:
	 - current price KPIs
	 - candlestick chart
	 - forecast + uncertainty table
	 - strategy-vs-buy-and-hold backtest chart

## Modeling Notes

- Features currently include:
	- `log_close`
	- `ma7`
	- `ma21`
	- `vol_ma7`
- The model predicts one-step price values and performs iterative multi-step forecasting.
- Future feature rows during iterative forecasting are approximated (naive update), which can accumulate error for longer horizons.
- Uncertainty is estimated by running the model multiple times with dropout enabled (`training=True`).

## Backtesting Notes

- The included strategy is a simple SMA crossover (short=7, long=21).
- Backtest output includes cumulative strategy return, buy-and-hold return, annualized return, max drawdown, and Sharpe ratio.
- This is educational/demo logic and does not model slippage, transaction cost, or realistic execution constraints.

## Limitations

- Forecasts are highly sensitive to feature engineering and hyperparameters.
- No persisted model storage is included by default (model lives in Streamlit session state).
- Auto-refresh behavior is basic and implemented via page reload logic.
- Not intended as financial advice or production-grade trading infrastructure.

## Ideas for Improvement

- Add richer indicators (RSI, MACD, Bollinger Bands, ATR).
- Add model persistence (`model.save`, artifact tracking).
- Support multiple model families and ensembling from the UI.
- Improve iterative feature simulation for more realistic multi-step forecasts.
- Add tests, experiment tracking, and robust evaluation metrics.

## Disclaimer

This project is for educational purposes only. Market predictions are uncertain, and past performance does not guarantee future results.
