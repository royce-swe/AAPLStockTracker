import pandas as pd
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plat_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import yfinance as yf

def run_forecast(symbol='AAPL'):
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period='6mo')

        if hist.empty or len(hist) < 30:
            return {'error': 'Not enough data for forecasting jit.'}
        

    except Exception as e:
        return {'error': str(e)}