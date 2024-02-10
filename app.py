import streamlit as st
import yfinance as yf
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots


def fetch_stock_data(ticker, start_date, end_date):
    """
    Fetch historical stock data for the specified ticker and time period.

    Args:
        ticker (str): The stock ticker symbol.
        start_date (str or datetime): The start date for the historical data (YYYY-MM-DD).
        end_date (str or datetime): The end date for the historical data (YYYY-MM-DD).

    Returns:
        pandas.DataFrame: A DataFrame containing the historical stock data.
    """
    # Fetch historical data using yfinance
    data = yf.download(ticker, start=start_date, end=end_date)
    return data


def calculate_technical_indicators(data):
    """
    Calculate technical indicators for the stock data.

    Args:
        data (pandas.DataFrame): Historical stock data.

    Returns:
        pandas.DataFrame: Dataframe with calculated technical indicators.
    """
    # Calculate MACD
    data['MACD'], data['Signal Line'], _ = data['Close']. \
        ewm(span=12, min_periods=0, adjust=False).mean(), \
        data['Close'].ewm(span=26, min_periods=0, adjust=False).mean(), \
        data['Close'].rolling(window=20).mean()

    # Calculate RSI
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    data['RSI'] = rsi

    # Calculate Bollinger Bands
    data['SMA'] = data['Close'].rolling(window=20).mean()
    data['Upper Band'] = data['SMA'] + 2 * data['Close'].rolling(window=20).std()
    data['Lower Band'] = data['SMA'] - 2 * data['Close'].rolling(window=20).std()

    return data


# Title of the web app
st.title('Stock Price Analysis and Prediction')

# Sidebar for user input
st.sidebar.header('User Input')
ticker = st.sidebar.text_input("Enter Stock Ticker", 'AAPL')  # Default to AAPL

# Date range selection
start_date = st.sidebar.date_input("Start Date", datetime(2010, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.now())

# Fetch stock data and calculate technical indicators
stock_data = fetch_stock_data(ticker, start_date, end_date)
technical_data = calculate_technical_indicators(stock_data)

# Plotting
fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                    subplot_titles=("Close Price", "MACD and Signal Line", "RSI and Bollinger Bands"))

# Close Price
fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Close Price'), row=1, col=1)

# MACD and Signal Line
fig.add_trace(go.Scatter(x=technical_data.index, y=technical_data['MACD'], mode='lines', name='MACD'), row=2, col=1)
fig.add_trace(go.Scatter(x=technical_data.index, y=technical_data['Signal Line'], mode='lines', name='Signal Line'),
              row=2, col=1)

# RSI
fig.add_trace(go.Scatter(x=technical_data.index, y=technical_data['RSI'], mode='lines', name='RSI'), row=3, col=1)

# Bollinger Bands
fig.add_trace(go.Scatter(x=technical_data.index, y=technical_data['Upper Band'], mode='lines', name='Upper Band'),
              row=3, col=1)
fig.add_trace(go.Scatter(x=technical_data.index, y=technical_data['Lower Band'], mode='lines', name='Lower Band'),
              row=3, col=1)

# Add titles and labels
fig.update_layout(title=f'{ticker} Stock Analysis', xaxis_title='Date', yaxis_title='Price/Value')

# Render the plot
st.plotly_chart(fig, use_container_width=True)
