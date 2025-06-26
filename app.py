import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from prophet import Prophet
from datetime import datetime
from bs4 import BeautifulSoup
import requests
import feedparser
import time

time.sleep(0.1)
st.set_page_config(page_title="AI-Powered Market Analyzer", layout="wide")

@st.cache_data
def get_data(ticker, period='1y'):
    try:
        ticker = ticker.upper().strip()
        if '-' in ticker and not ticker.endswith('-USD'):
            ticker += '-USD'
        data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if data.empty or 'Close' not in data.columns:
            raise ValueError("No valid data found for ticker.")
        return data
    except Exception as e:
        raise ValueError(f"Failed to fetch data for ticker: {str(e)}")

def get_company_name(ticker):
    try:
        info = yf.Ticker(ticker).info
        return info.get('longName') or info.get('shortName') or ticker
    except:
        return ticker

def calculate_indicators(df):
    if len(df) < 50:
        return df
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df.dropna(inplace=True)
    return df

def generate_signals(df):
    if len(df) == 0 or 'SMA_50' not in df or 'SMA_200' not in df:
        df['Signal'] = "Hold"
        df['Trend'] = 0
        df['Buy'] = False
        df['Sell'] = False
        return df
    df['Trend'] = np.where(df['SMA_50'] > df['SMA_200'], 1, -1)
    df['Buy'] = (df['RSI'] < 30) & (df['MACD'] > df['Signal_Line']) & (df['Trend'] == 1)
    df['Sell'] = (df['RSI'] > 70) & (df['MACD'] < df['Signal_Line']) & (df['Trend'] == -1)
    df['Signal'] = np.where(df['Buy'], 'Buy', np.where(df['Sell'], 'Sell', 'Hold'))
    return df

def get_news(ticker):
    try:
        url = f"https://news.google.com/search?q={ticker}+stock"
        html = requests.get(url, timeout=6).text
        soup = BeautifulSoup(html, 'html.parser')
        return [item.text for item in soup.select('article h3 a')][:5]
    except:
        return ["No news found."]

def get_fda_approvals():
    try:
        rss_url = "https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/press-releases/rss.xml"
        feed = feedparser.parse(rss_url)
        return [entry.title for entry in feed.entries[:5]]
    except:
        return ["No FDA data available"]

def forecast_prices(df, ticker):
    if len(df) < 30:
        return None
    prophet_df = df[['Close']].copy().reset_index()
    prophet_df.columns = ['ds', 'y']
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
    prophet_df['news_sentiment'] = 0

    model = Prophet(daily_seasonality=True)
    model.add_regressor('news_sentiment')
    try:
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=7)
        future['news_sentiment'] = 0
        forecast = model.predict(future)
        forecast_range = forecast.tail(7)
        x = np.arange(len(forecast_range))
        linear_coeffs = np.polyfit(x, forecast_range['yhat'], 1)
        linear_trend = np.polyval(linear_coeffs, x)
        forecast.loc[forecast.index[-7:], 'trendline'] = linear_trend
        return forecast[['ds', 'yhat', 'trendline']].tail(7)
    except:
        return None

def suggest_trading_
