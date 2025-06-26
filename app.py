import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup
import feedparser
from prophet import Prophet
from datetime import datetime, timedelta

st.set_page_config(page_title="Global Market Watcher", layout="wide")

# ---- UTILITIES ----
def resolve_ticker(ticker):
    try:
        info = yf.Ticker(ticker).info
        return info.get("longName") or info.get("shortName") or ticker
    except:
        return ticker

@st.cache_data(show_spinner=False)
def get_data(ticker, period='1mo', interval='1d'):
    ticker = ticker.upper().strip()
    if '-' in ticker and not ticker.endswith('-USD'):
        ticker += '-USD'
    if ticker.endswith('=X'):
        interval = '1h'
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError("No data found for ticker.")
    return df

def calc_indicators(df):
    df['SMA_14'] = df['Close'].rolling(14).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))
    df['MACD'] = df['Close'].ewm(12).mean() - df['Close'].ewm(26).mean()
    df['Signal'] = df['MACD'].ewm(9).mean()
    df.dropna(inplace=True)
    return df

def buy_hold_sell(df):
    last = df.iloc[-1]
    if last['RSI'] < 30 and last['MACD'] > last['Signal']:
        return "Buy"
    elif last['RSI'] > 70 and last['MACD'] < last['Signal']:
        return "Sell"
    else:
        return "Hold"

def prophet_forecast(df):
    prices = df['Close'][-60:]  # Use last 60 days for model
    hist = pd.DataFrame({'ds': prices.index, 'y': prices.values})
    model = Prophet(daily_seasonality=True)
    model.fit(hist)
    future = model.make_future_dataframe(periods=7)
    forecast = model.predict(future)
    f_dates = forecast['ds'][-7:]
    f_vals = forecast['yhat'][-7:]
    return f_dates, f_vals

def trading_strategy(suggestion):
    if suggestion == "Buy":
        return "- Market or Limit order to enter\n- Add Stop Loss below support\n- Consider Call Options if bullish"
    elif suggestion == "Sell":
        return "- Market or Limit sell\n- Add Stop Loss above resistance\n- Consider Put Options or short"
    else:
        return "- Hold: No new action\n- Review Stop Loss and existing positions"

def google_news_headlines(query, max_items=5):
    try:
        url = f"https://news.google.com/search?q={query.replace(' ', '+')}"
        html = requests.get(url, timeout=10).text
        soup = BeautifulSoup(html, "html.parser")
        return [item.text for item in soup.select('article h3 a')][:max_items]
    except:
        return ["No news found."]

def fda_drug_approvals(max_items=5):
    try:
        rss_url = "https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/press-releases/rss.xml"
        feed = feedparser.parse(rss_url)
        return [entry.title for entry in feed.entries[:max_items]]
    except:
        return ["No FDA data available"]

def yahoo_trending_stocks():
    try:
        url = "https://finance.yahoo.com/trending-tickers"
        html = requests.get(url, timeout=10).text
        soup = BeautifulSoup(html, "html.parser")
        return [a.text for a in soup.select('a[data-test=\"quoteLink\"]')][:7]
    except:
        return []

sector_keywords = {
    "Nuclear": ["BWXT", "CCJ", "SMR", "U", "CAMECO"],
    "AI": ["NVDA", "AMD", "GOOGL", "MSFT", "PLTR", "META", "TSLA"],
    "Tech": ["AAPL", "AMZN", "AVGO", "ORCL", "CRM"],
    "Healthcare": ["JNJ", "PFE", "LLY", "UNH", "CVS"],
    "Healthcare Tech": ["ISRG", "TDOC", "MDT", "DXCM"],
    "Pharmaceuticals": ["PFE", "MRNA", "BNTX", "SNY"],
    "Defense & Aerospace": ["LMT", "NOC", "RTX", "GD", "PLTR", "BA"]
}

# ---- STREAMLIT UI ----
tab1, tab2 = st.tabs(["Market Information", "Suggestions"])

with tab1:
    st.header("Market Information")
    user_ticker = st.text_input("Enter any ticker (Stock, ETF, Crypto, FOREX, Commodity):", "AAPL")
    if user_ticker:
        try:
            df = get_data(user_ticker, period='2mo', interval='1d')
            df = calc_indicators(df)
            company = resolve_ticker(user_ticker)
            st.subheader(f"{company} ({user_ticker.upper()})")
            st.metric("Current Price", f"${df['Close'].iloc[-1]:.2f}")
            st.metric("RSI", f"{df['RSI'].iloc[-1]:.2f}")
            st.metric("MACD", f"{df['MACD'].iloc[-1]:.2f}")
            suggestion = buy_hold_sell(df)
            st.metric("Signal", suggestion)
            st.line_chart(df['Close'], use_container_width=True)

            # 7-day Prophet forecast
            f_dates, f_vals = prophet_forecast(df)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Historical"))
            fig.add_trace(go.Scatter(x=f_dates, y=f_vals, name="7d Prophet Forecast", line=dict(dash='dot')))
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Trading Strategy Suggestion")
            st.markdown(trading_strategy(suggestion))

            st.subheader("Latest Headlines")
            for n in google_news_headlines(company):
                st.write("-", n)

        except Exception as e:
            st.error(f"Error: {e}")

with tab2:
    st.header("AI Suggestions (Trending Companies & Sectors)")
    st.markdown("**World-wide, with focus on Nuclear, AI, Tech, Healthcare, Defense, Pharma.**")
    st.divider()

    for sector, tickers in sector_keywords.items():
        st.subheader(f"🔷 {sector}")
        for t in tickers:
            cname = resolve_ticker(t)
            st.markdown(f"**{cname} ({t})**")
            news = google_news_headlines(cname)
            for n in news:
                st.write("-", n)

    st.divider()
    st.subheader("📰 News & Trends (Global)")
    st.markdown("**Trending Stocks (Yahoo):**")
    trending = yahoo_trending_stocks()
    if trending:
        st.write(", ".join(trending))
    else:
        st.write("No trending data found.")

    st.markdown("**FDA Drug Approvals:**")
    for item in fda_drug_approvals():
        st.write("-", item)

    st.markdown("**Recent News (Reddit/X):** *(web scraping limited for free, advanced integrations can be added later)*")
    st.write("Try searching companies or sectors of interest on Reddit/X for sentiment. For now, news headlines are shown above.")
