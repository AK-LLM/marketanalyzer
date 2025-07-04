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

# Slight delay to help render elements in Streamlit
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

def get_financials(ticker):
    try:
        ticker_obj = yf.Ticker(ticker)
        fin = ticker_obj.financials
        return fin.transpose() if not fin.empty else None
    except:
        return None

def get_earnings_calendar(ticker):
    try:
        cal = yf.Ticker(ticker).calendar
        return cal.transpose() if not cal.empty else None
    except:
        return None

def calculate_indicators(df):
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
    df['Trend'] = np.where(df['SMA_50'] > df['SMA_200'], 1, -1)
    df['Buy'] = (df['RSI'] < 30) & (df['MACD'] > df['Signal_Line']) & (df['Trend'] == 1)
    df['Sell'] = (df['RSI'] > 70) & (df['MACD'] < df['Signal_Line']) & (df['Trend'] == -1)
    df['Signal'] = np.where(df['Buy'], 'Buy', np.where(df['Sell'], 'Sell', 'Hold'))
    return df

def get_sentiment_score(ticker, dates):
    scores = []
    for _ in dates:
        try:
            news = get_news(ticker)
            score = sum(1 if any(kw in headline.lower() for kw in ['beat', 'record', 'growth', 'win', 'surge']) else -1 if any(kw in headline.lower() for kw in ['miss', 'loss', 'decline', 'fall']) else 0 for headline, _ in news)
            scores.append(score)
        except:
            scores.append(0)
    return pd.Series(scores[:len(dates)])

def fetch_event_impact(ticker):
    try:
        events = []
        calendar = get_earnings_calendar(ticker)
        if calendar is not None:
            events.append("Upcoming Earnings Call")
        return len(events)
    except:
        return 0

def forecast_prices(df, ticker):
    df = df[['Close']].copy().reset_index()
    df.columns = ['ds', 'y']
    df['ds'] = pd.to_datetime(df['ds'])
    df['news_sentiment'] = get_sentiment_score(ticker, df['ds'])
    df['event_impact'] = fetch_event_impact(ticker)

    model = Prophet(daily_seasonality=True)
    model.add_regressor('news_sentiment')
    model.fit(df)

    future = model.make_future_dataframe(periods=7)
    future['news_sentiment'] = 0
    forecast = model.predict(future)

    forecast_range = forecast.tail(7)
    x = np.arange(len(forecast_range))
    linear_coeffs = np.polyfit(x, forecast_range['yhat'], 1)
    linear_trend = np.polyval(linear_coeffs, x)
    forecast.loc[forecast.index[-7:], 'trendline'] = linear_trend

    return forecast[['ds', 'yhat', 'trendline']]

def get_news(ticker):
    try:
        url = f"https://news.google.com/search?q={ticker}+stock"
        html = requests.get(url).text
        soup = BeautifulSoup(html, 'html.parser')
        news_list = []
        for a in soup.select('article h3 a'):
            text = a.text
            href = a.get('href')
            link = f"https://news.google.com{href[1:]}" if href.startswith('.') else href
            news_list.append((text, link))
        return news_list[:5] if news_list else [("No news found.", "")]
    except:
        return [("No news found.", "")]

def get_fda_approvals():
    try:
        rss_url = "https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/press-releases/rss.xml"
        feed = feedparser.parse(rss_url)
        return [entry.title for entry in feed.entries[:5]]
    except:
        return ["No FDA data available"]

def get_yahoo_finance_news():
    try:
        rss_url = "https://finance.yahoo.com/rss/topstories"
        feed = feedparser.parse(rss_url)
        return [(entry.title, entry.link) for entry in feed.entries[:7]]
    except:
        return [("No Yahoo Finance RSS found.", "")]

# ========== UI ==========

st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] button { font-size: 1.1rem !important; }
    .stMetric label { font-size: 1.1rem !important; }
</style>
""", unsafe_allow_html=True)

tabs = st.tabs([
    "Market Information", "Suggestions",
    "Peer Comparison", "Watchlist", "Financial Health",
    "Options Analytics", "Macro Insights", "News Explorer"
])

ticker = "AAPL"  # Default value

# ----------------- MARKET INFORMATION TAB ------------------
with tabs[0]:
    st.header("Market Information")
    ticker = st.text_input("Enter any ticker (Stock, ETF, Crypto, FOREX, Commodity):", "AAPL")
    show_market_info = False
    df = pd.DataFrame()
    if ticker:
        try:
            df = get_data(ticker)
            df = calculate_indicators(df)
            df = generate_signals(df)
            forecast = forecast_prices(df.copy(), ticker)
            name = get_company_name(ticker)
            show_market_info = True
        except Exception as e:
            st.error(f"Error: {e}")

    if show_market_info and not df.empty:
        st.subheader(f"{name} ({ticker.upper()})")
        try:
            close_val = float(df['Close'].iloc[-1]) if not df['Close'].empty else float('nan')
            signal_val = df['Signal'].iloc[-1] if not df['Signal'].empty else "N/A"
            trend_val = "📈 Bullish" if not df['Trend'].empty and df['Trend'].iloc[-1] == 1 else "📉 Bearish"
        except Exception:
            close_val = float('nan')
            signal_val = "N/A"
            trend_val = "N/A"
        st.metric("Current Price", f"${close_val:.2f}" if not pd.isnull(close_val) else "N/A")
        st.metric("Signal", signal_val)
        st.metric("Trend", trend_val)

        st.subheader("7-Day Price Forecast")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['trendline'], name='Forecast Trendline', line=dict(shape='linear')))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("News Sentiment")
        for headline, link in get_news(ticker):
            if link:
                st.markdown(f"- [{headline}]({link})")
            else:
                st.write("-", headline)

# ----------------- SUGGESTIONS TAB -------------------
with tabs[1]:
    st.header("💡 AI Suggested Companies to Watch")

    ai_suggestions = {
        "AI": ["NVDA", "SMCI", "AMD"],
        "Tech": ["MSFT", "GOOGL", "AMZN", "AVGO", "TSLA"],
        "Defense": ["NOC", "LMT", "BWXT", "PLTR", "RTX", "BA"],
        "Healthcare Tech": ["UNH", "XBI", "CERN", "TDOC"],
        "Pharmaceuticals": ["PFE", "MRNA", "LLY", "NVO", "BNTX", "AZN"],
        "Nuclear": ["SMR", "BWXT", "LEU", "U"],
        "Biotech": ["ARKG", "VRTX", "REGN", "CRSP"],
        "Clean Energy": ["ICLN", "ENPH", "SEDG", "FSLR"],
        "Aerospace": ["LMT", "RTX", "BA", "AIR.PA"],
        "Broadcom & Chips": ["AVGO", "QCOM", "ASML", "TXN", "INTC"],
        "Other Trending": ["META", "NFLX", "SHOP", "SQ", "PYPL"]
    }

    for sector, tickers in ai_suggestions.items():
        st.subheader(f"🔷 {sector} Sector")
        for t in tickers:
            name = get_company_name(t)
            st.markdown(f"**{name} ({t})**")
            for n, link in get_news(t):
                if link:
                    st.markdown(f"- [{n}]({link})")
                else:
                    st.write("-", n)

    st.divider()
    st.markdown("### 📰 Sector-Wide News & Trends")

    st.subheader("Yahoo Finance News")
    for title, link in get_yahoo_finance_news():
        if link:
            st.markdown(f"- [{title}]({link})")
        else:
            st.write("-", title)

    st.subheader("🧪 FDA Drug Approval News")
    for fda in get_fda_approvals():
        st.write("-", fda)

# ------ The rest of the tabs ("Peer Comparison", "Watchlist", "Financial Health", etc.) are implemented as modular enhancements ------
# ------ You can add, comment out, or move them based on your needs. ------

# Example placeholder for Peer Comparison:
with tabs[2]:
    st.header("Peer Comparison (Coming Soon)")

with tabs[3]:
    st.header("Watchlist (Coming Soon)")

with tabs[4]:
    st.header("Financial Health (Coming Soon)")

with tabs[5]:
    st.header("Options Analytics (Coming Soon)")

with tabs[6]:
    st.header("Macro Insights (Coming Soon)")

with tabs[7]:
    st.header("News Explorer (Coming Soon)")
