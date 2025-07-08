import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
from prophet import Prophet
import requests
from bs4 import BeautifulSoup
import os

st.set_page_config(page_title="Market Terminal", layout="wide")

# ------------- Ticker DB -------------
@st.cache_data(show_spinner=True)
def load_ticker_db():
    if not os.path.exists("global_tickers.csv"):
        st.warning("Missing 'global_tickers.csv'. Please place it next to app.py.")
        return pd.DataFrame(columns=['symbol', 'name', 'exchange'])
    df = pd.read_csv("global_tickers.csv", dtype=str)
    df = df.dropna(subset=['symbol', 'name'])
    df['symbol'] = df['symbol'].str.upper()
    df['name'] = df['name'].str.title()
    return df

tickers_db = load_ticker_db()

def search_tickers_db(q):
    q = q.strip().upper()
    if not q:
        return []
    results = tickers_db[tickers_db.apply(
        lambda row: q in row['symbol'] or q in row['name'].upper(), axis=1)]
    return results[['symbol', 'name', 'exchange']].head(20).to_dict("records")

def resolve_symbol(q):
    df = tickers_db[tickers_db['symbol'] == q.upper()]
    if not df.empty:
        return df.iloc[0]['symbol'], df.iloc[0]['name'], df.iloc[0]['exchange']
    results = search_tickers_db(q)
    if results:
        return results[0]['symbol'], results[0]['name'], results[0]['exchange']
    return q.upper(), q.title(), ""

def get_company_name(symbol):
    df = tickers_db[tickers_db['symbol'] == symbol.upper()]
    if not df.empty:
        return df.iloc[0]['name']
    try:
        info = yf.Ticker(symbol).info
        return info.get('longName') or info.get('shortName') or symbol
    except:
        return symbol

@st.cache_data(show_spinner=True)
def get_data(symbol, period='1y'):
    try:
        data = yf.download(symbol, period=period, auto_adjust=True, progress=False)
        if data.empty or 'Close' not in data.columns:
            return pd.DataFrame()
        return data
    except Exception as e:
        return pd.DataFrame()

def calculate_indicators(df):
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    df['Signal_Line'] = df['MACD'].ewm(span=9).mean()
    df = df.dropna()
    return df

def generate_signals(df):
    df['Trend'] = np.where(df['SMA_50'] > df['SMA_200'], 1, -1)
    df['Buy'] = (df['RSI'] < 30) & (df['MACD'] > df['Signal_Line']) & (df['Trend'] == 1)
    df['Sell'] = (df['RSI'] > 70) & (df['MACD'] < df['Signal_Line']) & (df['Trend'] == -1)
    df['Signal'] = np.where(df['Buy'], 'Buy', np.where(df['Sell'], 'Sell', 'Hold'))
    return df

def forecast_prices(df, days=7):
    df_prophet = df[['Close']].reset_index()
    df_prophet.columns = ['ds', 'y']
    model = Prophet(daily_seasonality=True)
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']].tail(days)

def get_news(symbol, n=7):
    url = f"https://news.google.com/search?q={symbol}+stock"
    try:
        html = requests.get(url, timeout=5).text
        soup = BeautifulSoup(html, 'html.parser')
        headlines = []
        for h in soup.select('article h3 a')[:n]:
            href = h['href']
            if href.startswith('.'):
                href = "https://news.google.com" + href[1:]
            headlines.append((h.text, href))
        return headlines if headlines else [("No news found.", "")]
    except:
        return [("No news found.", "")]

# ========== Tabs ==========

def show_market_info():
    st.header("Market Information")
    query = st.text_input("Type company name or ticker:", "")
    options = search_tickers_db(query) if query else []
    symbol, name, exchange = "", "", ""
    if options:
        labels = [f"{x['name']} ({x['symbol']}) [{x['exchange']}]" for x in options]
        sel = st.selectbox("Select:", labels)
        idx = labels.index(sel)
        symbol, name, exchange = options[idx]['symbol'], options[idx]['name'], options[idx]['exchange']
    elif query:
        symbol, name, exchange = resolve_symbol(query)
    if symbol:
        st.subheader(f"{name} ({symbol})")
        data = get_data(symbol)
        if data.empty:
            st.warning("No price data available for this ticker. Please check the ticker symbol or try another.")
            return
        data = calculate_indicators(data)
        data = generate_signals(data)
        # Signal and Trend
        signal = str(data['Signal'].iloc[-1]) if not data.empty else "N/A"
        trend = "📈 Bullish" if data['Trend'].iloc[-1] == 1 else "📉 Bearish"
        st.metric("Signal", signal)
        st.metric("Trend", trend)
        # Forecast
        st.subheader("7-Day Price Forecast")
        forecast = forecast_prices(data)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast', line=dict(shape='linear')))
        st.plotly_chart(fig, use_container_width=True)
        # News
        st.subheader("Latest News")
        for title, url in get_news(symbol):
            st.write(f"- [{title}]({url})")
    else:
        st.info("Type a company name and select a valid ticker from the dropdown.")

def show_suggestions():
    st.header("AI Suggestions & News")
    ai_sectors = {
        "AI": ["NVDA", "MSFT", "GOOGL", "AMD", "TSLA", "AVGO", "SMCI"],
        "Healthcare": ["JNJ", "PFE", "MRK", "UNH", "RHHBY"],
        "Pharma": ["PFE", "MRNA", "BNTX", "GILD"],
        "Defense": ["LMT", "NOC", "RTX", "BA", "GD"],
        "Nuclear": ["BWXT", "SMR"],
        "Tech": ["AAPL", "AMZN", "META", "SAP", "SHOP"],
        "Commodities": ["GLD", "SLV", "BHP", "RIO", "VALE"]
    }
    for sector, tickers in ai_sectors.items():
        st.subheader(f"{sector} Sector")
        for t in tickers:
            st.markdown(f"**{get_company_name(t)} ({t})**")
            for title, url in get_news(t):
                st.write(f"- [{title}]({url})")
    st.divider()
    st.subheader("Sector/Global News")
    for t in ["^GSPC", "^IXIC", "^N225", "^FTSE", "BTC-USD"]:
        st.write(f"**News for {t}**")
        for title, url in get_news(t):
            st.write(f"- [{title}]({url})")

def peer_comparison_tab():
    st.header("Peer Comparison")
    st.info("Peer comparison is under development for global tickers.")

def watchlist_tab():
    st.header("Watchlist")
    if 'watchlist' not in st.session_state:
        st.session_state['watchlist'] = []
    addq = st.text_input("Add to Watchlist (company name or ticker):", "")
    if st.button("Add", key="add_watch"):
        symbol, name, exch = resolve_symbol(addq)
        if symbol and symbol not in st.session_state['watchlist']:
            st.session_state['watchlist'].append(symbol)
    if st.session_state['watchlist']:
        for sym in st.session_state['watchlist']:
            n = get_company_name(sym)
            col1, col2 = st.columns([4,1])
            with col1:
                st.write(f"{n} ({sym})")
            with col2:
                if st.button("Remove", key=f"rm_{sym}"):
                    st.session_state['watchlist'].remove(sym)
    else:
        st.info("No companies in watchlist.")

def financial_health_tab():
    st.header("Financial Health")
    st.info("Financial ratios and statements are being expanded for global tickers.")

def options_analytics_tab():
    st.header("Options Analytics")
    st.info("Options analytics (global) will be added as free data becomes available.")

def macro_insights_tab():
    st.header("Macro Insights")
    st.info("Macro indicators coming soon (will include major central banks, FX, indices).")

def news_explorer_tab():
    st.header("News Explorer")
    q = st.text_input("Search news for company, market, or sector:", "")
    if q:
        for title, url in get_news(q):
            st.write(f"- [{title}]({url})")
    else:
        st.info("Type a keyword, ticker, or company to see related news.")

tabs = {
    "Market Information": show_market_info,
    "Suggestions": show_suggestions,
    "Peer Comparison": peer_comparison_tab,
    "Watchlist": watchlist_tab,
    "Financial Health": financial_health_tab,
    "Options Analytics": options_analytics_tab,
    "Macro Insights": macro_insights_tab,
    "News Explorer": news_explorer_tab,
}
selection = st.sidebar.radio("Navigation", list(tabs.keys()), index=0)
tabs[selection]()
