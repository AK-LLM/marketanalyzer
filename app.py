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

# Load the tickers CSV
@st.cache_data(show_spinner=True)
def load_ticker_db():
    # This expects a file called global_tickers.csv in the app directory
    if not os.path.exists("global_tickers.csv"):
        st.warning("global_tickers.csv file not found. Please download a world company/ticker database and place it in the app directory.")
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
    # Match on symbol or name (case insensitive)
    results = tickers_db[tickers_db.apply(lambda row: q in row['symbol'] or q in row['name'].upper(), axis=1)]
    # Show up to 20 matches
    return results[['symbol', 'name', 'exchange']].head(20).to_dict("records")

def resolve_symbol(q):
    # Try exact symbol first
    df = tickers_db[tickers_db['symbol'] == q.upper()]
    if not df.empty:
        return df.iloc[0]['symbol'], df.iloc[0]['name'], df.iloc[0]['exchange']
    # Try fuzzy search
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
            raise ValueError("No price data available for this ticker.")
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

def get_news(symbol, n=5):
    # Google Finance News
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

def show_market_info():
    st.header("Market Information")
    query = st.text_input("Type company name or ticker (Name or Ticker):", "")
    options = []
    selected = ""
    if query:
        options = search_tickers_db(query)
        if options:
            selected = st.selectbox("Select:", [f"{x['name']} ({x['symbol']}) [{x['exchange']}]" for x in options])
        else:
            selected = ""
    symbol = ""
    name = ""
    exchange = ""
    if selected:
        idx = [i for i, x in enumerate(options) if f"{x['name']} ({x['symbol']}) [{x['exchange']}]" == selected]
        if idx:
            symbol, name, exchange = options[idx[0]]['symbol'], options[idx[0]]['name'], options[idx[0]]['exchange']
    elif query:
        symbol, name, exchange = resolve_symbol(query)
    if symbol:
        st.subheader(f"{name.upper()} ({symbol.upper()})")
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
    # Example: sector-sorted, you can expand this as needed
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
    st.write("Peer company analysis coming soon.")

def watchlist_tab():
    st.header("Watchlist")
    st.info("Watchlist functionality coming soon. (Will support add/remove for selected tickers.)")

def financial_health_tab():
    st.header("Financial Health")
    st.info("Balance sheet, PnL, ratios, and analytics coming soon.")

def options_analytics_tab():
    st.header("Options Analytics")
    st.info("Global options analytics coming soon (US only for now, global support if free data available).")

def macro_insights_tab():
    st.header("Macro Insights")
    st.info("Global macro indicators coming soon.")

def news_explorer_tab():
    st.header("News Explorer")
    st.info("Search and browse sector and market news.")

# -----------------
# Main navigation
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

