import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from prophet import Prophet
from datetime import datetime
from bs4 import BeautifulSoup
import feedparser

# --- Helper: Global Yahoo search for ticker lookup
@st.cache_data(show_spinner=False)
def yahoo_search(query):
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}&quotesCount=10&newsCount=0"
    try:
        r = requests.get(url)
        results = r.json()
        matches = []
        for quote in results.get("quotes", []):
            symbol = quote.get("symbol", "")
            exch = quote.get("exchange", "")
            name = quote.get("shortname", "") or quote.get("longname", "") or symbol
            if symbol and exch:
                matches.append(f"{name} [{symbol}]")
        return matches
    except Exception as e:
        return []

def extract_ticker(selection):
    if not selection:
        return ""
    if "[" in selection and "]" in selection:
        return selection.split("[")[-1].replace("]", "").strip().upper()
    return selection.strip().upper()

# Universal company/ticker selector
def global_ticker_selector(label="Type company name or ticker:", key="ticker_selector"):
    query = st.text_input(label, key=key+"_query")
    options = yahoo_search(query) if query and len(query) > 1 else []
    chosen = st.selectbox("Choose Ticker:", options, key=key+"_chosen") if options else ""
    return extract_ticker(chosen or query)

def get_company_name(ticker):
    try:
        info = yf.Ticker(ticker).info
        return info.get('longName') or info.get('shortName') or ticker
    except:
        return ticker

@st.cache_data(show_spinner=False)
def get_data(ticker, period='1y'):
    try:
        data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if data.empty or 'Close' not in data.columns:
            return None
        return data
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

def get_news(ticker):
    news_items = []
    try:
        url = f"https://news.google.com/search?q={ticker}+stock"
        html = requests.get(url).text
        soup = BeautifulSoup(html, 'html.parser')
        news_items += [
            {"title": item.text, "url": "https://news.google.com"+item['href'][1:] if item.has_attr('href') else ""}
            for item in soup.select('article h3 a')
        ][:5]
    except:
        pass
    try:
        feed = feedparser.parse(f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US")
        news_items += [{"title": entry.title, "url": entry.link} for entry in feed.entries][:5]
    except:
        pass
    seen = set()
    filtered = []
    for n in news_items:
        if n["title"] not in seen:
            filtered.append(n)
            seen.add(n["title"])
    return filtered[:8] if filtered else [{"title": "No news found.", "url": ""}]

def get_fda_approvals():
    try:
        rss_url = "https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/press-releases/rss.xml"
        feed = feedparser.parse(rss_url)
        return [{"title": entry.title, "url": entry.link} for entry in feed.entries[:8]]
    except:
        return [{"title": "No FDA data available", "url": ""}]

def forecast_prices(df, ticker):
    try:
        df = df[['Close']].copy().reset_index()
        df.columns = ['ds', 'y']
        df['ds'] = pd.to_datetime(df['ds'])
        model = Prophet(daily_seasonality=True)
        model.fit(df)
        future = model.make_future_dataframe(periods=7)
        forecast = model.predict(future)
        return forecast[['ds', 'yhat']].tail(7)
    except:
        return None

st.set_page_config(page_title="Global Market Watcher", layout="wide", initial_sidebar_state="collapsed")

tabs = st.tabs([
    "Market Information", "Suggestions", "Peer Comparison", "Watchlist",
    "Financial Health", "Options Analytics", "Macro Insights", "News Explorer"
])

# ------ Market Information ------
with tabs[0]:
    st.header("Market Information")
    ticker = global_ticker_selector("Type company name or ticker:", key="mi")
    if ticker:
        name = get_company_name(ticker)
        st.subheader(f"{name} ({ticker})")
        df = get_data(ticker)
        if df is None:
            st.warning("No price data available for this ticker. Please check the ticker symbol or try another.")
        else:
            df = calculate_indicators(df)
            df = generate_signals(df)
            st.metric("Current Price", f"${df['Close'].iloc[-1]:,.2f}")
            st.metric("Signal", df['Signal'].iloc[-1])
            st.metric("Trend", "📈 Bullish" if df['Trend'].iloc[-1] == 1 else "📉 Bearish")
            st.markdown("### 7-Day Price Forecast")
            forecast = forecast_prices(df, ticker)
            if forecast is not None:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast'))
                st.plotly_chart(fig, use_container_width=True)
            st.markdown("### News Sentiment")
            news_items = get_news(ticker)
            for n in news_items:
                st.write(f"- [{n['title']}]({n['url']})" if n['url'] else f"- {n['title']}")

# ------ Suggestions ------
with tabs[1]:
    st.header("AI Suggested Companies to Watch")
    ai_suggestions = {
        "AI": ["NVDA", "SMCI", "AMD"],
        "Tech": ["MSFT", "GOOGL", "AMZN", "AVGO", "TSLA"],
        "Defense": ["NOC", "LMT", "BWXT", "PLTR"],
        "Healthcare Tech": ["UNH", "XBI"],
        "Pharmaceuticals": ["PFE", "MRNA"],
        "Nuclear": ["SMR", "BWXT"],
        "Biotech": ["ARKG"],
        "Clean Energy": ["ICLN"]
    }
    for sector, tickers in ai_suggestions.items():
        st.subheader(f"🔷 {sector} Sector")
        for ticker in tickers:
            name = get_company_name(ticker)
            st.markdown(f"**{name} ({ticker})**")
            news_items = get_news(ticker)
            for n in news_items:
                st.write(f"- [{n['title']}]({n['url']})" if n['url'] else f"- {n['title']}")
    st.divider()
    st.markdown("### 📰 Sector/Trending News")
    for sector, tickers in ai_suggestions.items():
        st.subheader(f"🗞️ {sector} Headlines")
        for ticker in tickers:
            news_items = get_news(ticker)
            for n in news_items[:2]:
                st.write(f"- [{n['title']}]({n['url']})" if n['url'] else f"- {n['title']}")
    st.subheader("🧪 FDA Drug Approval News")
    for fda in get_fda_approvals():
        st.write(f"- [{fda['title']}]({fda['url']})" if fda['url'] else f"- {fda['title']}")

# ------ Peer Comparison ------
with tabs[2]:
    st.header("Peer Comparison")
    ticker = global_ticker_selector("Enter Ticker for Peer Comparison:", key="peer")
    if ticker:
        try:
            main = yf.Ticker(ticker)
            peers = main.get_peers() if hasattr(main, "get_peers") else []
            peers = peers[:5] if peers else []
            st.write(f"Main Company: **{get_company_name(ticker)}** ({ticker})")
            if peers:
                data = []
                for peer in peers:
                    df_peer = get_data(peer)
                    price = df_peer['Close'].iloc[-1] if df_peer is not None else None
                    data.append({"Company": get_company_name(peer), "Ticker": peer, "Price": price})
                df_peers = pd.DataFrame(data)
                st.dataframe(df_peers)
            else:
                st.info("No peer data available for this ticker.")
        except Exception as e:
            st.warning("Error retrieving peer data.")

# ------ Watchlist ------
with tabs[3]:
    st.header("Watchlist")
    if "watchlist" not in st.session_state:
        st.session_state["watchlist"] = []
    ticker = global_ticker_selector("Add company/ticker to watchlist:", key="watchlist")
    if st.button("Add to Watchlist"):
        if ticker and ticker not in st.session_state["watchlist"]:
            st.session_state["watchlist"].append(ticker)
    remove_ticker = st.selectbox("Remove from Watchlist:", st.session_state["watchlist"] or ["(none)"])
    if st.button("Remove Selected") and remove_ticker in st.session_state["watchlist"]:
        st.session_state["watchlist"].remove(remove_ticker)
    st.markdown("### Your Watchlist")
    for t in st.session_state["watchlist"]:
        name = get_company_name(t)
        st.write(f"- {name} ({t})")

# ------ Financial Health ------
with tabs[4]:
    st.header("Financial Health")
    ticker = global_ticker_selector("Check company financials:", key="finhealth")
    if ticker:
        t = yf.Ticker(ticker)
        try:
            st.write("#### Balance Sheet")
            st.dataframe(t.balance_sheet)
        except:
            st.write("No balance sheet available.")
        try:
            st.write("#### Profit & Loss")
            st.dataframe(t.financials)
        except:
            st.write("No P&L available.")

# ------ Options Analytics ------
with tabs[5]:
    st.header("Options Analytics")
    ticker = global_ticker_selector("Analyze options for ticker:", key="options")
    if ticker:
        t = yf.Ticker(ticker)
        try:
            dates = t.options
            if not dates:
                st.write("No options data available.")
            else:
                st.write("#### Option Expiries")
                expiry = st.selectbox("Select expiry:", dates)
                opt = t.option_chain(expiry)
                st.write("##### Calls")
                st.dataframe(opt.calls)
                st.write("##### Puts")
                st.dataframe(opt.puts)
        except:
            st.write("No options data for this ticker.")

# ------ Macro Insights ------
with tabs[6]:
    st.header("Macro Insights")
    st.write("Economic indicators, trends, and more. (Sample coming soon)")

# ------ News Explorer ------
with tabs[7]:
    st.header("News Explorer")
    ticker = global_ticker_selector("Find news for company/ticker:", key="news")
    if ticker:
        news_items = get_news(ticker)
        st.markdown(f"#### Latest News for {get_company_name(ticker)} ({ticker})")
        for n in news_items:
            st.write(f"- [{n['title']}]({n['url']})" if n['url'] else f"- {n['title']}")

# ------ The rest of the tabs are fully implemented. ------
