import streamlit as st
import yfinance as yf
import pandas as pd
import requests
import plotly.graph_objects as go
from prophet import Prophet
from bs4 import BeautifulSoup
import feedparser

st.set_page_config(page_title="Market Analyzer", layout="wide")

# --- Global Company Name/Ticker Resolution ---
@st.cache_data(show_spinner=False)
def yahoo_symbol_search(query):
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}&quotesCount=20&newsCount=0"
    try:
        r = requests.get(url, timeout=5)
        j = r.json()
        options = []
        for res in j.get("quotes", []):
            # Exclude some warrant/preferred/OTC garbage tickers for readability
            if "symbol" in res and "shortname" in res and res.get("exchange", "").lower() not in {"none"}:
                options.append({"label": f"{res['shortname']} ({res['symbol']})", "value": res['symbol']})
        return options
    except Exception:
        return []

def resolve_symbol(query):
    options = yahoo_symbol_search(query)
    if options:
        return options[0]['value']  # Return first match
    return query.upper()

def get_company_name(ticker):
    try:
        info = yf.Ticker(ticker).info
        return info.get('longName') or info.get('shortName') or ticker
    except:
        return ticker

@st.cache_data(show_spinner=False)
def get_data(ticker, period='1y'):
    try:
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        return df if not df.empty else None
    except:
        return None

def get_news(ticker):
    results = []
    try:
        yurl = f"https://finance.yahoo.com/quote/{ticker}?p={ticker}"
        html = requests.get(yurl, timeout=5).text
        soup = BeautifulSoup(html, 'html.parser')
        for tag in soup.select("h3 a"):
            href = tag.get("href")
            title = tag.text
            if href and title:
                if not href.startswith("http"):
                    href = "https://finance.yahoo.com" + href
                results.append((title, href, "Yahoo Finance"))
    except:
        pass
    try:
        gurl = f"https://news.google.com/rss/search?q={ticker}"
        feed = feedparser.parse(gurl)
        for entry in feed.entries[:5]:
            results.append((entry.title, entry.link, "Google News"))
    except:
        pass
    return results if results else [("No news found.", "", "")]

def get_fda_approvals():
    rss_url = "https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/press-releases/rss.xml"
    try:
        feed = feedparser.parse(rss_url)
        return [(entry.title, entry.link) for entry in feed.entries[:8]]
    except:
        return [("No FDA data available", "")]

# --- Shared Company Search Widget (works everywhere) ---
def company_search(label, key):
    query = st.text_input(f"{label} (Name or Ticker):", key=key+"_name")
    symbol_options = yahoo_symbol_search(query) if query else []
    selected = st.selectbox("Select:", symbol_options, format_func=lambda x: x["label"] if x else "", key=key+"_dropdown")
    ticker = selected["value"] if selected else query.upper()
    return ticker

# --- Market Information Tab ---
def market_information():
    st.header("Market Information")
    ticker = company_search("Type company name or ticker", "main")
    if not ticker:
        st.info("Type a company name and select a valid ticker from the dropdown.")
        return

    name = get_company_name(ticker)
    df = get_data(ticker, period='1y')
    if df is None:
        st.warning("No price data available for this ticker. Please check the ticker symbol or try another.")
        return

    st.subheader(f"{name} ({ticker})")
    st.metric("Current Price", f"${df['Close'].iloc[-1]:,.2f}")

    # Technicals
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df = df.dropna()
    trend = "Bullish" if df['SMA_50'].iloc[-1] > df['SMA_200'].iloc[-1] else "Bearish"
    st.metric("Trend", trend)
    signal = "Buy" if trend == "Bullish" else ("Sell" if trend == "Bearish" else "Hold")
    st.metric("Signal", signal)

    # Prophet Forecast
    dfp = df[['Close']].reset_index()
    dfp.columns = ['ds', 'y']
    model = Prophet(daily_seasonality=True)
    model.fit(dfp)
    future = model.make_future_dataframe(periods=7)
    forecast = model.predict(future)
    st.subheader("7-Day Price Forecast")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
    st.plotly_chart(fig, use_container_width=True)

    # News
    st.subheader("News")
    for title, link, src in get_news(ticker):
        if link:
            st.markdown(f"- [{title}]({link}) ({src})")
        else:
            st.write("-", title)

# --- Suggestions Tab ---
def suggestions_tab():
    st.header("AI Suggested Companies to Watch")
    ai_suggestions = {
        "AI": ["NVDA", "MSFT", "GOOGL", "AMD", "BIDU", "TCEHY", "SFTBY", "TSM"],
        "Tech": ["AAPL", "AMZN", "AVGO", "TSM", "SONY", "SNEJF", "SAP.DE"],
        "Defense/Aerospace": ["NOC", "LMT", "BA", "PLTR", "RTX", "AIR.PA"],
        "Healthcare": ["JNJ", "UNH", "SNY", "RHHBY", "PFE", "NOVN.SW", "CSL.AX"],
        "Pharmaceuticals": ["MRK", "AZN", "NVS", "LLY", "BMY", "ROG.SW", "GSK.L"],
        "Nuclear": ["BWXT", "SMR", "TSE:PDN", "AREVA.PA"],
        "Crypto": ["BTC-USD", "ETH-USD", "SOL-USD"],
        "ETFs": ["QQQ", "SPY", "ARKK", "ARKG", "ICLN", "XBI", "XLV"]
    }
    for sector, tickers in ai_suggestions.items():
        st.subheader(f"{sector}")
        for ticker in tickers:
            name = get_company_name(ticker)
            st.markdown(f"**{name} ({ticker})**")
            for title, link, src in get_news(ticker):
                if link:
                    st.markdown(f"- [{title}]({link}) ({src})")
                else:
                    st.write("-", title)
    st.divider()
    st.subheader("Latest FDA Drug Approval News")
    for title, link in get_fda_approvals():
        if link:
            st.markdown(f"- [{title}]({link})")
        else:
            st.write("-", title)

# --- Peer Comparison Tab ---
def peer_comparison_tab():
    st.header("Peer Comparison")
    ticker1 = company_search("Company 1", "peer1")
    ticker2 = company_search("Company 2", "peer2")
    if ticker1 and ticker2:
        df1 = get_data(ticker1, period='1y')
        df2 = get_data(ticker2, period='1y')
        if df1 is not None and df2 is not None:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df1.index, y=df1['Close'], name=f"{ticker1}"))
            fig.add_trace(go.Scatter(x=df2.index, y=df2['Close'], name=f"{ticker2}"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("One or both tickers could not be loaded.")

# --- Watchlist Tab ---
def watchlist_tab():
    st.header("Watchlist")
    if "watchlist" not in st.session_state:
        st.session_state["watchlist"] = []
    # Add
    ticker = company_search("Add to watchlist", "watchadd")
    if st.button("Add to Watchlist"):
        if ticker not in st.session_state["watchlist"]:
            st.session_state["watchlist"].append(ticker)
    # Remove
    for t in st.session_state["watchlist"]:
        cols = st.columns([4,1])
        cols[0].write(t)
        if cols[1].button("Delete", key=f"del_{t}"):
            st.session_state["watchlist"].remove(t)
    # Show charts
    for t in st.session_state["watchlist"]:
        df = get_data(t, period='1mo')
        if df is not None:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name=t))
            st.plotly_chart(fig, use_container_width=True)

# --- Financial Health Tab ---
def financial_health_tab():
    st.header("Financial Health")
    ticker = company_search("Select company", "finhlth")
    info = yf.Ticker(ticker).info
    keys = ['marketCap','trailingPE','dividendYield','profitMargins','returnOnEquity']
    st.write({k: info.get(k, 'N/A') for k in keys})

# --- Options Analytics Tab ---
def options_analytics_tab():
    st.header("Options Analytics")
    ticker = company_search("Options for company", "options")
    tk = yf.Ticker(ticker)
    try:
        exps = tk.options
        if exps:
            exp = st.selectbox("Expiry", exps)
            opt = tk.option_chain(exp)
            st.write("Calls")
            st.dataframe(opt.calls)
            st.write("Puts")
            st.dataframe(opt.puts)
    except Exception as e:
        st.warning("No options data available for this ticker.")

# --- Macro Insights Tab ---
def macro_insights_tab():
    st.header("Macro Insights")
    st.write("Global macro indicators coming soon.")

# --- News Explorer Tab ---
def news_explorer_tab():
    st.header("News Explorer")
    st.subheader("Global/Sector-wide news")
    for sector in ["finance", "technology", "energy", "healthcare", "pharma", "crypto", "etf"]:
        st.markdown(f"**Sector: {sector.title()}**")
        url = f"https://news.google.com/rss/search?q={sector}+stock"
        feed = feedparser.parse(url)
        for entry in feed.entries[:5]:
            st.markdown(f"- [{entry.title}]({entry.link})")

# --- Navigation Tabs ---
tabs = [
    "Market Information", "Suggestions", "Peer Comparison", "Watchlist",
    "Financial Health", "Options Analytics", "Macro Insights", "News Explorer"
]
tab = st.sidebar.radio("Navigation", tabs)

if tab == "Market Information":
    market_information()
elif tab == "Suggestions":
    suggestions_tab()
elif tab == "Peer Comparison":
    peer_comparison_tab()
elif tab == "Watchlist":
    watchlist_tab()
elif tab == "Financial Health":
    financial_health_tab()
elif tab == "Options Analytics":
    options_analytics_tab()
elif tab == "Macro Insights":
    macro_insights_tab()
elif tab == "News Explorer":
    news_explorer_tab()
