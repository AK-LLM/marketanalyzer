import streamlit as st
import requests
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from prophet import Prophet
import feedparser

st.set_page_config(page_title="Market Terminal", layout="wide")
st.title("Market Terminal")

TABS = [
    "Market Information", "Suggestions", "Peer Comparison", "Watchlist", "Financial Health",
    "Options Analytics", "Macro Insights", "News Explorer"
]
tab_idx = st.session_state.get("tab_idx", 0)
tab_names = st.columns(len(TABS))
for i, tab in enumerate(TABS):
    if tab_names[i].button(tab, key=f"nav_{tab}", use_container_width=True):
        st.session_state["tab_idx"] = i
        tab_idx = i

def yahoo_autocomplete(query, count=8):
    """Return list of dicts: [{'symbol':..., 'name':..., 'exchDisp':...}, ...]"""
    if not query:
        return []
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}&lang=en-US"
    try:
        data = requests.get(url, timeout=8).json()
        return [
            {
                "symbol": r.get("symbol", ""),
                "name": r.get("shortname", "") or r.get("longname", "") or "",
                "exchange": r.get("exchDisp", ""),
                "type": r.get("typeDisp", "")
            }
            for r in data.get("quotes", [])
            if r.get("symbol")
        ][:count]
    except Exception:
        return []

def resolve_symbol(query):
    # Try exact symbol, else autocomplete
    query = query.strip().upper()
    if query:
        ac = yahoo_autocomplete(query)
        for c in ac:
            if c["symbol"].upper() == query:
                return c["symbol"], c["name"], c["exchange"]
        if ac:
            return ac[0]["symbol"], ac[0]["name"], ac[0]["exchange"]
    return "", "", ""

def get_yf_data(ticker, period="1y"):
    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if df.empty:
            return None
        return df
    except Exception:
        return None

def google_news(ticker_or_name, count=7):
    q = ticker_or_name.replace("&", " ")
    url = f"https://news.google.com/rss/search?q={q}+stock"
    try:
        feed = feedparser.parse(url)
        return list({entry.link: entry.title for entry in feed.entries[:count]}.items())
    except Exception:
        return []

def yahoo_news(ticker, count=7):
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    try:
        feed = feedparser.parse(url)
        return list({entry.link: entry.title for entry in feed.entries[:count]}.items())
    except Exception:
        return []

def get_company_info(ticker):
    try:
        info = yf.Ticker(ticker).info
        return info
    except Exception:
        return {}

def plot_forecast(df, periods=7):
    if len(df) < 40:
        return None
    hist = df[["Close"]].reset_index().rename(columns={"Date":"ds", "Close":"y"})
    hist["ds"] = pd.to_datetime(hist["ds"])
    model = Prophet(daily_seasonality=True)
    model.fit(hist)
    fut = model.make_future_dataframe(periods=periods)
    forecast = model.predict(fut)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist['ds'], y=hist['y'], name='History'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name=f"{periods}-Day Forecast"))
    return fig

def news_section(symbol, name):
    news_items = []
    for news_feed in [google_news, yahoo_news]:
        news_items += news_feed(symbol) + news_feed(name)
    seen = set()
    deduped = []
    for url, title in news_items:
        if url not in seen:
            seen.add(url)
            deduped.append((url, title))
    return deduped[:8]

# --- TABS LOGIC ---
if tab_idx == 0: # Market Information
    st.header("Market Information")
    st.caption("Type company name or ticker (Name or Ticker):")
    query = st.text_input("", key="market_query")
    choices = yahoo_autocomplete(query)
    symbol = ""
    if choices:
        labels = [f"{c['name']} ({c['symbol']}, {c['exchange']})" for c in choices]
        choice = st.selectbox("Select:", labels, key="market_symbol_select")
        i = labels.index(choice)
        symbol = choices[i]["symbol"]
        name = choices[i]["name"]
    else:
        symbol = query.strip().upper()
        name = ""
    if symbol:
        name = name or symbol
        st.subheader(f"{name} ({symbol})")
        df = get_yf_data(symbol)
        if df is not None:
            st.metric("Current Price", f"${df['Close'].iloc[-1]:,.2f}")
            # Simple strategy suggestion
            change = df['Close'][-1] - df['Close'][-2] if len(df) > 1 else 0
            if change > 0:
                sig = "Buy"
                msg = "Momentum is positive"
            elif change < 0:
                sig = "Sell"
                msg = "Momentum is negative"
            else:
                sig = "Hold"
                msg = "No clear direction"
            st.metric("Signal", sig)
            st.caption(f"Rationale: {msg}")
            # 7-day forecast
            fc_fig = plot_forecast(df, periods=7)
            if fc_fig:
                st.plotly_chart(fc_fig, use_container_width=True)
            else:
                st.info("Not enough data for forecast.")
            # News
            st.subheader("News")
            for url, title in news_section(symbol, name):
                st.markdown(f"- [{title}]({url})")
        else:
            st.warning("No price data available for this ticker. Please check the ticker symbol or try another.")
    else:
        st.info("Type a company name and select a valid ticker from the dropdown.")

elif tab_idx == 1: # Suggestions
    st.header("AI Suggestions & News")
    # Example sectors and companies (expand as needed)
    sectors = {
        "AI": ["NVDA", "AMD", "SMCI", "GOOGL", "BIDU", "TSM"],
        "Nuclear": ["BWXT", "SMR", "CAMEF", "U", "PDN.TO", "CCJ"],
        "Tech": ["AAPL", "MSFT", "AVGO", "AMZN", "TSLA", "SHOP.TO", "SAP.DE", "SONY"],
        "Healthcare": ["JNJ", "PFE", "MRK", "RHHBY", "UNH", "BNTX", "XBI"],
        "Defense": ["LMT", "NOC", "RTX", "BA", "TDG", "BBD-B.TO"],
        "Pharma": ["MRNA", "AZN", "NVS", "GILD", "SNY"],
        "Canada": ["TD.TO", "BNS.TO", "ENB.TO", "SHOP.TO", "BCE.TO"]
    }
    for sec, tickers in sectors.items():
        st.subheader(f"{sec} Sector")
        for t in tickers:
            info = get_company_info(t)
            n = info.get("shortName") or info.get("longName") or t
            st.markdown(f"**{n} ({t})**")
            news = news_section(t, n)
            for url, title in news:
                st.write("-", f"[{title}]({url})")
        st.markdown("---")
    st.subheader("Market-Wide News")
    for url, title in google_news("stock market") + yahoo_news("stock market"):
        st.write("-", f"[{title}]({url})")

elif tab_idx == 2: # Peer Comparison
    st.header("Peer Comparison")
    st.caption("Type company name or ticker (Name or Ticker):")
    query = st.text_input("", key="peer_query")
    choices = yahoo_autocomplete(query)
    symbol = ""
    if choices:
        labels = [f"{c['name']} ({c['symbol']}, {c['exchange']})" for c in choices]
        choice = st.selectbox("Select:", labels, key="peer_symbol_select")
        i = labels.index(choice)
        symbol = choices[i]["symbol"]
        name = choices[i]["name"]
    else:
        symbol = query.strip().upper()
        name = ""
    if symbol:
        info = get_company_info(symbol)
        peers = info.get("sector") or ""
        # Simple example, real implementation may require better peer lookup
        st.write(f"Peers for {symbol} in {peers or 'N/A'} sector: (coming soon)")

elif tab_idx == 3: # Watchlist
    st.header("Watchlist")
    watchlist = st.session_state.get("watchlist", [])
    addq = st.text_input("Add company (Name or Ticker):", key="add_watch")
    if st.button("Add to Watchlist") and addq:
        s, n, _ = resolve_symbol(addq)
        if s and s not in watchlist:
            watchlist.append(s)
            st.session_state["watchlist"] = watchlist
    if watchlist:
        for w in watchlist:
            info = get_company_info(w)
            n = info.get("shortName") or info.get("longName") or w
            col1, col2 = st.columns([4,1])
            with col1:
                st.markdown(f"**{n} ({w})**")
            with col2:
                if st.button("Remove", key=f"rm_{w}"):
                    watchlist.remove(w)
                    st.session_state["watchlist"] = watchlist
        st.session_state["watchlist"] = watchlist
    else:
        st.info("No companies in your watchlist.")

elif tab_idx == 4: # Financial Health
    st.header("Financial Health")
    st.caption("Select company (Name or Ticker):")
    query = st.text_input("", key="fh_query")
    choices = yahoo_autocomplete(query)
    symbol = ""
    if choices:
        labels = [f"{c['name']} ({c['symbol']}, {c['exchange']})" for c in choices]
        choice = st.selectbox("Select:", labels, key="fh_symbol_select")
        i = labels.index(choice)
        symbol = choices[i]["symbol"]
        name = choices[i]["name"]
    else:
        symbol = query.strip().upper()
        name = ""
    if symbol:
        info = get_company_info(symbol)
        ratios = {
            "Market Cap": info.get("marketCap"),
            "PE Ratio": info.get("trailingPE"),
            "EPS": info.get("trailingEps"),
            "ROE": info.get("returnOnEquity"),
            "Debt/Equity": info.get("debtToEquity"),
            "Dividend Yield": info.get("dividendYield")
        }
        st.write({k: v for k, v in ratios.items() if v is not None})
    else:
        st.info("Select a company.")

elif tab_idx == 5: # Options Analytics
    st.header("Options Analytics")
    st.write("Type ticker (US stocks only):")
    query = st.text_input("", key="opt_query")
    symbol, name, _ = resolve_symbol(query)
    if symbol:
        try:
            ticker = yf.Ticker(symbol)
            opts = ticker.options
            st.write(f"Options available: {opts}")
        except Exception:
            st.warning("Options data only for major US stocks.")

elif tab_idx == 6: # Macro Insights
    st.header("Macro Insights")
    st.write("Global macro indicators coming soon.")

elif tab_idx == 7: # News Explorer
    st.header("News Explorer")
    q = st.text_input("Search news for:", key="newsq")
    if q:
        st.subheader("Google News")
        for url, title in google_news(q, count=10):
            st.write("-", f"[{title}]({url})")
        st.subheader("Yahoo Finance News")
        for url, title in yahoo_news(q, count=10):
            st.write("-", f"[{title}]({url})")

# ------ The rest of the tabs ("Peer Comparison", "Watchlist", "Financial Health", etc.) are implemented as modular enhancements ------
# ------ You can add, comment out, or move them based on your needs. ------
