import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import feedparser
from datetime import datetime
from prophet import Prophet
from bs4 import BeautifulSoup

st.set_page_config(page_title="AI-Powered Market Analyzer", layout="wide")

def format_large_number(n):
    if pd.isnull(n):
        return "N/A"
    abs_n = abs(n)
    if abs_n >= 1_000_000_000:
        return f"{n/1_000_000_000:.2f}B"
    elif abs_n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    elif abs_n >= 1_000:
        return f"{n/1_000:.2f}K"
    return f"{n:.2f}"

def get_company_name(ticker):
    try:
        info = yf.Ticker(ticker).info
        return info.get("longName") or info.get("shortName") or ticker
    except Exception:
        return ticker

def get_peers(ticker):
    # Dummy logic for peer lookup, replace with sector-based lookup for your prod
    sector_map = {
        "AAPL": ["MSFT", "GOOGL", "AMZN"],
        "MSFT": ["AAPL", "GOOGL", "AMZN"],
        "GOOGL": ["AAPL", "MSFT", "AMZN"],
        "NVDA": ["AMD", "QCOM", "AVGO"],
    }
    return sector_map.get(ticker.upper(), ["MSFT", "AAPL"])

def get_peer_comparison(ticker):
    # Get the target and peers
    peers = [ticker.upper()] + get_peers(ticker)
    data = []
    for t in peers:
        try:
            info = yf.Ticker(t).info
            name = info.get("shortName") or t
            market_cap = info.get("marketCap")
            pe = info.get("trailingPE")
            pb = info.get("priceToBook")
            div = info.get("dividendYield")
            data.append({
                "Ticker": t,
                "Company": name,
                "Market Cap": format_large_number(market_cap),
                "P/E": f"{pe:.2f}" if pe else "N/A",
                "P/B": f"{pb:.2f}" if pb else "N/A",
                "Dividend Yield": f"{100*div:.2f}%" if div else "N/A",
            })
        except Exception:
            data.append({"Ticker": t, "Company": "N/A", "Market Cap": "N/A", "P/E": "N/A", "P/B": "N/A", "Dividend Yield": "N/A"})
    return pd.DataFrame(data)

def get_news(ticker):
    try:
        url = f"https://news.google.com/search?q={ticker}+stock"
        html = requests.get(url).text
        soup = BeautifulSoup(html, 'html.parser')
        return [(item.text, "https://news.google.com" + item['href'][1:]) for item in soup.select('article h3 a')][:5]
    except:
        return [("No news found.", "")]

def get_fda_approvals():
    try:
        rss_url = "https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/press-releases/rss.xml"
        feed = feedparser.parse(rss_url)
        return [(entry.title, entry.link) for entry in feed.entries[:5]]
    except:
        return [("No FDA data available", "")]

# -------- UI Navigation ---------
PAGES = [
    "Market Information", "Suggestions", "Peer Comparison",
    "Watchlist", "Financial Health", "Options Analytics", "Macro Insights", "News Explorer"
]
st.sidebar.title("Navigation")
page = st.sidebar.radio("", PAGES)

# -------- Market Information Tab (unchanged, only format numbers) ---------
if page == "Market Information":
    st.title("Market Information")
    st.caption("Enter any ticker (Stock, ETF, Crypto, FOREX, Commodity):")
    ticker = st.text_input("", "AAPL")
    if ticker:
        df = yf.download(ticker, period="1y", auto_adjust=True, progress=False)
        info = yf.Ticker(ticker).info
        name = get_company_name(ticker)
        st.header(f"{name} ({ticker.upper()})")
        if not df.empty:
            st.metric("Current Price", f"${df['Close'].iloc[-1]:,.2f}")
            st.metric("Market Cap", format_large_number(info.get("marketCap")))
            st.metric("P/E Ratio", f"{info.get('trailingPE'):.3f}" if info.get("trailingPE") else "N/A")
        else:
            st.warning("No market data found for this ticker.")
        st.subheader("News Sentiment")
        for n, url in get_news(ticker):
            if url:
                st.markdown(f"- [{n}]({url})")
            else:
                st.write("-", n)

# -------- Suggestions Tab (do not change) ---------
elif page == "Suggestions":
    st.title("AI Company Suggestions")
    st.write("Latest trending and high-impact companies across sectors...")
    # (rest of your robust, previously frozen Suggestions tab code goes here)

# -------- Peer Comparison Tab (full patch) ---------
elif page == "Peer Comparison":
    st.title("Peer Comparison")
    st.caption("Enter Ticker for Peer Comparison:")
    peer_ticker = st.text_input("", "AAPL", key="peer_comp_ticker")
    if peer_ticker:
        try:
            comp_df = get_peer_comparison(peer_ticker)
            st.dataframe(comp_df)
            st.info("Compare key metrics: Market Cap, P/E, P/B, Dividend Yield")
        except Exception as e:
            st.error(f"Error: {e}")

# -------- Watchlist Tab ---------
elif page == "Watchlist":
    st.title("Watchlist")
    st.write("Add tickers to your watchlist and monitor them here.")
    # (Implement local/session storage, or just let user type)
    user_watchlist = st.text_area("Enter comma-separated tickers:", "AAPL,MSFT,GOOGL")
    tickers = [t.strip().upper() for t in user_watchlist.split(",") if t.strip()]
    if tickers:
        data = []
        for t in tickers:
            info = yf.Ticker(t).info
            data.append({
                "Ticker": t,
                "Price": info.get("currentPrice", "N/A"),
                "Market Cap": format_large_number(info.get("marketCap"))
            })
        st.dataframe(pd.DataFrame(data))

# -------- Financial Health Tab ---------
elif page == "Financial Health":
    st.title("Financial Health Dashboard")
    st.write("View financial ratios and balance sheet highlights for any company.")
    ticker = st.text_input("Enter ticker:", "AAPL", key="fh_ticker")
    if ticker:
        try:
            info = yf.Ticker(ticker).info
            st.metric("Current Ratio", f"{info.get('currentRatio', 'N/A')}")
            st.metric("Debt/Equity", f"{info.get('debtToEquity', 'N/A')}")
            st.metric("Return on Equity", f"{info.get('returnOnEquity', 'N/A')}")
        except Exception as e:
            st.error(f"Error: {e}")

# -------- Options Analytics Tab ---------
elif page == "Options Analytics":
    st.title("Options Analytics")
    st.write("Options prices and analytics for a given ticker.")
    ticker = st.text_input("Enter ticker for options analytics:", "AAPL", key="opt_ticker")
    if ticker:
        try:
            tk = yf.Ticker(ticker)
            opts = tk.options
            st.write("Available expiration dates:", opts)
            if opts:
                expiry = st.selectbox("Select expiration date:", opts)
                chain = tk.option_chain(expiry)
                st.write("Calls", chain.calls.head())
                st.write("Puts", chain.puts.head())
        except Exception as e:
            st.error(f"Error: {e}")

# -------- Macro Insights Tab ---------
elif page == "Macro Insights":
    st.title("Macro Insights")
    st.write("Economic indicators, trends, and more. Coming soon!")
    # For future expansion: add FRED, World Bank data

# -------- News Explorer Tab ---------
elif page == "News Explorer":
    st.title("News Explorer")
    st.write("Explore the latest financial news from multiple sources:")
    tick = st.text_input("Enter ticker or keyword for news:", "AAPL", key="news_ticker")
    if tick:
        for n, url in get_news(tick):
            if url:
                st.markdown(f"- [{n}]({url})")
            else:
                st.write("-", n)
    st.subheader("FDA Drug Approval News")
    for n, url in get_fda_approvals():
        if url:
            st.markdown(f"- [{n}]({url})")
        else:
            st.write("-", n)

# ---- End of script ----
